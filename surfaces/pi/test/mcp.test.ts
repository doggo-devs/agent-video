/**
 * Behavior tests for the pi video tool's MCP client (../lib/mcp.ts), driven
 * through an in-memory fake of the spawned server: real newline-delimited
 * JSON-RPC over streams, no child process. Covers the process-boundary
 * contract — parameter narrowing, consent-token flow riding one child,
 * failure unwrapping with fault-kind fidelity, byte-level framing, and
 * malformed responses failing diagnosably.
 */

import { describe, expect, test } from "bun:test";
import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";
import {
	failureMessage,
	type McpChild,
	parseCallResult,
	runVideoCall,
	VideoServer,
} from "../lib/mcp.ts";

/** In-memory stand-in for the spawned MCP server process. */
class FakeChild extends EventEmitter implements McpChild {
	stdin = new PassThrough();
	stdout = new PassThrough();
	stderr = new PassThrough();
	kill(): void {
		this.emit("exit", null, "SIGTERM");
	}
}

/** A scripted reply: a raw `result` value, or a JSON-RPC level error. */
class RpcError {
	constructor(readonly message: string) {}
}

type Reply = (
	call: Call,
	progress: (message: string) => void,
) => unknown | RpcError;

interface Call {
	tool: string;
	args: Record<string, unknown>;
}

/** Speaks the server side of the wire: answers initialize, records
 * tools/call requests, and replies with whatever the script returns.
 * `chunkBytes` splits every outgoing message into chunks of that many
 * bytes, as a pipe may. A reply promise can hold a call in flight. */
class FakeServer {
	readonly child = new FakeChild();
	readonly calls: Call[] = [];
	spawns = 0;

	constructor(
		private readonly reply: Reply,
		private readonly chunkBytes = Infinity,
	) {
		let buffer = "";
		this.child.stdin.on("data", (chunk: Buffer) => {
			buffer += chunk.toString("utf8");
			let newline = buffer.indexOf("\n");
			while (newline !== -1) {
				const line = buffer.slice(0, newline);
				buffer = buffer.slice(newline + 1);
				if (line.trim() !== "") void this.handle(JSON.parse(line));
				newline = buffer.indexOf("\n");
			}
		});
	}

	spawn = (): McpChild => {
		this.spawns += 1;
		return this.child;
	};

	private send(message: unknown): void {
		const bytes = Buffer.from(`${JSON.stringify(message)}\n`, "utf8");
		for (let at = 0; at < bytes.length; at += this.chunkBytes) {
			this.child.stdout.write(bytes.subarray(at, at + this.chunkBytes));
		}
	}

	private async handle(message: {
		id?: number;
		method?: string;
		params?: {
			name?: string;
			arguments?: Record<string, unknown>;
			_meta?: { progressToken?: number };
		};
	}): Promise<void> {
		if (message.method === "initialize") {
			this.send({ jsonrpc: "2.0", id: message.id, result: {} });
		} else if (message.method === "tools/call") {
			const call = { tool: message.params?.name ?? "", args: message.params?.arguments ?? {} };
			this.calls.push(call);
			const token = message.params?._meta?.progressToken;
			const progress = (text: string) =>
				this.send({
					jsonrpc: "2.0",
					method: "notifications/progress",
					params: { progressToken: token, message: text },
				});
			const reply = await this.reply(call, progress);
			if (reply instanceof RpcError) {
				this.send({
					jsonrpc: "2.0",
					id: message.id,
					error: { code: -32600, message: reply.message },
				});
			} else {
				this.send({ jsonrpc: "2.0", id: message.id, result: reply });
			}
		}
	}
}

function textResult(text: string, isError = false): unknown {
	return { content: [{ type: "text", text }], isError };
}

function failed(error: string, kind: string): unknown {
	return textResult(JSON.stringify({ status: "failed", error, kind }), true);
}

function run(fake: FakeServer, params: unknown): Promise<unknown> {
	return runVideoCall(new VideoServer(fake.spawn), "video", params, undefined, undefined);
}

describe("failure unwrapping", () => {
	test("preserves_fault_kind_on_failure", async () => {
		const fake = new FakeServer(() => failed("[INPUT_CORRUPT] bad moov", "bad_input"));
		await expect(run(fake, { source: "v.mp4" })).rejects.toThrow(
			"[INPUT_CORRUPT] bad moov (kind: bad_input — the source or arguments are wrong; " +
				"retrying the identical call cannot succeed)",
		);
	});

	test("marks_transient_failure_retriable", async () => {
		const fake = new FakeServer(() => failed("fetch stalled", "transient"));
		await expect(run(fake, { source: "http://v" })).rejects.toThrow(
			"fetch stalled (kind: transient — plausibly a passing condition; " +
				"retrying the same call may succeed)",
		);
	});

	test("surfaces_unknown_kind_verbatim", () => {
		const body = JSON.stringify({ status: "failed", error: "boom", kind: "flaky_moon" });
		expect(failureMessage(body)).toBe("boom (kind: flaky_moon)");
	});

	test("keeps_error_prose_when_kind_missing", () => {
		expect(failureMessage(JSON.stringify({ error: "just prose" }))).toBe("just prose");
	});

	test("falls_back_to_raw_text_when_failure_is_not_json", async () => {
		const fake = new FakeServer(() => textResult("plain disaster", true));
		await expect(run(fake, { source: "v.mp4" })).rejects.toThrow("plain disaster");
	});
});

describe("boundary validation", () => {
	test("rejects_non_object_params_before_sending", async () => {
		const fake = new FakeServer(() => textResult("unreachable"));
		await expect(run(fake, "v.mp4")).rejects.toThrow("tool parameters must be an object");
		expect(fake.spawns).toBe(0);
		expect(fake.calls).toHaveLength(0);
	});

	test("rejects_params_without_source", async () => {
		const fake = new FakeServer(() => textResult("unreachable"));
		await expect(run(fake, { from: 3 })).rejects.toThrow('"source" must be a string');
		expect(fake.calls).toHaveLength(0);
	});

	test("diagnoses_result_missing_content", async () => {
		const fake = new FakeServer(() => ({ ok: true }));
		await expect(run(fake, { source: "v.mp4" })).rejects.toThrow(
			'malformed MCP result: "content" is not an array',
		);
	});

	test("diagnoses_non_object_result", () => {
		expect(() => parseCallResult(null)).toThrow("malformed MCP result: expected an object");
	});

	test("diagnoses_block_without_type", () => {
		expect(() => parseCallResult({ content: [{ text: "x" }] })).toThrow(
			'content[0] lacks a string "type"',
		);
	});

	test("diagnoses_non_string_block_field", () => {
		expect(() => parseCallResult({ content: [{ type: "text", text: 42 }] })).toThrow(
			"content[0].text is not a string",
		);
	});

	test("diagnoses_non_boolean_is_error", () => {
		expect(() => parseCallResult({ content: [], isError: "yes" })).toThrow(
			'"isError" is not a boolean',
		);
	});

	test("passes_valid_result_through_unchanged", () => {
		const result = parseCallResult({
			content: [{ type: "text", text: "hi" }],
			isError: false,
		});
		expect(result).toEqual({ content: [{ type: "text", text: "hi" }], isError: false });
	});
});

describe("consent token flow", () => {
	test("reuses_one_server_so_consent_tokens_stay_valid", async () => {
		const fake = new FakeServer(({ tool }) =>
			tool === "video_confirm"
				? textResult("download started")
				: textResult(
						JSON.stringify({
							status: "confirm_required",
							consequence: "downloads from the network",
							tokens: [{ token: "tok-1" }],
						}),
					),
		);
		const server = new VideoServer(fake.spawn);

		const offer = await runVideoCall(
			server,
			"video",
			{ source: "http://v" },
			undefined,
			undefined,
		);
		expect(offer[0]?.type).toBe("text");
		expect(offer[0]?.type === "text" ? offer[0].text : "").toContain("tok-1");

		const confirmed = await runVideoCall(
			server,
			"video_confirm",
			{ source: "http://v", token: "tok-1" },
			undefined,
			undefined,
		);
		expect(confirmed).toEqual([{ type: "text", text: "download started" }]);
		expect(fake.spawns).toBe(1); // same child: in-process tokens survive the hop
		expect(fake.calls[1]).toEqual({
			tool: "video_confirm",
			args: { source: "http://v", token: "tok-1" },
		});
	});
});

describe("call mechanics", () => {
	test("maps_image_blocks_with_default_mime_type", async () => {
		const fake = new FakeServer(() => ({
			content: [
				{ type: "text", text: "grid 1" },
				{ type: "image", data: "b64" },
			],
		}));
		const blocks = await run(fake, { source: "v.mp4" });
		expect(blocks).toEqual([
			{ type: "text", text: "grid 1" },
			{ type: "image", data: "b64", mimeType: "image/jpeg" },
		]);
	});

	test("streams_progress_notifications_to_host", async () => {
		const fake = new FakeServer((_args, progress) => {
			progress("downloading 10%");
			return textResult("done");
		});
		const seen: string[] = [];
		const blocks = await runVideoCall(
			new VideoServer(fake.spawn),
			"video",
			{ source: "http://v" },
			(message) => seen.push(message),
			undefined,
		);
		expect(blocks).toEqual([{ type: "text", text: "done" }]);
		expect(seen).toEqual(["downloading 10%"]);
	});

	test("returns_cancel_notice_without_sending_when_pre_aborted", async () => {
		const fake = new FakeServer(() => textResult("unreachable"));
		const aborted = AbortSignal.abort();
		const blocks = await runVideoCall(
			new VideoServer(fake.spawn),
			"video",
			{ source: "v.mp4" },
			undefined,
			aborted,
		);
		expect(blocks[0]?.type === "text" ? blocks[0].text : "").toContain("cancelled");
		expect(fake.spawns).toBe(0);
		expect(fake.calls).toHaveLength(0);
	});

	test("decodes_text_split_mid_character_across_chunks", async () => {
		const fake = new FakeServer(() => textResult("中文😀 transcript"), 1);
		const blocks = await run(fake, { source: "v.mp4" });
		expect(blocks).toEqual([{ type: "text", text: "中文😀 transcript" }]);
	});

	test("receives_a_large_image_response_across_chunks", async () => {
		const data = "A".repeat(2 * 1024 * 1024);
		const fake = new FakeServer(() => ({ content: [{ type: "image", data }] }), 4096);

		expect(await run(fake, { source: "v.mp4" })).toEqual([
			{ type: "image", data, mimeType: "image/jpeg" },
		]);
	});

	test("cancels_one_waiter_without_closing_the_shared_reader", async () => {
		const started = Promise.withResolvers<void>();
		const reply = Promise.withResolvers<unknown>();
		const fake = new FakeServer(({ args }) => {
			if (args["source"] === "slow.mp4") {
				started.resolve();
				return reply.promise;
			}
			return textResult("done");
		});
		const server = new VideoServer(fake.spawn);
		const controller = new AbortController();
		const first = server.call("video", { source: "slow.mp4" }, undefined, controller.signal);
		await started.promise;
		const second = server.call("video", { source: "other.mp4" }, undefined, undefined);

		controller.abort();
		expect((await first).content[0]?.text).toContain("cancelled");
		expect(await second).toEqual(textResult("done"));
		reply.resolve(textResult("late reply"));
		expect(await server.call("video", { source: "next.mp4" }, undefined, undefined)).toEqual(
			textResult("done"),
		);
		expect(fake.spawns).toBe(1);
	});

	test.each(["exit", "error"] as const)("rejects_pending_calls_and_restarts_after_%s", async (event) => {
		const started = Promise.withResolvers<void>();
		const reply = Promise.withResolvers<unknown>();
		const first = new FakeServer(() => {
			started.resolve();
			return reply.promise;
		});
		const next = new FakeServer(() => textResult("restarted"));
		let spawns = 0;
		const server = new VideoServer(() => (spawns++ === 0 ? first.spawn() : next.spawn()));
		const pending = server.call("video", { source: "v.mp4" }, undefined, undefined).catch(
			(error: unknown) => error,
		);
		await started.promise;

		if (event === "exit") first.child.kill();
		else first.child.emit("error", new Error("spawn failed"));
		expect(await pending).toMatchObject({
			message: expect.stringContaining(event === "exit" ? "exited" : "failed"),
		});
		reply.resolve(textResult("late reply"));
		expect(await server.call("video", { source: "v.mp4" }, undefined, undefined)).toEqual(
			textResult("restarted"),
		);
		expect(spawns).toBe(2);
	});

	test("rejects_with_rpc_error_message", async () => {
		const fake = new FakeServer(() => new RpcError("no such tool"));
		await expect(run(fake, { source: "v.mp4" })).rejects.toThrow("no such tool");
	});
});
