/**
 * Pi extension: the `video` tool over the agent-video binary.
 *
 * A thin MCP stdio client, not an MCP framework: the extension lazily spawns
 * the platform binary's `mcp` subcommand once per session (consent tokens and
 * background downloads are in-process state there), speaks newline-delimited
 * JSON-RPC to it, and maps content blocks straight through — pi's
 * ToolResultMessage uses the same `{ type: "image", data, mimeType }` shape.
 *
 * Binary resolution: `AGENT_VIDEO_BIN` env override, then the exact-pinned
 * `@agent-video/<platform>` package installed via this package's dependency
 * on `@agent-video/cli`. No PATH fallback by design.
 */

import { type ChildProcess, spawn } from "node:child_process";
import { createRequire } from "node:module";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { PARAMETERS, PROMPT_SNIPPET, TOOL_DESCRIPTION } from "../lib/schema.ts";

const require = createRequire(import.meta.url);

const RELEASES_URL = "https://github.com/durvia/agent-video/releases";

/** Bound on the initialize handshake. Generous — startup is local process
 * spawn plus one JSON round-trip — so it only fires on a binary that wedges
 * before speaking MCP, which would otherwise block every video call forever
 * (the abort signal covers tool-call waiters, not the shared handshake). */
const INIT_TIMEOUT_MS = 30_000;

function resolveBinary(): string {
	const override = process.env["AGENT_VIDEO_BIN"];
	if (override) return override;
	const pkg = `@agent-video/${process.platform}-${process.arch}`;
	const bin = process.platform === "win32" ? "agent-video.exe" : "agent-video";
	try {
		return require.resolve(`${pkg}/bin/${bin}`);
	} catch {
		throw new Error(
			`agent-video core binary not found: ${pkg} is not installed. ` +
				`Your platform may be unsupported by the npm distribution, or optional ` +
				`dependencies were skipped at install time. Set AGENT_VIDEO_BIN to a ` +
				`binary from ${RELEASES_URL} as a manual override.`,
		);
	}
}

// ── minimal MCP stdio client ──

interface McpContentBlock {
	type: string;
	text?: string;
	data?: string;
	mimeType?: string;
}

interface McpCallResult {
	content: McpContentBlock[];
	isError?: boolean;
}

interface Pending {
	resolve: (result: McpCallResult) => void;
	reject: (error: Error) => void;
	onProgress?: (message: string) => void;
}

/** One MCP server child per pi session, restarted on death at the next call. */
class VideoServer {
	private child: ChildProcess | undefined;
	private ready: Promise<void> | undefined;
	private pending = new Map<number, Pending>();
	private nextId = 2; // 1 is the initialize request

	/** Call the `video` tool; progress notifications stream to `onProgress`. */
	async call(
		args: Record<string, unknown>,
		onProgress: ((message: string) => void) | undefined,
		signal: AbortSignal | undefined,
	): Promise<McpCallResult> {
		await this.start();
		const id = this.nextId++;
		const result = new Promise<McpCallResult>((resolve, reject) => {
			this.pending.set(id, { resolve, reject, onProgress });
			this.send({
				jsonrpc: "2.0",
				id,
				method: "tools/call",
				params: { name: "video", arguments: args, _meta: { progressToken: id } },
			});
		});
		// On abort the server keeps working (downloads stay pollable; a late
		// response finds no waiter and is dropped) — the waiter itself is
		// resolved with the cancel notice, so no promise is left dangling.
		signal?.addEventListener(
			"abort",
			() => {
				const waiter = this.pending.get(id);
				if (waiter === undefined) return; // already settled
				this.pending.delete(id);
				waiter.resolve({
					content: [
						{
							type: "text",
							text: "call cancelled — the server keeps working; repeat the same source to pick up the result",
						},
					],
				});
			},
			{ once: true },
		);
		return result;
	}

	/** Spawn + initialize once; concurrent callers share the handshake. */
	private start(): Promise<void> {
		if (this.child !== undefined && this.ready !== undefined) return this.ready;
		const child = spawn(resolveBinary(), ["mcp"], {
			stdio: ["pipe", "pipe", "pipe"],
		});
		this.child = child;
		this.wireStdout(child);
		// The server never logs to stderr, but crashes do: linker errors,
		// panic backtraces, signal messages. Keep a bounded tail so the exit
		// error can say why instead of just that.
		let stderrTail = "";
		child.stderr?.on("data", (chunk: Buffer) => {
			stderrTail = (stderrTail + chunk.toString("utf8")).slice(-4096);
		});
		child.stderr?.on("error", () => {});
		child.once("exit", (code, signal) => {
			if (this.child !== child) return; // superseded; a respawn owns the state now
			const tail = stderrTail.trim();
			const why = tail === "" ? "" : `\nstderr tail:\n${tail}`;
			this.fail(new Error(`agent-video mcp exited (code ${code}, signal ${signal})${why}`));
		});
		child.once("error", (e) => {
			if (this.child !== child) return;
			this.fail(new Error(`agent-video mcp failed: ${e.message}`));
		});
		// A write into the window between server death and the 'exit' event
		// surfaces as a stream error; without a listener Node throws it as an
		// uncaught exception into the host. The 'exit' handler owns recovery.
		child.stdin?.on("error", () => {});
		child.stdout?.on("error", () => {});

		this.ready = new Promise<void>((resolve, reject) => {
			// A binary that hangs at startup must not pin the first call (and
			// every call behind it) forever: kill it and fail the handshake;
			// `fail` clears the state, so the next call respawns cleanly.
			const deadline = setTimeout(() => {
				child.kill();
				this.fail(
					new Error(
						`agent-video mcp did not answer initialize within ${INIT_TIMEOUT_MS / 1000}s — killed; the next call restarts it`,
					),
				);
			}, INIT_TIMEOUT_MS);
			this.pending.set(1, {
				resolve: () => {
					clearTimeout(deadline);
					this.send({ jsonrpc: "2.0", method: "notifications/initialized" });
					resolve();
				},
				reject: (error) => {
					clearTimeout(deadline);
					reject(error);
				},
			});
			this.send({
				jsonrpc: "2.0",
				id: 1,
				method: "initialize",
				params: {
					protocolVersion: "2024-11-05",
					capabilities: {},
					clientInfo: { name: "pi-agent-video", version: "0" },
				},
			});
		});
		return this.ready;
	}

	/** Reject all in-flight calls and forget the child; next call respawns. */
	private fail(error: Error): void {
		for (const p of this.pending.values()) p.reject(error);
		this.pending.clear();
		this.child = undefined;
		this.ready = undefined;
	}

	private send(message: unknown): void {
		this.child?.stdin?.write(`${JSON.stringify(message)}\n`);
	}

	private wireStdout(child: ChildProcess): void {
		// Incremental line splitting: `scanned` marks the newline-free prefix
		// already searched, so a response arriving as one multi-megabyte line
		// (grids ride base64-inline) is scanned once, not re-scanned from the
		// start on every chunk.
		let buffer = "";
		let scanned = 0;
		child.stdout?.on("data", (chunk: Buffer) => {
			buffer += chunk.toString("utf8");
			let start = 0;
			let newline = buffer.indexOf("\n", scanned);
			while (newline !== -1) {
				const line = buffer.slice(start, newline);
				if (line.trim() !== "") this.dispatch(line);
				start = newline + 1;
				newline = buffer.indexOf("\n", start);
			}
			if (start > 0) buffer = buffer.slice(start);
			scanned = buffer.length;
		});
	}

	private dispatch(line: string): void {
		let message: {
			id?: number;
			method?: string;
			result?: unknown;
			error?: { message?: string };
			params?: { progressToken?: number; message?: string };
		};
		try {
			message = JSON.parse(line);
		} catch {
			return; // not ours to understand; skip
		}
		if (message.method === "notifications/progress") {
			const token = message.params?.progressToken;
			const text = message.params?.message;
			if (token !== undefined && text !== undefined) {
				this.pending.get(token)?.onProgress?.(text);
			}
			return;
		}
		if (message.id === undefined) return;
		const waiter = this.pending.get(message.id);
		if (waiter === undefined) return; // cancelled or unknown; drop
		this.pending.delete(message.id);
		if (message.error !== undefined) {
			waiter.reject(new Error(message.error.message ?? "MCP request failed"));
		} else {
			waiter.resolve((message.result ?? { content: [] }) as McpCallResult);
		}
	}
}

// ── tool registration ──

type Text = { type: "text"; text: string };
type Image = { type: "image"; data: string; mimeType: string };
type Block = Text | Image;

function toBlocks(content: McpContentBlock[]): Block[] {
	const blocks: Block[] = [];
	for (const c of content) {
		if (c.type === "text" && c.text !== undefined) {
			blocks.push({ type: "text", text: c.text });
		} else if (c.type === "image" && c.data !== undefined) {
			blocks.push({ type: "image", data: c.data, mimeType: c.mimeType ?? "image/jpeg" });
		}
	}
	return blocks;
}

export default function (pi: ExtensionAPI) {
	const server = new VideoServer();
	pi.registerTool({
		name: "video",
		label: "Video",
		description: TOOL_DESCRIPTION,
		promptSnippet: PROMPT_SNIPPET,
		parameters: PARAMETERS,
		async execute(_toolCallId, params, signal, onUpdate) {
			const result = await server.call(
				params as Record<string, unknown>,
				(message) => onUpdate?.({ content: [{ type: "text", text: message }] }),
				signal,
			);
			const content = toBlocks(result.content);
			if (result.isError === true) {
				// pi marks failure by throw; unwrap the Failed body so the model
				// sees the message, not an envelope.
				const text = content.find((b): b is Text => b.type === "text")?.text ?? "video call failed";
				let message = text;
				try {
					const body = JSON.parse(text) as { error?: string };
					if (typeof body.error === "string") message = body.error;
				} catch {
					// not JSON — use the raw text
				}
				throw new Error(message);
			}
			return { content, details: {} };
		},
	});
}
