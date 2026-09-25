/**
 * The MCP stdio client behind the pi `video` tool.
 *
 * Everything that crosses the process boundary lives here, free of pi and
 * schema imports so tests can drive it with an in-memory fake child. The
 * boundary rule: bytes from the server are validated, never asserted — a
 * malformed response becomes a diagnosable Error naming what was wrong,
 * not a TypeError three frames later. The server's Failed body carries a
 * machine-readable fault `kind` (crates/harness/src/schema.rs); this module
 * keeps it attached to the thrown message so retry decisions survive the
 * hop into the host.
 */

/** Bound on the initialize handshake. Generous — startup is local process
 * spawn plus one JSON round-trip — so it only fires on a binary that wedges
 * before speaking MCP, which would otherwise block every video call forever
 * (the abort signal covers tool-call waiters, not the shared handshake). */
const INIT_TIMEOUT_MS = 30_000;

// ── wire shapes ──

export interface McpContentBlock {
	type: string;
	text?: string;
	data?: string;
	mimeType?: string;
}

export interface McpCallResult {
	content: McpContentBlock[];
	isError?: boolean;
}

/** The slice of a spawned MCP server process this client touches —
 * structurally satisfied by `node:child_process.ChildProcess`, small
 * enough for tests to fake in memory. */
export interface McpChild {
	stdin: NodeJS.WritableStream | null;
	stdout: NodeJS.ReadableStream | null;
	stderr: NodeJS.ReadableStream | null;
	kill(): void;
	once(event: "exit", listener: (code: number | null, signal: string | null) => void): void;
	once(event: "error", listener: (error: Error) => void): void;
}

// ── boundary validation ──

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** A short, safe rendering of a foreign value for error messages. */
function render(value: unknown): string {
	const json = JSON.stringify(value);
	if (json === undefined) return typeof value;
	return json.length > 200 ? `${json.slice(0, 200)}…` : json;
}

/** Narrow a raw `tools/call` result from the wire. The server is a separate
 * process: its output is a claim, not a type. Rejections name the offending
 * field so a protocol bug is diagnosable from the error alone. */
export function parseCallResult(value: unknown): McpCallResult {
	if (!isRecord(value)) {
		throw new Error(`malformed MCP result: expected an object, got ${render(value)}`);
	}
	const content = value["content"];
	if (!Array.isArray(content)) {
		throw new Error(`malformed MCP result: "content" is not an array in ${render(value)}`);
	}
	const blocks: McpContentBlock[] = [];
	for (const [i, item] of content.entries()) {
		if (!isRecord(item) || typeof item["type"] !== "string") {
			throw new Error(`malformed MCP result: content[${i}] lacks a string "type": ${render(item)}`);
		}
		const block: McpContentBlock = { type: item["type"] };
		for (const key of ["text", "data", "mimeType"] as const) {
			const field = item[key];
			if (field === undefined) continue;
			if (typeof field !== "string") {
				throw new Error(`malformed MCP result: content[${i}].${key} is not a string`);
			}
			block[key] = field;
		}
		blocks.push(block);
	}
	const isError = value["isError"];
	if (isError !== undefined && typeof isError !== "boolean") {
		throw new Error(`malformed MCP result: "isError" is not a boolean: ${render(isError)}`);
	}
	return isError === undefined ? { content: blocks } : { content: blocks, isError };
}

/** Narrow tool parameters at the host boundary. pi hands them over shaped
 * by the schema, but this module never assumes the host validated; deeper
 * argument checking stays the server's job (single source of truth). */
export function parseArgs(tool: string, params: unknown): Record<string, unknown> {
	if (!isRecord(params)) {
		throw new Error(`${tool}: tool parameters must be an object, got ${render(params)}`);
	}
	if (typeof params["source"] !== "string") {
		throw new Error(`${tool}: "source" must be a string, got ${render(params["source"])}`);
	}
	return params;
}

// ── failure unwrapping ──

/** Retry guidance per fault kind, mirroring the contract of
 * `VideoBody::Failed` (crates/harness/src/schema.rs): each kind names who
 * can fix the failure and whether retrying can possibly succeed. */
const KIND_GUIDANCE: Record<string, string | undefined> = {
	bad_input: "the source or arguments are wrong; retrying the identical call cannot succeed",
	environment: "this installation lacks something — tell the user; retrying cannot help",
	transient: "plausibly a passing condition; retrying the same call may succeed",
	internal: "an agent-video bug; retrying cannot help",
};

/** Unwrap a Failed body into the message pi throws: the prose plus the
 * fault kind and its retry guidance, so the model can tell a retriable
 * transient from a hopeless bad_input. Unknown kinds pass through verbatim
 * (forward compatibility beats a lie); non-JSON text is already prose. */
export function failureMessage(text: string): string {
	let body: unknown;
	try {
		body = JSON.parse(text);
	} catch {
		return text; // not JSON — use the raw text
	}
	if (!isRecord(body) || typeof body["error"] !== "string") return text;
	const kind = body["kind"];
	if (typeof kind !== "string") return body["error"];
	const guidance = KIND_GUIDANCE[kind];
	return guidance === undefined
		? `${body["error"]} (kind: ${kind})`
		: `${body["error"]} (kind: ${kind} — ${guidance})`;
}

// ── content mapping ──

export type Text = { type: "text"; text: string };
export type Image = { type: "image"; data: string; mimeType: string };
export type Block = Text | Image;

/** Map validated MCP content into pi's ToolResultMessage blocks. */
export function toBlocks(content: McpContentBlock[]): Block[] {
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

// ── the client ──

interface Pending {
	resolve: (result: unknown) => void;
	reject: (error: Error) => void;
	onProgress?: (message: string) => void;
}

/** The one result an aborted call resolves to. The contract: the server
 * keeps working (downloads stay pollable; a late response finds no waiter
 * and is dropped) — cancellation only detaches this call's waiter, never
 * the shared child or other pending calls. Constructed per call: no
 * mutable object is ever shared across calls, so passing `result.content`
 * through to the host stays safe. */
function cancelled(): McpCallResult {
	return {
		content: [
			{
				type: "text",
				text: "call cancelled — the server keeps working; repeat the same source to pick up the result",
			},
		],
	};
}

/** One MCP server child per pi session, restarted on death at the next
 * call. The child is long-lived on purpose: consent tokens and background
 * downloads are in-process state over there, so every call must ride the
 * same process for tokens to stay valid. */
export class VideoServer {
	private child: McpChild | undefined;
	private ready: Promise<void> | undefined;
	private pending = new Map<number, Pending>();
	private nextId = 2; // 1 is the initialize request

	constructor(private readonly spawnServer: () => McpChild) {}

	/** Call one of the server's tools by name; progress notifications stream
	 * to `onProgress`. */
	async call(
		tool: string,
		args: Record<string, unknown>,
		onProgress: ((message: string) => void) | undefined,
		signal: AbortSignal | undefined,
	): Promise<McpCallResult> {
		// A listener added to an already-aborted signal never fires, so aborts
		// that land before the listener below exists are answered by checks:
		// once on entry, once after the shared handshake (which an abort must
		// ride out, not kill — other calls share the spawn). Either way the
		// request is never sent and no pending entry is left behind.
		if (signal?.aborted) return cancelled();
		await this.start();
		if (signal?.aborted) return cancelled();
		const id = this.nextId++;
		const result = new Promise<unknown>((resolve, reject) => {
			this.pending.set(id, { resolve, reject, onProgress });
			this.send({
				jsonrpc: "2.0",
				id,
				method: "tools/call",
				params: { name: tool, arguments: args, _meta: { progressToken: id } },
			});
		});
		// No await separates the post-handshake check from this attachment, so
		// an abort cannot slip between: every abort either returned above or
		// fires here, detaching the waiter and resolving it with the cancel
		// notice — no promise is left dangling.
		signal?.addEventListener(
			"abort",
			() => {
				const waiter = this.pending.get(id);
				if (waiter === undefined) return; // already settled
				this.pending.delete(id);
				waiter.resolve(cancelled());
			},
			{ once: true },
		);
		return parseCallResult(await result);
	}

	/** Spawn + initialize once; concurrent callers share the handshake. */
	private start(): Promise<void> {
		if (this.child !== undefined && this.ready !== undefined) return this.ready;
		const child = this.spawnServer();
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

	private wireStdout(child: McpChild): void {
		// Incremental line splitting: `scanned` marks the newline-free prefix
		// already searched, so a response arriving as one multi-megabyte line
		// (grids ride base64-inline) is scanned once, not re-scanned from the
		// start on every chunk.
		// One streaming decoder for the pipe: a multi-byte character split
		// across chunks must be joined, not decoded per chunk into U+FFFD.
		const decoder = new TextDecoder();
		let buffer = "";
		let scanned = 0;
		child.stdout?.on("data", (chunk: Buffer) => {
			buffer += decoder.decode(chunk, { stream: true });
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
		let message: unknown;
		try {
			message = JSON.parse(line);
		} catch {
			return; // not ours to understand; skip
		}
		if (!isRecord(message)) return;
		if (message["method"] === "notifications/progress") {
			const params = message["params"];
			if (!isRecord(params)) return;
			const token = params["progressToken"];
			const text = params["message"];
			if (typeof token === "number" && typeof text === "string") {
				this.pending.get(token)?.onProgress?.(text);
			}
			return;
		}
		const id = message["id"];
		if (typeof id !== "number") return;
		const waiter = this.pending.get(id);
		if (waiter === undefined) return; // cancelled or unknown; drop
		this.pending.delete(id);
		const error = message["error"];
		if (error !== undefined) {
			const detail =
				isRecord(error) && typeof error["message"] === "string"
					? error["message"]
					: `MCP request failed: ${render(error)}`;
			waiter.reject(new Error(detail));
		} else {
			// Raw and unvalidated on purpose: the initialize waiter ignores
			// its value, and `call` narrows through parseCallResult.
			waiter.resolve(message["result"]);
		}
	}
}

/** One tool call, unwrapped the way pi's tool contract expects: content
 * blocks on success, a thrown Error on failure — with the Failed body's
 * fault kind riding in the message so retriability reaches the model. */
export async function runVideoCall(
	server: VideoServer,
	tool: string,
	params: unknown,
	onProgress: ((message: string) => void) | undefined,
	signal: AbortSignal | undefined,
): Promise<Block[]> {
	const result = await server.call(tool, parseArgs(tool, params), onProgress, signal);
	const content = toBlocks(result.content);
	if (result.isError === true) {
		// pi marks failure by throw; unwrap the Failed body so the model
		// sees the message and the fault kind, not an envelope.
		const text = content.find((b): b is Text => b.type === "text")?.text ?? `${tool} call failed`;
		throw new Error(failureMessage(text));
	}
	return content;
}
