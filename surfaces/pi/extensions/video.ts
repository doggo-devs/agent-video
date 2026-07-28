/**
 * Pi extension: the `video` tool over the agent-video binary.
 *
 * A thin MCP stdio client, not an MCP framework: the extension lazily spawns
 * the platform binary's `mcp` subcommand once per session (consent tokens and
 * background downloads are in-process state there), speaks newline-delimited
 * JSON-RPC to it, and maps content blocks straight through — pi's
 * ToolResultMessage uses the same `{ type: "image", data, mimeType }` shape.
 * The client itself lives in ../lib/mcp.ts, dependency-free so its boundary
 * behavior is testable; this file is binary resolution plus pi glue.
 *
 * Binary resolution: `AGENT_VIDEO_BIN` env override, then the exact-pinned
 * `@agent-video/<platform>` package installed via this package's dependency
 * on `@agent-video/cli`. No PATH fallback by design.
 */

import { spawn } from "node:child_process";
import { createRequire } from "node:module";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { runVideoCall, VideoServer } from "../lib/mcp.ts";
import { PARAMETERS, PROMPT_SNIPPET, TOOL_DESCRIPTION } from "../lib/schema.ts";

const require = createRequire(import.meta.url);

const RELEASES_URL = "https://github.com/durvia/agent-video/releases";

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

export default function (pi: ExtensionAPI) {
	const server = new VideoServer(() =>
		spawn(resolveBinary(), ["mcp"], { stdio: ["pipe", "pipe", "pipe"] }),
	);
	pi.registerTool({
		name: "video",
		label: "Video",
		description: TOOL_DESCRIPTION,
		promptSnippet: PROMPT_SNIPPET,
		parameters: PARAMETERS,
		async execute(_toolCallId, params, signal, onUpdate) {
			const content = await runVideoCall(
				server,
				params,
				(message) => onUpdate?.({ content: [{ type: "text", text: message }] }),
				signal,
			);
			return { content, details: {} };
		},
	});
}
