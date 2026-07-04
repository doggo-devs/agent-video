/**
 * Public entry for embedding the harness (pi extension, custom hosts).
 * Re-exports the tool face and the call handler with production wiring.
 */

export type { Metadata } from './core/contract.ts'
export { defaultDeps, handleVideoCall, type ProgressReporter, type ServerDeps } from './mcp/server.ts'
export {
	LOOKS,
	type Look,
	VIDEO_INPUT_SCHEMA,
	VIDEO_TOOL_DESCRIPTION,
	type VideoArgs,
	type VideoResult,
} from './tool/schema.ts'
