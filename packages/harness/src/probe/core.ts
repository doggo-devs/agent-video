/**
 * Core probe adapter
 * Probes local files through `agent-video probe`, reducing its ProbeReport to
 * the model-facing ProbeSummary; ffprobe.ts remains the binary-less fallback.
 */

import type { CoreRun } from '../core/binary.ts'
import type { ProbeReport } from '../core/contract.ts'
import type { ProbeSummary } from '../tool/schema.ts'
import { ProbeError } from './ffprobe.ts'

// Reduce a core ProbeReport to the model-facing summary; pure.
export function summaryFromReport(report: ProbeReport): ProbeSummary {
	const subtitles = report.embedded_subtitles.map(
		t => `embedded #${t.index} (${t.lang !== undefined ? `${t.lang}, ` : ''}${t.codec})`
	)
	for (const name of report.sidecar_subtitles) subtitles.push(`sidecar ${name}`)
	return {
		duration: report.duration,
		width: report.width,
		height: report.height,
		sizeBytes: report.size_bytes,
		container: report.container,
		subtitles,
	}
}

// Probe a local file via the core binary. Throws ProbeError with the core's
// last stderr line (which carries its [ERROR_CODE] prefix) on failure.
export async function probeViaCore(
	runCore: (args: readonly string[]) => Promise<CoreRun>,
	file: string
): Promise<ProbeSummary> {
	const run = await runCore(['probe', file])
	if (run.code !== 0) {
		const detail = run.stderr.trim().split('\n').at(-1) ?? 'no error output'
		throw new ProbeError(`core probe failed (exit ${run.code}): ${detail}`)
	}
	let report: ProbeReport
	try {
		report = JSON.parse(run.stdout) as ProbeReport
	} catch {
		throw new ProbeError('core probe emitted invalid JSON — version mismatch between harness and core?')
	}
	return summaryFromReport(report)
}
