/**
 * Background download tracker
 * Per-process registry of in-flight downloads keyed by URL; polls read live
 * progress, settled jobs hand their stored response out exactly once.
 */

import type { CallToolResult } from '@modelcontextprotocol/sdk/types.js'

// Live view of one download job; the download loop mutates it in place
export interface JobStatus {
	downloaded: number
	total?: number
	// the final response, set when the job finished (success or failure)
	settled?: CallToolResult
}

export class DownloadTracker {
	private jobs = new Map<string, JobStatus>()

	// register a job; returns the mutable status handle
	start(key: string): JobStatus {
		const job: JobStatus = { downloaded: 0 }
		this.jobs.set(key, job)
		return job
	}

	peek(key: string): JobStatus | undefined {
		return this.jobs.get(key)
	}

	// hand out the settled response exactly once, clearing the job
	take(key: string): CallToolResult | undefined {
		const job = this.jobs.get(key)
		if (job?.settled === undefined) return undefined
		this.jobs.delete(key)
		return job.settled
	}

	// drop a job without reading it (sync-window completions)
	clear(key: string): void {
		this.jobs.delete(key)
	}
}
