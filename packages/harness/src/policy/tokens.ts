/**
 * Consent token registry, in-memory for one MCP stdio session.
 * A token only authorizes the plan and source it was issued for.
 */

import type { CoreInvocation } from './plan.ts'

// A self-contained download decision, no re-probing needed at exec time
export interface DownloadPlan {
	url: string
	// cache = disposable analysis copy; keep = user-owned file in keep_dir
	dest: 'cache' | 'keep'
	// name of the UrlSource that issued (and will execute) this plan
	source: string
	// source-private recipe describing exactly what to fetch
	recipe: unknown
	// best-effort size estimate; undefined = unknown before download
	estimatedBytes?: number
}

// A transcription decision: the media is local by probe time
export interface TranscribePlan {
	file: string
	// content fingerprint keying the cached transcript
	fingerprint: string
}

// The action authorized once a token is redeemed
export type PendingAction =
	| { kind: 'core'; invocation: CoreInvocation }
	| { kind: 'download'; plan: DownloadPlan }
	| { kind: 'transcribe'; plan: TranscribePlan }

// Storage interface the policy engine needs; hand-fakeable in tests
export interface PendingStore {
	// register an offer for key, wholesale replacing any prior offer
	offer(key: string, offers: Readonly<Record<string, PendingAction>>): void
	// redeem: return the action and consume the whole offer, else undefined
	take(key: string, token: string): PendingAction | undefined
}

export class TokenRegistry implements PendingStore {
	private pending = new Map<string, Record<string, PendingAction>>()

	offer(key: string, offers: Readonly<Record<string, PendingAction>>): void {
		this.pending.set(key, { ...offers })
	}

	take(key: string, token: string): PendingAction | undefined {
		const action = this.pending.get(key)?.[token]
		if (action === undefined) return undefined
		this.pending.delete(key)
		return action
	}
}
