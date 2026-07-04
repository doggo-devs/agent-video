import { describe, expect, test } from 'bun:test'
import type { PendingAction, PendingStore } from '../src/policy/tokens.ts'
import { decideUrl } from '../src/policy/url.ts'
import { directInfo } from '../src/source/direct.ts'
import type { ProbedUrl } from '../src/source/source.ts'
import { type SiteUrlInfo, siteSourceInfo } from '../src/source/ytdlp.ts'
import { defaultConfig } from '../src/store/config.ts'
import type { ProbeSummary } from '../src/tool/schema.ts'

const cfg = defaultConfig('/home/u') // maxBytes 2 GiB, analysisMaxHeight 480

class FakeStore implements PendingStore {
	offers: { key: string; offers: Record<string, PendingAction> }[] = []
	offer(key: string, offers: Readonly<Record<string, PendingAction>>): void {
		this.offers.push({ key, offers: { ...offers } })
	}
	take(): PendingAction | undefined {
		return undefined
	}
}

function probe(sizeBytes: number): ProbeSummary {
	return { duration: 300, width: 1280, height: 720, sizeBytes, subtitles: [] }
}

// A direct-link probe result tagged as its source claimed it
function directProbed(url: string, sizeBytes: number, c = cfg): ProbedUrl {
	return { source: 'direct', info: directInfo(url, probe(sizeBytes), c) }
}

function siteInfoFixture(overrides: Partial<SiteUrlInfo> = {}): SiteUrlInfo {
	return {
		title: 'A Talk',
		duration: 1800,
		subtitleLangs: ['en'],
		keepTiers: [
			{ height: 1080, estimatedBytes: 900 * 1024 ** 2 },
			{ height: 720, estimatedBytes: 500 * 1024 ** 2 },
			{ height: 480, estimatedBytes: 250 * 1024 ** 2 },
		],
		analysisEstimatedBytes: 80 * 1024 ** 2,
		...overrides,
	}
}

function siteProbed(info: SiteUrlInfo = siteInfoFixture(), c = cfg): ProbedUrl {
	return { source: 'ytdlp', info: siteSourceInfo(info, c) }
}

describe('decideUrl: direct links', () => {
	test('offers_analysis_and_single_keep_tier', () => {
		const store = new FakeStore()
		const url = 'https://cdn.example.com/talk.mp4'
		const d = decideUrl({ source: url }, directProbed(url, 700 * 1024 ** 2), cfg, store)
		if (d.kind !== 'confirm') throw new Error(`expected confirm, got ${d.kind}`)
		expect(d.tokens.map(t => t.token)).toEqual(['dl-analysis', 'dl-keep-720'])
		expect(d.consequence).toContain('700 MiB')
		expect(store.offers).toHaveLength(1)
		expect(Object.keys(store.offers[0]?.offers ?? {})).toEqual(['dl-analysis', 'dl-keep-720'])
	})

	test('offer_plan_round_trips_source_name_and_opaque_recipe', () => {
		const store = new FakeStore()
		const url = 'https://cdn.example.com/talk.mp4'
		decideUrl({ source: url }, directProbed(url, 700 * 1024 ** 2), cfg, store)
		const action = store.offers[0]?.offers['dl-analysis']
		if (action?.kind !== 'download') throw new Error('expected a download action')
		expect(action.plan.source).toBe('direct')
		expect(action.plan.dest).toBe('cache')
		expect(action.plan.recipe).toMatchObject({ ext: 'mp4' })
	})

	test('refuses_oversized_file_with_manual_hint', () => {
		const store = new FakeStore()
		const url = 'https://cdn.example.com/big.mkv'
		const d = decideUrl({ source: url }, directProbed(url, 3 * 1024 ** 3), cfg, store)
		if (d.kind !== 'reject') throw new Error(`expected reject, got ${d.kind}`)
		expect(d.message).toContain('manually')
		expect(store.offers).toHaveLength(0)
	})

	test('unknown_size_is_offered_with_a_label', () => {
		const url = 'https://cdn.example.com/talk.mp4'
		const d = decideUrl({ source: url }, directProbed(url, 0), cfg, new FakeStore())
		if (d.kind !== 'confirm') throw new Error(`expected confirm, got ${d.kind}`)
		expect(d.tokens[0]?.effect).toContain('size unknown')
	})
})

describe('decideUrl: site pages', () => {
	test('offers_analysis_plus_keep_tiers_with_estimates', () => {
		const store = new FakeStore()
		const d = decideUrl({ source: 'https://tube.example/w' }, siteProbed(), cfg, store)
		if (d.kind !== 'confirm') throw new Error(`expected confirm, got ${d.kind}`)
		expect(d.tokens.map(t => t.token)).toEqual(['dl-analysis', 'dl-keep-1080', 'dl-keep-720', 'dl-keep-480'])
		expect(d.consequence).toContain('A Talk')
		expect(d.consequence).toContain('analysis rip')
		expect(d.tokens[0]?.effect).toContain('480')
	})

	test('drops_tiers_over_the_cap_and_says_so', () => {
		const tight = defaultConfig('/home/u')
		tight.download.maxBytes = 600 * 1024 ** 2
		const d = decideUrl(
			{ source: 'https://tube.example/w' },
			siteProbed(siteInfoFixture(), tight),
			tight,
			new FakeStore()
		)
		if (d.kind !== 'confirm') throw new Error(`expected confirm, got ${d.kind}`)
		expect(d.tokens.map(t => t.token)).toEqual(['dl-analysis', 'dl-keep-720', 'dl-keep-480'])
		expect(d.consequence).toContain('not offered')
		expect(d.consequence).toContain('1080p (900 MiB)')
	})

	test('unknown_size_tiers_are_never_dropped', () => {
		const tiny = defaultConfig('/home/u')
		tiny.download.maxBytes = 1024
		const info = siteInfoFixture({ keepTiers: [{ height: 720 }], analysisEstimatedBytes: undefined })
		const d = decideUrl({ source: 'https://tube.example/w' }, siteProbed(info, tiny), tiny, new FakeStore())
		if (d.kind !== 'confirm') throw new Error(`expected confirm, got ${d.kind}`)
		expect(d.tokens.map(t => t.token)).toEqual(['dl-analysis', 'dl-keep-720'])
	})

	test('rejects_when_every_tier_breaches_the_cap', () => {
		const tiny = defaultConfig('/home/u')
		tiny.download.maxBytes = 1024
		const d = decideUrl(
			{ source: 'https://tube.example/w' },
			siteProbed(siteInfoFixture(), tiny),
			tiny,
			new FakeStore()
		)
		if (d.kind !== 'reject') throw new Error(`expected reject, got ${d.kind}`)
		expect(d.message).toContain('manually')
	})

	test('notes_that_intent_params_wait_for_the_download', () => {
		const d = decideUrl({ source: 'https://tube.example/w', look: 'overview' }, siteProbed(), cfg, new FakeStore())
		if (d.kind !== 'confirm') throw new Error(`expected confirm, got ${d.kind}`)
		expect(d.consequence).toContain('download comes first')
	})

	test('rejects_raw_on_an_undownloaded_url', () => {
		const d = decideUrl({ source: 'https://tube.example/w', raw: 'process' }, siteProbed(), cfg, new FakeStore())
		expect(d.kind).toBe('reject')
	})
})
