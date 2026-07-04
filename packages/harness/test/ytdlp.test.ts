import { afterAll, beforeAll, describe, expect, test } from 'bun:test'
import * as fs from 'node:fs'
import * as os from 'node:os'
import * as path from 'node:path'

import { parseProgress, probeSite, siteInfo, YtdlpError, type YtdlpJson } from '../src/source/ytdlp.ts'

const MIB = 1024 ** 2

/** Trimmed shape of a real -J document: youtube-style format ladder. */
const DOC: YtdlpJson = {
	title: 'Conference Talk',
	id: 'abc123',
	duration: 1800,
	subtitles: { en: [], ja: [] },
	formats: [
		// audio-only
		{ vcodec: 'none', acodec: 'opus', filesize: 20 * MIB },
		{ vcodec: 'none', acodec: 'mp4a.40.2', filesize: 28 * MIB },
		// video-only ladder
		{ height: 360, vcodec: 'avc1', acodec: 'none', filesize: 60 * MIB },
		{ height: 480, vcodec: 'avc1', acodec: 'none', filesize: 90 * MIB },
		{ height: 720, vcodec: 'avc1', acodec: 'none', filesize: 200 * MIB },
		{ height: 1080, vcodec: 'avc1', acodec: 'none', filesize: 450 * MIB },
		{ height: 1080, vcodec: 'vp9', acodec: 'none', filesize: 380 * MIB },
		// legacy muxed
		{ height: 360, vcodec: 'avc1', acodec: 'mp4a.40.2', filesize: 85 * MIB },
		// storyboard junk that must be ignored
		{ vcodec: 'none', acodec: 'none' },
	],
}

describe('siteInfo', () => {
	test('reduces_formats_to_descending_keep_tiers_capped_at_three', () => {
		const info = siteInfo(DOC, 480)
		expect(info.keepTiers.map(t => t.height)).toEqual([1080, 720, 480])
	})

	test('keep_estimate_adds_best_audio_to_video_only_tiers', () => {
		const info = siteInfo(DOC, 480)
		// 1080p: max video 450 MiB + best audio 28 MiB
		expect(info.keepTiers[0]?.estimatedBytes).toBe(478 * MIB)
	})

	test('muxed_tiers_do_not_double_count_audio', () => {
		const doc: YtdlpJson = { formats: [{ height: 360, vcodec: 'avc1', acodec: 'mp4a.40.2', filesize: 85 * MIB }] }
		expect(siteInfo(doc, 480).keepTiers[0]?.estimatedBytes).toBe(85 * MIB)
	})

	test('analysis_estimate_picks_best_video_only_under_the_cap', () => {
		const info = siteInfo(DOC, 480)
		expect(info.analysisEstimatedBytes).toBe(90 * MIB) // 480p video-only
	})

	test('carries_title_duration_and_subtitle_langs', () => {
		const info = siteInfo(DOC, 480)
		expect(info.title).toBe('Conference Talk')
		expect(info.duration).toBe(1800)
		expect(info.subtitleLangs).toEqual(['en', 'ja'])
	})

	test('missing_sizes_yield_undefined_estimates', () => {
		const doc: YtdlpJson = { formats: [{ height: 720, vcodec: 'avc1', acodec: 'none' }] }
		const info = siteInfo(doc, 480)
		expect(info.keepTiers[0]?.estimatedBytes).toBeUndefined()
		expect(info.analysisEstimatedBytes).toBeUndefined()
	})
})

describe('parseProgress', () => {
	test('reads_downloaded_and_total_bytes', () => {
		expect(parseProgress('AVDL 1048576 2097152 NA')).toEqual({ downloaded: 1048576, total: 2097152 })
	})

	test('falls_back_to_the_estimate_when_total_is_unknown', () => {
		expect(parseProgress('AVDL 10 NA 100')).toEqual({ downloaded: 10, total: 100 })
	})

	test('total_stays_undefined_when_nothing_is_known', () => {
		expect(parseProgress('AVDL 10 NA NA')).toEqual({ downloaded: 10, total: undefined })
	})

	test('ignores_foreign_lines', () => {
		expect(parseProgress('[download] 42.3% of ~123MiB')).toBeUndefined()
		expect(parseProgress('')).toBeUndefined()
	})
})

describe('probeSite without the binary', () => {
	let emptyDir: string
	const savedPath = process.env['PATH']

	beforeAll(() => {
		emptyDir = fs.mkdtempSync(path.join(os.tmpdir(), 'av-nopath-'))
		process.env['PATH'] = emptyDir
	})

	afterAll(() => {
		process.env['PATH'] = savedPath
	})

	test('fails_with_an_install_hint', async () => {
		expect.assertions(2)
		try {
			await probeSite('https://tube.example/w', 480)
		} catch (err) {
			expect(err).toBeInstanceOf(YtdlpError)
			expect((err as YtdlpError).message).toContain('install')
		}
	})
})
