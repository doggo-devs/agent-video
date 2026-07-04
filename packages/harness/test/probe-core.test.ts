import { describe, expect, test } from 'bun:test'

import type { ProbeReport } from '../src/core/contract.ts'
import { summaryFromReport } from '../src/probe/core.ts'

const BASE: ProbeReport = {
	duration: 93.5,
	width: 1920,
	height: 1080,
	fps: 30,
	video_codec: 'h264',
	container: 'mov,mp4,m4a',
	size_bytes: 1048576,
	has_audio: true,
	embedded_subtitles: [],
	sidecar_subtitles: [],
}

describe('summaryFromReport', () => {
	test('carries_metrics_and_container_verbatim', () => {
		const s = summaryFromReport(BASE)
		expect(s.duration).toBe(93.5)
		expect(s.width).toBe(1920)
		expect(s.height).toBe(1080)
		expect(s.sizeBytes).toBe(1048576)
		expect(s.container).toBe('mov,mp4,m4a')
		expect(s.subtitles).toEqual([])
	})

	test('renders_embedded_then_sidecar_subtitle_descriptors', () => {
		const s = summaryFromReport({
			...BASE,
			embedded_subtitles: [
				{ index: 0, lang: 'eng', codec: 'subrip' },
				{ index: 1, codec: 'ass' },
			],
			sidecar_subtitles: ['movie.srt'],
		})
		expect(s.subtitles).toEqual(['embedded #0 (eng, subrip)', 'embedded #1 (ass)', 'sidecar movie.srt'])
	})
})
