import { describe, expect, test } from 'bun:test'
import * as fs from 'node:fs'
import * as os from 'node:os'
import * as path from 'node:path'
import { ProbeError, scanSidecars, summarize } from '../src/probe/ffprobe.ts'

const VIDEO_STREAM = { codec_type: 'video', codec_name: 'h264', width: 1920, height: 1080 }

describe('summarize', () => {
	test('extracts_dimensions_duration_and_container', () => {
		const s = summarize(
			{ format: { duration: '93.5', size: '1048576', format_name: 'mov,mp4,m4a' }, streams: [VIDEO_STREAM] },
			0,
			[]
		)
		expect(s.duration).toBe(93.5)
		expect(s.width).toBe(1920)
		expect(s.height).toBe(1080)
		expect(s.sizeBytes).toBe(1048576)
		expect(s.container).toBe('mov,mp4,m4a')
		expect(s.subtitles).toEqual([])
	})

	test('describes_embedded_subtitle_tracks_with_zero_based_index', () => {
		const s = summarize(
			{
				format: { duration: '10' },
				streams: [
					VIDEO_STREAM,
					{ codec_type: 'audio', codec_name: 'aac' },
					{ codec_type: 'subtitle', codec_name: 'subrip', tags: { language: 'eng' } },
					{ codec_type: 'subtitle', codec_name: 'ass' },
				],
			},
			0,
			[]
		)
		expect(s.subtitles).toEqual(['embedded #0 (eng, subrip)', 'embedded #1 (ass)'])
	})

	test('appends_sidecars_after_embedded_tracks', () => {
		const s = summarize({ format: { duration: '10' }, streams: [VIDEO_STREAM] }, 0, ['movie.srt'])
		expect(s.subtitles).toEqual(['sidecar movie.srt'])
	})

	test('falls_back_to_stream_duration_and_stat_size', () => {
		const s = summarize({ format: {}, streams: [{ ...VIDEO_STREAM, duration: '42.0' }] }, 777, [])
		expect(s.duration).toBe(42)
		expect(s.sizeBytes).toBe(777)
	})

	test('rejects_files_without_a_video_stream', () => {
		expect(() => summarize({ format: { duration: '10' }, streams: [{ codec_type: 'audio' }] }, 0, [])).toThrow(
			ProbeError
		)
	})

	test('rejects_missing_duration', () => {
		expect(() => summarize({ format: {}, streams: [VIDEO_STREAM] }, 0, [])).toThrow('duration')
	})
})

describe('scanSidecars', () => {
	test('finds_same_basename_subtitle_files_only', async () => {
		const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'av-sidecar-'))
		const video = path.join(dir, 'talk.mp4')
		fs.writeFileSync(video, 'v')
		fs.writeFileSync(path.join(dir, 'talk.srt'), 's')
		fs.writeFileSync(path.join(dir, 'talk.vtt'), 's')
		fs.writeFileSync(path.join(dir, 'other.ass'), 's')

		expect(await scanSidecars(video)).toEqual(['talk.srt', 'talk.vtt'])
	})
})
