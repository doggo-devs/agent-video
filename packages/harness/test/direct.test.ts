import { afterAll, beforeAll, describe, expect, test } from 'bun:test'
import * as fs from 'node:fs'
import * as os from 'node:os'
import * as path from 'node:path'

import { DownloadError, downloadDirect, headContentLength } from '../src/source/direct.ts'

let server: ReturnType<typeof Bun.serve>
let base: string
let tmp: string

const BODY = Buffer.alloc(64 * 1024, 7)

beforeAll(() => {
	tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'av-direct-'))
	server = Bun.serve({
		port: 0,
		fetch(req) {
			const { pathname } = new URL(req.url)
			if (pathname === '/file.mp4') {
				return new Response(BODY, { headers: { 'content-length': String(BODY.length) } })
			}
			if (pathname === '/endless') {
				// No content-length; streams far past the cap under test but stays
				// bounded: Bun.serve drains pull streams with no backpressure, so a
				// truly infinite stream eats memory at wire speed (observed OOM).
				const chunk = new Uint8Array(4096)
				let sent = 0
				return new Response(
					new ReadableStream({
						pull(controller) {
							controller.enqueue(chunk)
							sent += chunk.byteLength
							if (sent >= 4 * 1024 ** 2) controller.close()
						},
					})
				)
			}
			return new Response('nope', { status: 404 })
		},
	})
	base = `http://localhost:${server.port}`
})

afterAll(() => {
	server.stop(true)
})

describe('downloadDirect', () => {
	test('downloads_via_part_file_and_renames', async () => {
		const dest = path.join(tmp, 'out.mp4')
		const { bytes } = await downloadDirect(`${base}/file.mp4`, dest, 1024 ** 2)
		expect(bytes).toBe(BODY.length)
		expect(fs.readFileSync(dest).equals(BODY)).toBe(true)
		expect(fs.existsSync(`${dest}.part`)).toBe(false)
	})

	test('refuses_known_oversize_before_reading_the_body', async () => {
		const dest = path.join(tmp, 'refused.mp4')
		await expect(downloadDirect(`${base}/file.mp4`, dest, 1024)).rejects.toThrow(DownloadError)
		expect(fs.existsSync(dest)).toBe(false)
		expect(fs.existsSync(`${dest}.part`)).toBe(false)
	})

	test('cuts_off_an_unbounded_stream_at_the_cap', async () => {
		const dest = path.join(tmp, 'endless.bin')
		expect.assertions(3)
		try {
			await downloadDirect(`${base}/endless`, dest, 64 * 1024)
		} catch (err) {
			expect(err).toBeInstanceOf(DownloadError)
			expect((err as DownloadError).message).toContain('exceeded')
		}
		expect(fs.existsSync(`${dest}.part`)).toBe(false)
	})

	test('fails_on_http_error_status', async () => {
		const dest = path.join(tmp, 'missing.mp4')
		await expect(downloadDirect(`${base}/gone`, dest, 1024)).rejects.toThrow('HTTP 404')
	})

	test('reports_final_progress_with_totals', async () => {
		const dest = path.join(tmp, 'progress.mp4')
		const reports: [number, number | undefined][] = []
		await downloadDirect(`${base}/file.mp4`, dest, 1024 ** 2, (b, t) => reports.push([b, t]))
		const last = reports.at(-1)
		expect(last).toEqual([BODY.length, BODY.length])
	})
})

describe('headContentLength', () => {
	test('reads_the_length_when_the_server_reveals_it', async () => {
		expect(await headContentLength(`${base}/file.mp4`)).toBe(BODY.length)
	})

	test('returns_undefined_on_unreachable_urls', async () => {
		expect(await headContentLength('http://127.0.0.1:1/x')).toBeUndefined()
	})
})
