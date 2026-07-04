/**
 * Disk cache under `cache.dir`.
 * `downloads/` holds analysis rips (LRU by size), `probe/` holds probe
 * documents (evicted by age); mtime is the LRU clock, disk is the index.
 */

import { createHash } from 'node:crypto'
import * as fs from 'node:fs/promises'
import * as path from 'node:path'
import type { Config } from './config.ts'

// probe documents older than this are dropped during eviction sweeps
export const PROBE_MAX_AGE_MS = 30 * 24 * 60 * 60 * 1000

// subtitle sidecar extensions living next to cached downloads
const SUBTITLE_EXTS = ['.srt', '.ass', '.vtt'] as const

// stable cache key for a URL: sha256 truncated to 16 bytes hex
export function urlKey(url: string): string {
	return createHash('sha256').update(url).digest('hex').slice(0, 32)
}

async function statOrNull(p: string): Promise<{ size: number; mtimeMs: number } | null> {
	try {
		const s = await fs.stat(p)
		return { size: s.size, mtimeMs: s.mtimeMs }
	} catch (err) {
		if ((err as NodeJS.ErrnoException).code === 'ENOENT') return null
		throw err
	}
}

// disk cache rooted at `config.cache.dir`; all methods create dirs lazily
export class Cache {
	private readonly downloadsDir: string
	private readonly probeDir: string

	constructor(private readonly cfg: Config['cache']) {
		this.downloadsDir = path.join(cfg.dir, 'downloads')
		this.probeDir = path.join(cfg.dir, 'probe')
	}

	// ── probe documents ──

	// read a cached probe document, refreshing its LRU/age clock
	async getProbe(fingerprint: string): Promise<unknown | undefined> {
		const p = this.probePath(fingerprint)
		let text: string
		try {
			text = await fs.readFile(p, 'utf8')
		} catch (err) {
			if ((err as NodeJS.ErrnoException).code === 'ENOENT') return undefined
			throw err
		}
		await touch(p)
		return JSON.parse(text)
	}

	// persist a probe document; caller must have verified file stability
	async putProbe(fingerprint: string, doc: unknown): Promise<void> {
		await fs.mkdir(this.probeDir, { recursive: true })
		await fs.writeFile(this.probePath(fingerprint), JSON.stringify(doc))
	}

	private probePath(fingerprint: string): string {
		return path.join(this.probeDir, `${fingerprint}.json`)
	}

	// ── downloads ──

	// target path for a download; existence check is the cache hit test
	downloadPath(key: string, ext: string): string {
		return path.join(this.downloadsDir, `${key}.${ext}`)
	}

	// look up a finished download by path; hits refresh the LRU clock (partials live at `<path>.part`)
	async getDownload(p: string): Promise<{ size: number } | undefined> {
		const s = await statOrNull(p)
		if (s === null) return undefined
		await touch(p)
		return { size: s.size }
	}

	// ensure the downloads directory exists; returns it
	async ensureDownloadsDir(): Promise<string> {
		await fs.mkdir(this.downloadsDir, { recursive: true })
		return this.downloadsDir
	}

	// locate the finished media download for a URL key (sidecars/`.part` never match); hits refresh LRU
	async findDownload(key: string): Promise<string | undefined> {
		let names: string[]
		try {
			names = await fs.readdir(this.downloadsDir)
		} catch (err) {
			if ((err as NodeJS.ErrnoException).code === 'ENOENT') return undefined
			throw err
		}
		const media = names.find(
			n => n.startsWith(`${key}.`) && !n.endsWith('.part') && !SUBTITLE_EXTS.some(ext => n.endsWith(ext))
		)
		if (media === undefined) return undefined
		const p = path.join(this.downloadsDir, media)
		await touch(p)
		return p
	}

	// ── eviction ──

	// evict downloads down to `cache.maxBytes` (oldest first) and drop probes past PROBE_MAX_AGE_MS; returns removed paths
	async evict(now: number = Date.now()): Promise<string[]> {
		const removed: string[] = []

		const downloads = await listWithStats(this.downloadsDir)
		let total = downloads.reduce((sum, f) => sum + f.size, 0)
		downloads.sort((a, b) => a.mtimeMs - b.mtimeMs)
		for (const f of downloads) {
			if (total <= this.cfg.maxBytes) break
			await fs.rm(f.path, { force: true })
			total -= f.size
			removed.push(f.path)
		}

		for (const f of await listWithStats(this.probeDir)) {
			if (now - f.mtimeMs > PROBE_MAX_AGE_MS) {
				await fs.rm(f.path, { force: true })
				removed.push(f.path)
			}
		}
		return removed
	}
}

async function touch(p: string): Promise<void> {
	const now = new Date()
	await fs.utimes(p, now, now)
}

async function listWithStats(dir: string): Promise<{ path: string; size: number; mtimeMs: number }[]> {
	let names: string[]
	try {
		names = await fs.readdir(dir)
	} catch (err) {
		if ((err as NodeJS.ErrnoException).code === 'ENOENT') return []
		throw err
	}
	const out: { path: string; size: number; mtimeMs: number }[] = []
	for (const name of names) {
		const p = path.join(dir, name)
		const s = await statOrNull(p)
		if (s !== null) out.push({ path: p, ...s })
	}
	return out
}
