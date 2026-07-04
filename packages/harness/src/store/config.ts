/**
 * Harness configuration.
 * JSON file at `~/.config/agent-video/config.json` overridden by
 * `AGENT_VIDEO_*` env vars; invalid values fail loudly.
 */

import * as fs from 'node:fs'
import * as os from 'node:os'
import * as path from 'node:path'

// ── config shape ──

// resolved harness configuration; all byte values are absolute counts
export interface Config {
	budget: {
		// predicted grid count at or below which a call runs immediately
		autoRun: number
		// predicted grid count above which the plan degrades / needs confirm
		max: number
	}
	download: {
		// hard refusal ceiling for any single download
		maxBytes: number
		// analysis rips are capped at this frame height (grid quality ceiling)
		analysisMaxHeight: number
		// where `dl-keep-*` downloads land; the file belongs to the user
		keepDir: string
	}
	cache: {
		dir: string
		// LRU eviction threshold over `downloads/`
		maxBytes: number
	}
}

const GIB = 1024 ** 3

// built-in defaults
export function defaultConfig(home: string = os.homedir()): Config {
	return {
		budget: { autoRun: 2, max: 8 },
		download: { maxBytes: 2 * GIB, analysisMaxHeight: 480, keepDir: path.join(home, 'Downloads') },
		cache: { dir: path.join(home, '.cache', 'agent-video'), maxBytes: 10 * GIB },
	}
}

// a config value failed validation; `message` names file/env key and value
export class ConfigError extends Error {
	constructor(message: string) {
		super(message)
		this.name = 'ConfigError'
	}
}

// ── file parsing ──

// JSON file shape: snake_case keys, every field optional, unknown keys rejected
interface ConfigFile {
	budget?: { auto_run?: number; max?: number }
	download?: { max_bytes?: number; analysis_max_height?: number; keep_dir?: string }
	cache?: { dir?: string; max_bytes?: number }
}

function requirePositiveInt(value: unknown, key: string): number {
	if (typeof value !== 'number' || !Number.isInteger(value) || value <= 0) {
		throw new ConfigError(`${key} must be a positive integer, got ${JSON.stringify(value)}`)
	}
	return value
}

function requireString(value: unknown, key: string): string {
	if (typeof value !== 'string' || value.length === 0) {
		throw new ConfigError(`${key} must be a non-empty string, got ${JSON.stringify(value)}`)
	}
	return value
}

function rejectUnknownKeys(obj: object, allowed: readonly string[], where: string): void {
	for (const key of Object.keys(obj)) {
		if (!allowed.includes(key)) {
			throw new ConfigError(`unknown key '${where}${key}' in config file`)
		}
	}
}

function applyFile(cfg: Config, raw: unknown): void {
	if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) {
		throw new ConfigError('config file root must be a JSON object')
	}
	const file = raw as ConfigFile
	rejectUnknownKeys(file, ['budget', 'download', 'cache'], '')

	if (file.budget !== undefined) {
		rejectUnknownKeys(file.budget, ['auto_run', 'max'], 'budget.')
		const b = file.budget
		if (b.auto_run !== undefined) cfg.budget.autoRun = requirePositiveInt(b.auto_run, 'budget.auto_run')
		if (b.max !== undefined) cfg.budget.max = requirePositiveInt(b.max, 'budget.max')
	}
	if (file.download !== undefined) {
		rejectUnknownKeys(file.download, ['max_bytes', 'analysis_max_height', 'keep_dir'], 'download.')
		const d = file.download
		if (d.max_bytes !== undefined) cfg.download.maxBytes = requirePositiveInt(d.max_bytes, 'download.max_bytes')
		if (d.analysis_max_height !== undefined) {
			cfg.download.analysisMaxHeight = requirePositiveInt(d.analysis_max_height, 'download.analysis_max_height')
		}
		if (d.keep_dir !== undefined) cfg.download.keepDir = requireString(d.keep_dir, 'download.keep_dir')
	}
	if (file.cache !== undefined) {
		rejectUnknownKeys(file.cache, ['dir', 'max_bytes'], 'cache.')
		const c = file.cache
		if (c.dir !== undefined) cfg.cache.dir = requireString(c.dir, 'cache.dir')
		if (c.max_bytes !== undefined) cfg.cache.maxBytes = requirePositiveInt(c.max_bytes, 'cache.max_bytes')
	}
}

function envInt(env: NodeJS.ProcessEnv, key: string): number | undefined {
	const raw = env[key]
	if (raw === undefined) return undefined
	const n = Number(raw)
	if (!Number.isInteger(n) || n <= 0) {
		throw new ConfigError(`env ${key} must be a positive integer, got '${raw}'`)
	}
	return n
}

function applyEnv(cfg: Config, env: NodeJS.ProcessEnv): void {
	cfg.budget.autoRun = envInt(env, 'AGENT_VIDEO_BUDGET_AUTO_RUN') ?? cfg.budget.autoRun
	cfg.budget.max = envInt(env, 'AGENT_VIDEO_BUDGET_MAX') ?? cfg.budget.max
	cfg.download.maxBytes = envInt(env, 'AGENT_VIDEO_DOWNLOAD_MAX_BYTES') ?? cfg.download.maxBytes
	cfg.download.analysisMaxHeight =
		envInt(env, 'AGENT_VIDEO_DOWNLOAD_ANALYSIS_MAX_HEIGHT') ?? cfg.download.analysisMaxHeight
	cfg.download.keepDir = env['AGENT_VIDEO_DOWNLOAD_KEEP_DIR'] ?? cfg.download.keepDir
	cfg.cache.dir = env['AGENT_VIDEO_CACHE_DIR'] ?? cfg.cache.dir
	cfg.cache.maxBytes = envInt(env, 'AGENT_VIDEO_CACHE_MAX_BYTES') ?? cfg.cache.maxBytes
}

// ── loading ──

// default config file location (`$AGENT_VIDEO_CONFIG` overrides the path)
export function configFilePath(env: NodeJS.ProcessEnv = process.env, home: string = os.homedir()): string {
	return env['AGENT_VIDEO_CONFIG'] ?? path.join(home, '.config', 'agent-video', 'config.json')
}

// load configuration: defaults ← config file (if present) ← env overrides
export function loadConfig(env: NodeJS.ProcessEnv = process.env, home: string = os.homedir()): Config {
	const cfg = defaultConfig(home)
	const file = configFilePath(env, home)
	let text: string | undefined
	try {
		text = fs.readFileSync(file, 'utf8')
	} catch (err) {
		if ((err as NodeJS.ErrnoException).code !== 'ENOENT') throw err
	}
	if (text !== undefined) {
		let parsed: unknown
		try {
			parsed = JSON.parse(text)
		} catch (err) {
			throw new ConfigError(`config file ${file} is not valid JSON: ${(err as Error).message}`)
		}
		applyFile(cfg, parsed)
	}
	applyEnv(cfg, env)
	return cfg
}
