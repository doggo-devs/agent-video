/**
 * Whisper transcription
 * PATH-detected whisper.cpp or openai-whisper turns subtitle-less videos into
 * cached srt transcripts; strictly optional, never a hard dependency.
 */

import { spawn } from 'node:child_process'
import * as fs from 'node:fs/promises'
import * as os from 'node:os'
import * as path from 'node:path'

import type { Config } from '../store/config.ts'

// Transcription failed or no tool is available; message is model-presentable.
export class TranscribeError extends Error {
	constructor(message: string) {
		super(message)
		this.name = 'TranscribeError'
	}
}

// A usable transcription tool; profile decides the flag dialect
export interface WhisperTool {
	// cpp = whisper.cpp (whisper-cli, needs a ggml model file); openai = openai-whisper CLI dialect
	profile: 'cpp' | 'openai'
	bin: string
	model?: string
}

// What the server needs from transcription; injectable for tests
export interface WhisperFacade {
	available(): Promise<boolean>
	transcribe(file: string, outSrt: string, onStep?: (message: string) => void): Promise<void>
}

async function exists(p: string): Promise<boolean> {
	try {
		await fs.access(p)
		return true
	} catch {
		return false
	}
}

// First executable named `name` on PATH, or undefined
async function findOnPath(name: string): Promise<string | undefined> {
	const exts = process.platform === 'win32' ? ['.exe', '.cmd', ''] : ['']
	for (const dir of (process.env.PATH ?? '').split(path.delimiter)) {
		if (dir === '') continue
		for (const ext of exts) {
			const p = path.join(dir, name + ext)
			try {
				await fs.access(p, fs.constants.X_OK)
				return p
			} catch {
				// keep walking
			}
		}
	}
	return undefined
}

// Detect a usable whisper tool. transcribe.bin overrides detection (dialect
// guessed from the basename); whisper-cli counts only with a model file.
export async function detectWhisper(cfg: Config['transcribe']): Promise<WhisperTool | undefined> {
	if (cfg.bin !== undefined) {
		const profile = path.basename(cfg.bin).startsWith('whisper-cli') ? 'cpp' : 'openai'
		return { profile, bin: cfg.bin, model: cfg.model }
	}
	const cli = await findOnPath('whisper-cli')
	if (cli !== undefined && cfg.model !== undefined && (await exists(cfg.model))) {
		return { profile: 'cpp', bin: cli, model: cfg.model }
	}
	const openai = await findOnPath('whisper')
	if (openai !== undefined) return { profile: 'openai', bin: openai, model: cfg.model }
	return undefined
}

function run(bin: string, args: readonly string[], what: string): Promise<void> {
	return new Promise((resolve, reject) => {
		const child = spawn(bin, args, { stdio: ['ignore', 'ignore', 'pipe'] })
		let stderr = ''
		child.stderr.on('data', (chunk: Buffer) => (stderr += chunk))
		child.once('error', (e: NodeJS.ErrnoException) => {
			reject(e.code === 'ENOENT' ? new TranscribeError(`${what} failed: ${bin} not found on PATH`) : e)
		})
		child.once('close', code => {
			if (code === 0) {
				resolve()
				return
			}
			const tail = stderr.trim().split('\n').slice(-5).join('\n') || 'no error output'
			reject(new TranscribeError(`${what} failed (exit ${code}): ${tail}`))
		})
	})
}

// Transcribe a media file to `outSrt`: ffmpeg extracts 16 kHz mono audio into a
// temp workspace, the whisper tool writes there, the srt is moved out on success.
export async function transcribe(
	tool: WhisperTool,
	file: string,
	outSrt: string,
	onStep?: (message: string) => void
): Promise<void> {
	const work = await fs.mkdtemp(path.join(os.tmpdir(), 'agent-video-whisper-'))
	try {
		const wav = path.join(work, 'audio.wav')
		onStep?.('extracting audio')
		await run('ffmpeg', ['-y', '-v', 'error', '-i', file, '-vn', '-ac', '1', '-ar', '16000', wav], 'audio extraction')
		onStep?.('transcribing (CPU-bound, may take minutes)')
		let produced: string
		if (tool.profile === 'cpp') {
			const prefix = path.join(work, 'transcript')
			// model presence is guaranteed by detectWhisper for the cpp profile
			await run(tool.bin, ['-m', tool.model ?? '', '-f', wav, '-osrt', '-of', prefix, '-np'], 'whisper.cpp')
			produced = `${prefix}.srt`
		} else {
			await run(
				tool.bin,
				[wav, '--model', tool.model ?? 'base', '--output_format', 'srt', '--output_dir', work, '--verbose', 'False'],
				'whisper'
			)
			produced = path.join(work, 'audio.srt')
		}
		await fs.mkdir(path.dirname(outSrt), { recursive: true })
		try {
			await fs.copyFile(produced, outSrt)
		} catch {
			throw new TranscribeError('whisper finished without producing an srt file — flag dialect mismatch?')
		}
	} finally {
		await fs.rm(work, { recursive: true, force: true })
	}
}

// Production facade: detection runs once per process and is cached
export function createWhisper(cfg: Config['transcribe']): WhisperFacade {
	let detected: Promise<WhisperTool | undefined> | undefined
	const detect = () => {
		detected ??= detectWhisper(cfg)
		return detected
	}
	return {
		available: async () => (await detect()) !== undefined,
		transcribe: async (file, outSrt, onStep) => {
			const tool = await detect()
			if (tool === undefined) {
				throw new TranscribeError(
					'no whisper tool available — install openai-whisper (pip install -U openai-whisper) ' +
						'or whisper.cpp (whisper-cli on PATH plus transcribe.model pointing at a ggml file)'
				)
			}
			await transcribe(tool, file, outSrt, onStep)
		},
	}
}
