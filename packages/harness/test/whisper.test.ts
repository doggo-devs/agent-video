import { afterEach, describe, expect, test } from 'bun:test'
import * as fs from 'node:fs'
import * as os from 'node:os'
import * as path from 'node:path'

import { detectWhisper } from '../src/transcribe/whisper.ts'

// Detection walks PATH, so each test owns a throwaway bin dir and restores PATH.
const savedPath = process.env['PATH']

afterEach(() => {
	process.env['PATH'] = savedPath
})

function binDirWith(...names: string[]): string {
	const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'av-whisper-'))
	for (const name of names) {
		const p = path.join(dir, name)
		fs.writeFileSync(p, '#!/bin/sh\nexit 0\n')
		fs.chmodSync(p, 0o755)
	}
	process.env['PATH'] = dir
	return dir
}

describe('detectWhisper', () => {
	test('explicit_bin_named_whisper_cli_selects_the_cpp_dialect', async () => {
		process.env['PATH'] = ''
		const tool = await detectWhisper({ bin: '/opt/whisper.cpp/whisper-cli', model: '/models/ggml.bin' })
		expect(tool).toEqual({ profile: 'cpp', bin: '/opt/whisper.cpp/whisper-cli', model: '/models/ggml.bin' })
	})

	test('any_other_explicit_bin_selects_the_openai_dialect', async () => {
		process.env['PATH'] = ''
		const tool = await detectWhisper({ bin: '/usr/local/bin/whisper-ctranslate2' })
		expect(tool?.profile).toBe('openai')
	})

	test('whisper_cli_on_path_counts_only_with_a_model_file', async () => {
		binDirWith('whisper-cli')
		const model = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'av-model-')), 'ggml-base.bin')
		fs.writeFileSync(model, 'weights')
		const tool = await detectWhisper({ model })
		expect(tool?.profile).toBe('cpp')
		expect(tool?.model).toBe(model)
	})

	test('whisper_cli_without_a_model_is_skipped_for_openai_whisper', async () => {
		binDirWith('whisper-cli', 'whisper')
		const tool = await detectWhisper({})
		expect(tool?.profile).toBe('openai')
	})

	test('whisper_cli_with_a_missing_model_file_is_skipped', async () => {
		binDirWith('whisper-cli', 'whisper')
		const tool = await detectWhisper({ model: '/no/such/model.bin' })
		expect(tool?.profile).toBe('openai')
	})

	test('no_whisper_anywhere_yields_undefined', async () => {
		binDirWith()
		expect(await detectWhisper({})).toBeUndefined()
	})
})
