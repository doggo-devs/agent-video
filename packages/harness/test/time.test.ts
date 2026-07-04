import { describe, expect, test } from 'bun:test'
import { ArgError } from '../src/policy/errors.ts'
import { formatTimestamp, parseTimestamp, resolveWindow } from '../src/policy/time.ts'

describe('parseTimestamp', () => {
	test('accepts_seconds_number_and_string', () => {
		expect(parseTimestamp(83, 'from')).toBe(83)
		expect(parseTimestamp('83', 'from')).toBe(83)
		expect(parseTimestamp('83.5', 'from')).toBe(83.5)
	})

	test('accepts_mm_ss_and_h_mm_ss', () => {
		expect(parseTimestamp('1:23', 'from')).toBe(83)
		expect(parseTimestamp('01:23', 'from')).toBe(83)
		expect(parseTimestamp('1:02:03', 'from')).toBe(3723)
	})

	test('rejects_garbage_with_examples', () => {
		expect(() => parseTimestamp('1m23s', 'from')).toThrow(ArgError)
		expect(() => parseTimestamp('1:99', 'from')).toThrow(ArgError)
		expect(() => parseTimestamp(-5, 'from')).toThrow(ArgError)
		expect(() => parseTimestamp(Number.POSITIVE_INFINITY, 'from')).toThrow(ArgError)
	})
})

describe('formatTimestamp', () => {
	test('matches_core_output_shape', () => {
		expect(formatTimestamp(83)).toBe('01:23')
		expect(formatTimestamp(3723)).toBe('1:02:03')
		expect(formatTimestamp(0)).toBe('00:00')
	})
})

describe('resolveWindow', () => {
	test('defaults_to_full_video', () => {
		const w = resolveWindow(undefined, undefined, 300)
		expect(w).toMatchObject({ start: 0, end: 300, full: true })
	})

	test('from_only_runs_to_end_and_until_only_from_start', () => {
		expect(resolveWindow('1:00', undefined, 300)).toMatchObject({ start: 60, end: 300 })
		expect(resolveWindow(undefined, '1:00', 300)).toMatchObject({ start: 0, end: 60 })
	})

	test('clamps_until_past_end_with_notice', () => {
		const w = resolveWindow(0, 400, 300)
		expect(w.end).toBe(300)
		expect(w.notices.length).toBe(1)
	})

	test('rejects_from_past_end_and_inverted_windows', () => {
		expect(() => resolveWindow(300, undefined, 300)).toThrow(ArgError)
		expect(() => resolveWindow(100, 50, 300)).toThrow(ArgError)
	})
})
