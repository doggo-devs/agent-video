/**
 * Content-based file identity.
 * Fingerprints key the probe/grid caches so moved or renamed files keep
 * their entries while in-place edits invalidate them.
 */

import { createHash } from 'node:crypto'
import * as fs from 'node:fs/promises'

// bytes hashed from each end of the file
export const SLICE_BYTES = 128 * 1024

// ── types ──

export interface Fingerprint {
	// 16-byte hex digest; equal hex means same identity for cache purposes
	hex: string
	// file size in bytes at the time of hashing
	size: number
	// false when the file changed mid-read; do not write cache entries
	stable: boolean
}

// minimal file I/O surface, injectable for tests
export interface SliceIo {
	// (size, mtimeMs) snapshot
	stat(path: string): Promise<{ size: number; mtimeMs: number }>
	// read exactly `length` bytes at `offset` (short only at EOF)
	readSlice(path: string, offset: number, length: number): Promise<Buffer>
}

const realIo: SliceIo = {
	async stat(path) {
		const s = await fs.stat(path)
		return { size: s.size, mtimeMs: s.mtimeMs }
	},
	async readSlice(path, offset, length) {
		const file = await fs.open(path, 'r')
		try {
			const buf = Buffer.alloc(length)
			const { bytesRead } = await file.read(buf, 0, length, offset)
			return buf.subarray(0, bytesRead)
		} finally {
			await file.close()
		}
	},
}

// pure digest over sampled parts: sha256(head ‖ tail ‖ size_le64), truncated to 16 bytes
export function digestParts(head: Buffer, tail: Buffer, size: number): string {
	const sizeLe = Buffer.alloc(8)
	sizeLe.writeBigUInt64LE(BigInt(size))
	return createHash('sha256').update(head).update(tail).update(sizeLe).digest('hex').slice(0, 32)
}

// fingerprint a file by content: first/last SLICE_BYTES plus size; check `stable` before caching
export async function fingerprintFile(path: string, io: SliceIo = realIo): Promise<Fingerprint> {
	const before = await io.stat(path)

	let head: Buffer
	let tail: Buffer
	if (before.size <= 2 * SLICE_BYTES) {
		head = await io.readSlice(path, 0, before.size)
		tail = Buffer.alloc(0)
	} else {
		head = await io.readSlice(path, 0, SLICE_BYTES)
		tail = await io.readSlice(path, before.size - SLICE_BYTES, SLICE_BYTES)
	}

	const after = await io.stat(path)
	const stable = before.size === after.size && before.mtimeMs === after.mtimeMs

	return { hex: digestParts(head, tail, before.size), size: before.size, stable }
}
