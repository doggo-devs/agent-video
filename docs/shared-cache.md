# Cross-process sharing patterns

Several agent-video server processes can run against the same cache directory
and the same cookie jar at once (one per MCP client, plus ad-hoc CLI runs).
The cache has no manifest and no lock server: disk is the index, and safety
comes from choosing, per artifact, one of exactly three legal sharing
patterns. Anything outside these three is a bug.

**The rule:** every shared on-disk artifact must name its pattern in its module
doc. A reviewer can then check the write path against the named pattern without
reconstructing the concurrency argument from scratch.

The three patterns, from cheapest to most involved:

## Pattern 1 — Content-addressed + idempotent write

**When it applies:** the artifact's name is a content fingerprint, so *equal key
implies equal bytes*. Two processes that write the same key are writing the same
payload.

**Why no coordination is needed:** because the bytes are identical, even a torn
interleave of two writers lands the same value at every offset, and a reader
that catches a half-written file simply fails to parse it, misses, and
regenerates. The write need not be atomic; correctness does not depend on it.

**Members:**

- **Probe documents** — `crate::mcp::probe_cache::cached_probe` +
  `crate::cache::Cache::put_probe`. Keyed by content fingerprint; only stable
  files are cached. The document is a deterministic function of the file's
  content, so the same fingerprint always renders the same JSON. Path-dependent
  facts stay out of the document: sidecar subtitle files (a property of the
  video's directory, not its bytes) are rescanned live on every lookup and
  appended to the summary after the cache read, alongside the
  whisper-transcript decoration — they never enter the cached document.
- **Whisper transcripts** — keyed by content fingerprint × model identity
  (`crate::config::WhisperModel::cache_id`). A transcript from one model must
  never answer for another, so the model is part of the key; the same
  content × model always produces the same cues.

## Pattern 2 — Per-writer staging + atomic rename + last-writer-wins

**When it applies:** the artifact is produced by a non-deterministic or
non-idempotent writer (a network download, an external tool), but *equal key
still implies equal recipe* — two writers for the same key are fetching the same
logical bytes, even if byte-for-byte identity is not guaranteed offset by offset.

**Why it is safe:** each writer stages into a private, uniquely-named location
(a pid+sequence `.part` file, or a pid+sequence staging directory) so no two
writers ever share a buffer. The publish step is an atomic rename on the same
filesystem. If two writers race to publish, last-writer-wins is harmless
*because equal keys mean equal recipes* — the loser's bytes were interchangeable
with the winner's. Per-writer uniqueness (not per-process) matters: two threads
of one server must not share a staging path either.

The staging spellings are owned by exactly one module,
`crate::cache::naming`: writers obtain their `.part` and `.staging-…` names
from its constructors, and the eviction sweep classifies directory entries
with its predicates (`is_part_file`, `is_staging_dir`). Constructor and
predicate are tested as inverses there, and the on-disk spellings are pinned
against literal strings so existing caches stay sweepable.

A crashed writer leaves its staging entry behind — a `.part` file or a
`.staging-…` directory. The eviction sweep reclaims both the same way: any
staging-named entry whose mtime is older than a day, in every directory
writers stage in (downloads and transcripts), has no owner left and is
removed. A live writer keeps its entry's mtime fresh, so age is the
ownership test; in-flight staging is exempt from the LRU budget.

**Members:**

- **Direct-link downloads** — `crate::fetch::direct`. Streams to a
  writer-unique `<dest>.<pid>-<seq>.part`, then `publish` runs `sync_all` on the
  writer's own handle before an atomic rename to the final cache name. A
  crashed writer's `.part` is left for the eviction sweep.
- **yt-dlp downloads** — `crate::fetch::ytdlp`. Each job downloads into its own
  `.staging-<key>-<pid>-<seq>/` directory; `adopt_into` fsyncs and renames each
  file up into the shared cache dir. A crashed job leaves its staging directory
  for the eviction sweep.
- **Whisper SRT output** — `crate::transcribe::write_srt`. The destination name
  is deterministic (fingerprint × route identity) and the cache is shared, so
  the staging file carries the pid; write, then rename.

## Pattern 3 — Three-way merge protocol

**When it applies:** genuinely shared *mutable* state that several processes
read-modify-write concurrently and that no key can partition. In this codebase
that is exactly one artifact.

**Member:** the **canonical cookie jar** — `crate::cookies`. yt-dlp both reads
the jar and rotates cookies back into it, without taking locks, so several
instances sharing the file directly would corrupt it. The protocol gives every
mutable copy one owner:

1. `JarLease::checkout` copies the canonical jar (under an exclusive lock) into
   a private staging directory: a pristine *snapshot* plus a *staged* copy for
   yt-dlp.
2. yt-dlp reads and rewrites only the staged copy.
3. `JarLease::commit` computes the delta `staged − snapshot` — exactly the
   cookies this run touched — and applies it to the canonical jar under the
   lock. A concurrent instance's rotations of other cookies survive. Deletions
   never propagate, so a killed child (whose staged file may be missing or
   truncated) yields an empty or additive delta, never a destructive one.

The lock is a sibling `.lock` file that is never replaced; the jar itself is
published by temp-file + rename (which would race a lock held on the jar node).

## Audit

Every named artifact's actual write path was checked against its declared
pattern.

| Artifact | Pattern | Write path | Conformant |
|----------|---------|------------|------------|
| Probe documents (`cache.rs::put_probe`) | 1 | `fs::create_dir_all` + `fs::write` (direct, non-atomic) | Yes — a non-atomic write is legal under pattern 1: same fingerprint ⇒ same JSON, so an interleave writes identical bytes and a partial read misses and re-probes. |
| Whisper transcripts (`cache.rs` keying) | 1 | keyed by fingerprint × `model_id`; written via `write_srt` (pattern 2) | Yes — the key guarantees equal bytes; the write itself is even stronger than pattern 1 requires (staged rename). |
| Direct downloads (`fetch/direct.rs`) | 2 | writer-unique `.part` (via `cache::naming`), `sync_all`, atomic `rename` | Yes. |
| yt-dlp downloads (`fetch/ytdlp.rs`) | 2 | per-writer `.staging-…` dir (via `cache::naming`), fsync + per-file `rename` in `adopt_into` | Yes. |
| Whisper SRT (`transcribe::write_srt`) | 2 | writer-unique `.part` (via `cache::naming`), `rename` | Yes — verified per prior review, no change needed. |
| Cookie jar (`cookies.rs`) | 3 | exclusive `.lock`, checkout/commit delta merge, temp + rename publish | Yes. |

No genuinely non-atomic write of a shared artifact outside its declared pattern
was found. The one direct `fs::write` (probe documents) is intentional and legal
under pattern 1.
