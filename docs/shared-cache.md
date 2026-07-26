# Cross-process sharing patterns

Several agent-video server processes can run against the same cache directory
and the same cookie jar at once (one per MCP client, plus ad-hoc CLI runs).
The cache has no manifest and no lock server: disk is the index, and safety
comes from choosing, per artifact, one of the legal sharing patterns below.
Anything outside them is a bug.

**The rule:** every shared on-disk artifact must name its pattern in its module
doc. A reviewer can then check the write path against the named pattern without
reconstructing the concurrency argument from scratch.

The patterns, from cheapest to most involved:

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
  The write path is stronger than the pattern requires: a writer-unique
  `.part` (via `cache::naming`) plus atomic rename, so a reader never sees a
  torn document at all. Classification follows the content guarantee —
  determinism alone would carry safety; the staged rename is defense in
  depth, not what safety rests on.

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
writers stage in (downloads, transcripts, and probe documents), has no owner
left and is removed. A live writer keeps its entry's mtime fresh, so age is the
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
- **Whisper transcripts** — `crate::transcribe::write_srt`. Keyed by content
  fingerprint × transcript identity (`crate::transcribe::transcript_id`:
  engine × model × conditioning), so one recipe's cues never answer for
  another's. Decoding is not byte-deterministic — GPU scheduling, beam
  search, and temperature fallback can all vary the cues — so equal key
  means equal recipe, not equal bytes: safety rests entirely on the write
  path. The destination name is deterministic and the cache is shared, so
  the staging file carries the writer id; write, then rename, removing the
  `.part` on failure.

## Pattern 2a — Per-writer staging + atomic hard-link + first-writer-wins

**When it applies:** the artifact is *stable-per-content, not a pure function
of the bytes* — any single writer's value would serve, but the value must
freeze at first publication because downstream identities derive from it.
Pattern 1's equal-bytes argument does not hold (two writers for the same key
may carry different values), and Pattern 2's last-writer-wins would break the
freeze (a re-publication changes the derived identity).

**Why it is safe:** staging is exactly Pattern 2's — a writer-unique `.part`
from `cache::naming`, so no two writers share a buffer and a crashed writer's
debris is age-swept like everyone else's. Only the publish step differs:
`fs::hard_link(part, dest)` instead of `rename`. The link atomically installs
the complete staged bytes or fails with `AlreadyExists` when a winner is in
place, so a reader can never observe a torn or empty artifact and no later
writer can dethrone the winner — losers remove their staging and report
success.

**Members:**

- **Title sidecars** — `crate::cache::Cache::put_title`. The first-sighted
  media title for a content fingerprint, an input to the transcript identity
  (`crate::transcribe::transcript_id`); freezing it closes the
  retitle → new transcript id → re-transcription fragmentation window. A
  blank title is never published, and a blank on-disk sidecar (a legacy torn
  write from the pre-atomic path) is no winner: `get_title` answers `None`
  without refreshing its clock so it ages out, and `put_title` unlinks it and
  contests the link again, so the next sighting heals it.

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
| Probe documents (`cache.rs::put_probe`) | 1 | writer-unique `.part` (via `cache::naming`), atomic `rename`, `.part` removed on failure | Yes — pattern 1 already licenses a plain write (same fingerprint ⇒ same JSON, so a torn interleave is harmless); the staged rename is stronger than the pattern requires, and a crashed writer's `.part` is age-swept. |
| Whisper transcripts (`cache.rs` keying + `transcribe::write_srt`) | 2 | keyed by fingerprint × `transcript_id`; writer-unique `.part` (via `cache::naming`), `rename`, `.part` removed on failure | Yes — the key rules out cross-recipe answers, but decoding is not byte-deterministic, so safety rests on the staged rename: exactly what pattern 2 provides. |
| Direct downloads (`fetch/direct.rs`) | 2 | writer-unique `.part` (via `cache::naming`), `sync_all`, atomic `rename` | Yes. |
| yt-dlp downloads (`fetch/ytdlp.rs`) | 2 | per-writer `.staging-…` dir (via `cache::naming`), fsync + per-file `rename` in `adopt_into` | Yes. |
| Title sidecars (`cache.rs::put_title`) | 2a | writer-unique `.part` (via `cache::naming`), atomic `hard_link` publish, `.part` removed win or lose | Yes — the link installs the whole title or loses to an existing winner, so first-writer-wins is atomic; blank legacy sidecars are dethroned on put and self-retire on read. |
| Cookie jar (`cookies.rs`) | 3 | exclusive `.lock`, checkout/commit delta merge, temp + rename publish | Yes. |

Every write path meets or exceeds its declared pattern. The one over-strength
member is probe documents: pattern 1 licenses a plain write, but the code
stages and renames anyway — the pattern names why the *content* is safe, the
audit records the stronger path actually taken.
