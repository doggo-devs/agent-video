# The `video` tool wire contract

The MCP server exposes exactly one tool, `video`. This document is the contract
its maintainer owns: the argument shape, the response states, the consent-token
lifecycle, and the fault taxonomy. The source of truth is `crate::schema`
(argument and body types), `crate::tokens` (consent registry), `crate::plan`
(decision engine), `crate::mcp::wire` (body → `CallToolResult`), and
`crate::fault` (fault kinds).

## Arguments

`VideoArgs` (see `crate::schema`). Only `source` is required; everything else is
optional and, when omitted, lets the server pick a lenient default.

| Field | Type | Meaning |
|-------|------|---------|
| `source` | string | Local file path or URL. Non-empty. |
| `from` | seconds (number) or `"MM:SS"` / `"H:MM:SS"` string | Window start; omit = start of video. |
| `until` | same as `from` | Window end; omit = end of video. |
| `look` | `overview` \| `detailed` \| `read_text` \| `track_motion` | How to look at the window (frame size/density); omit = balanced default. |
| `strategy` | `adaptive` \| `keyframes` \| `uniform` | Which moments become frames; omit = lenient adaptive. Orthogonal to `look`. |
| `raw` | string | Expert escape hatch: verbatim core CLI arguments (`process ...` / `extract ...`). Mutually exclusive with `from`/`until`/`look`/`strategy`. |
| `confirm` | string | Echo of a consent token from a previous response. Must be passed alone (only with `source`). |
| `transcribe` | boolean | `true` = generate (or reuse) a whisper transcript for the source. Must be passed alone. |

The published JSON Schema advertises `additionalProperties: false`, but clients
do not reliably enforce it, so the server re-checks the argument whitelist
defensively and rejects any unknown key as `bad_input`. The whitelist *is* the
schema's property set — it cannot drift.

## Response states

Every non-grid answer serializes as a `VideoBody`, tagged by a `status` field
(snake_case). A successful grid render is the one exception: it returns a text
block describing the grids followed by the JPEG image blocks, not a tagged body.

| `status` | Shape | Meaning | What the caller does |
|----------|-------|---------|----------------------|
| `plan` | `probe`, `guidance`, `recommended_calls` | Probe-only summary plus ready-to-paste follow-up calls. Returned for a bare source on anything non-trivial (decode later), and after an analysis download or a transcription completes. | Pick a recommended call and copy its args verbatim, or answer from the metadata alone. |
| `confirm_required` | `consequence`, `tokens` | The request needs explicit consent: an over-budget render, a large download. Each token carries an `effect` line spelling out what redeeming it does. | Retry with `confirm` set to one token, and nothing else but `source`. |
| `pending` | `message` | A download continues in the background after the sync grace window elapsed. | Call again with just the same source to poll; the result is served once it finishes. |
| `failed` | `error`, `kind` | The call failed. `error` is model-presentable prose; `kind` is the machine-readable class (see fault taxonomy). This is the one body rendered as an MCP *error* result. | Branch on `kind`: fix arguments, tell the user, retry, or report a bug. |

A ready grid response (not a `VideoBody`) carries one text block — ranges, key
frames, subtitles, and any notices — followed by one image block per grid.

### The probe summary

`plan` and the transcription/download follow-ups embed a `ProbeSummary`. Its
field names are the wire contract and stay camelCase where the TS harness set
the precedent (`sizeBytes`). An unknown size is *absent* on the wire, never a
fake `0`. `subtitles` lists human-readable descriptions (e.g.
`embedded #0 (eng, subrip)`, or `transcript (whisper)` once one is cached).

## Consent-token lifecycle

Consent tokens live in an in-memory registry (`crate::tokens::Tokens`) scoped
to one MCP stdio session. The design is three invariants:

1. **Commit on offer.** When policy decides a call needs consent, the same
   offers that render into the `confirm_required` body are committed to the
   registry in the same step — "the model saw it" and "it is redeemable" are
   one event, from one set of offers, so the wire rendering and the registry
   entry cannot drift apart. Offers are keyed by `source`; one live offer set
   per source.

2. **Idempotent redemption.** Redeeming a token returns the authorized action
   but does not consume the offer. A token stays valid until a newer plan for
   the same source replaces it, so a failed or repeated `confirm` never strands
   the model with a stale token, and sibling tokens in the same offer set
   survive when one is picked. This is safe because the executed actions are
   themselves idempotent: a re-confirmed analysis download serves the rip that
   already landed instead of fetching it again.

3. **A new plan supersedes.** Committing a fresh offer set for a source
   wholesale replaces the prior one. A token from the superseded plan is then
   unknown, and redeeming it is rejected with a hint to call again (without
   `confirm`) to see the current plan and its tokens.

A token authorizes exactly the plan and source it was issued for. Redeeming a
token against a different source fails. Tokens are opaque: policy mints them,
the registry keys on them, nothing parses them.

`confirm` must be passed alone (only with `source`). Combining it with `raw`,
any intent argument, or `transcribe` is rejected — it authorizes exactly the
pending plan, nothing else; the rule is enforced once, where arguments are
refined into an intent, so the model sees one story on every path.

## Fault classification

`crate::fault::FaultKind` is the failure class every adapter boundary (probe,
fetch, transcribe, the self-spawned core child) converges on. It is
deliberately coarser than core's `ErrorCode`: anything finer is prose in the
`error` field. Each kind names *who can fix the failure*.

| `kind` | Who fixes it | Semantics | Caller's move |
|--------|--------------|-----------|---------------|
| `bad_input` | the caller | The source or arguments are wrong — a missing file, a bad range, a stale token, corrupt or out-of-bounds input. Retrying the identical call cannot succeed. | Fix the arguments or the source and call again. |
| `environment` | the user | This installation lacks a capability (a protocol not compiled in, a missing decoder, absent yt-dlp or whisper weights) or the machine refuses (permissions, disk). | Tell the user; do not blindly retry. |
| `transient` | nobody | Plausibly a passing condition — network, a remote server, a stalled child. Retrying may succeed. This is the only retriable kind. | Retry, possibly after a delay. |
| `internal` | us | A bug in agent-video: the harness built core's arguments wrong, core output failed to parse, an uncoded child exit or a signal. | Report it; retrying is unlikely to help. |

### How core errors cross the boundary

A structured core error does not melt into prose when it enters the harness. Its
`ErrorCode` picks the `FaultKind` in exactly one place (`FaultKind::of`), the
`[CODE]`-prefixed rendering becomes the `error` message, and the typed error
survives intact in the source chain. The mapping:

- `DecoderMissing`, `ProtocolUnsupported` → `environment` (this build cannot do
  it; retrying cannot help).
- `DecodeFailed`, `InputNotFound`, `InputUnreadable`, `InputCorrupt`,
  `InputTooLong`, `InsufficientKeyframes`, `NoFramesExtracted`,
  `AmbiguousSubtitle` → `bad_input` (the source itself is unusable or out of
  bounds).
- `InvalidProfile`, `InvalidRange`, `InvalidConfig` → `internal` (the harness
  builds core's arguments; if core rejects them, the harness built them wrong).
- `OutputDirCreate`, `OutputWriteFailed` → `environment` (filesystem refusals on
  the output side).

When the core child is a subprocess, its exit-code decades are the same
published contract, so the subprocess seam loses no classification the in-process
one keeps; uncoded exits (bugs, panics, clap usage) and signals are `internal`.

When both URL sources (direct link, then yt-dlp) fail, the combined kind is the
join of the two failures — a plausibly transient side makes the pair transient,
else a missing capability makes it environment, so the caller is pointed at the
cheapest fruitful next step rather than being misdiagnosed.
