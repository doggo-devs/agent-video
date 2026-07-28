// SPDX-License-Identifier: Apache-2.0
//! Pure chunking state machine: judged frames in, chunk/static events out.
//!
//! Owns everything about placing frames on the timeline — time-chunk keying,
//! sparse-chunk merge-forward (the carry), scene-diff boundary context, and
//! static-stretch detection — and nothing about how chunks are processed:
//! no thread pool, no output session, no I/O. The shell
//! ([`super::state::ProcessState`]) interprets the emitted [`Event`]s; this
//! core is unit-testable with plain frames.
//!
//! Time chunks are a throttle, not a floor: a chunk whose surviving frames
//! cannot fill a grid is merged forward into the next chunk(s), so sparse
//! content produces fewer, fuller grids instead of one half-empty grid per
//! chunk.

use std::collections::BTreeMap;
use std::mem;
use std::sync::Arc;

use image::GrayImage;

use crate::types::{CandidateFrame, Ts};

use super::stretch::{Stretch, StretchDetector};

/// One effect the shell must perform, in emission order.
#[derive(Debug)]
pub(super) enum Event {
    /// A grid's worth of frames ready for a chunk worker, with the grayscale
    /// of the frame immediately preceding them (scene-diff boundary context;
    /// `None` for the first chunk).
    Chunk {
        frames: Vec<CandidateFrame>,
        prev_tail: Option<Arc<GrayImage>>,
    },
    /// A static stretch to render as a single representative image.
    Static(Stretch),
}

/// Streaming chunker over judged candidate frames.
///
/// Feed every dedup verdict in timestamp order via [`Chunker::push`], then
/// call [`Chunker::finish`] to flush the final (possibly under-full) chunk
/// and any trailing static stretch.
pub(super) struct Chunker {
    chunk_seconds: f32,
    /// Sampling fps: a frame's coverage extends `1/base_fps` past its ts,
    /// which is what the observed duration measures.
    base_fps: f32,
    pending: BTreeMap<usize, Vec<CandidateFrame>>,
    /// Frames from sealed time chunks that have not yet filled a grid. The
    /// time chunk is a throttle (at most one grid per `chunk_seconds`), not a
    /// floor: sparse chunks merge forward here until they can fill a grid.
    carry: Vec<CandidateFrame>,
    /// Grayscale of the previously emitted chunk's last frame — scene-diff
    /// boundary context, so the next chunk's first frame is scored against
    /// its true predecessor instead of a hardcoded zero. Shared, not copied:
    /// retaining the tail is a pointer clone, like the RGB representative.
    prev_chunk_tail: Option<Arc<GrayImage>>,
    high_water_mark: Option<usize>,
    /// Timestamp of the last placed frame; the monotone-input contract is
    /// asserted against it (see [`Chunker::push`]).
    last_ts: Option<Ts>,
    /// Static-stretch detector, fed every dedup verdict in timestamp order.
    stretches: StretchDetector,
    duration: f32,
}

impl Chunker {
    pub fn new(chunk_seconds: f32, base_fps: f32) -> Self {
        Self {
            chunk_seconds,
            base_fps,
            pending: BTreeMap::new(),
            carry: Vec::new(),
            prev_chunk_tail: None,
            high_water_mark: None,
            last_ts: None,
            stretches: StretchDetector::new(chunk_seconds),
            duration: 0.0,
        }
    }

    /// Video duration observed so far: last placed frame plus its sampling
    /// coverage.
    pub fn duration(&self) -> f32 {
        self.duration
    }

    /// Record one judged frame: a kept frame joins its time chunk, a dropped
    /// one feeds static-stretch evidence; both extend the observed duration.
    /// `capacity` is the current grid capacity — the fill level at which
    /// carried frames become a [`Event::Chunk`].
    ///
    /// Chunk keying, high-water-mark sealing, and static detection all
    /// assume non-decreasing timestamps. The decode pass guarantees exactly
    /// that ([`crate::media::decode`]'s timeline gate releases a
    /// non-decreasing, in-extent stream and fails the run otherwise), and
    /// every stage between it and this core preserves order — so a
    /// regression here is a bug in this crate, not an input property, and
    /// panics per this workspace's error contract (bugs are panics, never
    /// coded errors).
    pub fn push(&mut self, frame: CandidateFrame, kept: bool, capacity: usize) -> Vec<Event> {
        assert!(
            self.last_ts.is_none_or(|last| frame.ts >= last),
            "frames must arrive in non-decreasing timestamp order \
             (last placed {:?}, got {}); the decode gate guarantees this",
            self.last_ts.map(Ts::get),
            frame.ts.get()
        );
        self.last_ts = Some(frame.ts);
        self.duration = self.duration.max(frame.ts.get() + 1.0 / self.base_fps);
        if kept {
            let mut events = Vec::new();
            if let Some(stretch) = self.stretches.note_kept(frame.ts.get(), &frame.rgb) {
                events.push(Event::Static(stretch));
            }
            self.push_chunked_frame(frame, capacity, &mut events);
            events
        } else {
            self.stretches.note_dropped(frame.ts.get());
            Vec::new()
        }
    }

    /// Flush everything: a trailing static stretch, then the remaining
    /// frames — merged at `capacity` like any other seal — with whatever is
    /// left over as the final (possibly under-full) chunk.
    pub fn finish(&mut self, capacity: usize) -> Vec<Event> {
        let mut events = Vec::new();
        if let Some(stretch) = self.stretches.finish() {
            events.push(Event::Static(stretch));
        }
        self.seal_up_to(usize::MAX, capacity, &mut events);
        self.emit_carry(&mut events);
        events
    }

    fn push_chunked_frame(
        &mut self,
        frame: CandidateFrame,
        capacity: usize,
        events: &mut Vec<Event>,
    ) {
        // Pre-start frames (negative ts from edit lists / decoder pre-roll)
        // can only arrive before the first placed frame and belong to the
        // first chunk: clamp explicitly instead of leaning on as-usize
        // saturation.
        let key = (frame.ts.get() / self.chunk_seconds).floor().max(0.0) as usize;
        self.pending.entry(key).or_default().push(frame);

        if let Some(hwm) = self.high_water_mark
            && key > hwm
        {
            self.seal_up_to(hwm, capacity, events);
        }
        self.high_water_mark = Some(self.high_water_mark.map_or(key, |hwm| hwm.max(key)));
    }

    fn seal_up_to(&mut self, seal_key: usize, capacity: usize, events: &mut Vec<Event>) {
        let keys: Vec<usize> = self
            .pending
            .keys()
            .copied()
            .take_while(|&k| k <= seal_key)
            .collect();
        for key in keys {
            if let Some(chunk) = self.pending.remove(&key) {
                self.carry.extend(chunk);
            }
            // Lazy sealing: emit only once the carry can fill a grid, so a
            // sparse chunk merges forward instead of wasting a half-empty grid.
            if self.carry.len() >= capacity {
                self.emit_carry(events);
            }
        }
    }

    fn emit_carry(&mut self, events: &mut Vec<Event>) {
        if self.carry.is_empty() {
            return;
        }
        let frames = mem::take(&mut self.carry);
        let tail = frames.last().map(|f| f.gray.clone());
        let prev_tail = mem::replace(&mut self.prev_chunk_tail, tail);
        events.push(Event::Chunk { frames, prev_tail });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, RgbImage};
    use std::sync::Arc;

    /// A minimal candidate frame: the chunker reads only `ts` (placement) and
    /// `gray`/`rgb` handles (boundary context, static representative).
    fn frame(ts: f32) -> CandidateFrame {
        CandidateFrame {
            ts: Ts::new(ts).expect("finite test ts"),
            gray: Arc::new(GrayImage::new(1, 1)),
            rgb: Arc::new(RgbImage::new(1, 1)),
        }
    }

    /// Collect the timestamps of each emitted chunk.
    fn chunk_ts(events: &[Event]) -> Vec<Vec<f32>> {
        events
            .iter()
            .filter_map(|e| match e {
                Event::Chunk { frames, .. } => Some(frames.iter().map(|f| f.ts.get()).collect()),
                Event::Static(_) => None,
            })
            .collect()
    }

    #[test]
    fn seals_a_chunk_once_the_next_time_chunk_opens() {
        let mut chunker = Chunker::new(60.0, 1.0);

        let mut events = Vec::new();
        for ts in [0.0, 10.0, 20.0, 30.0] {
            events.extend(chunker.push(frame(ts), true, 4));
        }
        assert!(
            chunk_ts(&events).is_empty(),
            "the open chunk stays buffered until a later chunk starts"
        );

        events.extend(chunker.push(frame(65.0), true, 4));
        assert_eq!(
            chunk_ts(&events),
            vec![vec![0.0, 10.0, 20.0, 30.0]],
            "crossing the boundary seals the full previous chunk"
        );
    }

    #[test]
    fn places_pre_start_frames_in_the_first_chunk() {
        let mut chunker = Chunker::new(60.0, 1.0);

        let mut events = Vec::new();
        for ts in [-0.5, 0.0, 10.0, 20.0] {
            events.extend(chunker.push(frame(ts), true, 4));
        }
        events.extend(chunker.push(frame(65.0), true, 4));

        assert_eq!(
            chunk_ts(&events),
            vec![vec![-0.5, 0.0, 10.0, 20.0]],
            "pre-roll frames join the first chunk, not a phantom bucket"
        );
    }

    #[test]
    fn merges_sparse_chunks_until_capacity() {
        let mut chunker = Chunker::new(60.0, 1.0);

        // One frame per chunk at capacity 4: the throttle merges them forward.
        let mut events = Vec::new();
        for ts in [0.0, 65.0, 130.0, 195.0, 260.0] {
            events.extend(chunker.push(frame(ts), true, 4));
        }

        assert_eq!(
            chunk_ts(&events),
            vec![vec![0.0, 65.0, 130.0, 195.0]],
            "sparse chunks merge into one full grid"
        );
    }

    #[test]
    fn finish_flushes_the_underfull_tail_chunk() {
        let mut chunker = Chunker::new(60.0, 1.0);

        for ts in [0.0, 10.0] {
            chunker.push(frame(ts), true, 4);
        }
        let events = chunker.finish(4);

        assert_eq!(chunk_ts(&events), vec![vec![0.0, 10.0]]);
    }

    #[test]
    fn threads_boundary_context_between_chunks() {
        let mut chunker = Chunker::new(60.0, 1.0);

        let mut events = Vec::new();
        for ts in [0.0, 10.0, 65.0, 70.0] {
            events.extend(chunker.push(frame(ts), true, 2));
        }
        events.extend(chunker.finish(2));

        let tails: Vec<bool> = events
            .iter()
            .filter_map(|e| match e {
                Event::Chunk { prev_tail, .. } => Some(prev_tail.is_some()),
                Event::Static(_) => None,
            })
            .collect();
        assert_eq!(
            tails,
            vec![false, true],
            "first chunk has no predecessor; the second scores against the first's tail"
        );
    }

    #[test]
    #[should_panic(expected = "non-decreasing timestamp order")]
    fn panics_on_a_timestamp_regression_from_upstream() {
        // The decode gate guarantees non-decreasing input; a regression
        // reaching this core is a bug in this crate, never an input
        // property — bugs are panics, not coded errors.
        let mut chunker = Chunker::new(60.0, 1.0);

        chunker.push(frame(10.0), true, 4);
        chunker.push(frame(5.0), true, 4);
    }

    #[test]
    fn extends_duration_by_sampling_coverage() {
        let mut chunker = Chunker::new(60.0, 2.0);

        chunker.push(frame(9.0), true, 4);

        assert_eq!(chunker.duration(), 9.5, "ts plus 1/fps of coverage");
    }

    #[test]
    fn dropped_duplicates_become_a_static_event() {
        let mut chunker = Chunker::new(60.0, 1.0);

        let mut events = Vec::new();
        events.extend(chunker.push(frame(0.0), true, 8));
        for ts in 1..=70 {
            events.extend(chunker.push(frame(ts as f32), false, 8));
        }
        events.extend(chunker.push(frame(75.0), true, 8));

        let statics: Vec<(f32, f32)> = events
            .iter()
            .filter_map(|e| match e {
                Event::Static(s) => Some((s.range.start(), s.range.end())),
                Event::Chunk { .. } => None,
            })
            .collect();
        assert_eq!(
            statics,
            vec![(0.0, 75.0)],
            "the kept-frame gap with drop evidence becomes one static event"
        );
    }
}
