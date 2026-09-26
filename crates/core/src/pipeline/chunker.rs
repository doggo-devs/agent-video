// SPDX-License-Identifier: Apache-2.0
//! Scoring chunks over admitted frames: time buckets, sparse-bucket merging,
//! and boundary context. Under-full buckets merge until a grid can be filled.

use std::collections::BTreeMap;
use std::mem;
use std::sync::Arc;

use image::GrayImage;

use crate::types::CandidateFrame;

/// One scoring batch and the grayscale immediately preceding its frames.
#[derive(Debug)]
pub(super) struct Chunk {
    /// Admitted frames in timestamp order.
    pub frames: Vec<CandidateFrame>,
    /// Boundary context from the previous scoring batch, if any.
    pub prev_tail: Option<Arc<GrayImage>>,
}

/// Groups admitted frames in timestamp order without retaining rejected samples.
pub(super) struct Chunker {
    chunk_seconds: f32,
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
}

impl Chunker {
    /// Group frames by time, merging sparse buckets until capacity is reached.
    pub fn new(chunk_seconds: f32) -> Self {
        Self {
            chunk_seconds,
            pending: BTreeMap::new(),
            carry: Vec::new(),
            prev_chunk_tail: None,
            high_water_mark: None,
        }
    }

    /// Admit one selected frame to the scoring chunks.
    pub fn push(&mut self, frame: CandidateFrame, capacity: usize) -> Vec<Chunk> {
        let mut events = Vec::new();
        self.push_chunked_frame(frame, capacity, &mut events);
        events
    }

    /// Seal the remaining chunks and any under-full carry.
    pub fn finish(&mut self, capacity: usize) -> Vec<Chunk> {
        let mut events = Vec::new();
        self.seal_up_to(usize::MAX, capacity, &mut events);
        self.emit_carry(&mut events);
        events
    }

    fn push_chunked_frame(
        &mut self,
        frame: CandidateFrame,
        capacity: usize,
        events: &mut Vec<Chunk>,
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

    fn seal_up_to(&mut self, seal_key: usize, capacity: usize, events: &mut Vec<Chunk>) {
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

    fn emit_carry(&mut self, events: &mut Vec<Chunk>) {
        if self.carry.is_empty() {
            return;
        }
        let frames = mem::take(&mut self.carry);
        let tail = frames.last().map(|f| f.gray.clone());
        let prev_tail = mem::replace(&mut self.prev_chunk_tail, tail);
        events.push(Chunk { frames, prev_tail });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Ts;
    use image::{GrayImage, RgbImage};
    use std::sync::Arc;

    /// A minimal candidate frame: the chunker reads only `ts` (placement) and
    /// `gray`/`rgb` handles (boundary context and scoring payload).
    fn frame(ts: f32) -> CandidateFrame {
        CandidateFrame {
            ts: Ts::new(ts).expect("finite test ts"),
            gray: Arc::new(GrayImage::new(1, 1)),
            rgb: Arc::new(RgbImage::new(1, 1)),
        }
    }

    /// Collect the timestamps of each emitted chunk.
    fn chunk_ts(events: &[Chunk]) -> Vec<Vec<f32>> {
        events
            .iter()
            .map(|chunk| chunk.frames.iter().map(|f| f.ts.get()).collect())
            .collect()
    }

    #[test]
    fn seals_a_chunk_once_the_next_time_chunk_opens() {
        let mut chunker = Chunker::new(60.0);

        let mut events = Vec::new();
        for ts in [0.0, 10.0, 20.0, 30.0] {
            events.extend(chunker.push(frame(ts), 4));
        }
        assert!(
            chunk_ts(&events).is_empty(),
            "the open chunk stays buffered until a later chunk starts"
        );

        events.extend(chunker.push(frame(65.0), 4));
        assert_eq!(
            chunk_ts(&events),
            vec![vec![0.0, 10.0, 20.0, 30.0]],
            "crossing the boundary seals the full previous chunk"
        );
    }

    #[test]
    fn places_pre_start_frames_in_the_first_chunk() {
        let mut chunker = Chunker::new(60.0);

        let mut events = Vec::new();
        for ts in [-0.5, 0.0, 10.0, 20.0] {
            events.extend(chunker.push(frame(ts), 4));
        }
        events.extend(chunker.push(frame(65.0), 4));

        assert_eq!(
            chunk_ts(&events),
            vec![vec![-0.5, 0.0, 10.0, 20.0]],
            "pre-roll frames join the first chunk, not a phantom bucket"
        );
    }

    #[test]
    fn merges_sparse_chunks_until_capacity() {
        let mut chunker = Chunker::new(60.0);

        // One frame per chunk at capacity 4: the throttle merges them forward.
        let mut events = Vec::new();
        for ts in [0.0, 65.0, 130.0, 195.0, 260.0] {
            events.extend(chunker.push(frame(ts), 4));
        }

        assert_eq!(
            chunk_ts(&events),
            vec![vec![0.0, 65.0, 130.0, 195.0]],
            "sparse chunks merge into one full grid"
        );
    }

    #[test]
    fn finish_flushes_the_underfull_tail_chunk() {
        let mut chunker = Chunker::new(60.0);

        for ts in [0.0, 10.0] {
            chunker.push(frame(ts), 4);
        }
        let events = chunker.finish(4);

        assert_eq!(chunk_ts(&events), vec![vec![0.0, 10.0]]);
    }

    #[test]
    fn threads_boundary_context_between_chunks() {
        let mut chunker = Chunker::new(60.0);

        let mut events = Vec::new();
        for ts in [0.0, 10.0, 65.0, 70.0] {
            events.extend(chunker.push(frame(ts), 2));
        }
        events.extend(chunker.finish(2));

        let tails: Vec<bool> = events
            .iter()
            .map(|chunk| chunk.prev_tail.is_some())
            .collect();
        assert_eq!(
            tails,
            vec![false, true],
            "first chunk has no predecessor; the second scores against the first's tail"
        );
    }
}
