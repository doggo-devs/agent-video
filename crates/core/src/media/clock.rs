// SPDX-License-Identifier: Apache-2.0
//! Confirm container-clock runs before projecting decoded frames onto logical time.

use crate::types::Ts;
use ffmpeg_next as ffmpeg;

/// Seconds relative to the declared video origin, before clock repair.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub(super) struct ContainerTs(pub f64);

/// Seconds on the repaired, monotone playback clock.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub(super) struct LogicalTs(pub f64);

impl LogicalTs {
    /// Narrow to the finite frame timestamp type.
    pub fn to_ts(self) -> Option<Ts> {
        Ts::new(self.0 as f32)
    }
}

/// The first container and logical positions of a confirmed clock seam.
#[derive(Debug, Clone, Copy)]
pub(super) struct Discontinuity {
    pub container: ContainerTs,
    pub logical: LogicalTs,
}

/// Video tolerates reordered timestamps; audio supplies exact sample duration.
pub(super) trait ClockFrame {
    /// Whether backward jitter can be presentation reordering.
    const REORDERED: bool;
    /// Exact duration when known; otherwise learn cadence from timestamps.
    fn duration(&self) -> Option<f64>;
}

impl ClockFrame for ffmpeg::frame::Video {
    const REORDERED: bool = true;
    fn duration(&self) -> Option<f64> {
        None
    }
}

impl ClockFrame for ffmpeg::frame::Audio {
    const REORDERED: bool = false;
    fn duration(&self) -> Option<f64> {
        Some(self.samples() as f64 / f64::from(self.rate()))
    }
}

impl<F: ClockFrame> Default for SegStep<F> {
    fn default() -> Self {
        Self {
            released: Vec::new(),
            dropped: Vec::new(),
            cut: None,
        }
    }
}

/// A three-frame run defeats isolated one- and two-frame timestamp errors.
const SEGMENT_CONFIRM_FRAMES: usize = 3;

/// Allow imprecise duration headers without trusting arbitrary clock jumps.
const EXTENT_FENCE_FACTOR: f64 = 2.0;
const EXTENT_FENCE_SLACK: f64 = 300.0;

/// Starts below this offset trigger inspection of a potentially poisoned origin.
pub(super) const PREROLL_ALLOWANCE: f64 = -60.0;

/// Video cadence before the first confirmed inter-frame delta, in seconds.
pub(super) const FALLBACK_NOMINAL_GAP: f64 = 0.04;

#[derive(Debug, Clone, Copy)]
struct Segment {
    shift: f64,
    last: ContainerTs,
}

impl Segment {
    fn anchored(anchor: f64, first: ContainerTs) -> Segment {
        Segment {
            shift: anchor - first.0,
            last: first,
        }
    }

    fn project(&self, ts: ContainerTs) -> LogicalTs {
        LogicalTs(ts.0 + self.shift)
    }
}

/// A nonempty candidate run holding at most three decoded frames.
struct Buffered<F: ClockFrame> {
    frames: Vec<(ContainerTs, F)>,
}

impl<F: ClockFrame> Buffered<F> {
    fn new(ts: ContainerTs, frame: F) -> Buffered<F> {
        Buffered {
            frames: vec![(ts, frame)],
        }
    }

    fn push(&mut self, ts: ContainerTs, frame: F) {
        self.frames.push((ts, frame));
    }

    fn first(&self) -> ContainerTs {
        self.frames.first().expect("buffered runs are non-empty").0
    }

    fn last(&self) -> ContainerTs {
        self.frames.last().expect("buffered runs are non-empty").0
    }

    fn len(&self) -> usize {
        self.frames.len()
    }
}

enum Current<F: ClockFrame> {
    Head(Buffered<F>),
    Confirmed(Segment),
}

enum State<F: ClockFrame> {
    Unopened,
    Open {
        current: Current<F>,
        challenger: Option<Buffered<F>>,
    },
}

/// Frames released or discarded, with an optional confirmed seam.
pub(super) struct SegStep<F: ClockFrame> {
    pub released: Vec<(LogicalTs, F)>,
    pub dropped: Vec<ContainerTs>,
    pub cut: Option<Discontinuity>,
}

fn drop_run<F: ClockFrame>(buf: Buffered<F>, step: &mut SegStep<F>) {
    step.dropped
        .extend(buf.frames.into_iter().map(|(ts, _)| ts));
}

/// Project confirmed timestamp runs onto a monotone clock. Normal runs keep
/// their labels; backward or beyond-extent runs resume after the previous frame.
/// Unconfirmed challengers cannot advance the clock or teach its tolerance.
/// Each instance buffers at most a head and a challenger, independently of length.
pub(super) struct Segmenter<F: ClockFrame> {
    state: State<F>,
    /// Re-anchor point when the declared video origin proves corrupt.
    stream_head: Option<ContainerTs>,
    extent_fence: f64,
    /// Last confirmed video cadence: seam width and reordering tolerance.
    nominal_gap: f64,
    /// Largest confirmed video gap; audio uses its exact duration instead.
    max_released_gap: f64,
    /// Sample duration of the last released audio frame.
    last_duration: Option<f64>,
    #[cfg(debug_assertions)]
    watermark: f64,
}

impl<F: ClockFrame> Segmenter<F> {
    /// Fence jumps using the declared extent and a fallback video cadence.
    pub fn new(duration: Option<f32>, nominal_gap: f64) -> Segmenter<F> {
        let declared = duration
            .map(f64::from)
            .filter(|d| d.is_finite() && *d > 0.0)
            .unwrap_or(0.0);
        Segmenter {
            state: State::Unopened,
            stream_head: None,
            extent_fence: declared * EXTENT_FENCE_FACTOR + EXTENT_FENCE_SLACK,
            nominal_gap,
            max_released_gap: 0.0,
            last_duration: None,
            #[cfg(debug_assertions)]
            watermark: f64::NEG_INFINITY,
        }
    }

    /// Admit one finite timestamp and its decoded frame.
    pub fn admit(&mut self, ts: ContainerTs, frame: F) -> SegStep<F> {
        let mut step = SegStep::default();
        if self.stream_head.is_none() {
            self.stream_head = Some(ts);
        }
        match std::mem::replace(&mut self.state, State::Unopened) {
            State::Unopened => {
                self.state = State::Open {
                    current: Current::Head(Buffered::new(ts, frame)),
                    challenger: None,
                };
            }
            State::Open {
                current,
                challenger,
            } => {
                self.state = self.step(current, challenger, ts, frame, &mut step);
            }
        }
        step
    }

    /// Expected container position when an audio frame has no timestamp.
    pub fn next_timestamp(&self) -> Option<f64> {
        match &self.state {
            State::Unopened => None,
            State::Open {
                current: Current::Head(buf),
                ..
            } => {
                let (ts, frame) = buf.frames.last().expect("nonempty head");
                Some(ts.0 + frame.duration().unwrap_or(self.nominal_gap))
            }
            State::Open {
                current: Current::Confirmed(seg),
                ..
            } => Some(seg.last.0 + self.last_duration.unwrap_or(self.nominal_gap)),
        }
    }

    /// Release an uncontradicted head or plausible forward tail at EOF.
    /// Other unconfirmed tails are discarded without advancing the clock.
    pub fn finish(&mut self) -> SegStep<F> {
        let mut step = SegStep::default();
        if let State::Open {
            current,
            challenger,
        } = std::mem::replace(&mut self.state, State::Unopened)
        {
            match current {
                Current::Head(buf) => {
                    if let Some(ch) = challenger {
                        drop_run(ch, &mut step);
                    }
                    self.confirm_first(buf, &mut step);
                }
                Current::Confirmed(mut seg) => {
                    if let Some(ch) = challenger {
                        if self.continues_forward(&seg, &ch) {
                            self.extend(&mut seg, ch, &mut step);
                        } else {
                            drop_run(ch, &mut step);
                        }
                    }
                }
            }
        }
        step
    }
    fn step(
        &mut self,
        current: Current<F>,
        challenger: Option<Buffered<F>>,
        ts: ContainerTs,
        frame: F,
        step: &mut SegStep<F>,
    ) -> State<F> {
        if self.fits_current(&current, ts) {
            if let Some(ch) = challenger {
                drop_run(ch, step);
            }
            State::Open {
                current: self.advance(current, ts, frame, step),
                challenger: None,
            }
        } else if challenger.as_ref().is_some_and(|ch| self.fits_run(ch, ts)) {
            let mut ch = challenger.expect("checked by is_some_and");
            ch.push(ts, frame);
            if ch.len() < SEGMENT_CONFIRM_FRAMES {
                State::Open {
                    current,
                    challenger: Some(ch),
                }
            } else {
                State::Open {
                    current: Current::Confirmed(self.commit(current, ch, step)),
                    challenger: None,
                }
            }
        } else {
            if let Some(old) = challenger {
                drop_run(old, step);
            }
            State::Open {
                current,
                challenger: Some(Buffered::new(ts, frame)),
            }
        }
    }
    fn fits_current(&self, current: &Current<F>, ts: ContainerTs) -> bool {
        match current {
            Current::Head(buf) => self.fits_run(buf, ts),
            Current::Confirmed(seg) => {
                let gap = ts.0 - seg.last.0;
                (-self.jitter()..=self.forward_tolerance()).contains(&gap)
            }
        }
    }
    fn fits_run(&self, buf: &Buffered<F>, ts: ContainerTs) -> bool {
        let gap = ts.0 - buf.last().0;
        (0.0..=self.extent_fence).contains(&gap)
    }
    fn jitter(&self) -> f64 {
        if F::REORDERED {
            2.0 * self.nominal_gap
        } else {
            0.0
        }
    }
    fn forward_tolerance(&self) -> f64 {
        2.0 * self.last_duration.unwrap_or(self.max_released_gap)
    }
    fn continues_forward(&self, seg: &Segment, ch: &Buffered<F>) -> bool {
        let gap = ch.first().0 - seg.last.0;
        gap > 0.0 && gap <= self.extent_fence
    }
    fn advance(
        &mut self,
        current: Current<F>,
        ts: ContainerTs,
        frame: F,
        step: &mut SegStep<F>,
    ) -> Current<F> {
        match current {
            Current::Head(mut buf) => {
                buf.push(ts, frame);
                if buf.len() < SEGMENT_CONFIRM_FRAMES {
                    Current::Head(buf)
                } else {
                    Current::Confirmed(self.confirm_first(buf, step))
                }
            }
            Current::Confirmed(mut seg) => {
                if ts < seg.last {
                    step.dropped.push(ts);
                } else {
                    self.release(&mut seg, ts, frame, step);
                }
                Current::Confirmed(seg)
            }
        }
    }
    fn confirm_first(&mut self, buf: Buffered<F>, step: &mut SegStep<F>) -> Segment {
        let first = buf.first();
        let anchor = if !F::REORDERED && first.0 > self.extent_fence {
            0.0
        } else if first.0 >= PREROLL_ALLOWANCE {
            first.0
        } else {
            self.stream_head.expect("set on the first admitted frame").0
        };
        self.open_segment(Segment::anchored(anchor, first), buf, step)
    }
    fn commit(&mut self, current: Current<F>, ch: Buffered<F>, step: &mut SegStep<F>) -> Segment {
        match current {
            Current::Head(buf) => {
                drop_run(buf, step);
                self.confirm_first(ch, step)
            }
            Current::Confirmed(mut seg) => {
                if self.continues_forward(&seg, &ch) {
                    self.extend(&mut seg, ch, step);
                    seg
                } else {
                    let seam =
                        seg.project(seg.last).0 + self.last_duration.unwrap_or(self.nominal_gap);
                    let first = ch.first();
                    step.cut = Some(Discontinuity {
                        container: first,
                        logical: LogicalTs(seam),
                    });
                    self.open_segment(Segment::anchored(seam, first), ch, step)
                }
            }
        }
    }
    fn extend(&mut self, seg: &mut Segment, ch: Buffered<F>, step: &mut SegStep<F>) {
        for (ts, frame) in ch.frames {
            self.release(seg, ts, frame, step);
        }
    }
    fn open_segment(
        &mut self,
        mut seg: Segment,
        buf: Buffered<F>,
        step: &mut SegStep<F>,
    ) -> Segment {
        let mut frames = buf.frames.into_iter();
        let (first, frame) = frames.next().expect("buffered runs are non-empty");
        debug_assert!(first == seg.last, "a segment opens at its first frame");
        self.emit(&seg, first, frame, step);
        for (ts, frame) in frames {
            self.release(&mut seg, ts, frame, step);
        }
        seg
    }
    fn release(&mut self, seg: &mut Segment, ts: ContainerTs, frame: F, step: &mut SegStep<F>) {
        let gap = ts.0 - seg.last.0;
        debug_assert!(gap >= 0.0, "in-segment releases are monotone");
        // Untrusted jumps must not widen future admission tolerance.
        if gap > 0.0 && gap <= self.extent_fence {
            self.nominal_gap = gap;
            self.max_released_gap = self.max_released_gap.max(gap);
        }
        seg.last = ts;
        self.emit(seg, ts, frame, step);
    }

    fn emit(&mut self, seg: &Segment, ts: ContainerTs, frame: F, step: &mut SegStep<F>) {
        let logical = seg.project(ts);
        self.last_duration = frame.duration();
        #[cfg(debug_assertions)]
        {
            debug_assert!(
                logical.0 >= self.watermark,
                "the logical clock never rewinds"
            );
            self.watermark = logical.0;
        }
        step.released.push((logical, frame));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn places_audio_seams_and_missing_timestamps_after_the_last_sample() {
        let frame = |samples, rate| {
            let mut frame = ffmpeg::frame::Audio::empty();
            frame.set_rate(rate);
            frame.set_samples(samples);
            frame
        };
        let mut clock = Segmenter::new(Some(20.0), 0.02);
        for (at, samples, rate) in [(1.0, 160, 16_000), (1.01, 320, 16_000), (1.03, 960, 48_000)] {
            clock.admit(ContainerTs(at), frame(samples, rate));
        }
        assert!((clock.next_timestamp().expect("sample end") - 1.05).abs() < 1e-9);
        clock.admit(ContainerTs(0.0), frame(160, 16_000));
        clock.admit(ContainerTs(0.01), frame(160, 16_000));
        let step = clock.admit(ContainerTs(0.02), frame(160, 16_000));
        let times: Vec<_> = step.released.iter().map(|(ts, _)| ts.0).collect();
        assert_close(&times, &[1.05, 1.06, 1.07]);
        let next = clock.next_timestamp().expect("missing timestamp fallback");
        let step = clock.admit(ContainerTs(next), frame(160, 16_000));
        assert!((step.released[0].0.0 - 1.08).abs() < 1e-9);
    }

    /// A segmenter as a fixed-fps decode at 30 fps would build it, over a
    /// container declaring `duration` seconds (`None` = undeclared).
    fn segmenter(duration: Option<f32>) -> Segmenter<ffmpeg::frame::Video> {
        Segmenter::new(duration, 1.0 / 30.0)
    }

    fn admit(seg: &mut Segmenter<ffmpeg::frame::Video>, v: f64) -> SegStep<ffmpeg::frame::Video> {
        seg.admit(ContainerTs(v), ffmpeg::frame::Video::empty())
    }

    fn released(step: &SegStep<ffmpeg::frame::Video>) -> Vec<f64> {
        step.released.iter().map(|(l, _)| l.0).collect()
    }

    fn dropped(step: &SegStep<ffmpeg::frame::Video>) -> Vec<f64> {
        step.dropped.iter().map(|ts| ts.0).collect()
    }

    /// Feed `values` then finish; collect every released logical instant.
    fn drain_releases(seg: &mut Segmenter<ffmpeg::frame::Video>, values: &[f64]) -> Vec<f64> {
        let mut out = Vec::new();
        for &v in values {
            out.extend(released(&admit(seg, v)));
        }
        out.extend(released(&seg.finish()));
        out
    }

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 1e-9, "{actual:?} vs {expected:?}");
        }
    }

    #[test]
    fn projects_a_clean_stream_identically() {
        let mut seg = segmenter(Some(2.0));
        let out = drain_releases(&mut seg, &[0.0, 0.1, 0.2, 0.3, 0.4]);
        // Bit-exact, not approximate: the identity segment adds 0.0.
        assert_eq!(out, vec![0.0, 0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn buffers_the_head_until_confirmation() {
        let mut seg = segmenter(Some(2.0));
        assert!(released(&admit(&mut seg, 0.0)).is_empty());
        assert!(released(&admit(&mut seg, 0.1)).is_empty());
        assert_eq!(released(&admit(&mut seg, 0.2)), vec![0.0, 0.1, 0.2]);
        assert_eq!(released(&admit(&mut seg, 0.3)), vec![0.3], "streams on");
    }

    #[test]
    fn emits_no_signals_on_a_clean_pass() {
        let mut seg = segmenter(Some(2.0));
        for i in 0..60 {
            let step = admit(&mut seg, f64::from(i) * 0.1);
            assert!(step.dropped.is_empty(), "clean streams drop nothing");
            assert!(step.cut.is_none(), "clean streams commit no seams");
        }
        let step = seg.finish();
        assert!(step.dropped.is_empty() && step.cut.is_none());
    }

    #[test]
    fn drops_backward_jitter_inside_a_segment() {
        let mut seg = segmenter(Some(2.0));
        for v in [0.0, 0.1, 0.2] {
            admit(&mut seg, v);
        }
        // nominal gap is 0.1, jitter bound 0.2: a small backward step is
        // decoder noise, dropped with a signal, no challenger.
        let step = admit(&mut seg, 0.15);
        assert_eq!(dropped(&step), vec![0.15]);
        assert!(step.cut.is_none());
        assert_eq!(released(&admit(&mut seg, 0.3)), vec![0.3]);
    }

    #[test]
    fn commits_a_seam_on_a_confirmed_backward_jump() {
        let mut seg = segmenter(Some(100.0));
        for v in [5.0, 5.1, 5.2] {
            admit(&mut seg, v);
        }
        assert!(released(&admit(&mut seg, 1.0)).is_empty(), "buffered");
        assert!(released(&admit(&mut seg, 1.1)).is_empty(), "buffered");
        let step = admit(&mut seg, 1.2);
        let cut = step.cut.expect("a confirmed backward jump commits a seam");
        // The new segment continues one nominal gap (0.1) past the seal.
        assert!((cut.logical.0 - 5.3).abs() < 1e-9);
        assert!((cut.container.0 - 1.0).abs() < 1e-9);
        assert_close(&released(&step), &[5.3, 5.4, 5.5]);
        assert_close(&released(&admit(&mut seg, 1.3)), &[5.6]);
    }

    #[test]
    fn kills_an_unconfirmed_challenger_when_the_segment_resumes() {
        let mut seg = segmenter(Some(2.0));
        for v in [0.0, 0.1, 0.2] {
            admit(&mut seg, v);
        }
        assert!(released(&admit(&mut seg, 500.0)).is_empty(), "buffered");
        let step = admit(&mut seg, 0.3);
        assert_eq!(dropped(&step), vec![500.0], "the blip dies with a signal");
        assert_eq!(released(&step), vec![0.3], "the segment streams on");
        assert!(step.cut.is_none(), "no seam was committed");
    }

    #[test]
    fn bounds_alternating_poison_to_the_poisoned_half() {
        // The fragment-storm shape: every other frame jumps +24h. Each
        // poisoned frame opens a challenger that the next legal frame
        // kills, so loss is exactly the poisoned half, every drop is
        // signaled, no seam is committed, and the logical clock stays on
        // the legal half.
        let mut seg = segmenter(Some(2.0));
        let mut kept = Vec::new();
        let mut lost = 0;
        for i in 0..60 {
            let t = f64::from(i) * 0.033;
            let step = admit(&mut seg, t);
            kept.extend(released(&step));
            lost += step.dropped.len();
            assert!(step.cut.is_none());
            let step = admit(&mut seg, t + 86_400.0);
            kept.extend(released(&step));
            lost += step.dropped.len();
            assert!(step.cut.is_none());
        }
        let tail = seg.finish();
        lost += tail.dropped.len();
        kept.extend(released(&tail));
        assert_eq!(kept.len(), 60, "the legal half survives whole");
        assert_eq!(lost, 60, "loss is bounded to the poisoned half");
        assert!(kept.iter().all(|&t| t < 2.1), "duration does not explode");
        assert!(kept.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn extends_the_segment_over_a_confirmed_sparse_splice() {
        // Legal sparse pacing (a timelapse splice): the jump buffers as a
        // challenger and, once its successors vouch for it, continues the
        // same segment identically — no seam, no relabel, no drop.
        let mut seg = segmenter(Some(60.0));
        for v in [0.0, 0.1, 0.2] {
            admit(&mut seg, v);
        }
        assert!(released(&admit(&mut seg, 5.0)).is_empty(), "buffered");
        assert!(released(&admit(&mut seg, 5.1)).is_empty(), "buffered");
        let step = admit(&mut seg, 5.2);
        assert!(step.cut.is_none(), "sparse pacing is not a seam");
        assert!(step.dropped.is_empty(), "sparse pacing is not corruption");
        assert_eq!(released(&step), vec![5.0, 5.1, 5.2]);
        assert_eq!(
            released(&admit(&mut seg, 9.8)),
            vec![9.8],
            "the released splice gap is precedent for the next one"
        );
    }

    #[test]
    fn quarantines_an_unvouched_forward_blip() {
        // One frame jumps +9.5s — below the old 10s fence floor that let
        // it walk into the segment unconfirmed, drag the cursor forward,
        // and rewrite the jitter scale until ~49 legal successors fell
        // into the backward-drop path. The single door buffers it as a
        // challenger; the next legal frame kills it; nothing is learned
        // from it.
        let mut seg = segmenter(Some(2.0));
        let mut kept = Vec::new();
        let mut lost = Vec::new();
        for i in 0..60 {
            let t = f64::from(i) * 0.033;
            let v = if i == 10 { t + 9.5 } else { t };
            let step = admit(&mut seg, v);
            assert!(step.cut.is_none(), "a blip commits no seam");
            kept.extend(released(&step));
            lost.extend(dropped(&step));
        }
        let tail = seg.finish();
        kept.extend(released(&tail));
        lost.extend(dropped(&tail));
        assert_eq!(kept.len(), 59, "every legal frame survives: {kept:?}");
        assert!(
            kept.iter().all(|&t| t < 2.0),
            "labels stay at content scale"
        );
        assert_close(&lost, &[10.0 * 0.033 + 9.5]);
    }

    #[test]
    fn releases_a_plausible_sparse_tail_at_eof() {
        // A keyframe-only decode of a legal long-GOP recording: keyframes
        // 0,2,..,10 then 25 and 39.9 inside a declared 40s extent. The
        // tail is content at a plausible position with nothing left to
        // contradict it — released in place, not dropped as an
        // unconfirmed challenger.
        let mut seg = Segmenter::new(Some(40.0), FALLBACK_NOMINAL_GAP);
        let values = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 25.0, 39.9];
        let out = drain_releases(&mut seg, &values);
        assert_eq!(out, values.to_vec(), "the whole legal file is content");
    }

    #[test]
    fn commits_a_seam_on_a_confirmed_beyond_extent_jump() {
        // A confirmed forward run beyond the extent prior cannot keep its
        // labels: the position does not exist in the declared file, so
        // the clock seams and content continues at the seal.
        let mut seg = segmenter(Some(2.0));
        for v in [0.0, 0.1, 0.2] {
            admit(&mut seg, v);
        }
        assert!(released(&admit(&mut seg, 86_400.0)).is_empty(), "buffered");
        assert!(released(&admit(&mut seg, 86_400.1)).is_empty(), "buffered");
        let step = admit(&mut seg, 86_400.2);
        let cut = step.cut.expect("beyond the extent prior: a seam");
        assert!((cut.logical.0 - 0.3).abs() < 1e-9);
        assert_close(&released(&step), &[0.3, 0.4, 0.5]);
    }

    #[test]
    fn learns_the_forward_tolerance_from_released_gaps() {
        let mut seg = segmenter(Some(100.0));
        // Head learns an 8s cadence: tolerance becomes 2 × 8 = 16.
        for v in [0.0, 8.0, 16.0] {
            admit(&mut seg, v);
        }
        assert_eq!(
            released(&admit(&mut seg, 31.0)),
            vec![31.0],
            "a 15s gap fits the learned tolerance"
        );
        assert!(
            released(&admit(&mut seg, 100.0)).is_empty(),
            "a 69s gap beyond the tolerance opens a challenger"
        );
    }

    #[test]
    fn teaches_keyframe_cadence_through_the_head() {
        // A keyframe-only decode of a sparse stream: minute-scale gaps are
        // its legal cadence. The head fences against the declared extent,
        // not the flat floor, so the cadence confirms and then becomes its
        // own precedent.
        let mut seg = Segmenter::new(Some(500.0), FALLBACK_NOMINAL_GAP);
        let out = drain_releases(&mut seg, &[0.0, 100.0, 200.0, 300.0, 400.0]);
        assert_eq!(out, vec![0.0, 100.0, 200.0, 300.0, 400.0]);
    }

    #[test]
    fn fences_the_head_against_a_far_future_jump() {
        let mut seg = segmenter(Some(2.0));
        assert!(released(&admit(&mut seg, 0.0)).is_empty());
        assert!(
            released(&admit(&mut seg, 86_400.0)).is_empty(),
            "beyond the extent fence: a challenger, not a head frame"
        );
        let step = admit(&mut seg, 0.033);
        assert_eq!(dropped(&step), vec![86_400.0], "the blip dies signaled");
        assert_eq!(released(&admit(&mut seg, 0.066)), vec![0.0, 0.033, 0.066]);
    }

    #[test]
    fn replaces_an_unconfirmed_head_with_its_confirmed_challenger() {
        // The poisoned-head shape: the first packets dragged the declared
        // origin a day forward, so the head reads ~0 and legal content
        // rebases to -86398. The challenger confirms, the dead head is
        // dropped with signals, and the first segment re-anchors at the
        // stream head — content keeps sane labels, no verdict, no retry.
        let mut seg = segmenter(Some(0.1));
        admit(&mut seg, 0.0);
        admit(&mut seg, 0.033);
        assert!(released(&admit(&mut seg, -86_398.0)).is_empty());
        assert!(released(&admit(&mut seg, -86_397.966)).is_empty());
        let step = admit(&mut seg, -86_397.933);
        assert_eq!(
            dropped(&step),
            vec![0.0, 0.033],
            "the dead head is dropped with signals"
        );
        assert!(step.cut.is_none(), "no committed clock existed to seam");
        assert_close(&released(&step), &[0.0, 0.034, 0.067]);
    }

    #[test]
    fn anchors_an_opening_segment_within_preroll_identically() {
        // Edit-list pre-roll: legitimate negative timestamps close to zero
        // project identically — byte-for-byte what the container said.
        let mut seg = segmenter(Some(4.0));
        let out = drain_releases(&mut seg, &[-0.5, -0.4, -0.3, 0.0, 0.1]);
        assert_eq!(out, vec![-0.5, -0.4, -0.3, 0.0, 0.1]);
    }

    #[test]
    fn drops_an_unconfirmed_challenger_at_eof() {
        // A corrupt pts on the very last frame has no successor to vouch
        // for it: dropped with a signal, never released into a garbage
        // label.
        let mut seg = segmenter(Some(2.0));
        for v in [0.0, 0.1, 0.2] {
            admit(&mut seg, v);
        }
        admit(&mut seg, 86_400.0);
        let step = seg.finish();
        assert_eq!(dropped(&step), vec![86_400.0]);
        assert!(step.released.is_empty());
    }

    #[test]
    fn confirms_a_short_head_at_eof() {
        // A stream shorter than one confirmation is content, not a
        // fragment: nothing ever contradicted it.
        let mut seg = segmenter(Some(1.0));
        admit(&mut seg, 0.0);
        admit(&mut seg, 0.1);
        assert_eq!(released(&seg.finish()), vec![0.0, 0.1]);
    }

    #[test]
    fn releases_a_lone_frame_at_eof() {
        let mut seg = segmenter(Some(10.0));
        admit(&mut seg, 5.0);
        assert_eq!(released(&seg.finish()), vec![5.0]);
    }

    #[test]
    fn keeps_the_logical_clock_monotone_across_seams() {
        // Three clock domains, each internally sane: two seams, all
        // content preserved, and the concatenation is non-decreasing.
        let mut seg = segmenter(Some(3.0));
        let mut values = Vec::new();
        values.extend((0..10).map(|i| f64::from(i) * 0.033));
        values.extend((0..10).map(|i| 43_200.0 + f64::from(i) * 0.033));
        values.extend((0..10).map(|i| 20.0 + f64::from(i) * 0.033));
        let out = drain_releases(&mut seg, &values);
        assert_eq!(out.len(), 30, "every frame survives");
        assert!(out.windows(2).all(|w| w[0] <= w[1]), "monotone: {out:?}");
        assert!(out.iter().all(|&t| t < 2.0), "labels stay near content");
    }
}
