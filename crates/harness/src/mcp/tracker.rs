// SPDX-License-Identifier: Apache-2.0
//! Background download tracker.
//!
//! Per-process registry of in-flight downloads keyed by URL. The download
//! thread owns a [`JobReporter`] and sends events; the tracker owns the
//! receiving half plus a folded snapshot, so no state is ever mutated from
//! two sides. Polls fold pending events, settled jobs hand their stored
//! response out exactly once.
//!
//! Slot ownership has teeth in both directions. Downward: the slot owns
//! the job's [`CancelSignal`], and every removal — abandonment,
//! displacement, settled handout — trips it on drop, so a thread whose
//! slot was taken away observes the disownment instead of working on.
//! Upward: a [`Settlement`] may carry an effect that runs only if the slot
//! still owns the job when the settlement is folded — the channel receiver
//! *is* the ownership, so a disowned thread's late settlement (effect
//! included) is dropped unexecuted rather than checked against a flag.
//! Dropping it also fires the settlement's disowned compensation, the undo
//! of any durable effect the disowned job already published.

use std::collections::HashMap;
use std::sync::{Mutex, mpsc};
use std::time::{Duration, Instant};

use rmcp::model::CallToolResult;

use crate::cache::source::SourceId;
use crate::fetch::CancelSignal;
use crate::tokens::Dest;

/// One download job: canonical source and destination, independent of disk recipe keys.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DownloadKey {
    pub source: SourceId,
    pub dest: Dest,
}

/// What the download thread reports; the thread shares no state.
enum JobEvent {
    Progress { downloaded: u64, total: Option<u64> },
    Settled(Settlement),
}

/// A job's final response plus an optional effect that must land only
/// while the job still owns its slot. The pair travels through the slot's
/// channel as one event, so the effect can never run without the response
/// being adopted, nor the response be adopted with the effect lost. An
/// optional disowned compensation is the mirror image: it runs exactly
/// when the settlement is dropped unadopted.
///
/// Contract: adoption is decided inside the tracker's critical section,
/// but the adopted effect executes after it, outside the jobs lock — it
/// may block on I/O or call back into the tracker without wedging the
/// session (`Aftermath`).
pub struct Settlement {
    result: CallToolResult,
    on_owned: Option<Box<dyn FnOnce() + Send>>,
    /// Armed until adoption defuses it in [`JobSlot::fold`]; its `Drop`
    /// fires on every other path a settlement can take.
    on_disowned: Option<Unadopted>,
}

/// Drop bomb carrying a settlement's disowned compensation.
///
/// The channel *is* slot ownership, so "this settlement was never adopted"
/// has no single code path: the send can fail (slot already gone), a sent
/// event can be dropped with the receiver racing the send, or a queued
/// event can go down with the tracker at teardown. Owning the compensation
/// as a `Drop` impl covers all of them with one mechanism — only adoption
/// ([`JobSlot::fold`]) defuses it.
struct Unadopted(Option<Box<dyn FnOnce() + Send>>);

impl Unadopted {
    /// Adoption: the compensation must not run; drop it unexecuted.
    fn defuse(mut self) {
        self.0 = None;
    }
}

impl Drop for Unadopted {
    fn drop(&mut self) {
        if let Some(compensate) = self.0.take() {
            compensate();
        }
    }
}

impl Settlement {
    /// A settlement with a side effect that runs exactly when the tracker
    /// adopts the result — never for a job whose slot was abandoned or
    /// displaced. Used for the keep-download receipt, which a disowned
    /// zombie must not be able to write over a retry's.
    pub fn with_owned_effect(
        result: CallToolResult,
        effect: impl FnOnce() + Send + 'static,
    ) -> Self {
        Settlement {
            result,
            on_owned: Some(Box::new(effect)),
            on_disowned: None,
        }
    }

    /// Arm `compensate` to run exactly when this settlement is dropped
    /// without being adopted — its slot was abandoned, displaced, or torn
    /// down before the fold. Adoption defuses it, so the owned effect and
    /// the compensation are mutually exclusive by construction. Used by
    /// keep downloads to remove a published file whose receipt will never
    /// be written: "the file exists" and "its settlement was adopted" can
    /// then only diverge across a process crash.
    pub fn or_disowned(mut self, compensate: impl FnOnce() + Send + 'static) -> Self {
        self.on_disowned = Some(Unadopted(Some(Box::new(compensate))));
        self
    }
}

impl From<CallToolResult> for Settlement {
    fn from(result: CallToolResult) -> Self {
        Settlement {
            result,
            on_owned: None,
            on_disowned: None,
        }
    }
}

/// Write half of one job, owned by its download thread. Dropping it without
/// settling (a panic unwound the thread) is visible to the next poll as an
/// abandoned job.
#[derive(Debug)]
pub struct JobReporter {
    events: mpsc::Sender<JobEvent>,
    cancel: CancelSignal,
}

impl JobReporter {
    /// Report live progress; never blocks.
    pub fn progress(&self, downloaded: u64, total: Option<u64>) {
        let _ = self.events.send(JobEvent::Progress { downloaded, total });
    }

    /// The job's cancel signal, tripped by the tracker when the slot is
    /// given up. The download threads it into work that can observe it.
    pub fn cancel_signal(&self) -> CancelSignal {
        self.cancel.clone()
    }

    /// Deliver the final response. A job settles at most once; a settlement
    /// sent after the slot was given up is dropped whole, owned effect
    /// included.
    pub fn settle(self, settlement: impl Into<Settlement>) {
        let _ = self.events.send(JobEvent::Settled(settlement.into()));
    }
}

/// Read half plus folded snapshot of one job; owned by the tracker.
struct JobSlot {
    events: mpsc::Receiver<JobEvent>,
    /// Tripped on drop: no removal path can leave the thread unaware that
    /// its slot is gone. After a settled handout the thread is already done
    /// and the trip is a no-op.
    cancel: CancelSignal,
    downloaded: u64,
    total: Option<u64>,
    /// The final response, stashed by [`JobSlot::fold`] and taken exactly
    /// once by `poll`/`admit`. Stashing (rather than returning out of the
    /// fold) keeps "the result exists" a slot fact, so it always beats the
    /// disconnect that follows a settle.
    done: Option<CallToolResult>,
    /// Stamped by the reader on every folded progress event (job start until
    /// the first report). The poll path abandons jobs whose thread stopped
    /// reporting — the guarantee that a wedged download can never pin its
    /// URL on Pending forever.
    last_progress: Instant,
}

/// Whether a slot's reporter is still connected after a fold.
enum Fold {
    /// The reporter is gone without settling: the thread died.
    Dead,
    /// Still connected; the snapshot is current.
    Live,
}

impl JobSlot {
    fn new(events: mpsc::Receiver<JobEvent>, cancel: CancelSignal) -> Self {
        JobSlot {
            events,
            cancel,
            downloaded: 0,
            total: None,
            done: None,
            last_progress: Instant::now(),
        }
    }

    /// Drain pending events into the snapshot; a settled result is stashed
    /// in `done` (checked before liveness by every caller) and its owned
    /// effect is adopted into `aftermath` — the settlement was folded by a
    /// live slot, so the job provably still owned it. The effect executes
    /// only once the caller drops the aftermath, outside the jobs lock.
    fn fold(&mut self, aftermath: &mut Aftermath) -> Fold {
        loop {
            match self.events.try_recv() {
                Ok(JobEvent::Progress { downloaded, total }) => {
                    self.downloaded = downloaded;
                    if total.is_some() {
                        self.total = total;
                    }
                    self.last_progress = Instant::now();
                }
                Ok(JobEvent::Settled(settlement)) => {
                    // Folded by a live slot: adopted. The compensation is
                    // defused before the owned effect can run, so no panic
                    // or ordering accident can fire both.
                    if let Some(armed) = settlement.on_disowned {
                        armed.defuse();
                    }
                    if let Some(effect) = settlement.on_owned {
                        aftermath.effects.push(effect);
                    }
                    self.done = Some(settlement.result);
                }
                Err(mpsc::TryRecvError::Empty) => return Fold::Live,
                Err(mpsc::TryRecvError::Disconnected) => return Fold::Dead,
            }
        }
    }
}

/// Everything a critical section decided that must *execute* outside it:
/// adopted owned effects and evicted slots, whose teardown trips cancel
/// signals and can fire queued disowned compensations. Both run arbitrary
/// code — a callback that touched the tracker again would self-deadlock on
/// the non-reentrant jobs lock, and slow I/O would stall every poll in the
/// session — so the locked sections only ever *collect* into this value,
/// and their callers drop it after the guard is gone. Dropping is the
/// execution: adoption ordering (defuse, then effect) was already settled
/// under the lock, only the running happens here.
///
/// Execution lives in `Drop` so every exit path runs it and "dropped
/// without running" is unreachable; the double-panic risk of effects in a
/// drop does not arise under the workspace's `panic = "abort"`.
#[derive(Default)]
struct Aftermath {
    /// Owned effects of settlements adopted in this critical section.
    effects: Vec<Box<dyn FnOnce() + Send>>,
    /// Slots removed in this critical section; dropped here so their
    /// cancel trip and any queued compensations run without the lock.
    evicted: Vec<JobSlot>,
}

impl Drop for Aftermath {
    fn drop(&mut self) {
        for effect in self.effects.drain(..) {
            effect();
        }
        // `evicted` drops with self, after the effects.
    }
}

impl Drop for JobSlot {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

/// What one poll of a URL's download slot resolved to. The full state
/// machine of "is my download done?" — including the exactly-once handout
/// of the settled response and the abandonment of wedged jobs — runs inside
/// [`Tracker::poll`] under one lock, so callers cannot observe (or race)
/// intermediate states.
#[derive(Debug)]
pub enum Poll {
    /// No job registered for this key.
    Vacant,
    /// The finished response, handed out exactly once; the slot is cleared.
    Settled(CallToolResult),
    /// Still reporting progress; a snapshot for rendering.
    Running { downloaded: u64, total: Option<u64> },
    /// The job's thread died or stopped reporting past the stall limit; the
    /// slot is cleared — tripping the job's cancel signal — so the caller
    /// can start over without a zombie working underneath it.
    Abandoned,
}

/// Outcome of claiming a key's download slot ([`Tracker::admit`]), decided
/// in one critical section — there is no check-then-insert window, so two
/// concurrent confirms for the same URL admit exactly one writer.
#[derive(Debug)]
pub enum Admission {
    /// This caller won the slot and owns the job; the reporter is the only
    /// way to speak into it.
    Started(JobReporter),
    /// A live job already holds the slot; a snapshot for rendering.
    Running { downloaded: u64, total: Option<u64> },
    /// The previous job for this key finished unobserved; its response,
    /// handed out exactly once instead of a fresh start — re-running a
    /// successful keep download would duplicate the user's file, and a
    /// stored failure tells the caller what actually happened. The slot is
    /// cleared: the next admission starts fresh.
    Settled(CallToolResult),
}

/// In-flight downloads for one MCP session.
#[derive(Default)]
pub struct Tracker {
    /// The only lock; flat, values are plain owned data.
    jobs: Mutex<HashMap<DownloadKey, JobSlot>>,
}

impl Tracker {
    /// Claim the slot for `key`: fold the existing job first (settled →
    /// hand its response out, dead or stalled → evict), then insert a fresh
    /// job if the slot is free. The liveness check and the insertion share
    /// one critical section, so the loser of a concurrent double confirm
    /// sees `Running` instead of spawning a second download. Adopted
    /// effects and evicted slots execute after the lock, before returning.
    pub fn admit(&self, key: &DownloadKey, stall_limit: Duration) -> Admission {
        let mut aftermath = Aftermath::default();
        let admission = self.admit_locked(key, stall_limit, &mut aftermath);
        drop(aftermath); // the lock is gone: effects and teardown run here
        admission
    }

    fn admit_locked(
        &self,
        key: &DownloadKey,
        stall_limit: Duration,
        aftermath: &mut Aftermath,
    ) -> Admission {
        let mut jobs = self.lock();
        if let Some(slot) = jobs.get_mut(key) {
            let liveness = slot.fold(aftermath);
            if slot.done.is_some() {
                let mut slot = jobs.remove(key).expect("slot was just folded");
                let result = slot.done.take().expect("checked above");
                aftermath.evicted.push(slot);
                return Admission::Settled(result);
            }
            match liveness {
                // Dead or wedged: nothing to hand out; claim the slot anew.
                Fold::Dead => {}
                Fold::Live if slot.last_progress.elapsed() > stall_limit => {}
                Fold::Live => {
                    return Admission::Running {
                        downloaded: slot.downloaded,
                        total: slot.total,
                    };
                }
            }
        }
        let (tx, rx) = mpsc::channel();
        let cancel = CancelSignal::new();
        aftermath
            .evicted
            .extend(jobs.insert(key.clone(), JobSlot::new(rx, cancel.clone())));
        Admission::Started(JobReporter { events: tx, cancel })
    }

    /// Resolve one poll: settled beats dead beats stalled beats running,
    /// decided in a single critical section. Adopted effects and evicted
    /// slots execute after the lock, before this returns.
    pub fn poll(&self, key: &DownloadKey, stall_limit: Duration) -> Poll {
        let mut aftermath = Aftermath::default();
        let poll = self.poll_locked(key, stall_limit, &mut aftermath);
        drop(aftermath); // the lock is gone: effects and teardown run here
        poll
    }

    fn poll_locked(
        &self,
        key: &DownloadKey,
        stall_limit: Duration,
        aftermath: &mut Aftermath,
    ) -> Poll {
        let mut jobs = self.lock();
        let Some(slot) = jobs.get_mut(key) else {
            return Poll::Vacant;
        };
        let liveness = slot.fold(aftermath);
        if slot.done.is_some() {
            let mut slot = jobs.remove(key).expect("slot was just folded");
            let result = slot.done.take().expect("checked above");
            aftermath.evicted.push(slot);
            return Poll::Settled(result);
        }
        match liveness {
            Fold::Dead => {
                aftermath.evicted.extend(jobs.remove(key));
                Poll::Abandoned
            }
            Fold::Live if slot.last_progress.elapsed() > stall_limit => {
                aftermath.evicted.extend(jobs.remove(key));
                Poll::Abandoned
            }
            Fold::Live => Poll::Running {
                downloaded: slot.downloaded,
                total: slot.total,
            },
        }
    }

    /// Rewind a job's progress clock by `by`, as if it had been silent that
    /// long. Test-only: real staleness comes from the wall clock.
    #[cfg(test)]
    pub fn age(&self, key: &DownloadKey, by: Duration) {
        if let Some(slot) = self.lock().get_mut(key) {
            slot.last_progress = Instant::now().checked_sub(by).expect("clock has history");
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<DownloadKey, JobSlot>> {
        self.jobs.lock().expect("tracker poisoned")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmcp::model::ContentBlock;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc;

    /// Generous limit: these tests never exercise abandonment unless they
    /// rewind the clock explicitly.
    const LIMIT: Duration = Duration::from_secs(3600);

    fn key(source: &str) -> DownloadKey {
        DownloadKey {
            source: SourceId::from_spelling(source),
            dest: Dest::Cache,
        }
    }

    fn settled_result() -> CallToolResult {
        CallToolResult::success(vec![ContentBlock::text("done")])
    }

    /// Claim a vacant slot; the tests below own the invariant that it is.
    fn start(tracker: &Tracker, key: &DownloadKey) -> JobReporter {
        match tracker.admit(key, LIMIT) {
            Admission::Started(reporter) => reporter,
            other => panic!("slot must be vacant, got {other:?}"),
        }
    }

    #[test]
    fn polls_vacant_without_a_job() {
        let tracker = Tracker::default();
        assert!(matches!(tracker.poll(&key("k"), LIMIT), Poll::Vacant));
    }

    #[test]
    fn deduplicates_source_aliases_but_separates_destinations() {
        let tracker = Tracker::default();
        let _analysis = start(&tracker, &key("https://example.com/v.mp4"));
        let mut alias = key("HTTPS://Example.COM/v.mp4");
        assert!(matches!(
            tracker.admit(&alias, LIMIT),
            Admission::Running { .. }
        ));
        alias.dest = Dest::Keep;
        assert!(matches!(
            tracker.admit(&alias, LIMIT),
            Admission::Started(_)
        ));
    }

    #[test]
    fn polls_running_with_live_progress() {
        let tracker = Tracker::default();
        let reporter = start(&tracker, &key("k"));
        reporter.progress(512, Some(1024));
        let Poll::Running { downloaded, total } = tracker.poll(&key("k"), LIMIT) else {
            panic!("job is live");
        };
        assert_eq!((downloaded, total), (512, Some(1024)));
    }

    #[test]
    fn keeps_known_total_when_a_report_omits_it() {
        let tracker = Tracker::default();
        let reporter = start(&tracker, &key("k"));
        reporter.progress(512, Some(1024));
        reporter.progress(600, None);
        let Poll::Running { downloaded, total } = tracker.poll(&key("k"), LIMIT) else {
            panic!("job is live");
        };
        assert_eq!((downloaded, total), (600, Some(1024)));
    }

    #[test]
    fn hands_out_settled_response_exactly_once() {
        let tracker = Tracker::default();
        let reporter = start(&tracker, &key("k"));
        reporter.settle(settled_result());
        assert!(matches!(tracker.poll(&key("k"), LIMIT), Poll::Settled(_)));
        assert!(
            matches!(tracker.poll(&key("k"), LIMIT), Poll::Vacant),
            "cleared after handout"
        );
    }

    #[test]
    fn abandons_stalled_jobs_and_clears_the_slot() {
        let tracker = Tracker::default();
        let _reporter = start(&tracker, &key("k"));
        tracker.age(&key("k"), Duration::from_secs(120));
        assert!(matches!(
            tracker.poll(&key("k"), Duration::from_secs(60)),
            Poll::Abandoned
        ));
        assert!(
            matches!(tracker.poll(&key("k"), LIMIT), Poll::Vacant),
            "slot is free for a retry"
        );
    }

    #[test]
    fn abandons_a_job_whose_reporter_died_unsettled() {
        let tracker = Tracker::default();
        let reporter = start(&tracker, &key("k"));
        drop(reporter); // the download thread panicked
        assert!(matches!(tracker.poll(&key("k"), LIMIT), Poll::Abandoned));
        assert!(
            matches!(tracker.poll(&key("k"), LIMIT), Poll::Vacant),
            "slot is free for a retry"
        );
    }

    #[test]
    fn settled_beats_abandoned() {
        // Settled result queued against a stalled clock: the finished
        // download wins, clock staleness is irrelevant.
        let tracker = Tracker::default();
        let reporter = start(&tracker, &key("k"));
        tracker.age(&key("k"), Duration::from_secs(120));
        reporter.settle(settled_result());
        assert!(matches!(
            tracker.poll(&key("k"), Duration::from_secs(60)),
            Poll::Settled(_)
        ));
    }

    #[test]
    fn settled_beats_a_dead_reporter() {
        // The queued result must win over the disconnect that follows it.
        let tracker = Tracker::default();
        let reporter = start(&tracker, &key("k"));
        reporter.settle(settled_result());
        assert!(matches!(tracker.poll(&key("k"), LIMIT), Poll::Settled(_)));
    }

    #[test]
    fn abandonment_trips_the_jobs_cancel_signal() {
        let tracker = Tracker::default();
        let reporter = start(&tracker, &key("k"));
        let cancel = reporter.cancel_signal();
        tracker.age(&key("k"), Duration::from_secs(120));
        assert!(!cancel.is_cancelled(), "live job is not cancelled");
        assert!(matches!(
            tracker.poll(&key("k"), Duration::from_secs(60)),
            Poll::Abandoned
        ));
        assert!(
            cancel.is_cancelled(),
            "giving up the slot must tell the thread"
        );
    }

    #[test]
    fn displacing_admission_trips_the_old_jobs_cancel_signal() {
        let tracker = Tracker::default();
        let stalled = start(&tracker, &key("k"));
        let cancel = stalled.cancel_signal();
        tracker.age(&key("k"), Duration::from_secs(120));
        assert!(matches!(
            tracker.admit(&key("k"), Duration::from_secs(60)),
            Admission::Started(_)
        ));
        assert!(
            cancel.is_cancelled(),
            "the displaced job must observe its disownment"
        );
    }

    #[test]
    fn runs_owned_effects_outside_the_jobs_lock() {
        // The effect re-enters the tracker: with a non-reentrant mutex this
        // deadlocks unless the effect provably runs after the lock is
        // released. The poll runs on a helper thread so a regression fails
        // the recv timeout instead of hanging the suite.
        let tracker = Arc::new(Tracker::default());
        let reporter = start(&tracker, &key("k"));
        let inner = Arc::clone(&tracker);
        reporter.settle(Settlement::with_owned_effect(settled_result(), move || {
            assert!(matches!(inner.poll(&key("other"), LIMIT), Poll::Vacant));
        }));
        let (tx, rx) = mpsc::channel();
        let polling = Arc::clone(&tracker);
        std::thread::spawn(move || {
            let _ = tx.send(polling.poll(&key("k"), LIMIT));
        });
        let polled = rx
            .recv_timeout(Duration::from_secs(10))
            .expect("an effect re-entering the tracker must not deadlock the poll");
        assert!(matches!(polled, Poll::Settled(_)));
    }

    #[test]
    fn runs_the_owned_effect_when_a_settlement_is_adopted() {
        let tracker = Tracker::default();
        let reporter = start(&tracker, &key("k"));
        let ran = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&ran);
        reporter.settle(Settlement::with_owned_effect(settled_result(), move || {
            flag.store(true, Ordering::SeqCst);
        }));
        assert!(matches!(tracker.poll(&key("k"), LIMIT), Poll::Settled(_)));
        assert!(
            ran.load(Ordering::SeqCst),
            "an adopted settlement's effect must land with its result"
        );
    }

    #[test]
    fn drops_a_late_settlements_owned_effect_after_abandonment() {
        let tracker = Tracker::default();
        let reporter = start(&tracker, &key("k"));
        tracker.age(&key("k"), Duration::from_secs(120));
        assert!(matches!(
            tracker.poll(&key("k"), Duration::from_secs(60)),
            Poll::Abandoned
        ));
        let ran = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&ran);
        reporter.settle(Settlement::with_owned_effect(settled_result(), move || {
            flag.store(true, Ordering::SeqCst);
        }));
        assert!(
            matches!(tracker.poll(&key("k"), LIMIT), Poll::Vacant),
            "a disowned settlement must not resurrect the slot"
        );
        assert!(
            !ran.load(Ordering::SeqCst),
            "a zombie's effect must never run — it would overwrite a retry's receipt"
        );
    }

    #[test]
    fn fires_the_disowned_compensation_when_no_slot_adopts_the_settlement() {
        let tracker = Tracker::default();
        let reporter = start(&tracker, &key("k"));
        tracker.age(&key("k"), Duration::from_secs(120));
        assert!(matches!(
            tracker.poll(&key("k"), Duration::from_secs(60)),
            Poll::Abandoned
        ));
        let compensated = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&compensated);
        reporter.settle(Settlement::from(settled_result()).or_disowned(move || {
            flag.store(true, Ordering::SeqCst);
        }));
        assert!(
            compensated.load(Ordering::SeqCst),
            "a settlement nobody adopts must undo its published effect"
        );
    }

    #[test]
    fn adoption_defuses_the_disowned_compensation() {
        let tracker = Tracker::default();
        let reporter = start(&tracker, &key("k"));
        let compensated = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&compensated);
        reporter.settle(Settlement::from(settled_result()).or_disowned(move || {
            flag.store(true, Ordering::SeqCst);
        }));
        assert!(matches!(tracker.poll(&key("k"), LIMIT), Poll::Settled(_)));
        assert!(
            !compensated.load(Ordering::SeqCst),
            "an adopted settlement's compensation must never fire"
        );
    }

    #[test]
    fn tracker_teardown_fires_a_queued_settlements_compensation() {
        // The settlement was sent but never folded: dropping the tracker
        // drops the slot, the channel, and the queued event — the
        // compensation must ride that drop.
        let tracker = Tracker::default();
        let reporter = start(&tracker, &key("k"));
        let compensated = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&compensated);
        reporter.settle(Settlement::from(settled_result()).or_disowned(move || {
            flag.store(true, Ordering::SeqCst);
        }));
        drop(tracker);
        assert!(
            compensated.load(Ordering::SeqCst),
            "teardown must not strand a published effect without its receipt"
        );
    }

    #[test]
    fn admits_exactly_one_writer_under_contention() {
        use std::sync::Barrier;
        let tracker = Arc::new(Tracker::default());
        let barrier = Arc::new(Barrier::new(2));
        let admissions: Vec<Admission> = [(); 2]
            .map(|()| {
                let tracker = Arc::clone(&tracker);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    tracker.admit(&key("k"), LIMIT)
                })
            })
            .into_iter()
            .map(|t| t.join().expect("admit does not panic"))
            .collect();
        let started = admissions
            .iter()
            .filter(|a| matches!(a, Admission::Started(_)))
            .count();
        assert_eq!(started, 1, "one writer wins the slot");
        assert_eq!(admissions.len(), 2);
        assert!(
            admissions
                .iter()
                .any(|a| matches!(a, Admission::Running { .. })),
            "the loser observes the winner's live job"
        );
    }

    #[test]
    fn admission_over_a_live_job_reports_running() {
        let tracker = Tracker::default();
        let reporter = start(&tracker, &key("k"));
        reporter.progress(512, Some(1024));
        let Admission::Running { downloaded, total } = tracker.admit(&key("k"), LIMIT) else {
            panic!("live job must not be displaced");
        };
        assert_eq!((downloaded, total), (512, Some(1024)));
    }

    #[test]
    fn admission_hands_out_an_unobserved_settled_response() {
        let tracker = Tracker::default();
        let reporter = start(&tracker, &key("k"));
        reporter.settle(settled_result());
        assert!(matches!(
            tracker.admit(&key("k"), LIMIT),
            Admission::Settled(_)
        ));
        assert!(
            matches!(tracker.admit(&key("k"), LIMIT), Admission::Started(_)),
            "handed out once; the slot is free again"
        );
    }

    #[test]
    fn admission_displaces_a_dead_job() {
        let tracker = Tracker::default();
        drop(start(&tracker, &key("k"))); // the download thread panicked
        assert!(
            matches!(tracker.admit(&key("k"), LIMIT), Admission::Started(_)),
            "a dead job must not block retries"
        );
    }

    #[test]
    fn admission_displaces_a_stalled_job() {
        let tracker = Tracker::default();
        let _reporter = start(&tracker, &key("k"));
        tracker.age(&key("k"), Duration::from_secs(120));
        assert!(
            matches!(
                tracker.admit(&key("k"), Duration::from_secs(60)),
                Admission::Started(_)
            ),
            "a wedged job must not block retries"
        );
    }
}
