// SPDX-License-Identifier: Apache-2.0
//! Background download tracker.
//!
//! Per-process registry of in-flight downloads keyed by URL. The download
//! thread owns a [`JobReporter`] and sends events; the tracker owns the
//! receiving half plus a folded snapshot, so no state is ever mutated from
//! two sides. Polls fold pending events, settled jobs hand their stored
//! response out exactly once.

use std::collections::HashMap;
use std::sync::{Mutex, mpsc};
use std::time::{Duration, Instant};

use rmcp::model::CallToolResult;

/// What the download thread reports; the thread shares no state.
enum JobEvent {
    Progress { downloaded: u64, total: Option<u64> },
    Settled(CallToolResult),
}

/// Write half of one job, owned by its download thread. Dropping it without
/// settling (a panic unwound the thread) is visible to the next poll as an
/// abandoned job.
#[derive(Debug)]
pub struct JobReporter(mpsc::Sender<JobEvent>);

impl JobReporter {
    /// Report live progress; never blocks.
    pub fn progress(&self, downloaded: u64, total: Option<u64>) {
        let _ = self.0.send(JobEvent::Progress { downloaded, total });
    }

    /// Deliver the final response. Consumes the reporter: a job settles at
    /// most once, by construction.
    pub fn settle(self, result: CallToolResult) {
        let _ = self.0.send(JobEvent::Settled(result));
    }
}

/// Read half plus folded snapshot of one job; owned by the tracker.
struct JobSlot {
    events: mpsc::Receiver<JobEvent>,
    downloaded: u64,
    total: Option<u64>,
    /// Stamped by the reader on every folded progress event (job start until
    /// the first report). The poll path abandons jobs whose thread stopped
    /// reporting — the guarantee that a wedged download can never pin its
    /// URL on Pending forever.
    last_progress: Instant,
}

/// What folding a slot's pending events resolved to.
enum Fold {
    /// The final response arrived; it beats every other observation.
    Settled(CallToolResult),
    /// The reporter is gone without settling: the thread died.
    Dead,
    /// Still connected; the snapshot is current.
    Live,
}

impl JobSlot {
    fn new(events: mpsc::Receiver<JobEvent>) -> Self {
        JobSlot {
            events,
            downloaded: 0,
            total: None,
            last_progress: Instant::now(),
        }
    }

    /// Drain pending events into the snapshot; a settled result
    /// short-circuits.
    fn fold(&mut self) -> Fold {
        loop {
            match self.events.try_recv() {
                Ok(JobEvent::Progress { downloaded, total }) => {
                    self.downloaded = downloaded;
                    if total.is_some() {
                        self.total = total;
                    }
                    self.last_progress = Instant::now();
                }
                Ok(JobEvent::Settled(result)) => return Fold::Settled(result),
                Err(mpsc::TryRecvError::Empty) => return Fold::Live,
                Err(mpsc::TryRecvError::Disconnected) => return Fold::Dead,
            }
        }
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
    /// slot is cleared so the caller can start over.
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
    jobs: Mutex<HashMap<String, JobSlot>>,
}

impl Tracker {
    /// Claim the slot for `key`: fold the existing job first (settled →
    /// hand its response out, dead or stalled → evict), then insert a fresh
    /// job if the slot is free. The liveness check and the insertion share
    /// one critical section, so the loser of a concurrent double confirm
    /// sees `Running` instead of spawning a second download.
    pub fn admit(&self, key: &str, stall_limit: Duration) -> Admission {
        let mut jobs = self.lock();
        if let Some(slot) = jobs.get_mut(key) {
            match slot.fold() {
                Fold::Settled(result) => {
                    jobs.remove(key);
                    return Admission::Settled(result);
                }
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
        jobs.insert(key.to_string(), JobSlot::new(rx));
        Admission::Started(JobReporter(tx))
    }

    /// Resolve one poll: settled beats dead beats stalled beats running,
    /// decided in a single critical section.
    pub fn poll(&self, key: &str, stall_limit: Duration) -> Poll {
        let mut jobs = self.lock();
        let Some(slot) = jobs.get_mut(key) else {
            return Poll::Vacant;
        };
        match slot.fold() {
            Fold::Settled(result) => {
                jobs.remove(key);
                Poll::Settled(result)
            }
            Fold::Dead => {
                jobs.remove(key);
                Poll::Abandoned
            }
            Fold::Live if slot.last_progress.elapsed() > stall_limit => {
                jobs.remove(key);
                Poll::Abandoned
            }
            Fold::Live => Poll::Running {
                downloaded: slot.downloaded,
                total: slot.total,
            },
        }
    }

    /// Drop a job without reading it (sync-window completions).
    pub fn clear(&self, key: &str) {
        self.lock().remove(key);
    }

    /// Rewind a job's progress clock by `by`, as if it had been silent that
    /// long. Test-only: real staleness comes from the wall clock.
    #[cfg(test)]
    pub fn age(&self, key: &str, by: Duration) {
        if let Some(slot) = self.lock().get_mut(key) {
            slot.last_progress = Instant::now().checked_sub(by).expect("clock has history");
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, JobSlot>> {
        self.jobs.lock().expect("tracker poisoned")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmcp::model::ContentBlock;

    /// Generous limit: these tests never exercise abandonment unless they
    /// rewind the clock explicitly.
    const LIMIT: Duration = Duration::from_secs(3600);

    fn settled_result() -> CallToolResult {
        CallToolResult::success(vec![ContentBlock::text("done")])
    }

    /// Claim a vacant slot; the tests below own the invariant that it is.
    fn start(tracker: &Tracker, key: &str) -> JobReporter {
        match tracker.admit(key, LIMIT) {
            Admission::Started(reporter) => reporter,
            other => panic!("slot must be vacant, got {other:?}"),
        }
    }

    #[test]
    fn polls_vacant_without_a_job() {
        let tracker = Tracker::default();
        assert!(matches!(tracker.poll("k", LIMIT), Poll::Vacant));
    }

    #[test]
    fn polls_running_with_live_progress() {
        let tracker = Tracker::default();
        let reporter = start(&tracker, "k");
        reporter.progress(512, Some(1024));
        let Poll::Running { downloaded, total } = tracker.poll("k", LIMIT) else {
            panic!("job is live");
        };
        assert_eq!((downloaded, total), (512, Some(1024)));
    }

    #[test]
    fn keeps_known_total_when_a_report_omits_it() {
        let tracker = Tracker::default();
        let reporter = start(&tracker, "k");
        reporter.progress(512, Some(1024));
        reporter.progress(600, None);
        let Poll::Running { downloaded, total } = tracker.poll("k", LIMIT) else {
            panic!("job is live");
        };
        assert_eq!((downloaded, total), (600, Some(1024)));
    }

    #[test]
    fn hands_out_settled_response_exactly_once() {
        let tracker = Tracker::default();
        let reporter = start(&tracker, "k");
        reporter.settle(settled_result());
        assert!(matches!(tracker.poll("k", LIMIT), Poll::Settled(_)));
        assert!(
            matches!(tracker.poll("k", LIMIT), Poll::Vacant),
            "cleared after handout"
        );
    }

    #[test]
    fn abandons_stalled_jobs_and_clears_the_slot() {
        let tracker = Tracker::default();
        let _reporter = start(&tracker, "k");
        tracker.age("k", Duration::from_secs(120));
        assert!(matches!(
            tracker.poll("k", Duration::from_secs(60)),
            Poll::Abandoned
        ));
        assert!(
            matches!(tracker.poll("k", LIMIT), Poll::Vacant),
            "slot is free for a retry"
        );
    }

    #[test]
    fn abandons_a_job_whose_reporter_died_unsettled() {
        let tracker = Tracker::default();
        let reporter = start(&tracker, "k");
        drop(reporter); // the download thread panicked
        assert!(matches!(tracker.poll("k", LIMIT), Poll::Abandoned));
        assert!(
            matches!(tracker.poll("k", LIMIT), Poll::Vacant),
            "slot is free for a retry"
        );
    }

    #[test]
    fn settled_beats_abandoned() {
        // Settled result queued against a stalled clock: the finished
        // download wins, clock staleness is irrelevant.
        let tracker = Tracker::default();
        let reporter = start(&tracker, "k");
        tracker.age("k", Duration::from_secs(120));
        reporter.settle(settled_result());
        assert!(matches!(
            tracker.poll("k", Duration::from_secs(60)),
            Poll::Settled(_)
        ));
    }

    #[test]
    fn settled_beats_a_dead_reporter() {
        // The queued result must win over the disconnect that follows it.
        let tracker = Tracker::default();
        let reporter = start(&tracker, "k");
        reporter.settle(settled_result());
        assert!(matches!(tracker.poll("k", LIMIT), Poll::Settled(_)));
    }

    #[test]
    fn clear_drops_job_unread() {
        let tracker = Tracker::default();
        let reporter = start(&tracker, "k");
        reporter.settle(settled_result());
        tracker.clear("k");
        assert!(matches!(tracker.poll("k", LIMIT), Poll::Vacant));
    }

    #[test]
    fn admits_exactly_one_writer_under_contention() {
        use std::sync::{Arc, Barrier};
        let tracker = Arc::new(Tracker::default());
        let barrier = Arc::new(Barrier::new(2));
        let admissions: Vec<Admission> = [(); 2]
            .map(|()| {
                let tracker = Arc::clone(&tracker);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    tracker.admit("k", LIMIT)
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
        let reporter = start(&tracker, "k");
        reporter.progress(512, Some(1024));
        let Admission::Running { downloaded, total } = tracker.admit("k", LIMIT) else {
            panic!("live job must not be displaced");
        };
        assert_eq!((downloaded, total), (512, Some(1024)));
    }

    #[test]
    fn admission_hands_out_an_unobserved_settled_response() {
        let tracker = Tracker::default();
        let reporter = start(&tracker, "k");
        reporter.settle(settled_result());
        assert!(matches!(tracker.admit("k", LIMIT), Admission::Settled(_)));
        assert!(
            matches!(tracker.admit("k", LIMIT), Admission::Started(_)),
            "handed out once; the slot is free again"
        );
    }

    #[test]
    fn admission_displaces_a_dead_job() {
        let tracker = Tracker::default();
        drop(start(&tracker, "k")); // the download thread panicked
        assert!(
            matches!(tracker.admit("k", LIMIT), Admission::Started(_)),
            "a dead job must not block retries"
        );
    }

    #[test]
    fn admission_displaces_a_stalled_job() {
        let tracker = Tracker::default();
        let _reporter = start(&tracker, "k");
        tracker.age("k", Duration::from_secs(120));
        assert!(
            matches!(
                tracker.admit("k", Duration::from_secs(60)),
                Admission::Started(_)
            ),
            "a wedged job must not block retries"
        );
    }
}
