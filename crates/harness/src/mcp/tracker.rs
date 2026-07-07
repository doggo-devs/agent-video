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

/// In-flight downloads for one MCP session.
#[derive(Default)]
pub struct Tracker {
    /// The only lock; flat, values are plain owned data.
    jobs: Mutex<HashMap<String, JobSlot>>,
}

impl Tracker {
    /// Register a job; returns the reporter its download thread speaks
    /// through.
    pub fn start(&self, key: &str) -> JobReporter {
        let (tx, rx) = mpsc::channel();
        self.lock().insert(key.to_string(), JobSlot::new(rx));
        JobReporter(tx)
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

    #[test]
    fn polls_vacant_without_a_job() {
        let tracker = Tracker::default();
        assert!(matches!(tracker.poll("k", LIMIT), Poll::Vacant));
    }

    #[test]
    fn polls_running_with_live_progress() {
        let tracker = Tracker::default();
        let reporter = tracker.start("k");
        reporter.progress(512, Some(1024));
        let Poll::Running { downloaded, total } = tracker.poll("k", LIMIT) else {
            panic!("job is live");
        };
        assert_eq!((downloaded, total), (512, Some(1024)));
    }

    #[test]
    fn keeps_known_total_when_a_report_omits_it() {
        let tracker = Tracker::default();
        let reporter = tracker.start("k");
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
        let reporter = tracker.start("k");
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
        let _reporter = tracker.start("k");
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
        let reporter = tracker.start("k");
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
        let reporter = tracker.start("k");
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
        let reporter = tracker.start("k");
        reporter.settle(settled_result());
        assert!(matches!(tracker.poll("k", LIMIT), Poll::Settled(_)));
    }

    #[test]
    fn clear_drops_job_unread() {
        let tracker = Tracker::default();
        let reporter = tracker.start("k");
        reporter.settle(settled_result());
        tracker.clear("k");
        assert!(matches!(tracker.poll("k", LIMIT), Poll::Vacant));
    }
}
