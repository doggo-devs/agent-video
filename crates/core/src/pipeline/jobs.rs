// SPDX-License-Identifier: Apache-2.0
//! Bounded fan-out to a rayon pool with results delivered in spawn order.
//!
//! [`OrderedJobs`] assigns each job a sequence number, runs it on the caller's
//! pool, and re-orders completions so results come back exactly in spawn
//! order. `limit` is a hard cap on buffered jobs (running + completed but not
//! yet deliverable): [`OrderedJobs::spawn`] blocks until there is room, which
//! is what bounds memory when jobs carry large payloads.
//!
//! The dispatcher owes no trust to its jobs: every spawned job reports
//! through a completion guard that sends on drop — normal return and unwind
//! alike — so a sequence number can never stay outstanding with no message
//! coming. A job that dies before producing a result surfaces as an error
//! from [`OrderedJobs::spawn`] or [`OrderedJobs::finish`], never as a wedge.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::mpsc::{self, Receiver, Sender};

/// A spawned job unwound before producing a result: its completion guard
/// fired during the unwind and delivered nothing. By this module's contract
/// that is a bug in the job, surfaced as a typed error so a generic caller
/// can still fail its run instead of wedging.
#[derive(Debug)]
pub struct JobPanicked {
    /// Sequence number of the job that died.
    pub seq: usize,
}

impl fmt::Display for JobPanicked {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "job {} panicked before producing a result", self.seq)
    }
}

impl std::error::Error for JobPanicked {}

/// Order-preserving parallel job dispatcher with a hard buffering cap.
///
/// Methods must run outside the pool they spawn onto: they block on job
/// completions, so calling them from a pool thread can deadlock a
/// single-thread pool.
pub struct OrderedJobs<T> {
    /// Hard cap on buffered jobs (`outstanding + ready`).
    limit: usize,
    /// `None` once `finish` has run; jobs hold clones until they complete.
    tx: Option<Sender<(usize, Option<T>)>>,
    rx: Receiver<(usize, Option<T>)>,
    /// Completed out-of-order results waiting for their turn.
    ready: BTreeMap<usize, T>,
    outstanding: usize,
    next_spawn: usize,
    next_emit: usize,
    peak_buffered: usize,
}

impl<T: Send + 'static> OrderedJobs<T> {
    /// A dispatcher that never buffers more than `limit` jobs.
    pub fn new(limit: std::num::NonZeroUsize) -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            limit: limit.get(),
            tx: Some(tx),
            rx,
            ready: BTreeMap::new(),
            outstanding: 0,
            next_spawn: 0,
            next_emit: 0,
            peak_buffered: 0,
        }
    }

    /// Spawn `job` on `pool` once fewer than `limit` jobs are buffered,
    /// blocking on completions as needed. The job receives its sequence
    /// number. Returns any results that became deliverable while waiting, in
    /// spawn order.
    ///
    /// A job that unwinds without producing a result surfaces as an error
    /// here or in [`Self::finish`]; catching panics inside the job and
    /// encoding them into `T` still gives the richer per-job error.
    pub fn spawn(
        &mut self,
        pool: &rayon::ThreadPool,
        job: impl FnOnce(usize) -> T + Send + 'static,
    ) -> Result<Vec<T>, JobPanicked> {
        let mut done = Vec::new();
        self.recv_until(self.limit - 1, &mut done)?;
        let seq = self.next_spawn;
        self.next_spawn += 1;
        self.outstanding += 1;
        self.peak_buffered = self.peak_buffered.max(self.outstanding + self.ready.len());
        let tx = self.tx.as_ref().expect("sender lives until finish").clone();
        pool.spawn_fifo(move || {
            let mut completion = Completion {
                tx,
                seq,
                result: None,
            };
            completion.result = Some(job(seq));
        });
        Ok(done)
    }

    /// Wait for every outstanding job and return the remaining results in
    /// spawn order. Errors if a job died without producing a result.
    pub fn finish(&mut self) -> Result<Vec<T>, JobPanicked> {
        self.tx = None;
        let mut done = Vec::new();
        self.recv_until(0, &mut done)?;
        Ok(done)
    }

    /// High-water mark of buffered jobs, observed at each spawn. Lets tests
    /// assert that `limit` is a hard cap.
    #[cfg(test)]
    pub fn peak_buffered(&self) -> usize {
        self.peak_buffered
    }

    /// Absorb completions (blocking while more than `max_buffered` jobs are
    /// buffered) and move deliverable results into `done` in spawn order.
    fn recv_until(&mut self, max_buffered: usize, done: &mut Vec<T>) -> Result<(), JobPanicked> {
        loop {
            while let Ok((seq, result)) = self.rx.try_recv() {
                self.absorb(seq, result)?;
            }

            while let Some(result) = self.ready.remove(&self.next_emit) {
                self.next_emit += 1;
                done.push(result);
            }

            if self.outstanding + self.ready.len() <= max_buffered {
                return Ok(());
            }

            // Never a wedge: every outstanding job holds a completion guard
            // whose drop sends, so this recv is always fulfilled. Disconnect
            // with jobs outstanding is unreachable by construction — a
            // violated invariant here is a bug, and bugs are panics.
            let (seq, result) = self
                .rx
                .recv()
                .expect("every outstanding job holds a completion guard that sends on drop");
            self.absorb(seq, result)?;
        }
    }

    /// Account one completion. `None` is a job that died without a result —
    /// its guard fired during an unwind — reported as this dispatcher's
    /// error instead of a forever-missing sequence number.
    fn absorb(&mut self, seq: usize, result: Option<T>) -> Result<(), JobPanicked> {
        self.outstanding -= 1;
        let result = result.ok_or(JobPanicked { seq })?;
        self.ready.insert(seq, result);
        Ok(())
    }
}

/// Reports a job's completion on drop: a normal return carries
/// `Some(result)`, an unwind out of the job carries `None`. Tying the send
/// to drop removes the state the dispatcher could not recover from — a
/// spawned sequence number that never reports.
struct Completion<T> {
    tx: Sender<(usize, Option<T>)>,
    seq: usize,
    result: Option<T>,
}

impl<T> Drop for Completion<T> {
    fn drop(&mut self) {
        // A closed channel means the dispatcher already gave up on results.
        let _ = self.tx.send((self.seq, self.result.take()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroUsize;
    use std::sync::mpsc;

    fn pool(threads: usize) -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("test pool builds")
    }

    fn limit(n: usize) -> NonZeroUsize {
        NonZeroUsize::new(n).expect("test limit is nonzero")
    }

    /// A single-thread pool whose panic handler swallows unwinds, so a dead
    /// job leaves the pool alive.
    fn panic_swallowing_pool() -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .panic_handler(|_| {})
            .build()
            .expect("test pool builds")
    }

    /// A job that unwinds before producing a result. `resume_unwind` skips
    /// the panic hook: no backtrace noise in test output.
    fn die(_seq: usize) -> usize {
        std::panic::resume_unwind(Box::new("job dies before sending"))
    }

    #[test]
    fn delivers_results_in_spawn_order() {
        let pool = pool(2);
        let mut jobs = OrderedJobs::new(limit(2));
        // Job 0 blocks until job 1 has finished, forcing out-of-order
        // completion; delivery order must still be spawn order.
        let (unblock, blocked) = mpsc::channel::<()>();

        let mut delivered = jobs
            .spawn(&pool, move |seq| {
                blocked.recv().expect("job 1 unblocks job 0");
                seq
            })
            .expect("spawn job 0");
        delivered.extend(
            jobs.spawn(&pool, move |seq| {
                unblock.send(()).expect("job 0 still waiting");
                seq
            })
            .expect("spawn job 1"),
        );
        delivered.extend(jobs.finish().expect("finish"));

        assert_eq!(delivered, vec![0, 1]);
    }

    #[test]
    fn caps_buffered_jobs_at_limit() {
        let pool = pool(2);
        let mut jobs = OrderedJobs::new(limit(1));

        let mut delivered = Vec::new();
        for _ in 0..8 {
            delivered.extend(jobs.spawn(&pool, |seq| seq).expect("spawn"));
        }
        delivered.extend(jobs.finish().expect("finish"));

        assert!(
            jobs.peak_buffered() <= 1,
            "limit violated: peak {}",
            jobs.peak_buffered()
        );
        assert_eq!(delivered, (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn finish_drains_all_outstanding_jobs() {
        let pool = pool(2);
        let mut jobs = OrderedJobs::new(limit(4));

        for _ in 0..4 {
            jobs.spawn(&pool, |seq| seq * 10).expect("spawn");
        }
        let delivered = jobs.finish().expect("finish");

        assert_eq!(delivered, vec![0, 10, 20, 30]);
    }

    #[test]
    fn surfaces_a_vanished_job_instead_of_blocking_spawn() {
        // The job dies and the pool lives on. On the old shape the next
        // spawn blocked forever on the wedged sequence number; the
        // completion guard now reports the death and spawn errors out.
        let pool = panic_swallowing_pool();
        let mut jobs: OrderedJobs<usize> = OrderedJobs::new(limit(1));

        jobs.spawn(&pool, die).expect("first spawn has room");

        let err = jobs
            .spawn(&pool, |seq| seq)
            .expect_err("dead job must surface as an error, not a wedge");
        assert!(err.to_string().contains("panicked"), "got: {err:#}");
    }

    #[test]
    fn surfaces_a_vanished_job_through_finish() {
        // Same death as above, but with no further spawn the drain in
        // finish must be the path that reports it.
        let pool = panic_swallowing_pool();
        let mut jobs: OrderedJobs<usize> = OrderedJobs::new(limit(1));

        jobs.spawn(&pool, die).expect("first spawn has room");

        let err = jobs
            .finish()
            .expect_err("finish must surface the dead job, not return results");
        assert_eq!(err.seq, 0);
    }

    #[test]
    fn drops_completion_silently_after_dispatcher_is_gone() {
        // Sends on drop, so the test observes the moment strictly after the
        // completion guard has finished reporting into a closed channel.
        struct DoneOnDrop(mpsc::Sender<()>);
        impl Drop for DoneOnDrop {
            fn drop(&mut self) {
                let _ = self.0.send(());
            }
        }

        // No panic handler: if the guard's drop panicked, the spawned task
        // would abort the process and fail the test loudly.
        let pool = pool(1);
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let (done_tx, done_rx) = mpsc::channel::<()>();

        let mut jobs = OrderedJobs::new(limit(1));
        jobs.spawn(&pool, move |_| {
            release_rx.recv().expect("dispatcher drop releases the job");
            DoneOnDrop(done_tx)
        })
        .expect("first spawn has room");

        // The dispatcher gives up while the job is still running: its
        // receiver closes, so the guard's send has nowhere to land.
        drop(jobs);
        release_tx.send(()).expect("job is waiting for release");

        done_rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("guard completes without wedging or panicking");
    }
}
