// SPDX-License-Identifier: Apache-2.0
//! Bounded fan-out to a rayon pool with results delivered in spawn order.
//!
//! [`OrderedJobs`] assigns each job a sequence number, runs it on the caller's
//! pool, and re-orders completions so results come back exactly in spawn
//! order. `limit` is a hard cap on buffered jobs (running + completed but not
//! yet deliverable): [`OrderedJobs::spawn`] blocks until there is room, which
//! is what bounds memory when jobs carry large payloads.
//!
//! The dispatcher knows nothing about panics: a job that unwinds without
//! sending wedges its sequence number forever, so callers must catch unwinds
//! inside the job and encode them into `T`.

use std::collections::BTreeMap;
use std::sync::mpsc::{self, Receiver, Sender};

use anyhow::{Context, Result};

/// Order-preserving parallel job dispatcher with a hard buffering cap.
///
/// Methods must run outside the pool they spawn onto: they block on job
/// completions, so calling them from a pool thread can deadlock a
/// single-thread pool.
pub struct OrderedJobs<T> {
    /// Hard cap on buffered jobs (`outstanding + ready`).
    limit: usize,
    /// `None` once `finish` has run; jobs hold clones until they complete.
    tx: Option<Sender<(usize, T)>>,
    rx: Receiver<(usize, T)>,
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
    /// `job` must not unwind: catch panics inside and encode them into `T`.
    pub fn spawn(
        &mut self,
        pool: &rayon::ThreadPool,
        job: impl FnOnce(usize) -> T + Send + 'static,
    ) -> Result<Vec<T>> {
        let mut done = Vec::new();
        self.recv_until(self.limit - 1, &mut done)?;
        let seq = self.next_spawn;
        self.next_spawn += 1;
        self.outstanding += 1;
        self.peak_buffered = self.peak_buffered.max(self.outstanding + self.ready.len());
        let tx = self.tx.as_ref().expect("sender lives until finish").clone();
        pool.spawn_fifo(move || {
            let _ = tx.send((seq, job(seq)));
        });
        Ok(done)
    }

    /// Wait for every outstanding job and return the remaining results in
    /// spawn order. Errors if a job vanished without sending a result.
    pub fn finish(&mut self) -> Result<Vec<T>> {
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
    fn recv_until(&mut self, max_buffered: usize, done: &mut Vec<T>) -> Result<()> {
        loop {
            while let Ok((seq, result)) = self.rx.try_recv() {
                self.outstanding -= 1;
                self.ready.insert(seq, result);
            }

            while let Some(result) = self.ready.remove(&self.next_emit) {
                self.next_emit += 1;
                done.push(result);
            }

            if self.outstanding + self.ready.len() <= max_buffered {
                return Ok(());
            }

            let (seq, result) = self
                .rx
                .recv()
                .context("job disappeared without sending a result")?;
            self.outstanding -= 1;
            self.ready.insert(seq, result);
        }
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
}
