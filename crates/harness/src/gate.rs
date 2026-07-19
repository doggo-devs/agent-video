// SPDX-License-Identifier: Apache-2.0
//! Admission gates for expensive work.
//!
//! Two shapes, one discipline: an admitted caller holds an RAII guard for
//! its critical section, and the guard's drop — normal or unwinding —
//! frees the slot, so a panic can never leak admission.
//! [`KeyedGate`] admits one holder per key: the "does the result already
//! exist" check and the run that produces it become a single critical
//! section, which is the whole single-flight guarantee. Waiters sign their
//! own wall-clock ceiling ([`KeyedGate::admit_within`]) because the
//! holder's work has no bound of its own. [`Semaphore`]
//! admits at most N holders total, capping jobs that each scale to the
//! whole machine. Excess callers block ([`Semaphore::acquire`]) by
//! default — for an agent, one blocked call is cheaper than a chain of
//! pending polls (the `sync_wait_seconds` stance in docs/config.md) — or
//! are refused outright ([`Semaphore::try_acquire`]) where blocking would
//! hand an unauthenticated boundary a way to pile up threads.

use std::collections::HashSet;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// One admitted holder per key; waiters for a held key block until it
/// frees. Distinct keys never contend.
#[derive(Default)]
pub struct KeyedGate {
    held: Mutex<HashSet<String>>,
    freed: Condvar,
}

impl KeyedGate {
    /// Hold `key` exclusively, waiting at most `bound` while it is held by
    /// someone else: `None` acquired nothing. The bound is the waiter's
    /// signature on its own account — a holder's legitimate work has no
    /// wall-clock ceiling (it may be transcribing hours of audio), so a
    /// bounded waiter must not inherit the holder's total; it waits its
    /// own bound and reports instead. The returned guard frees the key on
    /// drop — including an unwind, so a panicking holder wakes its waiters
    /// instead of wedging the key forever.
    pub fn admit_within(&self, key: &str, bound: Duration) -> Option<KeyHold<'_>> {
        let deadline = Instant::now() + bound;
        let mut held = self.held.lock().expect("keyed gate poisoned");
        while held.contains(key) {
            let left = deadline.checked_duration_since(Instant::now())?;
            held = self
                .freed
                .wait_timeout(held, left)
                .expect("keyed gate poisoned")
                .0;
        }
        held.insert(key.to_string());
        Some(KeyHold {
            gate: self,
            key: key.to_string(),
        })
    }
}

/// Exclusive hold on one key; dropping it admits the next waiter.
pub struct KeyHold<'a> {
    gate: &'a KeyedGate,
    key: String,
}

impl Drop for KeyHold<'_> {
    fn drop(&mut self) {
        self.gate
            .held
            .lock()
            .expect("keyed gate poisoned")
            .remove(&self.key);
        // One condvar for all keys, so a release wakes every waiter and
        // the wrong-key ones re-sleep. Per-key parking would need a
        // per-key waiter registry with its own lifecycle; at this gate's
        // load (transcription single-flight, single-digit waiters) that
        // is machinery for a stampede that cannot form.
        self.gate.freed.notify_all();
    }
}

/// At most `capacity` concurrent holders; excess callers block
/// ([`Self::acquire`]) or are refused ([`Self::try_acquire`]).
///
/// Lives behind an `Arc`: a [`Permit`] owns a reference to its semaphore,
/// so it can travel into a spawned thread and outlive the scope that
/// acquired it — the relay hands each connection's permit to its handler
/// thread.
pub struct Semaphore {
    free: Mutex<usize>,
    freed: Condvar,
}

impl Semaphore {
    pub fn new(capacity: usize) -> Self {
        Semaphore {
            free: Mutex::new(capacity),
            freed: Condvar::new(),
        }
    }

    /// Block until a slot frees, then hold it. The permit frees the slot
    /// on drop — including an unwind, so a panicking holder cannot shrink
    /// the capacity for everyone after it. Wakeup order is the condvar's,
    /// not FIFO: a waiter can be overtaken by a later arrival.
    pub fn acquire(self: &Arc<Self>) -> Permit {
        let mut free = self.free.lock().expect("semaphore poisoned");
        while *free == 0 {
            free = self.freed.wait(free).expect("semaphore poisoned");
        }
        *free -= 1;
        Permit {
            sem: Arc::clone(self),
        }
    }

    /// A permit if a slot is free, `None` at the cap — never blocks. The
    /// refusal shape for callers fronting an unauthenticated boundary,
    /// where a blocked thread per excess caller is itself the resource
    /// leak the gate exists to prevent.
    pub fn try_acquire(self: &Arc<Self>) -> Option<Permit> {
        let mut free = self.free.lock().expect("semaphore poisoned");
        if *free == 0 {
            return None;
        }
        *free -= 1;
        Some(Permit {
            sem: Arc::clone(self),
        })
    }
}

/// One held slot of a [`Semaphore`]; dropping it admits the next waiter.
pub struct Permit {
    sem: Arc<Semaphore>,
}

impl Drop for Permit {
    fn drop(&mut self) {
        *self.sem.free.lock().expect("semaphore poisoned") += 1;
        self.sem.freed.notify_one();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;

    /// Long enough that a wrongly-blocked thread is clearly blocked, short
    /// enough to keep tests fast.
    const BLOCKED: Duration = Duration::from_millis(100);
    /// Generous bound for "the freed waiter proceeds".
    const FREED: Duration = Duration::from_secs(5);

    #[test]
    fn keyed_gate_blocks_a_second_holder_of_the_same_key() {
        let gate = KeyedGate::default();
        let hold = gate.admit_within("k", Duration::ZERO).expect("key is free");
        let (tx, rx) = mpsc::channel();
        std::thread::scope(|s| {
            s.spawn(|| {
                let _second = gate.admit_within("k", FREED).expect("freed within bound");
                tx.send(()).expect("test channel");
            });
            assert!(
                rx.recv_timeout(BLOCKED).is_err(),
                "the key is held; the second admit must block"
            );
            drop(hold);
            rx.recv_timeout(FREED)
                .expect("dropping the hold admits the waiter");
        });
    }

    #[test]
    fn keyed_gate_admits_distinct_keys_concurrently() {
        let gate = KeyedGate::default();
        let _a = gate.admit_within("a", Duration::ZERO).expect("key is free");
        // No contention across keys: acquired without waiting at all.
        let _b = gate.admit_within("b", Duration::ZERO).expect("key is free");
    }

    #[test]
    fn admit_within_acquires_a_free_key_immediately() {
        let gate = KeyedGate::default();
        let hold = gate.admit_within("k", Duration::ZERO);
        assert!(hold.is_some(), "a free key needs no waiting at all");
    }

    #[test]
    fn admit_within_gives_up_at_its_bound_while_the_key_is_held() {
        let gate = KeyedGate::default();
        let _hold = gate.admit_within("k", Duration::ZERO).expect("key is free");
        let started = std::time::Instant::now();
        assert!(
            gate.admit_within("k", BLOCKED).is_none(),
            "the waiter signs its own ceiling; it must not inherit the holder's"
        );
        assert!(
            started.elapsed() < FREED,
            "the refusal lands at the bound, not eventually: {:?}",
            started.elapsed()
        );
    }

    #[test]
    fn admit_within_acquires_when_freed_inside_the_bound() {
        let gate = KeyedGate::default();
        let hold = gate.admit_within("k", Duration::ZERO).expect("key is free");
        std::thread::scope(|s| {
            let waiter = s.spawn(|| gate.admit_within("k", FREED).is_some());
            std::thread::sleep(BLOCKED / 2);
            drop(hold);
            assert!(
                waiter.join().expect("waiter completes"),
                "a key freed inside the bound is acquired, not refused"
            );
        });
    }

    #[test]
    fn keyed_gate_frees_the_key_when_the_holder_panics() {
        let gate = KeyedGate::default();
        std::thread::scope(|s| {
            let holder = s.spawn(|| {
                let _hold = gate.admit_within("k", Duration::ZERO).expect("key is free");
                panic!("holder dies mid-flight");
            });
            assert!(holder.join().is_err(), "the holder must have panicked");
        });
        let reacquired = gate.admit_within("k", Duration::ZERO);
        assert!(reacquired.is_some(), "the unwind freed the key");
    }

    #[test]
    fn semaphore_blocks_past_capacity_and_frees_on_release() {
        let sem = Arc::new(Semaphore::new(2));
        let first = sem.acquire();
        let _second = sem.acquire();
        let (tx, rx) = mpsc::channel();
        std::thread::scope(|s| {
            s.spawn(|| {
                let _third = sem.acquire();
                tx.send(()).expect("test channel");
            });
            assert!(
                rx.recv_timeout(BLOCKED).is_err(),
                "at capacity; the third acquire must block"
            );
            drop(first);
            rx.recv_timeout(FREED)
                .expect("releasing a permit admits the waiter");
        });
    }

    #[test]
    fn semaphore_frees_the_permit_when_a_holder_panics() {
        let sem = Arc::new(Semaphore::new(1));
        std::thread::scope(|s| {
            let holder = s.spawn(|| {
                let _permit = sem.acquire();
                panic!("holder dies mid-flight");
            });
            assert!(holder.join().is_err(), "the holder must have panicked");
        });
        let _reacquired = sem.acquire(); // returns: the unwind freed it
    }

    #[test]
    fn try_acquire_refuses_at_cap_and_frees_on_drop() {
        let sem = Arc::new(Semaphore::new(1));
        let permit = sem.try_acquire().expect("first permit");
        assert!(sem.try_acquire().is_none(), "gate is full at the cap");
        drop(permit);
        assert!(sem.try_acquire().is_some(), "dropped permit frees a slot");
    }
}
