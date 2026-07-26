// SPDX-License-Identifier: Apache-2.0
//! Core invocation from the MCP server: the binary spawns itself.
//!
//! The server and the pipeline are one binary, but a `video` call still runs
//! `process`/`extract` in a child process rather than in-process: stdout
//! belongs to the MCP stdio transport, and this crate builds with
//! `panic = "abort"`, so an in-process decode panic would take the whole
//! server down. Self-spawn buys stdout isolation and crash containment for
//! the cost of one fork. Children are admitted through a small gate
//! (`MAX_CORE_CHILDREN`): each one scales to every core, so running them
//! unbounded would trample the machine, not parallelize it.

use std::num::NonZeroI32;
use std::process::{Command, ExitStatus};
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use crate::gate::Semaphore;

/// How long one core run may take before it is killed. Server-side core
/// work is local decode/compose on an already-fetched file — no network —
/// so even a long video on slow hardware finishes well inside this. A
/// child alive past it is wedged, and unbounded it would pin its server
/// thread (and the MCP call riding on it) until process exit.
const CORE_DEADLINE: Duration = Duration::from_secs(10 * 60);

/// How many core children may run at once. Each child is sized to the
/// whole machine — its ffmpeg decode threads and rayon compose pool both
/// scale to every core — so parallel children multiply oversubscription,
/// not throughput; two lets one child's compute run under another's I/O
/// stalls.
const MAX_CORE_CHILDREN: usize = 2;

/// Admission for [`SelfSpawn`] children. Lives at the spawn seam so every
/// production core run is gated while the trait's test fakes stay free.
static CORE_CHILDREN: LazyLock<Arc<Semaphore>> =
    LazyLock::new(|| Arc::new(Semaphore::new(MAX_CORE_CHILDREN)));

/// How a core child terminated. `Failure(0)` is unrepresentable, so
/// "succeeded" and "has exit code 0" cannot drift apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreExit {
    Success,
    /// Exited with a nonzero code.
    Failure(NonZeroI32),
    /// Killed by a signal; no exit code exists.
    Signal,
    /// Still running at `CORE_DEADLINE`; killed by the harness.
    Hung,
}

impl CoreExit {
    pub fn is_success(self) -> bool {
        matches!(self, CoreExit::Success)
    }
}

impl From<ExitStatus> for CoreExit {
    fn from(status: ExitStatus) -> Self {
        match status.code() {
            None => CoreExit::Signal,
            Some(code) => match NonZeroI32::new(code) {
                None => CoreExit::Success,
                Some(code) => CoreExit::Failure(code),
            },
        }
    }
}

impl std::fmt::Display for CoreExit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CoreExit::Success => f.write_str("exit 0"),
            CoreExit::Failure(code) => write!(f, "exit {code}"),
            CoreExit::Signal => f.write_str("killed by a signal"),
            CoreExit::Hung => write!(f, "hung for {}s and was killed", CORE_DEADLINE.as_secs()),
        }
    }
}

/// Captured outcome of one core run.
#[derive(Debug, Clone)]
pub struct CoreRun {
    pub exit: CoreExit,
    pub stdout: String,
    /// True when stdout blew the capture cap: the bytes are an arbitrary
    /// prefix and parsing must not be attempted, let alone diagnosed as
    /// a wire-contract bug.
    pub stdout_truncated: bool,
    pub stderr: String,
}

/// Runs core subcommands; injectable for tests.
pub trait CoreRunner: Send + Sync {
    fn run(&self, args: &[String]) -> std::io::Result<CoreRun>;
}

/// Production runner: spawn the current executable.
pub struct SelfSpawn;

impl CoreRunner for SelfSpawn {
    fn run(&self, args: &[String]) -> std::io::Result<CoreRun> {
        // RAII permit: held across the child's whole life, freed on drop —
        // panics included — so a failed run can never leak a slot.
        let _slot = CORE_CHILDREN.acquire();
        let exe = std::env::current_exe()?;
        // Deadline-bounded with capped captures (see [`crate::proc`]): a
        // wedged child cannot pin this thread, and runaway stderr cannot
        // balloon memory. Stdout truncation is carried explicitly so the
        // handler answers it as bad input (narrow the request) rather than
        // misdiagnosing the clipped bytes as a JSON parse bug.
        let out = crate::proc::output_with_deadline(Command::new(exe).args(args), CORE_DEADLINE)?;
        Ok(CoreRun {
            exit: out.status.map_or(CoreExit::Hung, CoreExit::from),
            stdout: String::from_utf8_lossy(&out.stdout.bytes).into_owned(),
            stdout_truncated: out.stdout.truncated,
            stderr: String::from_utf8_lossy(&out.stderr.bytes).into_owned(),
        })
    }
}
