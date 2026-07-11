// SPDX-License-Identifier: Apache-2.0
//! Bounded child processes.
//!
//! Every external tool invocation that can hang on the network gets a
//! deadline: a child that outlives it is killed and reaped, so a stuck
//! yt-dlp can never pin a server thread (and the MCP call riding on it)
//! until process exit.

use std::io::Read;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

/// How often deadline waits poll `try_wait`. Cheap against process
/// lifetimes measured in seconds.
const POLL: Duration = Duration::from_millis(50);

/// Grace for pipe drains after the child is gone. A group kill closes the
/// write ends immediately, so this only bites when a process outside the
/// kill group (its own setsid, or a straggler left behind by a normally
/// exiting child) still holds a pipe: that capture is abandoned rather than
/// blocking the caller forever.
const DRAIN_GRACE: Duration = Duration::from_secs(2);

/// Hard per-pipe ceiling on captured bytes. Legitimate tool output here is
/// megabytes at most; anything past this is a hostile or broken source, so
/// capture stops while the pipe is still drained to EOF.
const CAPTURE_LIMIT: u64 = 64 * 1024 * 1024;

/// One captured pipe: at most `CAPTURE_LIMIT` bytes, with overflow
/// recorded rather than silently dropped.
#[derive(Debug, Default)]
pub struct Capture {
    pub bytes: Vec<u8>,
    /// True when the pipe produced more than `CAPTURE_LIMIT` bytes; the
    /// excess was read and discarded so the child never blocked on a full
    /// pipe. Callers must treat truncated bytes as unparseable.
    pub truncated: bool,
}

/// Outcome of a deadline-bounded run.
#[derive(Debug)]
pub struct Outcome {
    /// The exit status, or `None` when the deadline killed the child.
    pub status: Option<ExitStatus>,
    pub stdout: Capture,
    pub stderr: Capture,
}

/// Run `cmd` to completion or `deadline`, whichever comes first. On expiry
/// the child — and, on Unix, its whole process group, so a grandchild
/// (yt-dlp's ffmpeg) inheriting the pipes dies with it — is killed and
/// reaped, and `status` is `None`. Both pipes are drained concurrently, so
/// a chatty child cannot deadlock against a full pipe buffer; each capture
/// is capped at `CAPTURE_LIMIT` bytes, so a hostile source cannot balloon
/// memory within the deadline window. A pipe held open past the child by a
/// process outside the kill group is abandoned after `DRAIN_GRACE` (its
/// capture comes back empty), so the deadline bounds the call regardless.
pub fn output_with_deadline(cmd: &mut Command, deadline: Duration) -> std::io::Result<Outcome> {
    let mut child = isolate_group(cmd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let stdout = drain(child.stdout.take());
    let stderr = drain(child.stderr.take());
    let status = wait_with_deadline(&mut child, deadline);
    Ok(Outcome {
        status,
        stdout: stdout.recv_timeout(DRAIN_GRACE).unwrap_or_default(),
        stderr: stderr.recv_timeout(DRAIN_GRACE).unwrap_or_default(),
    })
}

/// Make the spawned child lead its own process group (Unix), so
/// [`kill_tree`] can take down its entire tree. No-op elsewhere.
pub fn isolate_group(cmd: &mut Command) -> &mut Command {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    cmd
}

/// Kill `child` — and, on Unix, the process group it leads — then reap it.
/// Signaling the group only reaches grandchildren when the child was
/// spawned through [`isolate_group`]; a plain child is still killed
/// directly.
pub fn kill_tree(child: &mut Child) {
    #[cfg(unix)]
    // SAFETY: plain FFI signal send, no memory involved. The child is not
    // yet reaped (reaped below, and `&mut Child` excludes a concurrent
    // wait), so its pid — and thus the pgid it leads — cannot have been
    // recycled to an unrelated process.
    unsafe {
        libc::kill(-(child.id() as i32), libc::SIGKILL);
    }
    let _ = child.kill();
    let _ = child.wait();
}

/// Interrupt `child` — and, on Unix, the process group it leads — with
/// SIGINT first, giving cleanup duties `grace` to finish: yt-dlp's
/// KeyboardInterrupt path saves rotated cookies back to its jar and ffmpeg
/// finalizes its container. A child still alive past the grace is taken
/// down by [`kill_tree`]. Non-Unix has no group signal: hard kill at once.
pub fn kill_tree_graceful(child: &mut Child, grace: Duration) {
    #[cfg(unix)]
    {
        // SAFETY: as in [`kill_tree`] — a plain FFI signal send; the child
        // is not yet reaped (`&mut Child` excludes a concurrent wait), so
        // its pid/pgid cannot have been recycled.
        unsafe {
            libc::kill(-(child.id() as i32), libc::SIGINT);
        }
        // On expiry `wait_with_deadline` escalates to [`kill_tree`] itself.
        let _ = wait_with_deadline(child, grace);
    }
    #[cfg(not(unix))]
    kill_tree(child);
}

/// Read a pipe to EOF on a detached thread, keeping at most
/// [`CAPTURE_LIMIT`] bytes, and deliver the capture over a channel. The
/// thread is deliberately not joined: a receiver that gives up (see
/// [`DRAIN_GRACE`]) just drops the channel, and the thread exits whenever
/// the pipe finally closes.
fn drain(pipe: Option<impl Read + Send + 'static>) -> Receiver<Capture> {
    let (tx, rx) = mpsc::sync_channel(1);
    std::thread::spawn(move || {
        let mut bytes = Vec::new();
        let mut truncated = false;
        if let Some(mut pipe) = pipe {
            let _ = (&mut pipe).take(CAPTURE_LIMIT).read_to_end(&mut bytes);
            // Drain the overflow: a stopped reader would stall the child on a
            // full pipe and turn a size problem into a spurious deadline kill.
            truncated = std::io::copy(&mut pipe, &mut std::io::sink()).is_ok_and(|n| n > 0);
        }
        let _ = tx.send(Capture { bytes, truncated });
    });
    rx
}

/// Wait for `child` up to `deadline`; kill (the whole tree, see
/// [`kill_tree`]) and reap on expiry — or on a wait error, where the
/// child's state is unknowable.
///
/// Polling keeps ownership simple: `Child::wait` cannot be raced against a
/// timer without handing the child to another thread.
pub fn wait_with_deadline(child: &mut Child, deadline: Duration) -> Option<ExitStatus> {
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Some(status),
            Ok(None) if start.elapsed() < deadline => std::thread::sleep(POLL),
            _ => {
                kill_tree(child);
                return None;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collects_output_of_a_completing_child() {
        let out = output_with_deadline(
            Command::new("sh").args(["-c", "echo out; echo err >&2"]),
            Duration::from_secs(10),
        )
        .expect("spawn sh");
        assert!(out.status.expect("completed").success());
        assert_eq!(String::from_utf8_lossy(&out.stdout.bytes).trim(), "out");
        assert_eq!(String::from_utf8_lossy(&out.stderr.bytes).trim(), "err");
        assert!(!out.stdout.truncated);
    }

    #[test]
    fn caps_a_pipe_at_the_capture_limit() {
        let over = CAPTURE_LIMIT + 1024 * 1024;
        let out = output_with_deadline(
            Command::new("sh").args(["-c", &format!("head -c {over} /dev/zero")]),
            Duration::from_secs(30),
        )
        .expect("spawn sh");
        assert!(
            out.status
                .expect("the child must run to completion")
                .success(),
            "draining past the cap keeps the child from blocking on a full pipe"
        );
        assert_eq!(out.stdout.bytes.len() as u64, CAPTURE_LIMIT);
        assert!(out.stdout.truncated);
        assert!(!out.stderr.truncated);
    }

    #[test]
    fn kills_grandchildren_holding_the_pipe_at_the_deadline() {
        // The grandchild inherits stdout; only a group kill closes its
        // write end. A lone child.kill() would leave the drain blocked
        // until the DRAIN_GRACE fallback, well past the assertion below.
        let start = Instant::now();
        let out = output_with_deadline(
            Command::new("sh").args(["-c", "sleep 30 & wait"]),
            Duration::from_millis(200),
        )
        .expect("spawn sh");
        assert!(out.status.is_none(), "deadline reports no exit status");
        assert!(
            start.elapsed() < Duration::from_secs(1),
            "the grandchild must die with the group, not hold the pipes open"
        );
    }

    #[test]
    fn abandons_a_pipe_held_past_a_normal_exit() {
        // The child exits fine but leaves a straggler holding stdout: no
        // kill fires, so only the drain grace keeps the call bounded. The
        // capture is forfeited — unknowable EOF means unusable bytes.
        let start = Instant::now();
        let out = output_with_deadline(
            Command::new("sh").args(["-c", "echo lost; sleep 30 &"]),
            Duration::from_secs(10),
        )
        .expect("spawn sh");
        assert!(out.status.expect("the child itself exited").success());
        assert!(out.stdout.bytes.is_empty(), "an abandoned capture is empty");
        assert!(
            start.elapsed() < Duration::from_secs(8),
            "the straggler must not pin the call past the drain grace"
        );
    }

    #[cfg(unix)]
    #[test]
    fn graceful_kill_lets_the_child_run_its_int_trap() {
        // Models yt-dlp's KeyboardInterrupt cleanup (cookie jar save): the
        // trap must run to completion before the process dies.
        let dir = crate::testutil::TempDir::new("av-proc-int");
        let marker = dir.join("saved");
        let _ = std::fs::remove_file(&marker);
        let mut child = isolate_group(Command::new("sh").args([
            "-c",
            &format!(
                "trap 'echo done > {}; exit 0' INT; sleep 30",
                marker.display()
            ),
        ]))
        .spawn()
        .expect("spawn sh");
        // Let sh install the trap before the signal arrives.
        std::thread::sleep(Duration::from_millis(200));

        kill_tree_graceful(&mut child, Duration::from_secs(5));

        assert!(marker.is_file(), "the INT trap must have run");
    }

    #[test]
    fn kills_a_child_past_the_deadline() {
        let start = Instant::now();
        let out = output_with_deadline(Command::new("sleep").arg("30"), Duration::from_millis(200))
            .expect("spawn sleep");
        assert!(out.status.is_none(), "deadline reports no exit status");
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "the child must die at the deadline, not at its own pace"
        );
    }
}
