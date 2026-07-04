//! Child process management with stderr capture and clean shutdown.
//!
//! ffmpeg and ffprobe are external processes that can fail for many reasons
//! (missing codecs, corrupt files, permission errors). This module provides
//! [`TrackedChild`], a wrapper that captures stderr in a background thread
//! and includes it in error messages when the process fails.

use std::io::{BufReader, Read};
use std::process::{Child, ChildStdout, Command, ExitStatus, Stdio};
use std::thread::{self, JoinHandle};

use crate::error::{Error, ErrorCode, ErrorContext};

/// Cap on captured child stderr. ffmpeg on a corrupt long input can emit
/// per-frame errors indefinitely; only the tail carries the fatal message, so
/// we keep the last [`STDERR_CAP`] bytes and drop the rest.
const STDERR_CAP: usize = 64 * 1024;

/// A spawned child process with its stderr captured in a background thread.
///
/// The stdout handle is returned separately so the caller can read from it.
/// When the caller is done, call [`TrackedChild::wait`] to join the stderr
/// thread and check the exit status. If the process failed, the captured
/// stderr is included in the error message.
///
/// Implements [`Drop`] to kill and wait on the child if not explicitly consumed,
/// preventing zombie processes on early returns.
pub struct TrackedChild {
    child: Child,
    /// Joins to the captured stderr tail and whether earlier bytes were dropped.
    stderr_thread: Option<JoinHandle<(Vec<u8>, bool)>>,
}

/// Error from a failed child process, including captured stderr.
#[derive(Debug)]
pub struct ChildError {
    /// The child's exit status, or `None` when `wait()` itself failed.
    pub status: Option<ExitStatus>,
    pub stderr: String,
    pub context: &'static str,
}

impl std::fmt::Display for ChildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.status {
            Some(status) => write!(f, "{} ({status})", self.context)?,
            None => write!(f, "{} (wait failed)", self.context)?,
        }
        if !self.stderr.is_empty() {
            write!(f, ": {}", self.stderr)?;
        }
        Ok(())
    }
}

impl std::error::Error for ChildError {}

impl ChildError {
    /// Convert to structured Error with appropriate code.
    pub fn into_error(self, code: ErrorCode) -> Error {
        let message = self.to_string();
        let mut ctx = ErrorContext::default().with_stderr(self.stderr);
        if let Some(status) = self.status {
            ctx = ctx.with_exit_status(status);
        }
        Error::new(code, message).with_context(ctx)
    }
}

impl Drop for TrackedChild {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(t) = self.stderr_thread.take() {
            let _ = t.join();
        }
    }
}

impl TrackedChild {
    /// Spawn a command with stdout piped and stderr captured.
    ///
    /// Returns the tracked child and its stdout handle. The command's stdio is
    /// configured automatically (any prior configuration is overwritten):
    /// stdin is nulled so an interactive child — ffmpeg reads keyboard commands
    /// when stdin is a TTY — cannot consume terminal input or stall on it.
    ///
    /// `not_found_code` names the tool (typically `FfmpegNotFound` or
    /// `FfprobeNotFound`); spawn failures are classified by cause via
    /// [`spawn_error_code`].
    pub fn spawn(
        cmd: &mut Command,
        spawn_context: &'static str,
        not_found_code: ErrorCode,
    ) -> crate::error::Result<(Self, ChildStdout)> {
        let mut child = cmd
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                Error::new(
                    spawn_error_code(&e, not_found_code),
                    format!("failed to spawn {spawn_context}: {e}"),
                )
                .with_source(e)
            })?;

        let stdout = child.stdout.take().expect("stdout was piped");
        let stderr = child.stderr.take().expect("stderr was piped");

        let stderr_thread = thread::spawn(move || match capture_tail(BufReader::new(stderr)) {
            Ok(captured) => captured,
            Err(e) => {
                eprintln!("warning: failed to capture stderr: {e}");
                (Vec::new(), false)
            }
        });

        Ok((
            Self {
                child,
                stderr_thread: Some(stderr_thread),
            },
            stdout,
        ))
    }

    /// Wait for the child to exit and check its status.
    ///
    /// On success (exit code 0), returns `Ok(())`. On failure, returns an
    /// error that includes the captured stderr.
    pub fn wait(mut self, context: &'static str) -> Result<(), ChildError> {
        let wait_result = self.child.wait();
        let (stderr_bytes, truncated) = self
            .stderr_thread
            .take()
            .map(|t| {
                t.join().unwrap_or_else(|_| {
                    eprintln!("warning: stderr capture thread panicked");
                    (Vec::new(), false)
                })
            })
            .unwrap_or_default();
        let stderr = decode_stderr(&stderr_bytes, truncated);

        match wait_result {
            Ok(status) if status.success() => Ok(()),
            Ok(status) => Err(ChildError {
                status: Some(status),
                stderr,
                context,
            }),
            // waitpid can fail (e.g. ECHILD when a parent set SIGCHLD to
            // SIG_IGN); fold the io error into the message rather than panic.
            Err(e) => {
                let stderr = if stderr.is_empty() {
                    format!("wait() failed: {e}")
                } else {
                    format!("wait() failed: {e}; {stderr}")
                };
                Err(ChildError {
                    status: None,
                    stderr,
                    context,
                })
            }
        }
    }

    /// Kill the child process if still running, then wait.
    ///
    /// Consumes self; cleanup happens via Drop.
    pub fn kill_and_wait(self) {
        // Drop impl handles the actual cleanup
        drop(self);
    }

    /// Kill the child, reap it, and return whatever stderr it wrote.
    ///
    /// For error paths where the stdout consumer failed mid-stream: the child
    /// may still be alive and blocked writing to the full pipe, so a plain
    /// `wait` could deadlock — it must be killed first, and its stderr often
    /// carries the real cause of the consumer's failure.
    pub fn kill_collect_stderr(mut self) -> String {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let (bytes, truncated) = self
            .stderr_thread
            .take()
            .and_then(|t| t.join().ok())
            .unwrap_or_default();
        decode_stderr(&bytes, truncated)
    }
}

/// Run a tool to completion, returning its stdout lossily decoded.
///
/// The collect-mode sibling of [`TrackedChild::spawn`], for short probe
/// commands: stdout is capped at `stdout_cap` bytes — a stream past the cap
/// fails as runaway instead of buffering without bound — and stderr gets the
/// usual tail-capped capture, riding along on failures. A non-zero exit maps
/// to `failed_code`; spawn failures are classified via [`spawn_error_code`]
/// with `not_found_code`.
pub fn run_capped(
    cmd: &mut Command,
    context: &'static str,
    not_found_code: ErrorCode,
    failed_code: ErrorCode,
    stdout_cap: u64,
) -> crate::error::Result<String> {
    let (child, stdout) = TrackedChild::spawn(cmd, context, not_found_code)?;

    let mut buf = Vec::new();
    // One byte past the cap distinguishes "exactly cap" from "runaway".
    // Error paths return before `wait`, dropping the child: it may still be
    // blocked writing to the abandoned pipe, and Drop kills before reaping.
    if let Err(e) = stdout.take(stdout_cap + 1).read_to_end(&mut buf) {
        return Err(
            Error::new(failed_code, format!("reading {context} output: {e}")).with_source(e),
        );
    }
    if buf.len() as u64 > stdout_cap {
        return Err(Error::new(
            failed_code,
            format!(
                "{context} produced more than {stdout_cap} bytes on stdout; \
                 refusing to buffer runaway output"
            ),
        ));
    }

    child.wait(context).map_err(|e| e.into_error(failed_code))?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

/// Classify a process-launch failure: only [`std::io::ErrorKind::NotFound`]
/// means the binary is missing from PATH. Other OS failures (permission
/// denied, resource exhaustion) map to the tool's `Failed` code, so the
/// harness is not told to install a tool that is already present.
pub fn spawn_error_code(e: &std::io::Error, not_found_code: ErrorCode) -> ErrorCode {
    match (e.kind(), not_found_code) {
        (std::io::ErrorKind::NotFound, _) => not_found_code,
        (_, ErrorCode::FfprobeNotFound) => ErrorCode::FfprobeFailed,
        _ => ErrorCode::FfmpegFailed,
    }
}

/// Read all of `r`, retaining only the last [`STDERR_CAP`] bytes. Returns the
/// kept tail and whether earlier bytes were dropped.
fn capture_tail<R: Read>(mut r: R) -> std::io::Result<(Vec<u8>, bool)> {
    let mut buf = Vec::new();
    let mut truncated = false;
    let mut chunk = [0u8; 8192];
    loop {
        let n = r.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
        if buf.len() > STDERR_CAP {
            buf.drain(..buf.len() - STDERR_CAP);
            truncated = true;
        }
    }
    Ok((buf, truncated))
}

/// Decode captured stderr bytes to a trimmed string, marking dropped output.
fn decode_stderr(bytes: &[u8], truncated: bool) -> String {
    let text = String::from_utf8_lossy(bytes);
    let text = text.trim();
    if truncated {
        format!("[stderr truncated] {text}")
    } else {
        text.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_stderr_tail_when_capture_exceeds_cap() {
        // A stream past the cap keeps only the last STDERR_CAP bytes and flags
        // truncation; ffmpeg's fatal message sits at the tail, so it survives.
        let mut data = vec![b'a'; STDERR_CAP + 100];
        data.extend_from_slice(b"FATAL");
        let (kept, truncated) = capture_tail(&data[..]).expect("capture ok");
        assert!(truncated, "over-cap capture is flagged truncated");
        assert_eq!(kept.len(), STDERR_CAP);
        assert!(kept.ends_with(b"FATAL"), "the fatal tail is retained");
    }

    #[test]
    fn keeps_full_stderr_below_cap() {
        let (kept, truncated) = capture_tail(&b"short message"[..]).expect("capture ok");
        assert!(!truncated);
        assert_eq!(kept, b"short message");
    }

    #[test]
    fn classifies_spawn_failures_by_cause() {
        use std::io::{Error as IoError, ErrorKind};

        let missing = IoError::new(ErrorKind::NotFound, "no such file");
        let denied = IoError::new(ErrorKind::PermissionDenied, "permission denied");

        assert_eq!(
            spawn_error_code(&missing, ErrorCode::FfmpegNotFound),
            ErrorCode::FfmpegNotFound
        );
        assert_eq!(
            spawn_error_code(&denied, ErrorCode::FfmpegNotFound),
            ErrorCode::FfmpegFailed,
            "EACCES is not a missing binary"
        );
        assert_eq!(
            spawn_error_code(&denied, ErrorCode::FfprobeNotFound),
            ErrorCode::FfprobeFailed
        );
    }

    #[test]
    fn prefixes_marker_only_when_truncated() {
        assert_eq!(decode_stderr(b"  boom  ", false), "boom");
        assert_eq!(decode_stderr(b"boom", true), "[stderr truncated] boom");
    }

    #[test]
    fn run_capped_collects_stdout_below_cap() {
        let mut cmd = Command::new("sh");
        cmd.args(["-c", "printf ok"]);
        let out = run_capped(
            &mut cmd,
            "test echo",
            ErrorCode::FfprobeNotFound,
            ErrorCode::FfprobeFailed,
            1024,
        )
        .expect("small stdout is collected");
        assert_eq!(out, "ok");
    }

    #[test]
    fn run_capped_rejects_runaway_stdout() {
        let mut cmd = Command::new("sh");
        cmd.args(["-c", "head -c 4096 /dev/zero"]);
        let err = run_capped(
            &mut cmd,
            "test runaway",
            ErrorCode::FfprobeNotFound,
            ErrorCode::FfprobeFailed,
            1024,
        )
        .expect_err("over-cap stdout is runaway");
        assert_eq!(err.code, ErrorCode::FfprobeFailed);
    }

    #[test]
    fn run_capped_maps_nonzero_exit_with_stderr() {
        let mut cmd = Command::new("sh");
        cmd.args(["-c", "echo boom >&2; exit 3"]);
        let err = run_capped(
            &mut cmd,
            "test failure",
            ErrorCode::FfprobeNotFound,
            ErrorCode::FfprobeFailed,
            1024,
        )
        .expect_err("non-zero exit fails");
        assert_eq!(err.code, ErrorCode::FfprobeFailed);
        assert!(err.reason.contains("boom"), "stderr rides along: {err}");
    }
}
