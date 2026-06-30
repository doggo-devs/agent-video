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
    stderr_thread: Option<JoinHandle<Vec<u8>>>,
}

/// Error from a failed child process, including captured stderr.
#[derive(Debug)]
pub struct ChildError {
    pub status: ExitStatus,
    pub stderr: String,
    pub context: &'static str,
}

impl std::fmt::Display for ChildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.context, self.status)?;
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
        Error::new(code, self.to_string())
            .with_context(
                ErrorContext::default()
                    .with_exit_status(self.status)
                    .with_stderr(&self.stderr),
            )
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
    /// `not_found_code` is used when spawn fails (typically `FfmpegNotFound` or
    /// `FfprobeNotFound`).
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
                Error::new(not_found_code, format!("failed to spawn: {spawn_context}"))
                    .with_source(e)
            })?;

        let stdout = child.stdout.take().expect("stdout was piped");
        let stderr = child.stderr.take().expect("stderr was piped");

        let stderr_thread = thread::spawn(move || {
            let mut buf = Vec::new();
            if let Err(e) = BufReader::new(stderr).read_to_end(&mut buf) {
                eprintln!("warning: failed to capture stderr: {e}");
            }
            buf
        });

        Ok((Self { child, stderr_thread: Some(stderr_thread) }, stdout))
    }

    /// Wait for the child to exit and check its status.
    ///
    /// On success (exit code 0), returns `Ok(())`. On failure, returns an
    /// error that includes the captured stderr.
    pub fn wait(mut self, context: &'static str) -> Result<(), ChildError> {
        let status = self.child.wait().expect("child was spawned");
        let stderr_bytes = self
            .stderr_thread
            .take()
            .and_then(|t| t.join().ok())
            .unwrap_or_default();
        let stderr = String::from_utf8_lossy(&stderr_bytes)
            .trim()
            .to_string();

        if status.success() {
            Ok(())
        } else {
            Err(ChildError { status, stderr, context })
        }
    }

    /// Kill the child process if still running, then wait.
    ///
    /// Consumes self; cleanup happens via Drop.
    pub fn kill_and_wait(self) {
        // Drop impl handles the actual cleanup
        drop(self);
    }
}
