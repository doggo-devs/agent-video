//! Structured error types with codes for programmatic handling.
//!
//! The harness layer receives [`Error`] with a machine-readable [`ErrorCode`]
//! and human-readable reason. Internal modules may use `anyhow` for convenience;
//! public API boundaries convert to this type.

use std::fmt;
use std::path::PathBuf;
use std::process::ExitStatus;

/// Exit code categories, designed for harness-layer decision making.
///
/// A Unix exit status carries only 8 bits, so every code must fit in a `u8`
/// (the `repr` enforces this at compile time). The decades below also avoid
/// the conventional codes: 1 (uncoded errors: bugs, panics), 2 (CLI usage
/// errors from clap), and 126+ (shell/signal range).
///
/// Ranges:
/// - 10-19: environment/dependency issues (may resolve with PATH or install)
/// - 20-29: input issues (file not found, corrupt, constraints violated)
/// - 30-39: configuration issues (invalid profile, ambiguous options)
/// - 40-49: output issues (cannot write, permission denied)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ErrorCode {
    // Environment / external dependencies
    FfmpegNotFound = 10,
    FfprobeNotFound = 11,
    FfmpegFailed = 12,
    FfprobeFailed = 13,

    // Input problems
    InputNotFound = 20,
    InputUnreadable = 21,
    InputCorrupt = 22,
    InputTooLong = 23,
    InsufficientKeyframes = 24,
    NoFramesExtracted = 25,

    // Configuration problems
    InvalidProfile = 30,
    InvalidRange = 31,
    AmbiguousSubtitle = 32,
    InvalidConfig = 33,

    // Output problems
    OutputDirCreate = 40,
    OutputWriteFailed = 41,
}

impl ErrorCode {
    /// Map to process exit code.
    pub fn exit_code(self) -> u8 {
        self as u8
    }

    /// Short identifier for logs.
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorCode::FfmpegNotFound => "FFMPEG_NOT_FOUND",
            ErrorCode::FfprobeNotFound => "FFPROBE_NOT_FOUND",
            ErrorCode::FfmpegFailed => "FFMPEG_FAILED",
            ErrorCode::FfprobeFailed => "FFPROBE_FAILED",
            ErrorCode::InputNotFound => "INPUT_NOT_FOUND",
            ErrorCode::InputUnreadable => "INPUT_UNREADABLE",
            ErrorCode::InputCorrupt => "INPUT_CORRUPT",
            ErrorCode::InputTooLong => "INPUT_TOO_LONG",
            ErrorCode::InsufficientKeyframes => "INSUFFICIENT_KEYFRAMES",
            ErrorCode::NoFramesExtracted => "NO_FRAMES_EXTRACTED",
            ErrorCode::InvalidProfile => "INVALID_PROFILE",
            ErrorCode::InvalidRange => "INVALID_RANGE",
            ErrorCode::AmbiguousSubtitle => "AMBIGUOUS_SUBTITLE",
            ErrorCode::InvalidConfig => "INVALID_CONFIG",
            ErrorCode::OutputDirCreate => "OUTPUT_DIR_CREATE",
            ErrorCode::OutputWriteFailed => "OUTPUT_WRITE_FAILED",
        }
    }
}

/// Machine-readable context attached to errors.
#[derive(Debug, Default, Clone)]
pub struct ErrorContext {
    pub input_path: Option<PathBuf>,
    pub output_path: Option<PathBuf>,
    pub exit_status: Option<ExitStatus>,
    pub stderr_snippet: Option<String>,
}

impl ErrorContext {
    pub fn with_input(mut self, path: impl Into<PathBuf>) -> Self {
        self.input_path = Some(path.into());
        self
    }

    pub fn with_output(mut self, path: impl Into<PathBuf>) -> Self {
        self.output_path = Some(path.into());
        self
    }

    pub fn with_exit_status(mut self, status: ExitStatus) -> Self {
        self.exit_status = Some(status);
        self
    }

    pub fn with_stderr(mut self, stderr: impl Into<String>) -> Self {
        let s = stderr.into();
        // Truncate to first 500 chars to avoid bloating logs
        self.stderr_snippet = Some(if s.len() > 500 {
            format!("{}...", &s[..500])
        } else {
            s
        });
        self
    }
}

/// Structured error with code, reason, and optional context.
#[derive(Debug)]
pub struct Error {
    pub code: ErrorCode,
    pub reason: String,
    context: Option<Box<ErrorContext>>,
    source: Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
}

impl Error {
    pub fn new(code: ErrorCode, reason: impl Into<String>) -> Self {
        Self {
            code,
            reason: reason.into(),
            context: None,
            source: None,
        }
    }

    pub fn with_context(mut self, ctx: ErrorContext) -> Self {
        self.context = Some(Box::new(ctx));
        self
    }

    pub fn with_source(mut self, source: impl std::error::Error + Send + Sync + 'static) -> Self {
        self.source = Some(Box::new(source));
        self
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.code.as_str(), self.reason)?;
        if let Some(ctx) = &self.context {
            if let Some(ref path) = ctx.input_path {
                write!(f, " (input: {})", path.display())?;
            }
            if let Some(ref path) = ctx.output_path {
                write!(f, " (output: {})", path.display())?;
            }
        }
        Ok(())
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_ref().map(|e| e.as_ref() as _)
    }
}

/// Convenience type alias for Results using our Error.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_code_exit_codes_in_range() {
        assert_eq!(ErrorCode::FfmpegNotFound.exit_code(), 10);
        assert_eq!(ErrorCode::InputNotFound.exit_code(), 20);
        assert_eq!(ErrorCode::InvalidProfile.exit_code(), 30);
        assert_eq!(ErrorCode::OutputDirCreate.exit_code(), 40);
    }

    #[test]
    fn error_display_includes_code() {
        let e = Error::new(ErrorCode::InputNotFound, "file missing");
        assert!(e.to_string().contains("INPUT_NOT_FOUND"));
        assert!(e.to_string().contains("file missing"));
    }

    #[test]
    fn context_truncates_long_stderr() {
        let long_stderr = "x".repeat(1000);
        let ctx = ErrorContext::default().with_stderr(long_stderr);
        assert!(ctx.stderr_snippet.unwrap().len() < 600);
    }

    #[test]
    fn preserves_error_code_through_context() {
        // anyhow's `.context()` decorates the message but keeps the underlying
        // typed error downcastable — the mechanism the exit-code path relies on.
        let wrapped = anyhow::Error::from(Error::new(ErrorCode::FfmpegFailed, "boom"))
            .context("after 5 frames (possible truncation)");
        let code = wrapped
            .downcast_ref::<Error>()
            .map(|e| e.code)
            .expect("coded error survives .context()");
        assert_eq!(code, ErrorCode::FfmpegFailed);
    }
}
