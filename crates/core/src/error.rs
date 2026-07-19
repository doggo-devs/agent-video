// SPDX-License-Identifier: Apache-2.0
//! Structured error types with codes for programmatic handling.
//!
//! The harness layer receives [`Error`] with a machine-readable [`ErrorCode`]
//! and human-readable reason. Internal modules may use `anyhow` for convenience;
//! public API boundaries convert to this type.

use std::fmt;
use std::path::PathBuf;

/// Declares [`ErrorCode`] from a single variant list: each variant carries
/// its exit code and log identifier exactly once. [`ErrorCode::ALL`] and
/// [`ErrorCode::as_str`] expand from the same list, so a new variant cannot
/// be added without also joining the round-trip test's domain.
macro_rules! error_codes {
    ($( $(#[$meta:meta])* $variant:ident = $code:literal => $name:literal, )+) => {
        /// Exit code categories, designed for harness-layer decision making.
        ///
        /// A Unix exit status carries only 8 bits, so every code must fit in a `u8`
        /// (the `repr` enforces this at compile time). The decades below also avoid
        /// the conventional codes: 1 (uncoded errors: bugs, panics), 2 (CLI usage
        /// errors from clap), and 126+ (shell/signal range).
        ///
        /// Ranges:
        /// - 10-19: media-stack issues (a decoder this build lacks, decode failures)
        /// - 20-29: input issues (file not found, corrupt, constraints violated)
        /// - 30-39: configuration issues (invalid profile, ambiguous options)
        /// - 40-49: output issues (cannot write, permission denied)
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        #[repr(u8)]
        pub enum ErrorCode {
            $( $(#[$meta])* $variant = $code, )+
        }

        impl ErrorCode {
            /// Every variant, in declaration order.
            pub const ALL: &'static [ErrorCode] = &[ $(ErrorCode::$variant,)+ ];

            /// Short identifier for logs.
            pub fn as_str(self) -> &'static str {
                match self {
                    $( ErrorCode::$variant => $name, )+
                }
            }
        }
    };
}

error_codes! {
    // Media stack (in-process FFmpeg libraries)
    /// The linked libavcodec has no decoder for the stream's codec.
    DecoderMissing = 10 => "DECODER_MISSING",
    /// Demuxing or decoding failed mid-stream.
    DecodeFailed = 12 => "DECODE_FAILED",

    // Input problems
    InputNotFound = 20 => "INPUT_NOT_FOUND",
    InputUnreadable = 21 => "INPUT_UNREADABLE",
    InputCorrupt = 22 => "INPUT_CORRUPT",
    InputTooLong = 23 => "INPUT_TOO_LONG",
    InsufficientKeyframes = 24 => "INSUFFICIENT_KEYFRAMES",
    NoFramesExtracted = 25 => "NO_FRAMES_EXTRACTED",
    /// The input URL's protocol is not compiled into the linked FFmpeg.
    ProtocolUnsupported = 26 => "PROTOCOL_UNSUPPORTED",

    // Configuration problems
    InvalidProfile = 30 => "INVALID_PROFILE",
    InvalidRange = 31 => "INVALID_RANGE",
    AmbiguousSubtitle = 32 => "AMBIGUOUS_SUBTITLE",
    InvalidConfig = 33 => "INVALID_CONFIG",

    // Output problems
    OutputDirCreate = 40 => "OUTPUT_DIR_CREATE",
    OutputWriteFailed = 41 => "OUTPUT_WRITE_FAILED",
}

impl ErrorCode {
    /// Map to process exit code.
    pub fn exit_code(self) -> u8 {
        self as u8
    }

    /// Recover the code from a child process's exit status — the inverse of
    /// [`exit_code`](Self::exit_code). `None` for uncoded exits (1: bugs and
    /// panics, 2: CLI usage) and anything outside the assigned decades.
    pub fn from_exit_code(code: u8) -> Option<Self> {
        Self::ALL.iter().copied().find(|c| c.exit_code() == code)
    }
}

/// Machine-readable context attached to errors.
#[derive(Debug, Default, Clone)]
pub struct ErrorContext {
    /// The input the error concerns: a local path, or the verbatim URL of a
    /// remote source (URLs ride in the same `PathBuf`, byte-for-byte).
    pub input_path: Option<PathBuf>,
    pub output_path: Option<PathBuf>,
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

    /// Attach the input path, preserving any context fields already set
    /// (unlike [`with_context`](Self::with_context), which replaces them all).
    pub fn with_input(mut self, path: impl Into<PathBuf>) -> Self {
        self.context.get_or_insert_with(Default::default).input_path = Some(path.into());
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
        assert_eq!(ErrorCode::DecoderMissing.exit_code(), 10);
        assert_eq!(ErrorCode::InputNotFound.exit_code(), 20);
        assert_eq!(ErrorCode::InvalidProfile.exit_code(), 30);
        assert_eq!(ErrorCode::OutputDirCreate.exit_code(), 40);
    }

    #[test]
    fn exit_codes_round_trip_through_from_exit_code() {
        for &code in ErrorCode::ALL {
            assert_eq!(ErrorCode::from_exit_code(code.exit_code()), Some(code));
        }
        assert_eq!(ErrorCode::from_exit_code(1), None, "uncoded exit");
        assert_eq!(ErrorCode::from_exit_code(2), None, "CLI usage exit");
    }

    #[test]
    fn all_exit_codes_stay_in_assigned_decades() {
        for &code in ErrorCode::ALL {
            let n = code.exit_code();
            assert!(
                (10..=49).contains(&n),
                "{} = {n} escapes the assigned decades",
                code.as_str()
            );
        }
    }

    #[test]
    fn error_display_includes_code() {
        let e = Error::new(ErrorCode::InputNotFound, "file missing");
        assert!(e.to_string().contains("INPUT_NOT_FOUND"));
        assert!(e.to_string().contains("file missing"));
    }

    #[test]
    fn preserves_error_code_through_context() {
        // anyhow's `.context()` decorates the message but keeps the underlying
        // typed error downcastable — the mechanism the exit-code path relies on.
        let wrapped = anyhow::Error::from(Error::new(ErrorCode::DecodeFailed, "boom"))
            .context("after 5 frames (possible truncation)");
        let code = wrapped
            .downcast_ref::<Error>()
            .map(|e| e.code)
            .expect("coded error survives .context()");
        assert_eq!(code, ErrorCode::DecodeFailed);
    }
}
