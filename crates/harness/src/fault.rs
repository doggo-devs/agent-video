// SPDX-License-Identifier: Apache-2.0
//! The harness's single error currency.
//!
//! Core speaks structured [`agent_video_core::error::Error`] with a
//! machine-readable code; the adapters around it (probe, fetch, transcribe,
//! core self-spawn) each face failures core knows nothing about — HTTP
//! statuses, missing tools, stalled children. [`Fault`] is the one type
//! every adapter boundary converges on: a [`FaultKind`] at the granularity
//! downstream decisions are actually made, plus model-presentable prose.
//! A core error crossing the boundary keeps its `[CODE]` prefix in the
//! prose and survives intact in the source chain; its code decides the
//! kind in exactly one place ([`FaultKind::of`]).

use agent_video_core::error::{Error as CoreError, ErrorCode};

/// Failure class at the granularity anyone downstream branches on —
/// deliberately coarser than core's [`ErrorCode`]: anything finer is prose.
///
/// Each kind names who can fix the failure:
/// - [`BadInput`](Self::BadInput): the caller — the source or arguments are
///   wrong; retrying the identical call cannot succeed.
/// - [`Environment`](Self::Environment): the user — this installation lacks
///   a capability (protocol not compiled in, decoder missing, yt-dlp or
///   whisper weights absent) or the machine refuses (permissions, disk).
/// - [`Transient`](Self::Transient): nobody — plausibly a passing condition
///   (network, remote server, a stalled child); retrying may succeed.
/// - [`Internal`](Self::Internal): us — a bug in agent-video.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FaultKind {
    BadInput,
    Environment,
    Transient,
    Internal,
}

impl FaultKind {
    /// The single mapping from core's vocabulary into harness decisions.
    pub fn of(code: ErrorCode) -> FaultKind {
        match code {
            // This build cannot do it; retrying cannot help.
            ErrorCode::DecoderMissing | ErrorCode::ProtocolUnsupported => FaultKind::Environment,
            // The source itself is unusable or out of bounds.
            ErrorCode::DecodeFailed
            | ErrorCode::InputNotFound
            | ErrorCode::InputUnreadable
            | ErrorCode::InputCorrupt
            | ErrorCode::InputTooLong
            | ErrorCode::InsufficientKeyframes
            | ErrorCode::NoFramesExtracted
            | ErrorCode::AmbiguousSubtitle => FaultKind::BadInput,
            // The harness builds core's arguments; if core rejects them,
            // the harness built them wrong.
            ErrorCode::InvalidProfile | ErrorCode::InvalidRange | ErrorCode::InvalidConfig => {
                FaultKind::Internal
            }
            // Filesystem refusals on the output side.
            ErrorCode::OutputDirCreate | ErrorCode::OutputWriteFailed => FaultKind::Environment,
        }
    }

    /// Whether retrying the identical call can possibly succeed.
    pub fn retriable(self) -> bool {
        matches!(self, FaultKind::Transient)
    }
}

/// The wire names, for prose that must carry the kind (a test pins this
/// rendering to the serde one, so they cannot drift).
impl std::fmt::Display for FaultKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            FaultKind::BadInput => "bad_input",
            FaultKind::Environment => "environment",
            FaultKind::Transient => "transient",
            FaultKind::Internal => "internal",
        })
    }
}

/// A failed harness operation: class for decisions, prose for the model.
#[derive(Debug)]
pub struct Fault {
    pub kind: FaultKind,
    pub message: String,
    source: Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
}

impl Fault {
    pub fn new(kind: FaultKind, message: impl Into<String>) -> Self {
        Fault {
            kind,
            message: message.into(),
            source: None,
        }
    }

    /// The caller's source or arguments are wrong.
    pub fn bad_input(message: impl Into<String>) -> Self {
        Fault::new(FaultKind::BadInput, message)
    }

    /// This installation or machine cannot do it.
    pub fn environment(message: impl Into<String>) -> Self {
        Fault::new(FaultKind::Environment, message)
    }

    /// Plausibly a passing condition; retrying may succeed.
    pub fn transient(message: impl Into<String>) -> Self {
        Fault::new(FaultKind::Transient, message)
    }

    /// A bug in agent-video.
    pub fn internal(message: impl Into<String>) -> Self {
        Fault::new(FaultKind::Internal, message)
    }

    /// Attach the underlying typed error: `Display` stays the
    /// model-presentable prose, the chain keeps the evidence.
    pub fn with_source(mut self, source: impl std::error::Error + Send + Sync + 'static) -> Self {
        self.source = Some(Box::new(source));
        self
    }
}

/// Core errors cross the boundary by changing currency, not by melting into
/// prose: the code picks the kind, the `[CODE]`-prefixed rendering becomes
/// the message, and the typed error itself survives as the source.
impl From<CoreError> for Fault {
    fn from(e: CoreError) -> Self {
        Fault {
            kind: FaultKind::of(e.code),
            message: e.to_string(),
            source: Some(Box::new(e)),
        }
    }
}

impl std::fmt::Display for Fault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Fault {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_ref().map(|e| e.as_ref() as _)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_error_keeps_code_in_message_and_source() {
        let core = CoreError::new(ErrorCode::ProtocolUnsupported, "no rtsp in this build");
        let fault = Fault::from(core);
        assert_eq!(fault.kind, FaultKind::Environment);
        assert!(fault.message.contains("PROTOCOL_UNSUPPORTED"), "{fault}");
        let source = std::error::Error::source(&fault).expect("core error rides in the chain");
        assert_eq!(
            source
                .downcast_ref::<CoreError>()
                .expect("source is the typed core error")
                .code,
            ErrorCode::ProtocolUnsupported
        );
    }

    #[test]
    fn distinguishes_missing_capability_from_corrupt_input() {
        let missing = Fault::from(CoreError::new(ErrorCode::DecoderMissing, "no av1"));
        let corrupt = Fault::from(CoreError::new(ErrorCode::InputCorrupt, "bad moov"));
        assert_eq!(missing.kind, FaultKind::Environment);
        assert_eq!(corrupt.kind, FaultKind::BadInput);
    }

    #[test]
    fn only_transient_is_retriable() {
        assert!(FaultKind::Transient.retriable());
        assert!(!FaultKind::BadInput.retriable());
        assert!(!FaultKind::Environment.retriable());
        assert!(!FaultKind::Internal.retriable());
    }

    #[test]
    fn kind_serializes_snake_case() {
        let json = serde_json::to_string(&FaultKind::BadInput).expect("serializes");
        assert_eq!(json, "\"bad_input\"");
    }

    #[test]
    fn kind_display_matches_wire_names() {
        for kind in [
            FaultKind::BadInput,
            FaultKind::Environment,
            FaultKind::Transient,
            FaultKind::Internal,
        ] {
            let wire = serde_json::to_value(kind).expect("serializes");
            assert_eq!(wire, serde_json::Value::String(kind.to_string()));
        }
    }
}
