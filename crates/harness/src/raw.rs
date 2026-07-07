// SPDX-License-Identifier: Apache-2.0
//! `raw` escape hatch parsing: a strategy bypass, not a safety bypass.
//!
//! Enforces the subcommand and plumbing ban, and extracts what budgeting
//! needs (the implied chunk length).

use std::path::PathBuf;

use super::time::ArgError;
use agent_video_core::config::{DEFAULT_CHUNK_SECONDS, default_chunk_seconds};

/// Server-owned flags; rejected if they appear in raw.
const FORBIDDEN: [&str; 3] = ["-o", "--output", "--json"];

/// The core subcommand a raw string may start with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RawSubcommand {
    Process,
    Extract,
}

impl RawSubcommand {
    pub fn as_str(self) -> &'static str {
        match self {
            RawSubcommand::Process => "process",
            RawSubcommand::Extract => "extract",
        }
    }
}

/// A parsed raw invocation.
#[derive(Debug, Clone, PartialEq)]
pub struct RawInvocation {
    pub subcommand: RawSubcommand,
    /// Everything after the subcommand, tokenized — minus the flags the
    /// harness reasons about, which are typed fields below.
    pub flags: Vec<String>,
    /// Explicit `--subtitle` choice; its presence suppresses the server's
    /// cached-transcript attachment.
    pub subtitle: Option<PathBuf>,
    /// Chunk length implied by the raw args — the grid-count denominator.
    pub chunk_seconds: f64,
}

/// Tokenize a command string: split on whitespace, quotes group.
pub fn tokenize(input: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut chars = input.chars().peekable();
    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
        } else if c == '"' || c == '\'' {
            chars.next();
            let quoted: String = chars.by_ref().take_while(|&ch| ch != c).collect();
            tokens.push(quoted);
        } else {
            let mut word = String::new();
            while let Some(&ch) = chars.peek() {
                if ch.is_whitespace() {
                    break;
                }
                word.push(ch);
                chars.next();
            }
            tokens.push(word);
        }
    }
    tokens
}

/// Parse and validate a raw string; rejects an invalid subcommand or a
/// plumbing flag with a model-presentable message.
pub fn parse_raw(raw: &str) -> Result<RawInvocation, ArgError> {
    let tokens = tokenize(raw.trim());
    let subcommand = match tokens.first().map(String::as_str) {
        Some("process") => RawSubcommand::Process,
        Some("extract") => RawSubcommand::Extract,
        _ => {
            return Err(ArgError(format!(
                "raw must start with a core subcommand (\"process\" or \"extract\"), \
                 e.g. \"process --strategy keyframes\" — got \"{raw}\""
            )));
        }
    };
    let mut flags: Vec<String> = tokens[1..].to_vec();

    for t in &flags {
        if FORBIDDEN.contains(&t.as_str()) {
            return Err(ArgError(format!(
                "raw must not contain {t}: input/output plumbing is injected by the server. \
                 Pass only strategy flags."
            )));
        }
    }

    // The harness must know whether a subtitle was chosen (a cached whisper
    // transcript rides along otherwise), so `--subtitle` is lifted out of
    // the opaque flags into a typed field; last occurrence wins.
    let mut subtitle: Option<PathBuf> = None;
    while let Some(i) = flags.iter().position(|t| t == "--subtitle") {
        let Some(path) = flags.get(i + 1).filter(|p| !p.starts_with('-')) else {
            return Err(ArgError(
                "--subtitle needs a file path, e.g. --subtitle captions.srt".to_string(),
            ));
        };
        subtitle = Some(PathBuf::from(path));
        flags.drain(i..=i + 1);
    }

    // Profile resolution goes through core's own table, so the predicted
    // grid-count denominator cannot drift from what the run will do.
    let mut chunk_seconds = f64::from(DEFAULT_CHUNK_SECONDS);
    if let Some(i) = flags.iter().position(|t| t == "--profile")
        && let Some(profile) = flags.get(i + 1)
    {
        chunk_seconds = f64::from(default_chunk_seconds(Some(profile)));
    }
    if let Some(i) = flags.iter().position(|t| t == "--chunk-seconds")
        && let Some(v) = flags.get(i + 1).and_then(|s| s.parse::<f64>().ok())
        && v.is_finite()
        && v > 0.0
    {
        chunk_seconds = v;
    }

    Ok(RawInvocation {
        subcommand,
        flags,
        subtitle,
        chunk_seconds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizes_quoted_groups() {
        assert_eq!(
            tokenize(r#"process --subtitle "my file.srt" 'a b'"#),
            ["process", "--subtitle", "my file.srt", "a b"]
        );
    }

    #[test]
    fn parses_process_with_flags() {
        let inv = parse_raw("process --strategy keyframes").expect("valid");
        assert_eq!(inv.subcommand, RawSubcommand::Process);
        assert_eq!(inv.flags, ["--strategy", "keyframes"]);
        assert_eq!(inv.chunk_seconds, f64::from(DEFAULT_CHUNK_SECONDS));
    }

    #[test]
    fn rejects_missing_subcommand() {
        let err = parse_raw("--strategy keyframes").expect_err("no subcommand");
        assert!(err.0.contains("process"), "shows valid example: {}", err.0);
    }

    #[test]
    fn rejects_plumbing_flags() {
        for raw in ["process --json", "process -o /tmp", "extract --output x"] {
            let err = parse_raw(raw).expect_err("plumbing is server-owned");
            assert!(err.0.contains("plumbing"), "{}", err.0);
        }
    }

    #[test]
    fn derives_chunk_seconds_from_profile() {
        // Expectations route through core's table too: this locks the
        // parsing (which token names the profile), not the profile values.
        for profile in ["fast", "quality", "action"] {
            assert_eq!(
                parse_raw(&format!("process --profile {profile}"))
                    .expect("valid")
                    .chunk_seconds,
                f64::from(default_chunk_seconds(Some(profile))),
                "profile {profile}"
            );
        }
    }

    #[test]
    fn explicit_chunk_seconds_wins_over_profile() {
        let inv = parse_raw("process --profile fast --chunk-seconds 30").expect("valid");
        assert_eq!(inv.chunk_seconds, 30.0);
    }

    #[test]
    fn ignores_invalid_chunk_seconds_value() {
        let inv = parse_raw("process --chunk-seconds nope").expect("valid");
        assert_eq!(inv.chunk_seconds, f64::from(DEFAULT_CHUNK_SECONDS));
    }

    #[test]
    fn lifts_subtitle_out_of_flags() {
        let inv =
            parse_raw(r#"process --subtitle "my file.srt" --strategy keyframes"#).expect("valid");
        assert_eq!(inv.subtitle, Some(PathBuf::from("my file.srt")));
        assert_eq!(inv.flags, ["--strategy", "keyframes"], "no residue");
    }

    #[test]
    fn rejects_subtitle_without_a_path() {
        for raw in [
            "process --subtitle",
            "process --subtitle --strategy keyframes",
        ] {
            let err = parse_raw(raw).expect_err("dangling --subtitle");
            assert!(err.0.contains("file path"), "{}", err.0);
        }
    }
}
