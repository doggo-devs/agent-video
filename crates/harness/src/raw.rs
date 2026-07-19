// SPDX-License-Identifier: Apache-2.0
//! `raw` escape hatch parsing: a strategy bypass, not a safety bypass.
//!
//! Enforces the subcommand and plumbing ban, and extracts what budgeting
//! needs (the implied chunk length).

use std::path::PathBuf;

use super::time::ArgError;
use agent_video_core::config::{DEFAULT_CHUNK_SECONDS, Profile, default_chunk_seconds};

/// Server-owned long flags; rejected exact or in clap's `--flag=value`
/// form. The short `-o` is handled separately ([`forbidden_flag`]) because
/// clap also accepts its attached forms (`-oVALUE`, `-o=VALUE`).
const FORBIDDEN_LONG: [&str; 2] = ["--output", "--json"];

/// The server-owned flag `token` spells, or `None`.
///
/// The rule is precise: exact match (`-o`, `--output`, `--json`),
/// `flag=`-prefix for the long flags (`--output=/x`), and `-o` attachment
/// (`-o/x`, `-o=/x` — clap short-flag attachment). A flag that merely
/// shares a prefix (a hypothetical `--jsonx`) is not matched.
fn forbidden_flag(token: &str) -> Option<&'static str> {
    if token.starts_with("-o") && !token.starts_with("--") {
        return Some("-o");
    }
    FORBIDDEN_LONG.into_iter().find(|flag| {
        token == *flag
            || token
                .strip_prefix(flag)
                .is_some_and(|rest| rest.starts_with('='))
    })
}

/// One occurrence of an observed user flag, in either clap spelling:
/// `--flag value` (two tokens) or `--flag=value` (one token).
///
/// `span` covers exactly the tokens the occurrence owns, so removal via
/// [`FlagOccurrence::remove_from`] cannot leave residue or eat a
/// neighbour: one token for the `=` form, two for the space form.
struct FlagOccurrence {
    span: std::ops::Range<usize>,
    /// The flag's argument: the text after `=`, or the following token
    /// in the space form (`None` when the flag is the last token).
    value: Option<String>,
}

impl FlagOccurrence {
    /// Remove the occurrence from the token stream it was found in,
    /// yielding its value.
    fn remove_from(self, flags: &mut Vec<String>) -> Option<String> {
        flags.drain(self.span);
        self.value
    }
}

/// The first occurrence of `name` in `flags`, matching both clap
/// spellings. Exact-or-`=`-attached, never a bare prefix: `--profile`
/// does not match a hypothetical `--profilex`.
fn find_flag(flags: &[String], name: &str) -> Option<FlagOccurrence> {
    let i = flags.iter().position(|t| {
        t == name
            || t.strip_prefix(name)
                .is_some_and(|rest| rest.starts_with('='))
    })?;
    match flags[i]
        .strip_prefix(name)
        .and_then(|r| r.strip_prefix('='))
    {
        Some(attached) => Some(FlagOccurrence {
            span: i..i + 1,
            value: Some(attached.to_string()),
        }),
        None => match flags.get(i + 1) {
            Some(next) => Some(FlagOccurrence {
                span: i..i + 2,
                value: Some(next.clone()),
            }),
            None => Some(FlagOccurrence {
                span: i..i + 1,
                value: None,
            }),
        },
    }
}

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
        if let Some(flag) = forbidden_flag(t) {
            return Err(ArgError(format!(
                "raw must not contain {t} ({flag} is server-owned): input/output plumbing \
                 is injected by the server. Pass only strategy flags."
            )));
        }
    }

    // The harness must know whether a subtitle was chosen (a cached whisper
    // transcript rides along otherwise), so `--subtitle` is lifted out of
    // the opaque flags into a typed field; last occurrence wins.
    let mut subtitle: Option<PathBuf> = None;
    while let Some(occurrence) = find_flag(&flags, "--subtitle") {
        let Some(path) = occurrence
            .remove_from(&mut flags)
            .filter(|p| !p.is_empty() && !p.starts_with('-'))
        else {
            return Err(ArgError(
                "--subtitle needs a file path, e.g. --subtitle captions.srt".to_string(),
            ));
        };
        subtitle = Some(PathBuf::from(path));
    }

    // Profile resolution goes through core's own table, so the predicted
    // grid-count denominator cannot drift from what the run will do. An
    // unknown name dies here, with core's own message, instead of running
    // core just to watch it reject the flag.
    let mut chunk_seconds = f64::from(DEFAULT_CHUNK_SECONDS);
    if let Some(name) = find_flag(&flags, "--profile").and_then(|o| o.value) {
        let profile = name.parse::<Profile>().map_err(|e| ArgError(e.reason))?;
        chunk_seconds = f64::from(default_chunk_seconds(Some(profile)));
    }
    if let Some(v) = find_flag(&flags, "--chunk-seconds")
        .and_then(|o| o.value)
        .and_then(|s| s.parse::<f64>().ok())
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
    fn rejects_attached_output_flag() {
        for raw in ["process -o/tmp/x", "process -o=/tmp/x", "extract -ox.json"] {
            let err = parse_raw(raw).expect_err("clap accepts -o attachment");
            assert!(err.0.contains("plumbing"), "{raw}: {}", err.0);
        }
    }

    #[test]
    fn rejects_equals_form_of_forbidden_flags() {
        for raw in ["process --output=/tmp/x", "process --json=frames"] {
            let err = parse_raw(raw).expect_err("clap accepts --flag=value");
            assert!(err.0.contains("plumbing"), "{raw}: {}", err.0);
        }
    }

    #[test]
    fn permits_flags_that_merely_share_a_prefix() {
        // `--jsonx` is not `--json`: the ban is exact-or-`=`-attached, not
        // a prefix match.
        let inv = parse_raw("process --jsonx foo").expect("prefix-sharing flag passes");
        assert_eq!(inv.flags, ["--jsonx", "foo"]);
    }

    #[test]
    fn derives_chunk_seconds_from_profile() {
        // Expectations route through core's table too: this locks the
        // parsing (which token names the profile), not the profile values.
        for profile in [Profile::Fast, Profile::Quality, Profile::Action] {
            assert_eq!(
                parse_raw(&format!("process --profile {}", profile.cli_name()))
                    .expect("valid")
                    .chunk_seconds,
                f64::from(default_chunk_seconds(Some(profile))),
                "profile {profile:?}"
            );
        }
    }

    #[test]
    fn rejects_unknown_profile_naming_the_options() {
        let err = parse_raw("process --profile cinematic").expect_err("unknown profile");
        assert!(err.0.contains("cinematic"), "names the value: {}", err.0);
        assert!(err.0.contains("balanced"), "lists the options: {}", err.0);
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

    #[test]
    fn reads_equals_form_chunk_seconds() {
        let inv = parse_raw("process --chunk-seconds=1").expect("valid");
        assert_eq!(inv.chunk_seconds, 1.0);
    }

    #[test]
    fn derives_chunk_seconds_from_equals_form_profile() {
        assert_eq!(
            parse_raw("process --profile=fast")
                .expect("valid")
                .chunk_seconds,
            f64::from(default_chunk_seconds(Some(Profile::Fast)))
        );
    }

    #[test]
    fn lifts_equals_form_subtitle_out_of_flags() {
        let inv = parse_raw("process --subtitle=captions.srt --strategy keyframes").expect("valid");
        assert_eq!(inv.subtitle, Some(PathBuf::from("captions.srt")));
        assert_eq!(inv.flags, ["--strategy", "keyframes"], "no residue");
    }

    #[test]
    fn lifts_last_subtitle_across_mixed_spellings() {
        let inv = parse_raw("process --subtitle a.srt --subtitle=b.srt").expect("valid");
        assert_eq!(inv.subtitle, Some(PathBuf::from("b.srt")));
        assert!(inv.flags.is_empty(), "no residue: {:?}", inv.flags);
    }

    #[test]
    fn space_form_chunk_seconds_wins_over_equals_form_profile() {
        let inv = parse_raw("process --profile=fast --chunk-seconds 30").expect("valid");
        assert_eq!(inv.chunk_seconds, 30.0);
    }

    #[test]
    fn ignores_equals_form_chunk_seconds_with_empty_value() {
        let inv = parse_raw("process --chunk-seconds=").expect("valid");
        assert_eq!(inv.chunk_seconds, f64::from(DEFAULT_CHUNK_SECONDS));
    }

    #[test]
    fn rejects_equals_form_subtitle_with_empty_value() {
        let err = parse_raw("process --subtitle=").expect_err("empty subtitle path");
        assert!(err.0.contains("file path"), "{}", err.0);
    }
}
