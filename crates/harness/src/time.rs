// SPDX-License-Identifier: Apache-2.0
//! Timestamp parsing and window resolution.
//!
//! Lenient on format, strict on validity within the video bounds. Rejection
//! messages are model-presentable and carry a literal example of valid input.

use super::schema::TimeValue;
use agent_video_core::types::format_ts;

/// Argument-level rejection; the message goes to the model verbatim.
#[derive(Debug, PartialEq)]
pub struct ArgError(pub String);

/// Parse a user-provided timestamp into seconds: bare seconds (`83`, `"83.5"`),
/// `"MM:SS"`, or `"H:MM:SS"`. `param` names the argument in errors.
pub fn parse_timestamp(value: &TimeValue, param: &str) -> Result<f64, ArgError> {
    match value {
        TimeValue::Number(n) => {
            if !n.is_finite() || *n < 0.0 {
                return Err(ArgError(format!(
                    "{param} must be a non-negative number of seconds, got {n}"
                )));
            }
            Ok(*n)
        }
        TimeValue::Text(s) => {
            let s = s.trim();
            parse_text_timestamp(s).ok_or_else(|| {
                ArgError(format!(
                    "{param} must be seconds or a timestamp — e.g. 83, \"1:23\", or \"1:02:03\" — got \"{s}\""
                ))
            })
        }
    }
}

fn parse_text_timestamp(s: &str) -> Option<f64> {
    let parts: Vec<&str> = s.split(':').collect();
    match parts.as_slice() {
        [secs] => parse_seconds(secs, false),
        [m, s] => Some(parse_int(m)? * 60.0 + parse_seconds(s, true)?),
        [h, m, s] => {
            Some(parse_int(h)? * 3600.0 + parse_sexagesimal(m)? * 60.0 + parse_seconds(s, true)?)
        }
        _ => None,
    }
}

/// `\d+` as f64.
fn parse_int(s: &str) -> Option<f64> {
    (!s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())).then(|| s.parse().ok())?
}

/// `\d+(\.\d+)?`, optionally constrained under 60 (`[0-5]?\d`).
fn parse_seconds(s: &str, under_60: bool) -> Option<f64> {
    let (int_part, frac) = match s.split_once('.') {
        Some((i, f)) => (i, Some(f)),
        None => (s, None),
    };
    parse_int(int_part)?;
    if let Some(f) = frac {
        parse_int(f)?;
    }
    if under_60 && (int_part.len() > 2 || int_part.parse::<u32>().ok()? > 59) {
        return None;
    }
    s.parse().ok()
}

/// `[0-5]?\d` — an integer under 60.
fn parse_sexagesimal(s: &str) -> Option<f64> {
    let v = parse_int(s)?;
    (s.len() <= 2 && v < 60.0).then_some(v)
}

/// A resolved window of interest — a valid sub-range of the video by
/// construction ([`resolve_window`] is the only door):
/// `0 <= start < end <= duration`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Window {
    start: f64,
    end: f64,
    /// The video duration the bounds were validated against; carried so
    /// fullness is derived on demand instead of cached.
    duration: f64,
}

impl Window {
    /// Start in seconds; `< end` by construction.
    pub fn start(&self) -> f64 {
        self.start
    }

    /// End in seconds; `<= duration` by construction.
    pub fn end(&self) -> f64 {
        self.end
    }

    /// Seconds covered; positive by construction.
    pub fn span(&self) -> f64 {
        self.end - self.start
    }

    /// Whether the window covers the entire video.
    pub fn is_full(&self) -> bool {
        self.start == 0.0 && self.end == self.duration
    }
}

/// Resolve `[from ?? 0, until ?? duration]` against the video duration into
/// a [`Window`] plus resolution notices. Out-of-bounds `until` is clamped
/// with a notice; out-of-bounds `from` errors. A zero (or negative)
/// `duration` is the probe reporting a zero-length video — rejected naming
/// the file, not the arguments, as the problem.
pub fn resolve_window(
    from: Option<&TimeValue>,
    until: Option<&TimeValue>,
    duration: f64,
) -> Result<(Window, Vec<String>), ArgError> {
    if duration <= 0.0 {
        return Err(ArgError(
            "the probe reports a zero-length video — the file has no playable duration, \
             so no window can be resolved (the file, not from/until, is the problem)"
                .to_string(),
        ));
    }
    let mut notices = Vec::new();
    let start = match from {
        None => 0.0,
        Some(v) => parse_timestamp(v, "from")?,
    };
    let mut end = match until {
        None => duration,
        Some(v) => parse_timestamp(v, "until")?,
    };

    if start >= duration {
        return Err(ArgError(format!(
            "from ({}) is past the end of the video ({})",
            format_ts(start as f32),
            format_ts(duration as f32)
        )));
    }
    if end > duration {
        notices.push(format!(
            "until clamped to video end ({})",
            format_ts(duration as f32)
        ));
        end = duration;
    }
    if end <= start {
        return Err(ArgError(format!(
            "until ({}) must be after from ({})",
            format_ts(end as f32),
            format_ts(start as f32)
        )));
    }
    Ok((
        Window {
            start,
            end,
            duration,
        },
        notices,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(s: &str) -> TimeValue {
        TimeValue::Text(s.to_string())
    }

    #[test]
    fn parses_bare_seconds() {
        assert_eq!(parse_timestamp(&TimeValue::Number(83.0), "from"), Ok(83.0));
        assert_eq!(parse_timestamp(&text("83"), "from"), Ok(83.0));
        assert_eq!(parse_timestamp(&text("83.5"), "from"), Ok(83.5));
    }

    #[test]
    fn parses_mm_ss() {
        assert_eq!(parse_timestamp(&text("1:23"), "from"), Ok(83.0));
        assert_eq!(parse_timestamp(&text("00:05"), "from"), Ok(5.0));
        assert_eq!(parse_timestamp(&text("90:00"), "from"), Ok(5400.0));
        assert_eq!(parse_timestamp(&text("0:59.5"), "from"), Ok(59.5));
    }

    #[test]
    fn parses_h_mm_ss() {
        assert_eq!(parse_timestamp(&text("1:02:03"), "from"), Ok(3723.0));
        assert_eq!(parse_timestamp(&text("10:00:00"), "from"), Ok(36000.0));
    }

    #[test]
    fn rejects_out_of_range_sexagesimal() {
        assert!(parse_timestamp(&text("1:60"), "from").is_err());
        assert!(parse_timestamp(&text("1:61:00"), "from").is_err());
    }

    #[test]
    fn rejects_garbage_with_examples() {
        let err = parse_timestamp(&text("half past"), "from").expect_err("invalid");
        assert!(err.0.contains("from"), "names the param: {}", err.0);
        assert!(err.0.contains("1:23"), "shows an example: {}", err.0);
    }

    #[test]
    fn rejects_negative_number() {
        assert!(parse_timestamp(&TimeValue::Number(-1.0), "until").is_err());
        assert!(parse_timestamp(&TimeValue::Number(f64::NAN), "until").is_err());
    }

    #[test]
    fn resolves_full_window_when_unbounded() {
        let (w, notices) = resolve_window(None, None, 120.0).expect("valid");
        assert_eq!((w.start(), w.end(), w.is_full()), (0.0, 120.0, true));
        assert_eq!(w.span(), 120.0);
        assert!(notices.is_empty());
    }

    #[test]
    fn derives_partial_window_as_not_full() {
        let (w, _) = resolve_window(Some(&text("0:10")), None, 120.0).expect("valid");
        assert!(!w.is_full(), "a window starting past 0 is not full");
    }

    #[test]
    fn clamps_overlong_until_with_notice() {
        let (w, notices) = resolve_window(None, Some(&text("10:00")), 120.0).expect("valid");
        assert_eq!(w.end(), 120.0);
        assert!(w.is_full(), "clamped-to-end window covers the whole video");
        assert_eq!(notices.len(), 1);
    }

    #[test]
    fn rejects_from_past_video_end() {
        let err = resolve_window(Some(&text("3:00")), None, 120.0).expect_err("past end");
        assert!(err.0.contains("past the end"), "{}", err.0);
    }

    #[test]
    fn names_zero_duration_as_the_culprit() {
        let err = resolve_window(None, None, 0.0).expect_err("zero-length video");
        assert!(err.0.contains("zero-length"), "names the file: {}", err.0);
        assert!(
            !err.0.contains("past the end"),
            "must not blame the arguments: {}",
            err.0
        );
    }

    #[test]
    fn rejects_inverted_window() {
        let err =
            resolve_window(Some(&text("1:00")), Some(&text("0:30")), 120.0).expect_err("inverted");
        assert!(err.0.contains("must be after"), "{}", err.0);
    }
}
