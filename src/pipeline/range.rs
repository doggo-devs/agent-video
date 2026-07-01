//! Time range parsing for the `extract` CLI subcommand.

use anyhow::{Context, Result, bail};

use crate::error::{Error, ErrorCode};
use crate::types::TimeRange;

/// Parse a `MM:SS-MM:SS` (or `SS-SS`) range into a [`TimeRange`].
pub fn parse_range(s: &str) -> Result<TimeRange> {
    let (a, b) = s.split_once('-').ok_or_else(|| {
        Error::new(
            ErrorCode::InvalidRange,
            format!("range '{s}' must be START-END"),
        )
    })?;
    let start = parse_clock(a)?;
    let end = parse_clock(b)?;
    // extract requires a positive-length range, stricter than TimeRange's
    // `start <= end`.
    if end <= start {
        return Err(Error::new(
            ErrorCode::InvalidRange,
            format!("range end ({end}s) must be after start ({start}s)"),
        )
        .into());
    }
    Ok(TimeRange::new(start, end).expect("end > start implies a valid range"))
}

fn parse_clock(s: &str) -> Result<f32> {
    let parts: Vec<&str> = s.split(':').collect();
    let parse = |p: &str| {
        p.parse::<f32>()
            .with_context(|| format!("invalid time '{s}'"))
    };
    match parts.as_slice() {
        [sec] => parse(sec),
        [m, sec] => Ok(parse(m)? * 60.0 + parse(sec)?),
        [h, m, sec] => Ok(parse(h)? * 3600.0 + parse(m)? * 60.0 + parse(sec)?),
        _ => bail!("expected MM:SS or seconds, got '{s}'"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_mmss_range() {
        let r = parse_range("01:23-01:35").expect("valid range");
        assert_eq!(r.start(), 83.0);
        assert_eq!(r.end(), 95.0);
    }

    #[test]
    fn parses_plain_seconds_range() {
        let r = parse_range("5-12").expect("valid range");
        assert_eq!((r.start(), r.end()), (5.0, 12.0));
    }

    #[test]
    fn rejects_inverted_range() {
        assert!(parse_range("01:35-01:23").is_err());
    }
}
