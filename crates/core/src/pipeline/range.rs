// SPDX-License-Identifier: Apache-2.0
//! Time range parsing (`extract --range`) and the `process` attention window.

use crate::error::{Error, ErrorCode, Result};
use crate::types::TimeRange;

/// Parse a `MM:SS-MM:SS` (or `SS-SS`) range into a [`TimeRange`]. Each bound
/// is the crate's shared clock grammar (`crate::types::parse_clock`), so `.`
/// and `,` fractions are both accepted in any component.
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
        ));
    }
    Ok(TimeRange::new(start, end).expect("end > start implies a valid range"))
}

/// Absolute attention window for `process` (`--from`/`--until`): decoding is
/// clamped to it while frame timestamps stay absolute to the file.
///
/// By construction (the [`Window::from_args`] door): `start` is finite and
/// non-negative, and `end`, when present, is finite and after `start`.
#[derive(Debug, Clone, Copy)]
pub struct Window {
    start: f32,
    end: Option<f32>,
}

impl Window {
    /// Start in seconds.
    pub fn start(&self) -> f32 {
        self.start
    }

    /// End in seconds; `None` reads to the end of the video.
    pub fn end(&self) -> Option<f32> {
        self.end
    }

    /// Build from the CLI flags; `None` when neither is given. Bounds must be
    /// finite, non-negative, and ordered.
    pub fn from_args(from: Option<f32>, until: Option<f32>) -> Result<Option<Self>> {
        if from.is_none() && until.is_none() {
            return Ok(None);
        }
        let start = from.unwrap_or(0.0);
        if !start.is_finite() || start < 0.0 {
            return Err(Error::new(
                ErrorCode::InvalidRange,
                format!("--from must be a finite, non-negative number of seconds, got {start}"),
            ));
        }
        if let Some(end) = until
            && (!end.is_finite() || end <= start)
        {
            return Err(Error::new(
                ErrorCode::InvalidRange,
                format!("--until ({end}s) must be finite and after --from ({start}s)"),
            ));
        }
        Ok(Some(Self { start, end: until }))
    }

    /// Seconds of a `duration`-long video this window actually covers.
    pub fn effective_span(&self, duration: f32) -> f32 {
        (self.end.unwrap_or(duration).min(duration) - self.start).max(0.0)
    }
}

/// The shared clock grammar with `--range`'s error currency: a rejection
/// must carry [`ErrorCode::InvalidRange`] (exit 31) — a contract the typed
/// [`Result`] now states in the signature.
fn parse_clock(s: &str) -> Result<f32> {
    crate::types::parse_clock(s).ok_or_else(|| {
        Error::new(
            ErrorCode::InvalidRange,
            format!("invalid time '{s}': expected [HH:]MM:SS or seconds, finite and non-negative"),
        )
    })
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

    #[test]
    fn rejects_negative_clock_components() {
        assert!(parse_range("0:-30-1:00").is_err());
    }

    #[test]
    fn parses_comma_fraction_range() {
        // The shared clock grammar accepts `,` fractions (SRT heritage);
        // range bounds inherit the leniency by construction.
        let r = parse_range("1:23,5-1:24").expect("comma fraction is valid");
        assert_eq!((r.start(), r.end()), (83.5, 84.0));
    }

    #[test]
    fn rejects_overflowing_clock() {
        // Finite components can still overflow f32 when scaled to seconds.
        assert!(parse_range("1e38:0:0-2e38:0:0").is_err());
    }

    #[test]
    fn maps_every_parse_failure_to_invalid_range_code() {
        // main() picks the exit code from the typed Error; the signature
        // guarantees a code, this pins that it is the right one.
        for s in ["abc-5", "1:2:3:4-9", "NaN-1", "0-inf", "01:35-01:23", "5"] {
            let err = parse_range(s).expect_err("malformed range must fail");
            assert_eq!(
                err.code,
                ErrorCode::InvalidRange,
                "range '{s}' must carry the InvalidRange code"
            );
        }
    }
}
