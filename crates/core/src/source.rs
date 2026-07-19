// SPDX-License-Identifier: Apache-2.0
//! Input source classification: local filesystem path vs protocol URL.
//!
//! [`Source::classify`] is the single entry point, applied once at the CLI
//! boundary. The variants encode which validations apply downstream: a
//! [`Source::Local`] input is subject to filesystem checks and sidecar
//! subtitle discovery; a [`Source::Remote`] input skips both and its bytes go
//! verbatim to `avformat_open_input` — FFmpeg owns protocol resolution, and a
//! protocol this build lacks surfaces from open as a coded error.

use std::borrow::Cow;
use std::fmt;
use std::path::{Path, PathBuf};

/// A video input, classified once by [`Source::classify`].
#[derive(Debug, Clone)]
pub enum Source {
    /// A local file: validated against the filesystem before any media work,
    /// and scanned for sidecar subtitles. When a path is already known local
    /// by type (a cache entry, a temp file), construct this variant directly
    /// — re-classifying its string form would re-guess settled knowledge and
    /// misread colon-named files (`clip:1.mp4`) as protocol URLs.
    Local(PathBuf),
    /// A protocol URL, handed verbatim to FFmpeg's demuxer.
    Remote(RemoteUrl),
}

/// A protocol URL, held verbatim — no parsing, validation, or normalization,
/// because the payload is FFmpeg's to interpret (`http://…`, `subfile,…`,
/// nested protocols like `concat:`). Constructed only by [`Source::classify`],
/// so a [`Source::Remote`] can never hold a bare local path.
#[derive(Debug, Clone)]
pub struct RemoteUrl(String);

impl RemoteUrl {
    /// The verbatim URL string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Source {
    /// Classify `input` as a local path or a protocol URL.
    ///
    /// No scheme whitelist: which protocols are usable is a property of the
    /// FFmpeg build, so protocol existence is deferred to open and reported
    /// as [`crate::error::ErrorCode::ProtocolUnsupported`].
    pub fn classify(input: &str) -> Self {
        if is_protocol_url(input) {
            Source::Remote(RemoteUrl(input.to_string()))
        } else {
            Source::Local(PathBuf::from(input))
        }
    }

    /// The verbatim bytes of the input, shaped as a [`Path`] — for error
    /// context and display; for a remote URL the "path" is simply the URL
    /// string.
    pub fn as_path(&self) -> &Path {
        match self {
            Source::Local(path) => path,
            Source::Remote(url) => Path::new(url.as_str()),
        }
    }

    /// The path handed to `avformat_open_input`.
    ///
    /// A remote URL goes verbatim — FFmpeg owns protocol resolution. A local
    /// path is already classified, but FFmpeg re-parses whatever string it
    /// receives with its own protocol grammar, so a relative file named like
    /// a scheme (`clip:1.mp4`) would be routed to a nonexistent `clip`
    /// protocol. Such paths are prefixed with `./`, which FFmpeg reads as a
    /// plain file path — the classification decided here stays decided.
    pub fn ffmpeg_path(&self) -> Cow<'_, Path> {
        match self {
            Source::Local(path) if is_protocol_url(&path.to_string_lossy()) => {
                Cow::Owned(Path::new(".").join(path))
            }
            source => Cow::Borrowed(source.as_path()),
        }
    }
}

impl fmt::Display for Source {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Source::Local(path) => path.display().fmt(f),
            Source::Remote(url) => f.write_str(url.as_str()),
        }
    }
}

/// Whether `input` names an FFmpeg protocol, following the shape FFmpeg's own
/// `url_find_protocol` (libavformat/avio.c) recognizes: a leading scheme over
/// `[A-Za-z0-9+.-]` terminated by `:`, plus the `subfile,` special case.
///
/// This is a deliberate approximation, not a byte-exact port. It tightens two
/// inputs FFmpeg would route to nonexistent protocols — a scheme here must
/// start with a letter, and a single-letter scheme is a Windows drive path —
/// so odd local names like `1:23.mp4` fail with a precise filesystem error
/// instead of FFmpeg's opaque "Protocol not found". Every real FFmpeg
/// protocol name matches this shape, so no remote input is misrouted.
fn is_protocol_url(input: &str) -> bool {
    // `subfile,,start,N,end,M,,:media.ts`: comma-separated options precede the
    // nested target, so the scheme scan below would stop at the comma. FFmpeg
    // special-cases the prefix and requires a `:` later; mirror both.
    if let Some(rest) = input.strip_prefix("subfile,") {
        return rest.contains(':');
    }
    let bytes = input.as_bytes();
    if !bytes.first().is_some_and(u8::is_ascii_alphabetic) {
        return false;
    }
    let scheme_len = bytes
        .iter()
        .take_while(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'.' | b'-'))
        .count();
    // A single-letter scheme is a Windows drive path (`C:\video.mp4`), which
    // FFmpeg likewise treats as a local file; no real protocol is one letter.
    scheme_len > 1 && bytes.get(scheme_len) == Some(&b':')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_https_as_remote() {
        let source = Source::classify("https://example.com/video.mp4");
        assert!(
            matches!(&source, Source::Remote(url) if url.as_str() == "https://example.com/video.mp4"),
            "https URL must classify as remote, verbatim: {source:?}"
        );
    }

    #[test]
    fn classifies_bare_path_as_local() {
        for path in ["video.mp4", "./video.mp4", "/data/clips/video.mp4"] {
            assert!(
                matches!(Source::classify(path), Source::Local(_)),
                "{path} must classify as local"
            );
        }
    }

    #[test]
    fn treats_windows_drive_as_local() {
        assert!(matches!(
            Source::classify(r"C:\videos\talk.mp4"),
            Source::Local(_)
        ));
    }

    #[test]
    fn treats_subfile_prefix_as_remote() {
        assert!(matches!(
            Source::classify("subfile,,start,32815239,end,0,,:video.ts"),
            Source::Remote(_)
        ));
    }

    #[test]
    fn treats_digit_leading_scheme_as_local() {
        // FFmpeg would look up protocol "1" and fail opaquely; classifying as
        // local yields a precise filesystem error instead.
        assert!(matches!(Source::classify("1:23.mp4"), Source::Local(_)));
    }

    #[test]
    fn treats_colonless_input_as_local() {
        assert!(matches!(Source::classify("https"), Source::Local(_)));
    }

    #[test]
    fn displays_remote_url_verbatim() {
        let url = "https://example.com/v.mp4?token=a+b.c";
        assert_eq!(Source::classify(url).to_string(), url);
    }

    #[test]
    fn shields_scheme_like_local_filename_from_ffmpeg() {
        // A file literally named `clip:1.mp4`: known local by type, but
        // FFmpeg would re-parse the bare string as a `clip:` protocol URL.
        let source = Source::Local(PathBuf::from("clip:1.mp4"));
        assert_eq!(source.ffmpeg_path().as_ref(), Path::new("./clip:1.mp4"));
        assert_eq!(
            source.as_path(),
            Path::new("clip:1.mp4"),
            "errors stay verbatim"
        );
    }

    #[test]
    fn hands_unambiguous_paths_to_ffmpeg_verbatim() {
        for input in [
            "video.mp4",
            "/data/clip:1.mp4", // absolute: no leading scheme
            "dir/clip:1.mp4",   // colon after a separator: no scheme either
            "https://example.com/v.mp4",
        ] {
            let source = Source::classify(input);
            assert_eq!(source.ffmpeg_path().as_ref(), Path::new(input), "{input}");
        }
    }
}
