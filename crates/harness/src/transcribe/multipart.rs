// SPDX-License-Identifier: Apache-2.0
//! Minimal multipart/form-data encoder — ureq 3.x has none built in.
//! A general-purpose codec: parts are the caller's policy, framing is
//! this module's. Knows nothing about transcription.

/// A multipart/form-data body under construction.
///
/// The boundary derives from one payload's SHA-256 ([`Form::keyed_to`]),
/// so no RNG is needed: text that does not know the payload cannot
/// contain the boundary short of a preimage, and therefore cannot break
/// the framing.
pub(super) struct Form {
    boundary: String,
    body: Vec<u8>,
}

impl Form {
    /// Start a form whose boundary is derived from `payload`'s hash. Pass
    /// the dominant binary part: the framing guarantee holds against any
    /// field value that cannot predict `payload`'s bytes.
    pub(super) fn keyed_to(payload: &[u8]) -> Form {
        use sha2::{Digest, Sha256};
        let digest = Sha256::digest(payload);
        Form {
            boundary: format!("----avform{}", crate::identity::hex(&digest[..16])),
            body: Vec::new(),
        }
    }

    /// Append a file part.
    pub(super) fn file(&mut self, name: &str, filename: &str, content_type: &str, data: &[u8]) {
        self.part(name, Some(filename), content_type, data);
    }

    /// Append a plain-text field.
    pub(super) fn text(&mut self, name: &str, value: &str) {
        self.part(name, None, "text/plain", value.as_bytes());
    }

    fn part(&mut self, name: &str, filename: Option<&str>, content_type: &str, data: &[u8]) {
        self.body
            .extend_from_slice(format!("--{}\r\n", self.boundary).as_bytes());
        let disposition = match filename {
            Some(fname) => {
                format!("Content-Disposition: form-data; name=\"{name}\"; filename=\"{fname}\"\r\n")
            }
            None => format!("Content-Disposition: form-data; name=\"{name}\"\r\n"),
        };
        self.body.extend_from_slice(disposition.as_bytes());
        self.body
            .extend_from_slice(format!("Content-Type: {content_type}\r\n\r\n").as_bytes());
        self.body.extend_from_slice(data);
        self.body.extend_from_slice(b"\r\n");
    }

    /// Close the form: appends the final delimiter.
    pub(super) fn finish(mut self) -> EncodedForm {
        self.body
            .extend_from_slice(format!("--{}--\r\n", self.boundary).as_bytes());
        EncodedForm {
            body: self.body,
            boundary: self.boundary,
        }
    }
}

/// A finished multipart body and the boundary that frames it.
pub(super) struct EncodedForm {
    pub(super) body: Vec<u8>,
    pub(super) boundary: String,
}

impl EncodedForm {
    /// The `Content-Type` header value announcing the boundary.
    pub(super) fn content_type(&self) -> String {
        format!("multipart/form-data; boundary={}", self.boundary)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundary_is_a_function_of_the_keyed_payload() {
        let a = Form::keyed_to(b"payload A").finish();
        let b = Form::keyed_to(b"payload B").finish();
        assert_ne!(
            a.boundary, b.boundary,
            "different payloads, different frames"
        );
        assert_eq!(
            Form::keyed_to(b"payload A").finish().boundary,
            a.boundary,
            "deterministic: no RNG in the frame"
        );
        assert!(a.content_type().ends_with(&a.boundary));
    }
}
