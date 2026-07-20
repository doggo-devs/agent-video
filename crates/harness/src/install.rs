// SPDX-License-Identifier: Apache-2.0
//! `install-whisper`: explicit provisioning of whisper weights.
//!
//! Fetches the pinned HuggingFace ggml model for the configured (or
//! overridden) model name plus the Silero VAD model into `paths.models`,
//! sha256-verified, staged-then-renamed. Weights are runtime data, not
//! dependencies: nothing is ever auto-downloaded mid-run, and this command
//! is the only writer of the models directory — though several instances
//! of it may run at once, so staging names are writer-unique
//! ([`crate::cache::naming::part_path`]) and publication is an atomic
//! last-writer-wins rename of a stream each writer verified alone.

use std::io::{Read, Write};
use std::path::Path;
use std::time::{Duration, Instant, SystemTime};

use sha2::{Digest, Sha256};

use super::config::{Config, WhisperModel};
use super::plan::format_bytes;
use super::transcribe::{VAD_FILENAME, model_filename};

/// A pinned artifact: URL, sha256, and size for progress display.
struct Pin {
    filename: &'static str,
    url: &'static str,
    sha256: &'static str,
    bytes: u64,
}

const MEDIUM: Pin = Pin {
    filename: "ggml-medium.bin",
    url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
    sha256: "6c14d5adee5f86394037b4e4e8b59f1673b6cee10e3cf0b11bbdbee79c156208",
    bytes: 1_533_763_059,
};

const LARGE_V3: Pin = Pin {
    filename: "ggml-large-v3.bin",
    url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin",
    sha256: "64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2",
    bytes: 3_095_033_483,
};

const LARGE_V3_TURBO: Pin = Pin {
    filename: "ggml-large-v3-turbo.bin",
    url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
    sha256: "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69",
    bytes: 1_624_555_275,
};

const SILERO_VAD: Pin = Pin {
    filename: VAD_FILENAME,
    url: "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin",
    sha256: "29940d98d42b91fbd05ce489f3ecf7c72f0a42f027e4875919a28fb4c04ea2cf",
    bytes: 885_098,
};

/// Installation failed; the message names the artifact and cause.
#[derive(Debug)]
pub struct InstallError(pub String);

impl InstallError {
    /// Exit code for CLI paths. Core owns decades 10–40; the harness claims
    /// 50 for provisioning failures (network, checksum, disk) — expected
    /// runtime conditions, not the uncoded-bug exit 1.
    pub fn exit_code(&self) -> u8 {
        50
    }
}

impl std::fmt::Display for InstallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for InstallError {}

/// Run `install-whisper`: provision the model named by `override_model`
/// (falling back to the config) plus the VAD model into `paths.models`.
/// Present artifacts are verified and skipped; progress goes to stderr.
pub fn install_whisper(cfg: &Config, override_model: Option<&str>) -> Result<(), InstallError> {
    let model = match override_model {
        Some("medium") => WhisperModel::Medium,
        Some("large-v3") => WhisperModel::LargeV3,
        Some("large-v3-turbo") => WhisperModel::LargeV3Turbo,
        Some(other) => {
            return Err(InstallError(format!(
                "unknown model '{other}': install-whisper provisions the pinned models \
                 (medium, large-v3, large-v3-turbo); a self-supplied ggml file needs no \
                 installation — point whisper.model at it"
            )));
        }
        None => cfg.whisper.model.clone(),
    };
    let pin = match &model {
        WhisperModel::Medium => MEDIUM,
        WhisperModel::LargeV3 => LARGE_V3,
        WhisperModel::LargeV3Turbo => LARGE_V3_TURBO,
        WhisperModel::File(p) => {
            return Err(InstallError(format!(
                "whisper.model points at a self-supplied file ({}); nothing to install — pass \
                 --model medium|large-v3|large-v3-turbo to provision a pinned model instead",
                p.display()
            )));
        }
    };
    debug_assert_eq!(Some(pin.filename), model_filename(&model));

    sweep_stale_parts(&cfg.paths.models);
    for pin in [pin, SILERO_VAD] {
        provision(&cfg.paths.models, &pin)?;
    }
    Ok(())
}

/// A `.part` in the models dir older than this has no live writer left.
/// Two installs may run concurrently, so the sweep cannot treat every
/// partial as debris: a downloading rival touches its staging file with
/// every 256 KiB chunk — even a crawling link updates the mtime every few
/// seconds — while a killed run's leftover only ages. One hour is orders
/// of magnitude past any live write gap, and multi-GiB debris still gets
/// collected by the next install.
const STALE_PART_AGE: Duration = Duration::from_secs(60 * 60);

/// Remove abandoned `*.part` files before provisioning.
///
/// Staging names are writer-unique, so a partial file here is either a
/// rival install's live transfer or the debris of a run that was killed
/// mid-download (SIGKILL, power loss) — the normal error path already
/// unlinks its own `.part`. The two are told apart by age
/// ([`STALE_PART_AGE`]): only what no live writer can still be touching
/// is collected. Best-effort: a directory that does not exist yet, or a
/// file we cannot stat or unlink, is skipped.
fn sweep_stale_parts(models_dir: &Path) {
    let now = SystemTime::now();
    let Ok(entries) = std::fs::read_dir(models_dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_none_or(|ext| ext != "part") {
            continue;
        }
        let abandoned = entry
            .metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|mtime| now.duration_since(mtime).ok())
            .is_some_and(|age| age > STALE_PART_AGE);
        if abandoned && std::fs::remove_file(&path).is_ok() {
            eprintln!("removed stale partial download {}", path.display());
        }
    }
}

/// Fetch one pinned artifact unless a verified copy already exists.
fn provision(models_dir: &Path, pin: &Pin) -> Result<(), InstallError> {
    let dest = models_dir.join(pin.filename);
    if dest.is_file() {
        eprint!("verifying {} ... ", pin.filename);
        if file_sha256(&dest)? == pin.sha256 {
            eprintln!("ok (already installed)");
            return Ok(());
        }
        eprintln!("checksum mismatch, re-downloading");
    }
    std::fs::create_dir_all(models_dir)
        .map_err(|e| InstallError(format!("cannot create {}: {e}", models_dir.display())))?;

    eprintln!(
        "downloading {} ({}) ...",
        pin.filename,
        format_bytes(pin.bytes)
    );
    // The staging name comes from the cache's one writer-unique grammar
    // (crate::cache::naming) rather than a third local spelling: the models
    // dir is not the cache, but the hazard is the same — two concurrent
    // installs of one model must never share a staging file, or their
    // interleaved writes could pass each writer's own checksum and be
    // renamed into a corrupt final file. With exclusive staging, the
    // rename below is atomic per writer and last-writer-wins is sound:
    // both candidates were verified whole against the same pin.
    let part = crate::cache::naming::part_path(&dest);
    match fetch_verified(pin, &part) {
        Ok(()) => {
            std::fs::rename(&part, &dest)
                .map_err(|e| InstallError(format!("cannot finalize {}: {e}", dest.display())))?;
            eprintln!("installed {}", dest.display());
            Ok(())
        }
        Err(e) => {
            let _ = std::fs::remove_file(&part);
            Err(e)
        }
    }
}

/// Stream the artifact to `part`, hashing on the fly; fails on any digest
/// mismatch so a bad `.part` never survives. Verify-during-stream is sound
/// because `part` is writer-unique: this stream is the only writer of that
/// file, so the digest covers exactly the bytes on disk.
fn fetch_verified(pin: &Pin, part: &Path) -> Result<(), InstallError> {
    let mut res = ureq::get(pin.url).call().map_err(|e| match e {
        ureq::Error::StatusCode(code) => {
            InstallError(format!("download failed: GET {} → HTTP {code}", pin.url))
        }
        other => InstallError(format!("download failed: GET {} → {other}", pin.url)),
    })?;

    let mut file = std::fs::File::create(part)
        .map_err(|e| InstallError(format!("cannot write {}: {e}", part.display())))?;
    let mut reader = res.body_mut().as_reader();
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 256 * 1024];
    let mut bytes: u64 = 0;
    let mut last_report = Instant::now();
    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|e| InstallError(format!("download failed mid-stream: {e}")))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        file.write_all(&buf[..n])
            .map_err(|e| InstallError(format!("cannot write {}: {e}", part.display())))?;
        bytes += n as u64;
        if last_report.elapsed() >= Duration::from_secs(1) {
            last_report = Instant::now();
            eprint!(
                "\r  {} / {} ({}%)",
                format_bytes(bytes),
                format_bytes(pin.bytes),
                bytes * 100 / pin.bytes.max(1)
            );
        }
    }
    eprintln!("\r  {} downloaded          ", format_bytes(bytes));

    let digest = super::identity::hex(&hasher.finalize());
    if digest != pin.sha256 {
        return Err(InstallError(format!(
            "sha256 mismatch for {}: expected {}, got {digest} — the upstream file changed or \
             the download corrupted; nothing was installed",
            pin.filename, pin.sha256
        )));
    }
    Ok(())
}

/// sha256 of a file's contents as lowercase hex.
fn file_sha256(path: &Path) -> Result<String, InstallError> {
    let mut file = std::fs::File::open(path)
        .map_err(|e| InstallError(format!("cannot read {}: {e}", path.display())))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 256 * 1024];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| InstallError(format!("cannot read {}: {e}", path.display())))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(super::identity::hex(&hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{default_config, test_dirs};
    use std::path::PathBuf;

    fn cfg() -> Config {
        default_config(&test_dirs(Path::new("/home/u")))
    }

    #[test]
    fn hashes_file_contents() {
        let dir = crate::tempdir::TempDir::new("av-install");
        let p = dir.join("x.bin");
        std::fs::write(&p, b"abc").expect("write");
        assert_eq!(
            file_sha256(&p).expect("hash"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    fn age(path: &Path, by: Duration) {
        std::fs::File::options()
            .append(true)
            .open(path)
            .and_then(|f| f.set_modified(SystemTime::now() - by))
            .expect("age the file");
    }

    #[test]
    fn sweeps_stale_part_files_but_keeps_finished_weights() {
        let dir = crate::tempdir::TempDir::new("av-sweep");
        let part = crate::cache::naming::part_path(&dir.join("ggml-medium.bin"));
        std::fs::write(&part, b"partial").expect("write part");
        std::fs::write(dir.join("ggml-medium.bin"), b"whole").expect("write bin");
        age(&part, STALE_PART_AGE + Duration::from_secs(60));

        sweep_stale_parts(&dir);

        assert!(!part.exists(), "the killed-run partial is collected");
        assert!(
            dir.join("ggml-medium.bin").exists(),
            "a finished weight file is untouched"
        );
    }

    #[test]
    fn sweep_spares_a_live_rivals_fresh_part() {
        // A concurrent install's transfer keeps its staging file's mtime
        // fresh; the sweep must not pull it out from under the rival.
        let dir = crate::tempdir::TempDir::new("av-sweep-live");
        let part = crate::cache::naming::part_path(&dir.join("ggml-medium.bin"));
        std::fs::write(&part, b"in flight").expect("write part");

        sweep_stale_parts(&dir);

        assert!(part.exists(), "a fresh part is a live rival's download");
    }

    #[test]
    fn staging_names_are_writer_unique() {
        let dest = Path::new("/models/ggml-medium.bin");
        let a = crate::cache::naming::part_path(dest);
        let b = crate::cache::naming::part_path(dest);
        assert_ne!(a, b, "two installs must never share a staging file");
        for p in [&a, &b] {
            assert!(
                p.extension().is_some_and(|e| e == "part"),
                "the sweep must still recognize the staging name: {p:?}"
            );
        }
    }

    #[test]
    fn concurrent_installs_stage_and_publish_without_interleaving() {
        // The defect scenario at the naming level: two installs of the same
        // model. Each streams into its own staging file (so each digest
        // covers exactly its own bytes) and each rename is atomic; the last
        // writer wins with a file it verified whole.
        let dir = crate::tempdir::TempDir::new("av-install-race");
        let dest = dir.join("ggml-medium.bin");
        let a = crate::cache::naming::part_path(&dest);
        let b = crate::cache::naming::part_path(&dest);

        std::fs::write(&a, b"writer-a bytes").expect("stage a");
        std::fs::write(&b, b"writer-b bytes").expect("stage b");
        assert_eq!(
            std::fs::read(&a).expect("read a"),
            b"writer-a bytes",
            "a rival's writes never reach this writer's stream"
        );
        std::fs::rename(&a, &dest).expect("a publishes");
        std::fs::rename(&b, &dest).expect("b publishes over a");

        assert_eq!(
            std::fs::read(&dest).expect("read final"),
            b"writer-b bytes",
            "last writer wins with an unmixed file"
        );
    }

    #[test]
    fn sweep_tolerates_missing_directory() {
        // First-ever install: the models dir does not exist yet.
        sweep_stale_parts(Path::new("/nonexistent/av-models-xyz"));
    }

    #[test]
    fn rejects_unknown_model_override() {
        let err = install_whisper(&cfg(), Some("base")).expect_err("not a pinned model");
        assert!(err.0.contains("medium, large-v3, large-v3-turbo"), "{err}");
    }

    #[test]
    fn rejects_install_for_self_supplied_file() {
        let mut cfg = cfg();
        cfg.whisper.model = WhisperModel::File(PathBuf::from("/opt/m.bin"));
        let err = install_whisper(&cfg, None).expect_err("nothing to install");
        assert!(err.0.contains("self-supplied"), "{err}");
    }
}
