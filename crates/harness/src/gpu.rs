// SPDX-License-Identifier: Apache-2.0
//! Crash-isolated GPU device probing for local whisper backends.
//!
//! A compiled-in CUDA/Vulkan backend proves nothing about the host: the
//! loader may be absent (the release tarball bundles `libvulkan.so.1` via
//! an `$ORIGIN` rpath, so the binary always *starts*), the driver may be
//! broken, or no device may be present. ggml's failure paths include hard
//! aborts, so enumeration never runs in the serving process: the parent
//! re-spawns itself as `agent-video probe-gpu <backend>` and reads only the
//! exit status. Metal and CoreML are ambient on Apple hardware and skip
//! the probe entirely.

use std::process::{Command, Stdio};
use std::sync::OnceLock;

use crate::config::Backend;

/// Child exit code: the backend enumerated zero devices.
const NO_DEVICE: i32 = 1;
/// Child exit code: the argument named no known backend.
const UNKNOWN_BACKEND: i32 = 2;

/// Whether `backend` has a usable device on this host.
///
/// Ambient backends answer `true` immediately; probed backends
/// (CUDA/Vulkan) are asked once per process via a child probe — device
/// presence does not change within a process lifetime.
pub fn device_available(backend: Backend) -> bool {
    if !backend.needs_probe() {
        return true;
    }
    static CUDA: OnceLock<bool> = OnceLock::new();
    static VULKAN: OnceLock<bool> = OnceLock::new();
    let cell = match backend {
        Backend::Cuda => &CUDA,
        Backend::Vulkan => &VULKAN,
        Backend::Metal | Backend::Coreml => unreachable!("ambient backends skip the probe"),
    };
    *cell.get_or_init(|| {
        let available = spawn_probe(backend);
        tracing::info!(backend = backend.name(), available, "GPU device probe");
        available
    })
}

/// Run the probe in a disposable child. A failed spawn, a non-zero exit,
/// or a crash inside the driver all read the same way: not available.
fn spawn_probe(backend: Backend) -> bool {
    let Ok(exe) = std::env::current_exe() else {
        return false;
    };
    Command::new(exe)
        .args(["probe-gpu", backend.name()])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Child-side entry for the hidden `probe-gpu` subcommand: enumerate the
/// named backend's devices in-process and answer through the exit code
/// (0 = at least one device). May abort inside a broken driver — that is
/// precisely why it runs in a disposable child.
pub fn probe_child(backend: &str) -> i32 {
    let Some(backend) = Backend::from_name(backend) else {
        return UNKNOWN_BACKEND;
    };
    if device_count(backend) > 0 {
        0
    } else {
        NO_DEVICE
    }
}

/// ggml device count for a probed backend; 0 when the backend is not
/// compiled into this build (there is nothing to enumerate) or is ambient
/// (never probed through this path).
fn device_count(backend: Backend) -> i32 {
    match backend {
        #[cfg(feature = "whisper-cuda")]
        // SAFETY: no-argument FFI query into the statically linked ggml;
        // it only reads driver state.
        Backend::Cuda => unsafe { ggml_backend_cuda_get_device_count() },
        #[cfg(feature = "whisper-vulkan")]
        // SAFETY: as above, for the Vulkan backend.
        Backend::Vulkan => unsafe { ggml_backend_vk_get_device_count() },
        _ => 0,
    }
}

#[cfg(feature = "whisper-cuda")]
unsafe extern "C" {
    fn ggml_backend_cuda_get_device_count() -> std::ffi::c_int;
}
#[cfg(feature = "whisper-vulkan")]
unsafe extern "C" {
    fn ggml_backend_vk_get_device_count() -> std::ffi::c_int;
}
