// SPDX-License-Identifier: Apache-2.0
//! Shared test scaffolding for the `mcp` handler modules: hand-written
//! fakes for the injected capabilities (prober, core, fetcher, whisper), a
//! `TestEnv` that wires them into `ServerDeps` over a temp dir, and small
//! helpers to drive `handle_video_call` and read its wire result.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::mpsc::{Receiver, Sender};

use rmcp::model::{CallToolResult, ContentBlock};

use super::core_run::{CoreExit, CoreRun, CoreRunner};
use super::deps::{FileProber, ServerDeps, UrlFetcher, Whisper, null_notifier};
use super::handler::handle_video_call;
use super::tracker::Tracker;
use crate::config::{Config, default_config, test_dirs};
use crate::fault::Fault;
use crate::fetch::{
    CancelSignal, DownloadIo, DownloadOption, Downloaded, ProgressFn, Recipe, SourceInfo,
    direct::DirectRecipe,
};
use crate::relay::PolicyRelay;
use crate::schema::ProbeSummary;
use crate::tokens::{Dest, DownloadPlan, Tokens};

// ── fakes ──

struct FakeProber(ProbeSummary);

impl FileProber for FakeProber {
    fn probe(&self, _file: &Path) -> Result<ProbeSummary, Fault> {
        Ok(self.0.clone())
    }
}

pub(crate) struct FakeCore {
    stdout: String,
    /// Tests flip this after env construction to simulate a run whose
    /// stdout blew the capture cap.
    pub(crate) truncate: std::sync::atomic::AtomicBool,
    pub(crate) calls: Mutex<Vec<Vec<String>>>,
}

impl CoreRunner for FakeCore {
    fn run(&self, args: &[String]) -> std::io::Result<CoreRun> {
        self.calls.lock().expect("calls").push(args.to_vec());
        Ok(CoreRun {
            exit: CoreExit::Success,
            stdout: self.stdout.clone(),
            stdout_truncated: self.truncate.load(std::sync::atomic::Ordering::SeqCst),
            stderr: String::new(),
        })
    }
}

/// Shared handles let tests inspect fakes after `ServerDeps` boxes them.
impl<T: CoreRunner> CoreRunner for Arc<T> {
    fn run(&self, args: &[String]) -> std::io::Result<CoreRun> {
        (**self).run(args)
    }
}

impl<T: UrlFetcher> UrlFetcher for Arc<T> {
    fn probe_url(&self, url: &str, cfg: &Config, relay: &PolicyRelay) -> Result<SourceInfo, Fault> {
        (**self).probe_url(url, cfg, relay)
    }

    fn download(
        &self,
        plan: &DownloadPlan,
        io: DownloadIo<'_>,
        cfg: &Config,
        relay: &PolicyRelay,
        cancel: &CancelSignal,
        progress: ProgressFn<'_>,
    ) -> Result<Downloaded, Fault> {
        (**self).download(plan, io, cfg, relay, cancel, progress)
    }
}

impl<T: Whisper> Whisper for Arc<T> {
    fn available(&self, cfg: &Config) -> bool {
        (**self).available(cfg)
    }

    fn transcribe(
        &self,
        cfg: &Config,
        file: &Path,
        out_srt: &Path,
        prompt: &str,
        on_step: &mut dyn FnMut(&str),
    ) -> Result<crate::transcribe::Transcribed, Fault> {
        (**self).transcribe(cfg, file, out_srt, prompt, on_step)
    }
}

pub(crate) struct FakeWhisper {
    available: bool,
    pub(crate) calls: Mutex<u32>,
    /// The effective conditioning text of every `transcribe` call, in
    /// order, so tests can pin what whisper was primed with.
    pub(crate) prompts: Mutex<Vec<String>>,
    /// Decoded-audio seconds the fake reports; `INFINITY` = "covers
    /// everything", so only truncation tests set a finite value.
    pub(crate) audio_seconds: Mutex<f64>,
    /// When present, `transcribe` announces itself on the sender and then
    /// blocks on the receiver until released, so tests can hold one whisper
    /// run open while a concurrent request arrives.
    pub(crate) gate: Mutex<Option<(Sender<()>, Receiver<()>)>>,
}

impl Whisper for FakeWhisper {
    fn available(&self, _cfg: &Config) -> bool {
        self.available
    }

    fn transcribe(
        &self,
        _cfg: &Config,
        _file: &Path,
        out_srt: &Path,
        prompt: &str,
        on_step: &mut dyn FnMut(&str),
    ) -> Result<crate::transcribe::Transcribed, Fault> {
        *self.calls.lock().expect("calls") += 1;
        self.prompts
            .lock()
            .expect("prompts")
            .push(prompt.to_string());
        if let Some((entered, release)) = &*self.gate.lock().expect("whisper gate") {
            let _ = entered.send(());
            let _ = release.recv();
        }
        on_step("transcribing");
        std::fs::create_dir_all(out_srt.parent().expect("parent")).expect("transcripts dir");
        std::fs::write(out_srt, "1\n00:00:00,000 --> 00:00:01,000\nhi\n\n").expect("write srt");
        Ok(crate::transcribe::Transcribed {
            audio_seconds: *self.audio_seconds.lock().expect("audio_seconds"),
        })
    }
}

/// Downloads block on `gate` when present, making slow downloads
/// deterministic in tests. Records each job's cancel signal so tests can
/// observe that abandonment reaches the fetch, and counts finished
/// fetches so tests can sequence against a download thread they cannot
/// join.
pub(crate) struct FakeFetcher {
    pub(crate) info: SourceInfo,
    pub(crate) gate: Option<Mutex<Receiver<()>>>,
    pub(crate) warning: Option<String>,
    pub(crate) downloads: Mutex<u32>,
    /// Fetches that ran to completion (the file is on disk when this
    /// ticks); the surrounding settle may still be in flight.
    pub(crate) finished: Mutex<u32>,
    pub(crate) last_cancel: Mutex<Option<CancelSignal>>,
}

impl UrlFetcher for FakeFetcher {
    fn probe_url(
        &self,
        _url: &str,
        _cfg: &Config,
        _relay: &PolicyRelay,
    ) -> Result<SourceInfo, Fault> {
        Ok(self.info.clone())
    }

    fn download(
        &self,
        _plan: &DownloadPlan,
        io: DownloadIo<'_>,
        _cfg: &Config,
        _relay: &PolicyRelay,
        cancel: &CancelSignal,
        progress: ProgressFn<'_>,
    ) -> Result<Downloaded, Fault> {
        *self.downloads.lock().expect("downloads") += 1;
        *self.last_cancel.lock().expect("last_cancel") = Some(cancel.clone());
        if let Some(gate) = &self.gate {
            progress(1024, Some(4096));
            gate.lock()
                .expect("gate")
                .recv()
                .map_err(|_| Fault::transient("gate closed"))?;
        }
        let dest = match io {
            DownloadIo::Cache { dir, key } => dir.join(format!("{key}.mp4")),
            DownloadIo::Keep { dir } => dir.join("v.mp4"),
        };
        let mut f = std::fs::File::create(&dest).expect("create download");
        f.write_all(b"fake video bytes").expect("write");
        progress(16, Some(16));
        *self.finished.lock().expect("finished") += 1;
        Ok(Downloaded {
            path: dest,
            warning: self.warning.clone(),
        })
    }
}

// ── scaffolding ──

pub(crate) fn probe(duration: f64, subtitles: Vec<String>) -> ProbeSummary {
    ProbeSummary {
        duration,
        width: 1920,
        height: 1080,
        size_bytes: Some(10 * 1024 * 1024),
        container: Some("mp4".to_string()),
        subtitles,
    }
}

/// Built by serializing the real wire types: the fake core's stdout is
/// in-contract by construction, not by hand-maintained JSON.
pub(crate) fn core_json() -> String {
    use agent_video_core::types::{
        Detail, GridImage, GridPosition, KeyFrameMeta, LayoutKind, Quality, Settings,
    };
    use agent_video_core::wire::{Grid, Metadata};
    serde_json::to_string(&Metadata::process(
        "v.mp4".to_string(),
        90.0,
        Settings {
            max_edge: 2048,
            quality: Quality::Low,
            detail: Detail::Standard,
        },
        vec![Grid {
            image: GridImage::Base64("QUJD".to_string()),
            time_range: "00:00-01:30".to_string(),
            total_frames: 12,
            key_frames: vec![KeyFrameMeta {
                timestamp: "00:10".to_string(),
                score: 5,
                grid_position: GridPosition::Hero,
            }],
            layout: LayoutKind::Uniform,
            subtitles: Some("00:01 hi".to_string()),
        }],
    ))
    .expect("wire metadata serializes")
}

pub(crate) struct TestEnv {
    pub(crate) deps: Arc<ServerDeps>,
    pub(crate) core: Arc<FakeCore>,
    pub(crate) fetcher: Arc<FakeFetcher>,
    pub(crate) whisper: Arc<FakeWhisper>,
    // Last field: dropped after everything above so its files outlive
    // any path-holding dependency during teardown.
    pub(crate) dir: crate::testutil::TempDir,
}

pub(crate) fn env(probe_summary: ProbeSummary, whisper_available: bool) -> TestEnv {
    env_with_fetcher(
        probe_summary,
        whisper_available,
        FakeFetcher {
            info: SourceInfo {
                headline: "A video.".to_string(),
                title: None,
                options: vec![analysis_option()],
            },
            gate: None,
            warning: None,
            downloads: Mutex::new(0),
            finished: Mutex::new(0),
            last_cancel: Mutex::new(None),
        },
    )
}

pub(crate) fn env_with_fetcher(
    probe_summary: ProbeSummary,
    whisper_available: bool,
    fetcher: FakeFetcher,
) -> TestEnv {
    let dir = crate::testutil::TempDir::new("av-server");
    let mut config = default_config(&test_dirs(&dir));
    config.paths.cache = dir.join("cache");
    config.paths.keep = dir.join("keep");
    let cache = crate::cache::Cache::new(&config.paths.cache, config.cache.max_bytes);
    let core = Arc::new(FakeCore {
        stdout: core_json(),
        truncate: std::sync::atomic::AtomicBool::new(false),
        calls: Mutex::new(Vec::new()),
    });
    let fetcher = Arc::new(fetcher);
    let whisper = Arc::new(FakeWhisper {
        available: whisper_available,
        calls: Mutex::new(0),
        prompts: Mutex::new(Vec::new()),
        audio_seconds: Mutex::new(f64::INFINITY),
        gate: Mutex::new(None),
    });
    let deps = Arc::new(ServerDeps {
        cache,
        tokens: Tokens::default(),
        downloads: Tracker::default(),
        transcriptions: crate::gate::KeyedGate::default(),
        // Generous enough that only tests which shrink it observe the
        // bound; real waits in tests are released in milliseconds.
        transcribe_wait: std::time::Duration::from_secs(60),
        prober: Box::new(FakeProber(probe_summary)),
        fetcher: Box::new(Arc::clone(&fetcher)),
        whisper: Box::new(Arc::clone(&whisper)),
        core: Box::new(Arc::clone(&core)),
        relay: crate::mcp::deps::RelaySlot::new(),
        config,
    });
    TestEnv {
        deps,
        core,
        fetcher,
        whisper,
        dir,
    }
}

pub(crate) fn analysis_option() -> DownloadOption {
    DownloadOption {
        dest: Dest::Cache,
        label: "480p".to_string(),
        effect: "download the analysis rip".to_string(),
        estimated_bytes: Some(1024),
        recipe: Recipe::Direct(DirectRecipe {
            ext: "mp4".to_string(),
            filename: "v.mp4".to_string(),
        }),
    }
}

pub(crate) fn keep_option() -> DownloadOption {
    DownloadOption {
        dest: Dest::Keep,
        label: "1080p".to_string(),
        effect: "save the full-quality file to the keep dir".to_string(),
        estimated_bytes: Some(4096),
        recipe: Recipe::Direct(DirectRecipe {
            ext: "mp4".to_string(),
            filename: "v.mp4".to_string(),
        }),
    }
}

pub(crate) fn media_file(env: &TestEnv, name: &str) -> PathBuf {
    let p = env.dir.join(name);
    std::fs::write(&p, b"pretend this is a video").expect("write media");
    p
}

pub(crate) fn call(env: &TestEnv, args: serde_json::Value) -> CallToolResult {
    call_with_notify(env, args, &null_notifier())
}

/// Drive one call with a caller-owned notifier, for tests pinning who may
/// send progress notifications (and when).
pub(crate) fn call_with_notify(
    env: &TestEnv,
    args: serde_json::Value,
    notify: &super::deps::Notifier,
) -> CallToolResult {
    let serde_json::Value::Object(map) = args else {
        panic!("args must be an object");
    };
    handle_video_call(&map, &env.deps, notify)
}

pub(crate) fn text_of(result: &CallToolResult) -> String {
    result
        .content
        .iter()
        .filter_map(|c| match c {
            ContentBlock::Text(t) => Some(t.text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub(crate) fn body_of(result: &CallToolResult) -> serde_json::Value {
    serde_json::from_str(&text_of(result)).expect("body is JSON")
}

/// The offered token strings of a `confirm_required` body, in offer order.
/// Tokens are opaque and minted per commit, so tests read them from the
/// body the model would see instead of assuming any spelling.
pub(crate) fn offered_tokens(body: &serde_json::Value) -> Vec<String> {
    body["tokens"]
        .as_array()
        .expect("confirm_required body carries tokens")
        .iter()
        .map(|t| t["token"].as_str().expect("token is a string").to_string())
        .collect()
}
