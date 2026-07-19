// SPDX-License-Identifier: Apache-2.0
//! Manual benchmark: does whisper conditioning (prompt / language hint)
//! improve transcription quality on real media? Never runs in CI —
//! examples build on demand and this one needs a provisioned route
//! (local GPU model or API key) plus ground-truth media.
//!
//! ```text
//! cargo run -p agent-video-harness --example bench_transcribe \
//!     [--features whisper-vulkan] --release -- \
//!     <media> <reference.srt> [--prompt "..."] [--language zh]
//! ```
//!
//! Defaults to `test2.mp4` / `test2.zh.srt` at the repo root (Chinese
//! speech). The metric is CER over normalized characters; see
//! [`normalize`]. Do not feed reference text into `--prompt` — that is
//! leakage, and the numbers would flatter conditioning it never earns
//! on unseen media.

use std::path::{Path, PathBuf};
use std::time::Instant;

use agent_video_core::subtitle::parse_srt;
use agent_video_harness::config::{BaseDirs, Config, load_config};
use agent_video_harness::transcribe::{Route, effective_prompt, transcribe};

fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();

    // The local-route GPU probe re-spawns `current_exe()` as
    // `probe-gpu <backend>`; in an example that binary is us, so answer
    // the probe the same way the CLI does.
    if args.first().map(String::as_str) == Some("probe-gpu") {
        let backend = args.get(1).map(String::as_str).unwrap_or("");
        std::process::exit(agent_video_harness::gpu::probe_child(backend));
    }

    let cli = match Cli::parse(&args) {
        Ok(cli) => cli,
        Err(msg) => {
            eprintln!("{msg}\n\n{USAGE}");
            return std::process::ExitCode::FAILURE;
        }
    };
    match run(&cli) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("bench_transcribe: {msg}");
            std::process::ExitCode::FAILURE
        }
    }
}

const USAGE: &str = "usage: bench_transcribe [<media> <reference.srt>] \
                     [--prompt \"...\"] [--language zh]";

/// Parsed invocation: ground-truth pair plus the conditioning under test.
struct Cli {
    media: PathBuf,
    reference: PathBuf,
    /// Conditioning text for the prompt rows; `None` skips those rows
    /// (deriving one from the reference would leak ground truth).
    prompt: Option<String>,
    language: String,
}

impl Cli {
    fn parse(args: &[String]) -> Result<Cli, String> {
        let mut positional = Vec::new();
        let mut prompt = None;
        let mut language = None;
        let mut it = args.iter();
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--prompt" => {
                    prompt = Some(
                        it.next()
                            .ok_or_else(|| "--prompt needs a value".to_string())?
                            .clone(),
                    );
                }
                "--language" => {
                    language = Some(
                        it.next()
                            .ok_or_else(|| "--language needs a value".to_string())?
                            .clone(),
                    );
                }
                flag if flag.starts_with("--") => return Err(format!("unknown flag {flag}")),
                positional_arg => positional.push(positional_arg.to_string()),
            }
        }
        if positional.len() == 1 || positional.len() > 2 {
            return Err("provide both <media> and <reference.srt>, or neither".to_string());
        }
        let mut positional = positional.into_iter();
        Ok(Cli {
            media: positional
                .next()
                .map_or_else(|| PathBuf::from("test2.mp4"), PathBuf::from),
            reference: positional
                .next()
                .map_or_else(|| PathBuf::from("test2.zh.srt"), PathBuf::from),
            prompt,
            // The bundled reference is Chinese speech.
            language: language.unwrap_or_else(|| "zh".to_string()),
        })
    }
}

fn run(cli: &Cli) -> Result<(), String> {
    // The operator's real config supplies paths (installed model, API key);
    // the conditioning knobs are cleared so every row states its own.
    let mut cfg = load_config(&BaseDirs::detect()).map_err(|e| e.to_string())?;
    cfg.whisper.prompt = None;
    cfg.whisper.language = None;

    if let Err(fault) = Route::resolve(&cfg) {
        // Not a bench failure: the machine simply has no transcription
        // route. Say how to provision one and leave quietly.
        println!("no transcription route available — nothing to benchmark.\n{fault}");
        return Ok(());
    }

    let reference = std::fs::read_to_string(&cli.reference)
        .map_err(|e| format!("reading {}: {e}", cli.reference.display()))?;
    let ref_chars = normalize(&concat_cues(&reference));
    if ref_chars.is_empty() {
        return Err(format!(
            "{} normalized to zero characters — not a usable reference",
            cli.reference.display()
        ));
    }

    let mut rows: Vec<RowSpec> = vec![
        RowSpec::new("bare", None, false),
        RowSpec::new("language", None, true),
    ];
    match &cli.prompt {
        Some(prompt) => {
            rows.push(RowSpec::new("prompt", Some(prompt.clone()), false));
            rows.push(RowSpec::new("prompt+language", Some(prompt.clone()), true));
        }
        None => println!(
            "note: no --prompt given — skipping the prompt rows. Deriving one from the \
             reference SRT would leak ground truth into the conditioning.\n"
        ),
    }

    // Each row writes its SRT into a disposable directory: the real
    // transcript cache must never see benchmark artifacts.
    let work_dir = std::env::temp_dir().join(format!("bench-transcribe-{}", std::process::id()));
    std::fs::create_dir_all(&work_dir)
        .map_err(|e| format!("creating {}: {e}", work_dir.display()))?;
    let outcome = run_rows(&cfg, cli, &rows, &ref_chars, &work_dir);
    let _ = std::fs::remove_dir_all(&work_dir);
    let results = outcome?;

    print_table(&rows, &results, ref_chars.len());
    Ok(())
}

/// One matrix row: which conditioning the transcription receives.
struct RowSpec {
    label: &'static str,
    prompt: Option<String>,
    language: bool,
}

impl RowSpec {
    fn new(label: &'static str, prompt: Option<String>, language: bool) -> RowSpec {
        RowSpec {
            label,
            prompt,
            language,
        }
    }
}

/// Measured facts for one finished row.
struct RowResult {
    cer: f64,
    cues: usize,
    chars: usize,
    seconds: f64,
}

fn run_rows(
    cfg: &Config,
    cli: &Cli,
    rows: &[RowSpec],
    ref_chars: &[char],
    work_dir: &Path,
) -> Result<Vec<RowResult>, String> {
    let mut results = Vec::with_capacity(rows.len());
    for (i, row) in rows.iter().enumerate() {
        let mut row_cfg = cfg.clone();
        row_cfg.whisper.prompt = row.prompt.clone();
        row_cfg.whisper.language = row.language.then(|| cli.language.clone());
        // Route production shaping (NUL strip, byte cap) over the raw
        // argument — the bench must measure what real runs would send.
        let prompt = effective_prompt(&row_cfg, None);

        let out_srt = work_dir.join(format!("row-{i}.srt"));
        eprintln!("[{}/{}] {} ...", i + 1, rows.len(), row.label);
        let started = Instant::now();
        transcribe(&row_cfg, &cli.media, &out_srt, &prompt, &mut |_| {})
            .map_err(|e| format!("row '{}': {e}", row.label))?;
        let seconds = started.elapsed().as_secs_f64();

        let srt = std::fs::read_to_string(&out_srt)
            .map_err(|e| format!("reading {}: {e}", out_srt.display()))?;
        let cues = parse_srt(&srt).len();
        let hyp_chars = normalize(&concat_cues(&srt));
        let dist = levenshtein(ref_chars, &hyp_chars);
        results.push(RowResult {
            cer: dist as f64 / ref_chars.len() as f64,
            cues,
            chars: hyp_chars.len(),
            seconds,
        });
    }
    Ok(results)
}

/// Cue texts joined in time order — CER scores the transcript as one
/// stream; cue segmentation differences must not count as errors.
fn concat_cues(srt: &str) -> String {
    let mut cues = parse_srt(srt);
    cues.sort_by(|a, b| a.start.total_cmp(&b.start));
    cues.into_iter().map(|c| c.text).collect()
}

/// Normalize for CER: keep only alphanumeric characters (Unicode-aware —
/// CJK ideographs are `Alphabetic`, so they survive), lowercase ASCII.
/// Whitespace and punctuation carry no recognition signal and their
/// conventions differ between human and machine transcripts.
fn normalize(text: &str) -> Vec<char> {
    text.chars()
        .filter(|c| c.is_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

/// Levenshtein distance in the two-row form: O(min(|a|,|b|)) memory,
/// fine for the tens of thousands of characters a real transcript holds.
fn levenshtein(a: &[char], b: &[char]) -> usize {
    let (short, long) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    let mut prev: Vec<usize> = (0..=short.len()).collect();
    let mut cur = vec![0usize; short.len() + 1];
    for (i, &lc) in long.iter().enumerate() {
        cur[0] = i + 1;
        for (j, &sc) in short.iter().enumerate() {
            let subst = prev[j] + usize::from(lc != sc);
            cur[j + 1] = subst.min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[short.len()]
}

fn print_table(rows: &[RowSpec], results: &[RowResult], ref_len: usize) {
    let bare_cer = results[0].cer;
    println!("\nreference: {ref_len} normalized chars");
    println!(
        "{:<16} {:>8} {:>10} {:>6} {:>7} {:>8}",
        "row", "CER", "Δ vs bare", "cues", "chars", "seconds"
    );
    for (row, r) in rows.iter().zip(results) {
        let delta = if std::ptr::eq(r, &results[0]) {
            "—".to_string()
        } else {
            format!("{:+.2}pp", (r.cer - bare_cer) * 100.0)
        };
        println!(
            "{:<16} {:>7.2}% {:>10} {:>6} {:>7} {:>7.1}s",
            row.label,
            r.cer * 100.0,
            delta,
            r.cues,
            r.chars,
            r.seconds
        );
    }
    println!(
        "\nCER = Levenshtein / reference length over cue text concatenated in time \
         order, whitespace and punctuation stripped (Unicode alphanumeric only, \
         ASCII lowercased)."
    );
}
