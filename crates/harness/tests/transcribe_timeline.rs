// SPDX-License-Identifier: Apache-2.0
//! Real decoding through an HTTP transcription peer, without remote inference.

#[allow(dead_code)]
#[path = "../../core/tests/support/fixtures.rs"]
mod fixtures;

use agent_video_harness::config::{BaseDirs, default_config};
use agent_video_harness::transcribe::transcribe;
use std::io::{BufRead, Read, Write};

#[test]
fn coalesces_short_audio_gaps_without_shifting_source_cues() {
    let Some(fixture) = fixtures::sparse_audio(44_100) else {
        return;
    };
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("local API");
    let addr = listener.local_addr().expect("address");
    let peer = std::thread::spawn(move || {
        let (mut socket, _) = listener.accept().expect("request");
        socket
            .set_read_timeout(Some(std::time::Duration::from_secs(10)))
            .expect("timeout");
        let mut reader = std::io::BufReader::new(&mut socket);
        let mut len = None;
        loop {
            let mut line = String::new();
            assert!(reader.read_line(&mut line).expect("header") > 0);
            if line == "\r\n" {
                break;
            }
            if let Some((name, value)) = line.split_once(':')
                && name.eq_ignore_ascii_case("content-length")
            {
                len = Some(value.trim().parse::<usize>().expect("length"));
            }
        }
        let mut body = vec![0; len.expect("body length")];
        reader.read_exact(&mut body).expect("multipart body");
        let riff = body
            .windows(4)
            .position(|s| s == b"RIFF")
            .expect("WAV upload");
        let wav = &body[riff..];
        let data_bytes = u32::from_le_bytes(wav[40..44].try_into().expect("data size")) as usize;
        assert_eq!(
            data_bytes,
            64_000 * 2,
            "one request includes the one-second gap"
        );
        let pcm = &wav[44..44 + data_bytes];
        assert!(pcm.as_chunks::<2>().0.iter().enumerate().all(|(i, p)| {
            let amplitude = if i < 16_000 {
                0.25
            } else if i < 32_000 {
                0.0
            } else {
                0.75
            };
            let value = i16::from_le_bytes(*p) as f32 / 32767.0;
            (value - amplitude).abs() < 0.002
        }));
        let json = r#"{"segments":[{"start":0.25,"end":0.75,"text":"speech"},{"start":2.25,"end":2.75,"text":"speech"}]}"#;
        write!(socket, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", json.len(), json).expect("response");
    });
    let root = std::env::temp_dir().join(format!("av-transcript-timeline-{}", std::process::id()));
    std::fs::create_dir_all(&root).expect("scratch directory");
    let mut cfg = default_config(&BaseDirs::detect());
    cfg.whisper.api_key = Some("test-key".into());
    cfg.whisper.api_url = format!("http://{addr}/transcribe");
    let output = root.join("transcript.srt");
    let result = transcribe(&cfg, &fixture.media, &output, "", &mut |_| {}).expect("transcription");
    peer.join().expect("API expectations");
    assert_eq!(
        result.decoded_end_seconds, 5.0,
        "three seconds of audio cover through second five"
    );
    let srt = std::fs::read_to_string(&output).expect("SRT");
    assert_eq!(
        srt,
        "1\n00:00:01,250 --> 00:00:01,750\nspeech\n\n2\n00:00:03,250 --> 00:00:03,750\nspeech\n\n"
    );
    std::fs::remove_dir_all(root).expect("scratch cleanup");
}
