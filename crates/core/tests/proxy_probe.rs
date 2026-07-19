// SPDX-License-Identifier: Apache-2.0
//! The `http_proxy` option forwarded through
//! [`media::open_input_interruptible`] must route libav's own connections
//! through the given proxy rather than dialing the target directly. This is
//! the contract the harness relies on to keep the egress policy enforced on
//! every demuxer connection, so it is asserted against the linked libav for
//! both http and https targets.

use std::io::Read;
use std::net::TcpListener;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::time::Duration;

use agent_video_core::media;
use agent_video_core::source::Source;

/// Point libav at a recording proxy and open a `scheme://` target through
/// it; returns the first request line the proxy received, or `None` if no
/// connection arrived (libav dialed the target directly).
fn proxied_request_line(scheme: &str) -> Option<String> {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind recording proxy");
    let addr = listener.local_addr().expect("local addr");
    let (tx, rx) = mpsc::channel();
    let recorder = std::thread::spawn(move || {
        // One connection is enough to prove the route; drop it afterwards so
        // libav's open fails fast instead of negotiating with a dead peer.
        let line = match listener.accept() {
            Ok((mut s, _)) => {
                s.set_read_timeout(Some(Duration::from_secs(2))).ok();
                let mut buf = [0u8; 256];
                let n = s.read(&mut buf).unwrap_or(0);
                String::from_utf8_lossy(&buf[..n])
                    .lines()
                    .next()
                    .unwrap_or("")
                    .to_string()
            }
            Err(_) => String::new(),
        };
        let _ = tx.send(line);
    });

    let proxy = format!("http://127.0.0.1:{}", addr.port());
    // A literal global IP: routing through the proxy means the proxy sees
    // the request; dialing directly would never touch the recorder.
    let source = Source::classify(&format!("{scheme}://93.184.216.34/video.mp4"));
    let stop = Arc::new(AtomicBool::new(false));
    let disarm = Arc::clone(&stop);
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs(4));
        disarm.store(true, Ordering::Relaxed);
    });
    let _ = media::open_input_interruptible(&source, stop, &[("http_proxy", &proxy)]);
    let seen = rx.recv_timeout(Duration::from_secs(6)).ok();
    recorder.join().ok();
    seen.filter(|s| !s.is_empty())
}

#[test]
fn routes_http_targets_through_the_proxy_option() {
    let line = proxied_request_line("http").expect("the proxy must see the http request");
    assert!(
        line.starts_with("GET http://93.184.216.34"),
        "http goes through as an absolute-form GET: {line}"
    );
}

#[test]
fn routes_https_targets_through_the_proxy_option() {
    let line = proxied_request_line("https").expect("the proxy must see the https CONNECT");
    assert!(
        line.starts_with("CONNECT 93.184.216.34:443"),
        "https tunnels through as CONNECT: {line}"
    );
}
