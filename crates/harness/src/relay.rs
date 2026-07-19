// SPDX-License-Identifier: Apache-2.0
//! A loopback forward relay that applies the netguard address policy to
//! every connection it carries.
//!
//! Two egress paths resolve hostnames and follow redirects themselves and
//! never re-check the policy after [`crate::netguard::vet`]: the yt-dlp
//! subprocess and the in-process libav demuxer. Pointing both at this relay
//! (`--proxy` for yt-dlp, the `http_proxy` demuxer option for libav) brings
//! their traffic — API calls, manifests, fragments, redirect targets,
//! CONNECT tunnels — under the same per-connection screen the direct-http
//! client already enforces through [`crate::netguard::GuardedResolver`].
//!
//! The relay speaks the minimum a forward proxy needs: `CONNECT host:port`
//! for TLS targets (screen, dial the screened address, splice bytes without
//! inspecting the tunnel) and absolute-form http requests for plain-http
//! clients (screen, rewrite to origin-form, forward). It screens through
//! `crate::netguard::screen`, so the address that passed the check is the
//! address it dials — the hostname never reaches the OS resolver a second
//! time, which would reopen the check-then-use window.
//!
//! The listener is loopback-only and unauthenticated, so any local process
//! may use it. That grants no new authority — the screen applies to every
//! connection, and a local process could dial out directly anyway — but
//! traffic through it egresses with this process's identity, worth knowing
//! on shared machines.

use std::io::{Read, Write};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::netguard::{self, ScreenRefusal};

/// Cap on the request head read before the request line is understood: a
/// legitimate proxy request head is well under this. A client that sends
/// more without a complete head is hostile or broken and is refused.
const MAX_HEAD_BYTES: usize = 16 * 1024;

/// Read timeout while waiting for the request head. Bounds a client that
/// connects and then stalls before stating its intent; widened to
/// [`TUNNEL_IDLE_TIMEOUT`] once a tunnel is established, since long
/// downloads inside it are legitimate.
const HEAD_TIMEOUT: Duration = Duration::from_secs(10);

/// Idle bound per direction on an established tunnel. A client that
/// vanished without FIN/RST (a hard-killed subprocess, a dropped link)
/// used to pin its handler thread and both sockets until process exit;
/// with the bound, a direction that moves no bytes for this long errors
/// out of its `io::copy` and the tunnel closes. Far above every consumer's
/// own liveness bound (yt-dlp's watchdog aborts at 480s of no progress,
/// CDNs idle out well under this), so it only reaps connections nothing is
/// waiting on.
const TUNNEL_IDLE_TIMEOUT: Duration = Duration::from_secs(15 * 60);

/// Connection-establishment bound when dialing a screened upstream.
const DIAL_TIMEOUT: Duration = Duration::from_secs(15);

/// A loopback forward relay applying the netguard address policy to every
/// connection — the single egress point for subprocess and demuxer traffic,
/// whose own connections otherwise never re-check the policy after
/// [`crate::netguard::vet`].
///
/// Bind and start with [`PolicyRelay::start`], hand [`PolicyRelay::url`] to
/// yt-dlp (`--proxy`) or libav (`http_proxy`), and drop it to stop.
pub struct PolicyRelay {
    local: SocketAddr,
    shutdown: Arc<AtomicBool>,
    accept: Option<JoinHandle<()>>,
}

impl PolicyRelay {
    /// Bind `127.0.0.1:0` and start the accept loop on its own thread. Each
    /// accepted connection is handled on a fresh thread, so concurrent
    /// downloads (yt-dlp opens many) never serialize behind one another.
    pub fn start(allow_private: bool) -> std::io::Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let local = listener.local_addr()?;
        let shutdown = Arc::new(AtomicBool::new(false));
        let accept = {
            let shutdown = Arc::clone(&shutdown);
            std::thread::spawn(move || accept_loop(listener, shutdown, allow_private))
        };
        Ok(PolicyRelay {
            local,
            shutdown,
            accept: Some(accept),
        })
    }

    /// Proxy URL for yt-dlp's `--proxy` and libav's `http_proxy` option:
    /// `http://127.0.0.1:<port>`.
    pub fn url(&self) -> String {
        format!("http://127.0.0.1:{}", self.local.port())
    }
}

impl Drop for PolicyRelay {
    fn drop(&mut self) {
        // Stop accepting, then wake the blocked `accept` with a throwaway
        // connection so the loop observes the flag and returns. In-flight
        // tunnels are left to their own EOF — process teardown reaps them.
        self.shutdown.store(true, Ordering::Relaxed);
        let woke = TcpStream::connect_timeout(&self.local, Duration::from_secs(1)).is_ok();
        if let Some(handle) = self.accept.take() {
            // If the wake-up connect failed the loop may never observe the
            // flag; leak the thread rather than hang this drop on it.
            if woke {
                let _ = handle.join();
            }
        }
    }
}

/// Accept until the shutdown flag is raised, handling each connection on its
/// own thread.
fn accept_loop(listener: TcpListener, shutdown: Arc<AtomicBool>, allow_private: bool) {
    for stream in listener.incoming() {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }
        let stream = match stream {
            Ok(stream) => stream,
            // Accept errors (fd exhaustion, aborted handshakes) tend to
            // arrive in storms; back off briefly instead of spinning.
            Err(_) => {
                std::thread::sleep(Duration::from_millis(50));
                continue;
            }
        };
        std::thread::spawn(move || {
            let _ = handle(stream, allow_private);
        });
    }
}

/// One proxied connection: read the request head, then either tunnel a
/// CONNECT or forward an absolute-form http request. Every error path
/// closes the client stream.
fn handle(client: TcpStream, allow_private: bool) -> std::io::Result<()> {
    // Bound both directions until intent is known: the read timeout caps a
    // client that stalls before stating its request, the write timeout caps
    // one that requests and then never drains our error response. Both are
    // lifted once a tunnel is established (see the handlers).
    client.set_read_timeout(Some(HEAD_TIMEOUT))?;
    client.set_write_timeout(Some(HEAD_TIMEOUT))?;
    let head = match read_request_head(&client) {
        Ok(head) => head,
        Err(HeadError::TooLarge) => {
            return respond(&client, "400 Bad Request", "request head too large");
        }
        Err(HeadError::Io) => return Ok(()),
    };
    let text = String::from_utf8_lossy(&head);
    let mut lines = text.split("\r\n");
    let request_line = lines.next().unwrap_or("");
    let mut parts = request_line.split(' ');
    let method = parts.next().unwrap_or("");
    let target = parts.next().unwrap_or("");
    let version = parts.next().unwrap_or("HTTP/1.1");

    if method.eq_ignore_ascii_case("CONNECT") {
        handle_connect(client, target, allow_private)
    } else if target.starts_with("http://") {
        handle_absolute(client, method, target, version, lines, allow_private)
    } else {
        respond(
            &client,
            "400 Bad Request",
            "this proxy serves only CONNECT and absolute-form http requests",
        )
    }
}

/// Tunnel a `CONNECT host:port`: screen the authority, dial the screened
/// address, acknowledge, then splice raw bytes without inspecting them.
fn handle_connect(client: TcpStream, authority: &str, allow_private: bool) -> std::io::Result<()> {
    let (host, port) = match split_authority(authority) {
        Some(hp) => hp,
        None => return respond(&client, "400 Bad Request", "malformed CONNECT authority"),
    };
    let upstream = match dial_screened(&host, port, allow_private) {
        Ok(stream) => stream,
        Err(refusal) => return refuse(&client, &refusal),
    };
    set_tunnel_timeouts(&client, &upstream)?;
    (&client).write_all(b"HTTP/1.1 200 Connection Established\r\n\r\n")?;
    splice(client, upstream);
    Ok(())
}

/// Forward an absolute-form http request: screen the target, dial it,
/// rewrite the request line to origin-form, drop hop-by-hop proxy headers,
/// force `Connection: close`, then splice both directions until EOF.
fn handle_absolute(
    client: TcpStream,
    method: &str,
    target: &str,
    version: &str,
    header_lines: std::str::Split<'_, &str>,
    allow_private: bool,
) -> std::io::Result<()> {
    let uri: ureq::http::Uri = match target.parse() {
        Ok(uri) => uri,
        Err(_) => return respond(&client, "400 Bad Request", "malformed request target"),
    };
    let host = match uri.host() {
        Some(host) => host.to_string(),
        None => return respond(&client, "400 Bad Request", "request target has no host"),
    };
    let port = uri.port_u16().unwrap_or(80);
    let path_and_query = uri
        .path_and_query()
        .map(|p| p.as_str())
        .filter(|p| !p.is_empty())
        .unwrap_or("/");

    let mut upstream = match dial_screened(&host, port, allow_private) {
        Ok(stream) => stream,
        Err(refusal) => return refuse(&client, &refusal),
    };

    // Origin-form request line plus every original header except the
    // hop-by-hop proxy ones, with a forced `Connection: close`.
    let mut rewritten = format!("{method} {path_and_query} {version}\r\n");
    for line in header_lines {
        if line.is_empty() {
            break;
        }
        let name = line.split(':').next().unwrap_or("").trim();
        if name.eq_ignore_ascii_case("proxy-connection") || name.eq_ignore_ascii_case("connection")
        {
            continue;
        }
        rewritten.push_str(line);
        rewritten.push_str("\r\n");
    }
    rewritten.push_str("Connection: close\r\n\r\n");
    upstream.write_all(rewritten.as_bytes())?;

    set_tunnel_timeouts(&client, &upstream)?;
    splice(client, upstream);
    Ok(())
}

/// Screen `host:port` and dial the address that passed. Under
/// `allow_private` the policy screen is skipped — mirroring
/// [`crate::netguard::vet`] under the same knob — but resolution still
/// happens so the relay has an address to reach.
fn dial_screened(host: &str, port: u16, allow_private: bool) -> Result<TcpStream, DialRefusal> {
    let addrs = if allow_private {
        let bare = host.trim_start_matches('[').trim_end_matches(']');
        (bare, port)
            .to_socket_addrs()
            .map(|it| it.collect::<Vec<_>>())
            .map_err(|e| DialRefusal::Unreachable(e.to_string()))?
    } else {
        netguard::screen(host, port).map_err(DialRefusal::from)?
    };
    dial(&addrs).map_err(|e| DialRefusal::Unreachable(e.to_string()))
}

/// A relay-side refusal to dial: a policy verdict or a plain reachability
/// failure. Shapes the status the client sees.
enum DialRefusal {
    /// The address policy banned the target; carries the range name.
    Forbidden { range: &'static str },
    /// Resolution or connection failed.
    Unreachable(String),
}

impl From<ScreenRefusal> for DialRefusal {
    fn from(refusal: ScreenRefusal) -> Self {
        match refusal {
            ScreenRefusal::Forbidden { range, .. } => DialRefusal::Forbidden { range },
            ScreenRefusal::Dns { detail } => DialRefusal::Unreachable(detail),
        }
    }
}

/// Answer a refused dial: `403` for a policy verdict (with the range name so
/// the failure is diagnosable), `502` for an unreachable target.
fn refuse(client: &TcpStream, refusal: &DialRefusal) -> std::io::Result<()> {
    match refusal {
        DialRefusal::Forbidden { range } => respond(
            client,
            "403 Forbidden",
            &format!("refusing to connect: the target resolves into {range}"),
        ),
        DialRefusal::Unreachable(detail) => respond(
            client,
            "502 Bad Gateway",
            &format!("cannot reach the target: {detail}"),
        ),
    }
}

/// Dial the first reachable address in `addrs`, with a connect timeout so a
/// black-holed address cannot pin the handler.
fn dial(addrs: &[SocketAddr]) -> std::io::Result<TcpStream> {
    let mut last = std::io::Error::new(std::io::ErrorKind::NotFound, "no addresses to dial");
    for addr in addrs {
        match TcpStream::connect_timeout(addr, DIAL_TIMEOUT) {
            Ok(stream) => return Ok(stream),
            Err(e) => last = e,
        }
    }
    Err(last)
}

/// Widen both sockets from the head-phase bounds to the tunnel idle bound:
/// long transfers are legitimate, immortal half-open tunnels are not.
fn set_tunnel_timeouts(client: &TcpStream, upstream: &TcpStream) -> std::io::Result<()> {
    for stream in [client, upstream] {
        stream.set_read_timeout(Some(TUNNEL_IDLE_TIMEOUT))?;
        stream.set_write_timeout(Some(TUNNEL_IDLE_TIMEOUT))?;
    }
    Ok(())
}

/// Copy bytes both ways between `a` and `b` until each direction hits EOF
/// or sits idle past [`TUNNEL_IDLE_TIMEOUT`], half-closing the write side
/// as each source drains. The relay never inspects what flows; this is a
/// raw byte pump.
fn splice(a: TcpStream, b: TcpStream) {
    let a_read = a.try_clone().expect("clone client stream");
    let b_write = b.try_clone().expect("clone upstream stream");
    let pump = std::thread::spawn(move || {
        let _ = std::io::copy(&mut &a_read, &mut &b_write);
        let _ = b_write.shutdown(Shutdown::Write);
    });
    let _ = std::io::copy(&mut &b, &mut &a);
    let _ = a.shutdown(Shutdown::Write);
    let _ = pump.join();
}

/// Failure reading the request head: an oversized head is a client verdict
/// worth a 400, an I/O error (EOF, timeout, reset) just ends the connection
/// silently.
enum HeadError {
    TooLarge,
    Io,
}

/// Read up to and including the terminating `\r\n\r\n`, one byte at a time
/// so no tunnel byte past the head is ever consumed (a buffered reader would
/// swallow the client's first TLS bytes). Capped at [`MAX_HEAD_BYTES`].
fn read_request_head(stream: &TcpStream) -> Result<Vec<u8>, HeadError> {
    let mut reader = stream;
    let mut head = Vec::with_capacity(1024);
    let mut byte = [0u8; 1];
    loop {
        match reader.read(&mut byte) {
            Ok(0) => return Err(HeadError::Io),
            Ok(_) => {
                head.push(byte[0]);
                if head.ends_with(b"\r\n\r\n") {
                    return Ok(head);
                }
                if head.len() > MAX_HEAD_BYTES {
                    return Err(HeadError::TooLarge);
                }
            }
            Err(_) => return Err(HeadError::Io),
        }
    }
}

/// Split a `host:port` authority, handling a bracketed IPv6 literal
/// (`[::1]:443`). The port must be present and numeric.
fn split_authority(authority: &str) -> Option<(String, u16)> {
    if let Some(rest) = authority.strip_prefix('[') {
        let (host, after) = rest.split_once(']')?;
        let port = after.strip_prefix(':')?;
        Some((format!("[{host}]"), port.parse().ok()?))
    } else {
        let (host, port) = authority.rsplit_once(':')?;
        Some((host.to_string(), port.parse().ok()?))
    }
}

/// Write a minimal error response and close: status line, a plain-text
/// body, `Connection: close`.
fn respond(stream: &TcpStream, status: &str, body: &str) -> std::io::Result<()> {
    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    (&*stream).write_all(response.as_bytes())?;
    let _ = stream.shutdown(Shutdown::Write);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::BufRead;

    /// A local listener that serves one canned response and records the
    /// request line it received.
    fn canned_upstream(response: &'static [u8]) -> (SocketAddr, std::sync::mpsc::Receiver<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind upstream");
        let addr = listener.local_addr().expect("addr");
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut reader = std::io::BufReader::new(stream.try_clone().expect("clone"));
                let mut request_line = String::new();
                let _ = reader.read_line(&mut request_line);
                // Drain the rest of the head so the client's write completes.
                let mut line = String::new();
                loop {
                    line.clear();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                        break;
                    }
                }
                let _ = stream.write_all(response);
                let _ = stream.shutdown(Shutdown::Write);
                let _ = tx.send(request_line);
            }
        });
        (addr, rx)
    }

    /// Send `request` to the relay and return the raw bytes it answers with.
    fn round_trip(relay: &PolicyRelay, request: &[u8]) -> Vec<u8> {
        let mut client = TcpStream::connect(relay.local).expect("connect relay");
        client.write_all(request).expect("write request");
        let mut response = Vec::new();
        client.read_to_end(&mut response).expect("read response");
        response
    }

    #[test]
    fn refuses_connect_to_a_non_global_target() {
        let relay = PolicyRelay::start(false).expect("relay starts");
        let response = round_trip(&relay, b"CONNECT 127.0.0.1:1 HTTP/1.1\r\n\r\n");
        let text = String::from_utf8_lossy(&response);
        assert!(text.starts_with("HTTP/1.1 403"), "{text}");
        assert!(text.contains("loopback"), "names the banning range: {text}");
    }

    #[test]
    fn refuses_absolute_form_to_a_non_global_target() {
        let relay = PolicyRelay::start(false).expect("relay starts");
        let response = round_trip(
            &relay,
            b"GET http://127.0.0.1:1/x HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n",
        );
        let text = String::from_utf8_lossy(&response);
        assert!(text.starts_with("HTTP/1.1 403"), "{text}");
        assert!(text.contains("loopback"), "{text}");
    }

    #[test]
    fn tunnels_connect_end_to_end() {
        let relay = PolicyRelay::start(true).expect("relay starts");
        let (upstream, _rx) = canned_upstream(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nhi");
        let mut client = TcpStream::connect(relay.local).expect("connect relay");
        client
            .write_all(format!("CONNECT {upstream} HTTP/1.1\r\n\r\n").as_bytes())
            .expect("write connect");
        // The relay's own 200, then the tunneled upstream bytes.
        let mut established = [0u8; 39];
        client
            .read_exact(&mut established)
            .expect("read established");
        assert!(
            String::from_utf8_lossy(&established)
                .starts_with("HTTP/1.1 200 Connection Established"),
            "{:?}",
            String::from_utf8_lossy(&established)
        );
        // Drive a byte through the tunnel: the canned upstream reads a
        // request line before it answers, so without this both ends block
        // (client awaiting the body, upstream awaiting the request).
        client
            .write_all(b"GET / HTTP/1.1\r\n\r\n")
            .expect("write through tunnel");
        let mut tunneled = Vec::new();
        client.read_to_end(&mut tunneled).expect("read tunneled");
        assert!(
            tunneled.ends_with(b"hi"),
            "canned upstream body arrives intact: {:?}",
            String::from_utf8_lossy(&tunneled)
        );
    }

    #[test]
    fn relays_absolute_form_end_to_end() {
        let relay = PolicyRelay::start(true).expect("relay starts");
        let (upstream, rx) = canned_upstream(b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\n\r\nbody");
        let request = format!(
            "GET http://{upstream}/path?q=1 HTTP/1.1\r\nHost: {upstream}\r\nProxy-Connection: keep-alive\r\n\r\n"
        );
        let response = round_trip(&relay, request.as_bytes());
        let text = String::from_utf8_lossy(&response);
        assert!(text.contains("body"), "upstream body relayed back: {text}");
        let request_line = rx
            .recv_timeout(Duration::from_secs(2))
            .expect("upstream saw request");
        assert!(
            request_line.starts_with("GET /path?q=1 HTTP/1.1"),
            "request line rewritten to origin-form: {request_line:?}"
        );
    }

    #[test]
    fn rejects_oversized_request_head() {
        let relay = PolicyRelay::start(false).expect("relay starts");
        let mut client = TcpStream::connect(relay.local).expect("connect relay");
        // A never-terminating head past the cap.
        let mut blob = b"GET http://example.com/".to_vec();
        blob.extend(std::iter::repeat_n(b'a', MAX_HEAD_BYTES + 1024));
        let _ = client.write_all(&blob);
        let mut response = Vec::new();
        let _ = client.read_to_end(&mut response);
        let text = String::from_utf8_lossy(&response);
        assert!(text.starts_with("HTTP/1.1 400"), "{text}");
    }

    #[test]
    fn rejects_unknown_verbs() {
        let relay = PolicyRelay::start(false).expect("relay starts");
        let response = round_trip(&relay, b"PUT /local/path HTTP/1.1\r\nHost: x\r\n\r\n");
        let text = String::from_utf8_lossy(&response);
        assert!(text.starts_with("HTTP/1.1 400"), "{text}");
        assert!(text.contains("CONNECT"), "explains what it serves: {text}");
    }
}
