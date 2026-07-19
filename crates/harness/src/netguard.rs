// SPDX-License-Identifier: Apache-2.0
//! Egress policy: which hosts outbound requests may contact.
//!
//! [`vet`] is the single constructor of [`VettedUrl`], and every network
//! entry point in [`crate::fetch`] demands one — an unchecked string cannot
//! reach the network by construction. The policy refuses hosts that resolve
//! to loopback, private (RFC 1918), link-local (cloud metadata), or any
//! other non-global address, unless `network.allow_private_hosts` is set.
//!
//! The policy is one function — `screen` resolves a host and rejects the
//! whole set if any address is non-global — enforced up front and again per
//! connection everywhere a connection is made:
//! - [`vet`] runs `screen` once, up front, so a refusal reaches the model
//!   with a clear reason before any packet leaves.
//! - [`GuardedResolver`] re-runs the same classification on every address
//!   the direct-http client is about to connect to — redirect targets and
//!   DNS answers that changed since [`vet`] included — closing the
//!   check-then-use window on that path.
//! - [`crate::relay::PolicyRelay`] runs `screen` on every connection the
//!   yt-dlp subprocess and the libav demuxer make, then connects to the
//!   already-screened address. Those two paths resolve hostnames
//!   themselves and never re-check after [`vet`], so their traffic is
//!   routed through the relay (`--proxy` for yt-dlp, the `http_proxy`
//!   demuxer option for libav) to bring them under the same per-connection
//!   check. An attacker who rebinds a passing name to an internal address
//!   between [`vet`] and the connection is refused at the relay, not
//!   trusted from the earlier resolution.
//!
//! `no_proxy` would punch a hole in the relay routing — libav consults the
//! env var at connect time even when a proxy is explicitly configured — so
//! the MCP server scrubs it from its own environment at boot and blanks it
//! in every yt-dlp child's environment.

use std::fmt;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, ToSocketAddrs};

use ureq::http::Uri;
use ureq::unversioned::resolver::{DefaultResolver, ResolvedSocketAddrs, Resolver};
use ureq::unversioned::transport::NextTimeout;

use crate::config::Config;

/// A URL whose host passed the egress policy at construction time.
///
/// The field is private and [`vet`] is the only constructor, so holding one
/// is proof the policy ran. The vetting is point-in-time — DNS may change
/// afterwards — which is why the direct-http path re-screens per connection
/// via [`GuardedResolver`] and download execution re-vets before spawning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VettedUrl(String);

impl VettedUrl {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for VettedUrl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Why a URL was refused; every message is model-presentable and names the
/// way out (fix the URL, flip the config knob, or download manually).
#[derive(Debug)]
pub enum UrlRejection {
    /// Not an absolute http(s) URL with a host.
    NotFetchable { detail: String },
    /// The host resolved to an address outside the permitted ranges.
    ForbiddenAddress {
        host: String,
        ip: IpAddr,
        range: &'static str,
    },
    /// The host did not resolve at all.
    DnsFailure { host: String, detail: String },
}

impl fmt::Display for UrlRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UrlRejection::NotFetchable { detail } => {
                write!(f, "not a fetchable http(s) URL: {detail}")
            }
            UrlRejection::ForbiddenAddress { host, ip, range } => write!(
                f,
                "refusing to contact {host}: it resolves to {ip}, which is in {range}. \
                 Private and internal destinations are blocked; if this is intentional, set \
                 network.allow_private_hosts = true in config.toml, or download the file \
                 manually and pass the local path as source"
            ),
            UrlRejection::DnsFailure { host, detail } => write!(
                f,
                "cannot resolve {host}: {detail} — check the URL, or download the file \
                 manually and pass the local path as source"
            ),
        }
    }
}

impl std::error::Error for UrlRejection {}

/// Why [`screen`] refused a host: a resolvable name whose addresses fail
/// the policy, or a name that would not resolve. [`vet`] and the relay map
/// this to their own model-facing errors.
#[derive(Debug)]
pub(crate) enum ScreenRefusal {
    /// An address in the set is non-global; `range` names the banning range.
    Forbidden { ip: IpAddr, range: &'static str },
    /// The host did not resolve to any address.
    Dns { detail: String },
}

/// Resolve `host:port` and apply the egress policy to every answer,
/// returning the screened addresses on success.
///
/// One forbidden address fails the whole set — a host resolving to both
/// public and private addresses is exactly the rebinding shape, not
/// something to connect to on its good side. Callers connect to one of the
/// returned addresses directly rather than re-resolving, so the address
/// that passed the check is the address that gets used — no check-then-use
/// window. `host` may carry IPv6 brackets; they are stripped for the
/// resolver form.
pub(crate) fn screen(host: &str, port: u16) -> Result<Vec<std::net::SocketAddr>, ScreenRefusal> {
    // `Uri::host` keeps IPv6 brackets; the (host, port) resolver form wants
    // the bare address.
    let bare = host.trim_start_matches('[').trim_end_matches(']');
    let addrs: Vec<std::net::SocketAddr> = (bare, port)
        .to_socket_addrs()
        .map_err(|e| ScreenRefusal::Dns {
            detail: e.to_string(),
        })?
        .collect();
    if addrs.is_empty() {
        return Err(ScreenRefusal::Dns {
            detail: "resolved to no addresses".to_string(),
        });
    }
    for addr in &addrs {
        if let Some(range) = forbidden(addr.ip()) {
            return Err(ScreenRefusal::Forbidden {
                ip: addr.ip(),
                range,
            });
        }
    }
    Ok(addrs)
}

/// Parse, resolve, classify. Shares `screen`'s "one forbidden address
/// fails the whole URL" policy with the per-connection relay, so a model
/// sees the same verdict up front that the relay would enforce later.
///
/// With `network.allow_private_hosts` set, only URL well-formedness is
/// checked; no resolution happens here (the probe resolves anyway).
pub fn vet(source: &str, cfg: &Config) -> Result<VettedUrl, UrlRejection> {
    let not_fetchable = |detail: String| UrlRejection::NotFetchable { detail };
    let uri: Uri = source.parse().map_err(|e| not_fetchable(format!("{e}")))?;
    let default_port = match uri.scheme_str() {
        Some("http") => 80,
        Some("https") => 443,
        other => {
            return Err(not_fetchable(format!(
                "scheme must be http or https, got '{}'",
                other.unwrap_or("none")
            )));
        }
    };
    let host = uri
        .host()
        .ok_or_else(|| not_fetchable("no host".to_string()))?
        .to_string();

    if cfg.network.allow_private_hosts {
        return Ok(VettedUrl(source.to_string()));
    }

    let port = uri.port_u16().unwrap_or(default_port);
    match screen(&host, port) {
        Ok(_) => Ok(VettedUrl(source.to_string())),
        Err(ScreenRefusal::Forbidden { ip, range }) => {
            Err(UrlRejection::ForbiddenAddress { host, ip, range })
        }
        Err(ScreenRefusal::Dns { detail }) => Err(UrlRejection::DnsFailure { host, detail }),
    }
}

// ── address classification ──

/// The range name that bans `ip`, or `None` for global unicast.
fn forbidden(ip: IpAddr) -> Option<&'static str> {
    match ip {
        IpAddr::V4(v4) => forbidden_v4(v4),
        IpAddr::V6(v6) => forbidden_v6(v6),
    }
}

fn forbidden_v4(ip: Ipv4Addr) -> Option<&'static str> {
    let o = ip.octets();
    if ip.is_loopback() {
        return Some("the loopback range");
    }
    if ip.is_private() {
        return Some("a private (RFC 1918) range");
    }
    if ip.is_link_local() {
        return Some("the link-local range (cloud metadata lives here)");
    }
    if o[0] == 0 {
        return Some("the 'this network' range");
    }
    if o[0] == 100 && (o[1] & 0xc0) == 64 {
        return Some("the carrier-grade NAT range");
    }
    if o[0] == 192 && o[1] == 0 && o[2] == 0 {
        return Some("the IETF protocol assignments range");
    }
    if ip.is_documentation() {
        return Some("a documentation range");
    }
    if o[0] == 198 && (o[1] & 0xfe) == 18 {
        return Some("the benchmarking range");
    }
    if ip.is_multicast() {
        return Some("a multicast range");
    }
    if o[0] >= 240 {
        // 240.0.0.0/4 including the broadcast address.
        return Some("a reserved range");
    }
    None
}

/// The IPv4 address packed into two adjacent IPv6 segments.
fn embedded_v4(hi: u16, lo: u16) -> Ipv4Addr {
    Ipv4Addr::new(
        (hi >> 8) as u8,
        (hi & 0xff) as u8,
        (lo >> 8) as u8,
        (lo & 0xff) as u8,
    )
}

fn forbidden_v6(ip: Ipv6Addr) -> Option<&'static str> {
    // Honest reasons for the two addresses `to_ipv4` would misfile
    // (`::` → 0.0.0.0, `::1` → 0.0.0.1).
    if ip.is_loopback() {
        return Some("the loopback range");
    }
    if ip.is_unspecified() {
        return Some("the unspecified address");
    }
    // Embedded-IPv4 forms answer for the embedded address:
    // `::ffff:127.0.0.1` is loopback wearing a coat.
    if let Some(v4) = ip.to_ipv4() {
        return forbidden_v4(v4);
    }
    let s = ip.segments();
    if s[..6] == [0x64, 0xff9b, 0, 0, 0, 0] {
        // NAT64 (64:ff9b::/96): judge the embedded IPv4.
        return forbidden_v4(embedded_v4(s[6], s[7]));
    }
    if s[0] == 0x2002 {
        // 6to4 (2002::/16): bits 16..48 embed the IPv4 the packet is
        // routed to; judge it, as NAT64.
        return forbidden_v4(embedded_v4(s[1], s[2]));
    }
    if s[0] == 0x2001 && s[1] == 0 {
        // Teredo (2001:0::/32): the server IPv4 sits plain in bits 32..64,
        // the client IPv4 bit-inverted in the last 32; either being
        // forbidden bans the address.
        return forbidden_v4(embedded_v4(s[2], s[3]))
            .or_else(|| forbidden_v4(embedded_v4(!s[6], !s[7])));
    }
    if (s[0] & 0xfe00) == 0xfc00 {
        return Some("the unique-local range");
    }
    if (s[0] & 0xffc0) == 0xfe80 {
        return Some("the link-local range");
    }
    if s[0] == 0x2001 && s[1] == 0xdb8 {
        return Some("a documentation range");
    }
    if (s[0] & 0xff00) == 0xff00 {
        return Some("a multicast range");
    }
    None
}

// ── per-connection enforcement for the direct-http client ──

/// [`DefaultResolver`] plus the egress screen on every answer it returns.
///
/// Runs per connection attempt, so redirect targets and names that
/// re-resolved differently since [`vet`] cannot smuggle in a forbidden
/// address. `ureq::unversioned` is semver-exempt — re-check this impl on
/// ureq upgrades.
#[derive(Debug)]
pub struct GuardedResolver {
    inner: DefaultResolver,
    allow_private: bool,
}

impl GuardedResolver {
    pub fn new(cfg: &Config) -> Self {
        GuardedResolver {
            inner: DefaultResolver::default(),
            allow_private: cfg.network.allow_private_hosts,
        }
    }
}

impl Resolver for GuardedResolver {
    fn resolve(
        &self,
        uri: &Uri,
        config: &ureq::config::Config,
        timeout: NextTimeout,
    ) -> Result<ResolvedSocketAddrs, ureq::Error> {
        let addrs = self.inner.resolve(uri, config, timeout)?;
        if self.allow_private {
            return Ok(addrs);
        }
        for addr in addrs.iter() {
            if let Some(range) = forbidden(addr.ip()) {
                return Err(ureq::Error::Io(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    format!(
                        "refusing to connect to {}: {} is in {range} \
                         (network.allow_private_hosts permits private destinations)",
                        uri.host().unwrap_or("<no host>"),
                        addr.ip(),
                    ),
                )));
            }
        }
        Ok(addrs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{default_config, test_dirs};
    use std::path::Path;

    fn cfg() -> Config {
        default_config(&test_dirs(Path::new("/home/u")))
    }

    fn rejection(url: &str) -> UrlRejection {
        vet(url, &cfg()).expect_err("must be refused")
    }

    #[test]
    fn rejects_loopback_literal() {
        let e = rejection("http://127.0.0.1:8080/v.mp4");
        assert!(matches!(e, UrlRejection::ForbiddenAddress { .. }), "{e}");
        assert!(e.to_string().contains("loopback"), "{e}");
    }

    #[test]
    fn rejects_cloud_metadata_endpoint() {
        let e = rejection("http://169.254.169.254/latest/meta-data/");
        assert!(e.to_string().contains("link-local"), "{e}");
    }

    #[test]
    fn rejects_private_range_literal() {
        let e = rejection("https://192.168.1.5/cam.m3u8");
        assert!(e.to_string().contains("RFC 1918"), "{e}");
    }

    #[test]
    fn rejects_v4_mapped_loopback() {
        let e = rejection("http://[::ffff:127.0.0.1]/v.mp4");
        assert!(e.to_string().contains("loopback"), "{e}");
    }

    #[test]
    fn rejects_ipv6_loopback() {
        let e = rejection("http://[::1]/v.mp4");
        assert!(e.to_string().contains("loopback"), "{e}");
    }

    #[test]
    fn rejects_ipv6_unique_local() {
        let e = rejection("http://[fd12:3456::1]/v.mp4");
        assert!(e.to_string().contains("unique-local"), "{e}");
    }

    #[test]
    fn rejects_localhost_by_name() {
        assert!(vet("http://localhost:9000/v.mp4", &cfg()).is_err());
    }

    #[test]
    fn rejects_decimal_ip_literal() {
        // getaddrinfo reads "2130706433" as 127.0.0.1 (inet_aton form);
        // where it does not, the name fails resolution — refused either way.
        assert!(vet("http://2130706433/v.mp4", &cfg()).is_err());
    }

    #[test]
    fn rejects_non_http_scheme() {
        assert!(matches!(
            rejection("file:///etc/passwd"),
            UrlRejection::NotFetchable { .. }
        ));
        assert!(matches!(
            rejection("ftp://ex.com/v.mp4"),
            UrlRejection::NotFetchable { .. }
        ));
    }

    #[test]
    fn allows_public_literal() {
        let v = vet("https://1.1.1.1/v.mp4", &cfg()).expect("public address is permitted");
        assert_eq!(v.as_str(), "https://1.1.1.1/v.mp4");
    }

    #[test]
    fn allows_private_when_configured() {
        let mut cfg = cfg();
        cfg.network.allow_private_hosts = true;
        assert!(vet("http://192.168.1.5/cam.m3u8", &cfg).is_ok());
        assert!(
            vet("file:///etc/passwd", &cfg).is_err(),
            "scheme check survives the knob"
        );
    }

    #[test]
    fn classifies_special_v4_ranges() {
        let banned = [
            "0.0.0.0",
            "100.64.0.1",  // carrier-grade NAT
            "192.0.0.170", // IETF protocol assignments
            "192.0.2.1",   // documentation
            "198.18.0.1",  // benchmarking
            "224.0.0.251", // multicast
            "255.255.255.255",
        ];
        for ip in banned {
            let ip: Ipv4Addr = ip.parse().expect("literal");
            assert!(forbidden_v4(ip).is_some(), "{ip} must be forbidden");
        }
        for ip in ["8.8.8.8", "93.184.216.34"] {
            let ip: Ipv4Addr = ip.parse().expect("literal");
            assert!(forbidden_v4(ip).is_none(), "{ip} must be permitted");
        }
    }

    #[test]
    fn rejects_6to4_embedded_private_v4() {
        // 2002:c0a8:0105:: embeds 192.168.1.5.
        let ip: Ipv6Addr = "2002:c0a8:105::1".parse().expect("literal");
        assert_eq!(forbidden_v6(ip), Some("a private (RFC 1918) range"));
    }

    #[test]
    fn permits_6to4_of_public_v4() {
        // 2002:0808:0808:: embeds 8.8.8.8 — same policy as NAT64.
        let ip: Ipv6Addr = "2002:808:808::1".parse().expect("literal");
        assert_eq!(forbidden_v6(ip), None);
    }

    #[test]
    fn rejects_teredo_embedding_private_client_v4() {
        // Client 192.168.1.5 = c0a8:0105, bit-inverted → 3f57:fefa;
        // server is public 8.8.8.8.
        let ip: Ipv6Addr = "2001:0:808:808::3f57:fefa".parse().expect("literal");
        assert_eq!(forbidden_v6(ip), Some("a private (RFC 1918) range"));
    }

    #[test]
    fn rejects_teredo_embedding_private_server_v4() {
        // Server 10.0.0.1 = 0a00:0001; inverted client is public 8.8.4.4.
        let ip: Ipv6Addr = "2001:0:a00:1::f7f7:fbfb".parse().expect("literal");
        assert_eq!(forbidden_v6(ip), Some("a private (RFC 1918) range"));
    }

    #[test]
    fn permits_teredo_of_public_endpoints() {
        // Server 1.1.1.1, client 8.8.4.4 (inverted f7f7:fbfb).
        let ip: Ipv6Addr = "2001:0:101:101::f7f7:fbfb".parse().expect("literal");
        assert_eq!(forbidden_v6(ip), None);
    }

    #[test]
    fn classifies_special_v6_ranges() {
        let banned = ["::", "fe80::1", "ff02::1", "2001:db8::1", "64:ff9b::7f00:1"];
        for ip in banned {
            let ip: Ipv6Addr = ip.parse().expect("literal");
            assert!(forbidden_v6(ip).is_some(), "{ip} must be forbidden");
        }
        let global: Ipv6Addr = "2606:4700:4700::1111".parse().expect("literal");
        assert!(forbidden_v6(global).is_none());
        let nat64_public: Ipv6Addr = "64:ff9b::808:808".parse().expect("literal");
        assert!(
            forbidden_v6(nat64_public).is_none(),
            "NAT64 of a public address is fine"
        );
    }
}
