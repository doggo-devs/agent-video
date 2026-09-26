// SPDX-License-Identifier: Apache-2.0
//! The canonical cookie jar and its concurrency protocol.
//!
//! One Netscape-format jar under the config dir holds cookies for every
//! site. yt-dlp reads it and rotates cookies back on exit but takes no
//! locks, so every mutable copy gets exactly one owner:
//!
//! 1. [`JarLease::checkout`] copies the canonical jar under an exclusive
//!    lock into a private staging dir: a pristine *snapshot* plus a
//!    *staged* copy handed to yt-dlp.
//! 2. yt-dlp reads and rewrites only the staged copy.
//! 3. [`JarLease::commit`] applies the delta staged − snapshot to the
//!    canonical jar under the lock, so concurrent runs' rotations of other
//!    cookies survive.
//!
//! Deletions never propagate: a killed child's missing or truncated staged
//! file is an empty or additive delta, never a destructive one. A user's
//! [`clear`] does propagate: it is logged, and a lease checked out before
//! it drops the cleared scope from its delta.
//!
//! A canonical jar that does not exist is empty; one that exists but
//! fails to read aborts the operation before any rewrite. Only the yt-dlp
//! source consumes cookies.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::config::CookiesCfg;

/// How cookies reach yt-dlp for the current operation, resolved from
/// config and jar presence at call time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CookieMode {
    /// No cookies configured; downloads are anonymous.
    Anonymous,
    /// `--cookies-from-browser`: read live from the browser store.
    Browser(String),
    /// The canonical jar exists; lease a copy per invocation.
    Jar(PathBuf),
}

/// Resolve the active mode: explicit browser config wins, then an existing
/// jar, then anonymous. Jar presence is re-checked per call, so an import
/// takes effect without restarting the server.
pub fn mode(cfg: &CookiesCfg) -> CookieMode {
    if let Some(browser) = &cfg.from_browser {
        return CookieMode::Browser(browser.clone());
    }
    if cfg.jar.is_file() {
        return CookieMode::Jar(cfg.jar.clone());
    }
    CookieMode::Anonymous
}

/// Cookie forwarding for one yt-dlp invocation: acquired before spawn,
/// committed after the child settles (any verdict — a gracefully aborted
/// child saved its staged jar too; an unwritten one is an empty delta).
#[derive(Debug)]
pub enum CookieHandle {
    Anonymous,
    Browser(String),
    Jar(JarLease),
}

impl CookieHandle {
    /// Acquire the configured cookie source. A jar that cannot be staged
    /// is an error — silently degrading to anonymous would turn into a
    /// confusing "login required" failure downstream.
    pub fn acquire(cfg: &CookiesCfg) -> io::Result<CookieHandle> {
        match mode(cfg) {
            CookieMode::Anonymous => Ok(CookieHandle::Anonymous),
            CookieMode::Browser(b) => Ok(CookieHandle::Browser(b)),
            CookieMode::Jar(jar) => Ok(CookieHandle::Jar(JarLease::checkout(&jar)?)),
        }
    }

    /// The yt-dlp arguments carrying this handle's cookies.
    pub fn args(&self) -> Vec<String> {
        match self {
            CookieHandle::Anonymous => Vec::new(),
            CookieHandle::Browser(b) => {
                vec!["--cookies-from-browser".to_string(), b.clone()]
            }
            CookieHandle::Jar(lease) => vec![
                "--cookies".to_string(),
                lease.staged().to_string_lossy().into_owned(),
            ],
        }
    }

    /// Adopt rotated cookies back into the canonical jar (jar mode only).
    pub fn commit(self) -> io::Result<()> {
        match self {
            CookieHandle::Anonymous | CookieHandle::Browser(_) => Ok(()),
            CookieHandle::Jar(lease) => lease.commit(),
        }
    }
}

/// Single-owner lease on the canonical jar: a private staged copy for
/// yt-dlp plus a pristine snapshot for the commit-time delta. The staging
/// dir is removed on drop; commit consumes the lease.
#[derive(Debug)]
pub struct JarLease {
    staging: PathBuf,
    canonical: PathBuf,
    /// The last [`clear`] recorded at checkout; later clears bind the
    /// commit.
    epoch: u64,
}

/// Writer-unique `pid-seq` token for lease staging dirs, mirroring the
/// grammar of `cache::naming::writer_id` (kept private there): a pid
/// alone — or pid plus a timestamp, which Windows quantizes coarsely —
/// would let two concurrent leases of one process share a directory,
/// and the first lease's Drop would then delete it under the other's
/// live child.
fn writer_id() -> String {
    static WRITER_SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = WRITER_SEQ.fetch_add(1, Ordering::Relaxed);
    format!("{}-{seq}", std::process::id())
}

impl JarLease {
    /// Copy the canonical jar into a fresh staging dir under the lock.
    /// Fails when the jar is missing — resolve the mode first.
    pub fn checkout(canonical: &Path) -> io::Result<JarLease> {
        let parent = canonical
            .parent()
            .ok_or_else(|| io::Error::other("cookie jar path has no parent directory"))?;
        let staging = parent.join(format!(".cookie-lease-{}", writer_id()));
        fs::create_dir_all(&staging)?;
        let mut lease = JarLease {
            staging,
            canonical: canonical.to_path_buf(),
            epoch: 0,
        };
        {
            let _lock = JarLock::exclusive(canonical)?;
            // fs::copy clones the source mode; a user-supplied canonical
            // jar may be 0644, so these credential copies are created born
            // 0600 instead — no world-readable window.
            copy_private(canonical, &lease.staged())?;
            copy_private(canonical, &lease.snapshot())?;
            lease.epoch = ClearLog::read(canonical)?.epoch();
        }
        Ok(lease)
    }

    /// The private jar copy to hand yt-dlp via `--cookies`.
    pub fn staged(&self) -> PathBuf {
        self.staging.join("jar.txt")
    }

    fn snapshot(&self) -> PathBuf {
        self.staging.join("snapshot.txt")
    }

    /// Apply this run's delta (staged − snapshot) to the canonical jar
    /// under the lock. Cookies untouched by this run — including ones a
    /// concurrent instance rotated meanwhile — are left as the canonical
    /// jar has them. Cookies in a scope [`clear`]ed after checkout are
    /// dropped: the run's copy predates the user's revocation.
    pub fn commit(self) -> io::Result<()> {
        // A killed child may have left the staged file unwritten; the
        // snapshot then equals it and the delta is empty.
        let staged = fs::read_to_string(self.staged()).unwrap_or_default();
        let snapshot = fs::read_to_string(self.snapshot()).unwrap_or_default();
        let snapshot = parse_lenient(&snapshot);
        let delta: Vec<(EntryKey, String)> = parse_lenient(&staged)
            .into_iter()
            .filter(|(key, line)| snapshot.get(key) != Some(line))
            .collect();
        if delta.is_empty() {
            return Ok(());
        }
        let _lock = JarLock::exclusive(&self.canonical)?;
        let clears = ClearLog::read(&self.canonical)?;
        let delta: Vec<(EntryKey, String)> = delta
            .into_iter()
            .filter(|((domain, _, _), _)| !clears.revokes_since(self.epoch, domain))
            .collect();
        if delta.is_empty() {
            return Ok(());
        }
        let mut canonical = read_jar_if_present(&self.canonical)?;
        canonical.extend(delta);
        write_jar(&self.canonical, &canonical)
    }
}

impl Drop for JarLease {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.staging);
    }
}

/// Every [`clear`] still able to bind a lease: `(epoch, scope)` in epoch
/// order, where scope `None` is the whole jar. Kept in a sibling file
/// because a clear-all removes the jar itself. A clear-all subsumes every
/// earlier record, so the log compacts to it.
struct ClearLog(Vec<(u64, Option<String>)>);

impl ClearLog {
    fn path(canonical: &Path) -> PathBuf {
        canonical.with_extension("cleared")
    }

    /// Read the log; a missing one is empty. Callers hold the lock.
    fn read(canonical: &Path) -> io::Result<ClearLog> {
        let text = match fs::read_to_string(Self::path(canonical)) {
            Ok(text) => text,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(ClearLog(Vec::new())),
            Err(e) => return Err(e),
        };
        let records = text
            .lines()
            .map(|line| {
                let malformed =
                    || io::Error::other(format!("malformed cookie clear log: {line:?}"));
                let (epoch, scope) = line.split_once('\t').ok_or_else(malformed)?;
                let epoch = epoch.parse().map_err(|_| malformed())?;
                Ok((epoch, (!scope.is_empty()).then(|| scope.to_string())))
            })
            .collect::<io::Result<_>>()?;
        Ok(ClearLog(records))
    }

    fn epoch(&self) -> u64 {
        self.0.last().map_or(0, |&(epoch, _)| epoch)
    }

    /// Whether a clear after `epoch` covers cookies of `domain`.
    fn revokes_since(&self, epoch: u64, domain: &str) -> bool {
        self.0
            .iter()
            .any(|(e, scope)| *e > epoch && scope.as_deref().is_none_or(|s| covers(s, domain)))
    }

    /// Append a clear of `scope` and persist the log. Callers hold the lock.
    fn record(mut self, canonical: &Path, scope: Option<&str>) -> io::Result<()> {
        let epoch = self.epoch() + 1;
        if scope.is_none() {
            self.0.clear();
        }
        self.0.push((epoch, scope.map(str::to_string)));
        let text: String = self
            .0
            .iter()
            .map(|(e, s)| format!("{e}\t{}\n", s.as_deref().unwrap_or("")))
            .collect();
        let path = Self::path(canonical);
        let tmp = path.with_extension(format!("cleared-tmp-{}", std::process::id()));
        fs::write(&tmp, text)?;
        fs::rename(&tmp, &path)
    }
}

/// Whether clearing `scope` removes a cookie of `domain`: the domain
/// itself and its subdomains, leading dot ignored.
fn covers(scope: &str, domain: &str) -> bool {
    let d = domain.trim_start_matches('.');
    d == scope || d.ends_with(&format!(".{scope}"))
}

// ── jar format ──

/// Merge identity of one cookie: yt-dlp's write-back and a concurrent
/// import agree on these three fields naming "the same cookie".
type EntryKey = (String, String, String); // (domain, path, name)

/// Entries keyed for merging; each value is the verbatim jar line, so
/// flags this module does not model (secure, httponly) survive untouched.
type Entries = BTreeMap<EntryKey, String>;

/// curl and yt-dlp mark HttpOnly cookies with this prefix on the domain
/// field; such lines look like comments but are entries.
const HTTPONLY_PREFIX: &str = "#HttpOnly_";

/// Parse one jar line into its merge key; `None` for comments, blank
/// lines, and anything that is not a 7-field cookie line.
fn parse_line(line: &str) -> Option<EntryKey> {
    let body = line.strip_prefix(HTTPONLY_PREFIX).unwrap_or(line);
    if body.starts_with('#') || body.trim().is_empty() {
        return None;
    }
    let fields: Vec<&str> = body.splitn(7, '\t').collect();
    let [domain, _, path, _, _, name, _] = fields.as_slice() else {
        return None;
    };
    Some((domain.to_string(), path.to_string(), name.to_string()))
}

/// Read and leniently parse a jar that may not exist yet. NotFound is an
/// empty jar; any other error propagates — every caller follows a read
/// with a rewrite, and mistaking an unreadable jar for an empty one would
/// destroy every cookie the read missed.
fn read_jar_if_present(path: &Path) -> io::Result<Entries> {
    match fs::read_to_string(path) {
        Ok(text) => Ok(parse_lenient(&text)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Entries::new()),
        Err(e) => Err(e),
    }
}

/// Parse a jar, skipping anything unrecognizable: inputs here are
/// machine-written (yt-dlp, this module), and a truncated trailing line
/// from a killed writer must not poison the rest.
fn parse_lenient(text: &str) -> Entries {
    text.lines()
        .filter_map(|line| Some((parse_line(line)?, line.to_string())))
        .collect()
}

/// Parse a user-supplied jar; the first malformed non-comment line is an
/// error naming itself, so importing a JSON export fails loudly instead
/// of importing nothing.
fn parse_strict(text: &str) -> Result<Entries, String> {
    let mut entries = Entries::new();
    for line in text.lines() {
        let body = line.strip_prefix(HTTPONLY_PREFIX).unwrap_or(line);
        if body.starts_with('#') || body.trim().is_empty() {
            continue;
        }
        let Some(key) = parse_line(line) else {
            return Err(format!(
                "not a Netscape cookie line (expected 7 tab-separated fields): {line:?}"
            ));
        };
        entries.insert(key, line.to_string());
    }
    Ok(entries)
}

/// Write `entries` as a jar: temp file + rename (atomic against the
/// crash-torn write), owner-only permissions. Callers hold the lock.
fn write_jar(path: &Path, entries: &Entries) -> io::Result<()> {
    let mut text = String::from("# Netscape HTTP Cookie File\n");
    for line in entries.values() {
        text.push_str(line);
        text.push('\n');
    }
    let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
    let mut file = create_private(&tmp)?;
    file.write_all(text.as_bytes())?;
    // Close before rename: Windows refuses to rename a file with an open
    // handle, and the release matrix ships win32-x64.
    drop(file);
    fs::rename(&tmp, path)
}

/// Create `path` for writing with owner-only permissions from its first
/// byte. A credential file must never exist — not even momentarily —
/// wider than 0600: `fs::write` at the umask default lands at 0644
/// (group/world-readable) until a later chmod, a window an attacker can
/// read through. On Unix the mode is set atomically at `open`; a stale
/// file from a crashed run is removed first so `create_new` cannot reuse
/// its loose permissions. Off Unix, [`restrict_to_owner`] is a
/// best-effort fallback.
fn create_private(path: &Path) -> io::Result<File> {
    let _ = fs::remove_file(path);
    let mut opts = fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let file = opts.open(path)?;
    restrict_to_owner(path)?;
    Ok(file)
}

/// Copy `src` to `dst` with owner-only permissions from creation.
/// [`fs::copy`] clones the source's mode (a user-supplied jar may be
/// 0644), so it cannot bear a credential copy without a world-readable
/// window; this streams the bytes into a file born 0600. Returns the
/// number of bytes copied.
fn copy_private(src: &Path, dst: &Path) -> io::Result<u64> {
    let mut reader = File::open(src)?;
    let mut writer = create_private(dst)?;
    io::copy(&mut reader, &mut writer)
}

/// Owner-only permissions for credential files. On Unix files are born
/// 0600 via [`create_private`], so this only re-affirms the mode; off
/// Unix it is the sole (best-effort, currently no-op) protection.
fn restrict_to_owner(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

/// Exclusive lock over the canonical jar, held via a sibling `.lock` file
/// that is never replaced — locking the jar itself would race the
/// rename-replace in [`write_jar`]. Released on drop (close unlocks).
struct JarLock {
    _file: File,
}

impl JarLock {
    fn exclusive(canonical: &Path) -> io::Result<JarLock> {
        let path = canonical.with_extension("lock");
        let file = File::options().create(true).append(true).open(&path)?;
        file.lock()?;
        Ok(JarLock { _file: file })
    }
}

// ── CLI operations ──

/// A failed jar operation; the message is user-presentable.
#[derive(Debug)]
pub struct CookieError(pub String);

impl CookieError {
    /// Exit code for CLI paths. Core owns decades 10–40 and
    /// [`crate::install::InstallError`] claims 50; jar operations claim 51
    /// — a malformed import or a locked-out jar is an expected runtime
    /// condition, not the uncoded-bug exit 1.
    pub fn exit_code(&self) -> u8 {
        51
    }
}

impl std::fmt::Display for CookieError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cookies: {}", self.0)
    }
}

impl std::error::Error for CookieError {}

impl From<io::Error> for CookieError {
    fn from(e: io::Error) -> Self {
        CookieError(e.to_string())
    }
}

/// Outcome of an import: how many cookies arrived and the jar's new total.
#[derive(Debug, PartialEq, Eq)]
pub struct ImportReport {
    pub imported: usize,
    pub total: usize,
}

/// Merge a user-exported Netscape jar into the canonical jar; imported
/// entries win per cookie. Creates the jar (and its directory) on first
/// import.
pub fn import(cfg: &CookiesCfg, file: &Path) -> Result<ImportReport, CookieError> {
    let text = fs::read_to_string(file)
        .map_err(|e| CookieError(format!("cannot read {}: {e}", file.display())))?;
    let incoming =
        parse_strict(&text).map_err(|e| CookieError(format!("{}: {e}", file.display())))?;
    if incoming.is_empty() {
        return Err(CookieError(format!(
            "{} contains no cookies",
            file.display()
        )));
    }
    if let Some(parent) = cfg.jar.parent() {
        fs::create_dir_all(parent)?;
    }
    let _lock = JarLock::exclusive(&cfg.jar)?;
    let mut entries = read_jar_if_present(&cfg.jar)?;
    let imported = incoming.len();
    entries.extend(incoming);
    write_jar(&cfg.jar, &entries)?;
    Ok(ImportReport {
        imported,
        total: entries.len(),
    })
}

/// The jar's contents grouped for display: `(domain, cookie count)`,
/// domain-sorted, with leading dots stripped.
pub fn status(cfg: &CookiesCfg) -> Result<Vec<(String, usize)>, CookieError> {
    let mut by_domain: BTreeMap<String, usize> = BTreeMap::new();
    for (domain, _, _) in read_jar_if_present(&cfg.jar)?.into_keys() {
        let display = domain.trim_start_matches('.').to_string();
        *by_domain.entry(display).or_insert(0) += 1;
    }
    Ok(by_domain.into_iter().collect())
}

/// Remove all cookies, or only one domain's (matching the domain and its
/// subdomains); returns how many entries were removed. Leases checked out
/// before the clear cannot write the cleared scope back.
pub fn clear(cfg: &CookiesCfg, domain: Option<&str>) -> Result<usize, CookieError> {
    if domain.is_some_and(|domain| domain.trim().is_empty() || domain.chars().any(char::is_control))
    {
        return Err(CookieError(
            "cookie domain must be nonempty and contain no control characters".into(),
        ));
    }
    if !cfg.jar.is_file() {
        // Nothing to clear, and no lease can be outstanding: checkout
        // needs the jar. Also keeps the lock from creating the config dir.
        return Ok(0);
    }
    let _lock = JarLock::exclusive(&cfg.jar)?;
    let mut entries = read_jar_if_present(&cfg.jar)?;
    let clears = ClearLog::read(&cfg.jar)?;
    let Some(domain) = domain else {
        clears.record(&cfg.jar, None)?;
        fs::remove_file(&cfg.jar)?;
        return Ok(entries.len());
    };
    let before = entries.len();
    entries.retain(|(entry_domain, _, _), _| !covers(domain, entry_domain));
    let removed = before - entries.len();
    clears.record(&cfg.jar, Some(domain))?;
    if removed > 0 {
        write_jar(&cfg.jar, &entries)?;
    }
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_cfg(tag: &str) -> (CookiesCfg, crate::tempdir::TempDir) {
        let dir = crate::tempdir::TempDir::new(&format!("av-cookies-{tag}"));
        let cfg = CookiesCfg {
            from_browser: None,
            jar: dir.join("cookies.txt"),
        };
        (cfg, dir)
    }

    fn line(domain: &str, name: &str, value: &str) -> String {
        format!("{domain}\tTRUE\t/\tFALSE\t0\t{name}\t{value}")
    }

    #[test]
    fn mode_prefers_browser_over_jar() {
        let (mut cfg, _dir) = temp_cfg("mode");
        fs::write(&cfg.jar, line(".a.com", "n", "v")).expect("write");
        assert_eq!(mode(&cfg), CookieMode::Jar(cfg.jar.clone()));
        cfg.from_browser = Some("firefox".to_string());
        assert_eq!(mode(&cfg), CookieMode::Browser("firefox".to_string()));
    }

    #[test]
    fn mode_is_anonymous_without_jar_or_browser() {
        let (cfg, _dir) = temp_cfg("anon");
        assert_eq!(mode(&cfg), CookieMode::Anonymous);
        assert!(matches!(
            CookieHandle::acquire(&cfg).expect("acquire"),
            CookieHandle::Anonymous
        ));
    }

    #[test]
    fn parses_httponly_prefixed_lines_as_entries() {
        let text = format!(
            "# a comment\n\n{}\n{HTTPONLY_PREFIX}{}\n",
            line(".a.com", "plain", "1"),
            line(".a.com", "hidden", "2")
        );
        let entries = parse_lenient(&text);
        assert_eq!(entries.len(), 2, "HttpOnly lines are cookies, not comments");
    }

    #[test]
    fn commit_applies_only_this_runs_delta() {
        // Instance A leases the jar; meanwhile instance B rotates a
        // youtube cookie in the canonical jar. A's run rotates a bilibili
        // cookie in its staged copy. A's commit must adopt the bilibili
        // rotation without clobbering B's youtube rotation.
        let (cfg, _dir) = temp_cfg("delta");
        let youtube = line(".youtube.com", "SID", "old");
        let bilibili = line(".bilibili.com", "SESSDATA", "old");
        fs::write(&cfg.jar, format!("{youtube}\n{bilibili}\n")).expect("write");

        let lease = JarLease::checkout(&cfg.jar).expect("checkout");
        // B's concurrent rotation, committed directly to canonical.
        fs::write(
            &cfg.jar,
            format!("{}\n{bilibili}\n", line(".youtube.com", "SID", "b-new")),
        )
        .expect("write");
        // A's run rotates bilibili in the staged copy.
        fs::write(
            lease.staged(),
            format!(
                "{youtube}\n{}\n",
                line(".bilibili.com", "SESSDATA", "a-new")
            ),
        )
        .expect("write");
        lease.commit().expect("commit");

        let final_jar = fs::read_to_string(&cfg.jar).expect("read");
        assert!(final_jar.contains("b-new"), "B's rotation survives");
        assert!(final_jar.contains("a-new"), "A's rotation lands");
        assert!(!final_jar.contains("\told\n"), "no stale value remains");
    }

    #[test]
    fn commit_with_unwritten_staged_jar_is_a_noop() {
        // A SIGKILLed child never wrote the staged file: the canonical
        // jar must come through byte-identical.
        let (cfg, _dir) = temp_cfg("noop");
        fs::write(&cfg.jar, format!("{}\n", line(".a.com", "n", "v"))).expect("write");
        let before = fs::read_to_string(&cfg.jar).expect("read");

        let lease = JarLease::checkout(&cfg.jar).expect("checkout");
        lease.commit().expect("commit");

        assert_eq!(fs::read_to_string(&cfg.jar).expect("read"), before);
    }

    #[test]
    fn commit_never_propagates_deletions() {
        // A truncated staged file (killed writer) must not delete cookies
        // from the canonical jar.
        let (cfg, _dir) = temp_cfg("keep");
        fs::write(
            &cfg.jar,
            format!(
                "{}\n{}\n",
                line(".a.com", "n", "v"),
                line(".b.com", "m", "w")
            ),
        )
        .expect("write");

        let lease = JarLease::checkout(&cfg.jar).expect("checkout");
        fs::write(lease.staged(), format!("{}\n", line(".a.com", "n", "v"))).expect("truncate");
        lease.commit().expect("commit");

        let final_jar = fs::read_to_string(&cfg.jar).expect("read");
        assert!(final_jar.contains(".b.com"), "absent ≠ deleted");
    }

    #[cfg(unix)]
    #[test]
    fn commit_rejects_unreadable_canonical_jar() {
        // A transient read failure (EIO, permission flip) on the canonical
        // jar must abort the commit, not masquerade as an empty jar that
        // the delta then wholesale replaces.
        use std::os::unix::fs::PermissionsExt;
        let (cfg, _dir) = temp_cfg("eio-commit");
        let before = format!(
            "{}\n{}\n",
            line(".a.com", "n", "v"),
            line(".b.com", "m", "w")
        );
        fs::write(&cfg.jar, &before).expect("write");

        let lease = JarLease::checkout(&cfg.jar).expect("checkout");
        fs::write(lease.staged(), format!("{}\n", line(".c.com", "k", "x"))).expect("write staged");
        fs::set_permissions(&cfg.jar, fs::Permissions::from_mode(0o000)).expect("chmod");
        if fs::read_to_string(&cfg.jar).is_ok() {
            return; // privileged (e.g. root): unreadable cannot be simulated
        }

        lease
            .commit()
            .expect_err("unreadable canonical jar must abort the commit");

        fs::set_permissions(&cfg.jar, fs::Permissions::from_mode(0o600)).expect("chmod back");
        assert_eq!(
            fs::read_to_string(&cfg.jar).expect("read"),
            before,
            "a failed commit leaves the canonical jar untouched"
        );
    }

    #[test]
    fn commit_after_clear_all_leaves_the_jar_cleared() {
        let (cfg, _dir) = temp_cfg("gone");
        fs::write(&cfg.jar, format!("{}\n", line(".a.com", "n", "v"))).expect("write");
        let lease = JarLease::checkout(&cfg.jar).expect("checkout");
        fs::write(
            lease.staged(),
            format!("{}\n", line(".a.com", "n", "rotated")),
        )
        .expect("write staged");

        clear(&cfg, None).expect("clear");
        lease.commit().expect("commit");

        assert!(
            !cfg.jar.exists(),
            "a pre-clear lease must not revive the jar"
        );
        assert_eq!(mode(&cfg), CookieMode::Anonymous);
    }

    #[test]
    fn commit_after_domain_clear_drops_only_that_domain() {
        let (cfg, _dir) = temp_cfg("gone-domain");
        fs::write(
            &cfg.jar,
            format!(
                "{}\n{}\n",
                line(".a.com", "n", "v"),
                line(".b.org", "m", "w")
            ),
        )
        .expect("write");
        let lease = JarLease::checkout(&cfg.jar).expect("checkout");
        fs::write(
            lease.staged(),
            format!(
                "{}\n{}\n",
                line("sub.a.com", "n", "rotated"),
                line(".b.org", "m", "rotated")
            ),
        )
        .expect("write staged");

        clear(&cfg, Some("a.com")).expect("clear");
        lease.commit().expect("commit");

        let jar = fs::read_to_string(&cfg.jar).expect("read");
        assert!(
            !jar.contains("a.com"),
            "cleared domain stays cleared: {jar}"
        );
        assert!(jar.contains(&line(".b.org", "m", "rotated")), "{jar}");
    }

    #[test]
    fn a_lease_after_a_clear_commits_normally() {
        let (cfg, _dir) = temp_cfg("post-clear");
        fs::write(&cfg.jar, format!("{}\n", line(".a.com", "n", "v"))).expect("write");
        clear(&cfg, Some("a.com")).expect("clear");
        let lease = JarLease::checkout(&cfg.jar).expect("checkout");
        fs::write(
            lease.staged(),
            format!("{}\n", line(".a.com", "n", "fresh")),
        )
        .expect("write staged");

        lease.commit().expect("commit");

        let jar = fs::read_to_string(&cfg.jar).expect("read");
        assert!(
            jar.contains("fresh"),
            "only earlier leases are bound: {jar}"
        );
    }

    #[test]
    fn concurrent_leases_never_share_a_staging_dir() {
        // A shared dir would let the first lease's Drop delete it under
        // the other's live child.
        let (cfg, _dir) = temp_cfg("unique");
        fs::write(&cfg.jar, line(".a.com", "n", "v")).expect("write");

        let leases: Vec<JarLease> = (0..8)
            .map(|_| JarLease::checkout(&cfg.jar).expect("checkout"))
            .collect();

        let distinct: std::collections::BTreeSet<&Path> =
            leases.iter().map(|l| l.staging.as_path()).collect();
        assert_eq!(
            distinct.len(),
            leases.len(),
            "every lease owns a private staging dir"
        );
    }

    #[test]
    fn dropping_a_lease_removes_its_staging_dir() {
        let (cfg, _dir) = temp_cfg("drop");
        fs::write(&cfg.jar, line(".a.com", "n", "v")).expect("write");
        let lease = JarLease::checkout(&cfg.jar).expect("checkout");
        let staging = lease.staging.clone();
        assert!(staging.is_dir());
        drop(lease);
        assert!(!staging.exists(), "credential copies must not linger");
    }

    #[test]
    fn import_rejects_a_non_jar_file() {
        let (cfg, _dir) = temp_cfg("badimport");
        let bogus = cfg.jar.with_file_name("export.json");
        fs::write(&bogus, "{\"cookies\":[]}").expect("write");
        let err = import(&cfg, &bogus).expect_err("json is not a jar");
        assert!(err.0.contains("Netscape"), "explains the format: {err}");
        assert!(!cfg.jar.exists(), "a failed import must not create a jar");
    }

    #[cfg(unix)]
    #[test]
    fn import_rejects_unreadable_existing_jar() {
        // An existing-but-unreadable jar is not a first-time empty jar;
        // importing over it would replace every other site's cookies.
        use std::os::unix::fs::PermissionsExt;
        let (cfg, _dir) = temp_cfg("eio-import");
        let before = format!("{}\n", line(".a.com", "n", "v"));
        fs::write(&cfg.jar, &before).expect("write");
        let incoming = cfg.jar.with_file_name("export.txt");
        fs::write(&incoming, format!("{}\n", line(".b.com", "m", "w"))).expect("write");
        fs::set_permissions(&cfg.jar, fs::Permissions::from_mode(0o000)).expect("chmod");
        if fs::read_to_string(&cfg.jar).is_ok() {
            return; // privileged (e.g. root): unreadable cannot be simulated
        }

        import(&cfg, &incoming).expect_err("unreadable existing jar must abort the import");

        fs::set_permissions(&cfg.jar, fs::Permissions::from_mode(0o600)).expect("chmod back");
        assert_eq!(
            fs::read_to_string(&cfg.jar).expect("read"),
            before,
            "a failed import leaves the jar untouched"
        );
    }

    #[test]
    fn import_creates_jar_when_absent() {
        // NotFound is a genuinely empty jar: first import creates it.
        let (cfg, _dir) = temp_cfg("first");
        let incoming = cfg.jar.with_file_name("export.txt");
        fs::write(&incoming, format!("{}\n", line(".a.com", "n", "v"))).expect("write");

        let report = import(&cfg, &incoming).expect("first import");

        assert_eq!(
            report,
            ImportReport {
                imported: 1,
                total: 1
            }
        );
        assert!(cfg.jar.is_file());
    }

    #[test]
    fn import_merges_with_incoming_winning() {
        let (cfg, _dir) = temp_cfg("import");
        fs::write(
            &cfg.jar,
            format!(
                "{}\n{}\n",
                line(".a.com", "n", "old"),
                line(".b.com", "m", "w")
            ),
        )
        .expect("write");
        let incoming = cfg.jar.with_file_name("export.txt");
        fs::write(&incoming, format!("{}\n", line(".a.com", "n", "new"))).expect("write");

        let report = import(&cfg, &incoming).expect("import");

        assert_eq!(
            report,
            ImportReport {
                imported: 1,
                total: 2
            }
        );
        let jar = fs::read_to_string(&cfg.jar).expect("read");
        assert!(jar.contains("new"), "incoming wins per cookie");
        assert!(jar.contains(".b.com"), "unrelated cookies survive");
    }

    #[test]
    fn status_groups_by_domain() {
        let (cfg, _dir) = temp_cfg("status");
        assert_eq!(status(&cfg).expect("no jar is empty"), Vec::new());
        fs::write(
            &cfg.jar,
            format!(
                "{}\n{}\n{}\n",
                line(".a.com", "x", "1"),
                line(".a.com", "y", "2"),
                line("b.org", "z", "3")
            ),
        )
        .expect("write");
        assert_eq!(
            status(&cfg).expect("status"),
            vec![("a.com".to_string(), 2), ("b.org".to_string(), 1)]
        );
    }

    #[test]
    fn clear_domain_removes_it_and_its_subdomains_only() {
        let (cfg, _dir) = temp_cfg("clear");
        fs::write(
            &cfg.jar,
            format!(
                "{}\n{}\n{}\n",
                line(".a.com", "x", "1"),
                line("sub.a.com", "y", "2"),
                line("b.org", "z", "3")
            ),
        )
        .expect("write");

        let removed = clear(&cfg, Some("a.com")).expect("clear");

        assert_eq!(removed, 2);
        let jar = fs::read_to_string(&cfg.jar).expect("read");
        assert!(!jar.contains("a.com"));
        assert!(jar.contains("b.org"));
    }

    #[test]
    fn clear_all_removes_the_jar_file() {
        let (cfg, _dir) = temp_cfg("clearall");
        fs::write(&cfg.jar, line(".a.com", "n", "v")).expect("write");
        clear(&cfg, None).expect("clear");
        assert!(!cfg.jar.exists());
        assert_eq!(mode(&cfg), CookieMode::Anonymous, "back to anonymous");
    }

    #[test]
    fn rejects_invalid_clear_scopes_without_revoking_leases_or_corrupting_the_log() {
        let (cfg, _dir) = temp_cfg("invalid-clear");
        let original = line("a.com", "session", "old");
        fs::write(&cfg.jar, &original).expect("jar");
        let lease = JarLease::checkout(&cfg.jar).expect("lease");
        fs::write(lease.staged(), line("a.com", "session", "new")).expect("rotation");
        for domain in ["", " ", "a.com\nb.org", "a.com\r", "a.com\t", "a.com\0"] {
            assert!(clear(&cfg, Some(domain)).is_err(), "{domain:?}");
        }
        assert_eq!(
            fs::read_to_string(&cfg.jar).expect("jar unchanged"),
            original
        );
        assert!(!ClearLog::path(&cfg.jar).exists());
        lease
            .commit()
            .expect("invalid clears did not revoke the lease");
        assert!(fs::read_to_string(&cfg.jar).expect("jar").contains("new"));
        assert_eq!(clear(&cfg, Some("a.com")).expect("valid clear"), 1);
        JarLease::checkout(&cfg.jar).expect("clear log remains readable");
    }

    #[cfg(unix)]
    #[test]
    fn jar_and_staged_copies_are_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let (cfg, _dir) = temp_cfg("perms");
        let incoming = cfg.jar.with_file_name("export.txt");
        fs::write(&incoming, format!("{}\n", line(".a.com", "n", "v"))).expect("write");
        import(&cfg, &incoming).expect("import");
        let jar_mode = fs::metadata(&cfg.jar).expect("meta").permissions().mode();
        assert_eq!(jar_mode & 0o777, 0o600, "jar is owner-only");

        let lease = JarLease::checkout(&cfg.jar).expect("checkout");
        let staged_mode = fs::metadata(lease.staged())
            .expect("meta")
            .permissions()
            .mode();
        assert_eq!(staged_mode & 0o777, 0o600, "staged copy is owner-only");
    }

    #[cfg(unix)]
    #[test]
    fn jar_copies_and_writes_are_born_owner_only() {
        // A canonical jar left world-readable (0644, the umask default a
        // user's `cp` produces) must never propagate that mode: every
        // copy and rewrite lands at 0600 from creation, not after a chmod.
        use std::os::unix::fs::PermissionsExt;
        let (cfg, _dir) = temp_cfg("born-private");
        fs::write(&cfg.jar, format!("{}\n", line(".a.com", "n", "v"))).expect("write");
        fs::set_permissions(&cfg.jar, fs::Permissions::from_mode(0o644)).expect("chmod");

        let mode = |p: PathBuf| fs::metadata(p).expect("meta").permissions().mode() & 0o777;
        let lease = JarLease::checkout(&cfg.jar).expect("checkout");
        assert_eq!(mode(lease.staged()), 0o600, "staged copy is owner-only");
        assert_eq!(mode(lease.snapshot()), 0o600, "snapshot copy is owner-only");

        // The write_jar path (via import) must also be born 0600.
        let incoming = cfg.jar.with_file_name("in.txt");
        fs::write(&incoming, format!("{}\n", line(".b.com", "m", "w"))).expect("write");
        import(&cfg, &incoming).expect("import");
        assert_eq!(mode(cfg.jar.clone()), 0o600, "written jar is owner-only");
    }
}
