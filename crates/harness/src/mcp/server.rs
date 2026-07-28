// SPDX-License-Identifier: Apache-2.0
//! MCP `video` tool server over stdio: rmcp assembly around the handler.
//!
//! The handler itself is synchronous; rmcp's tokio runtime hosts it via
//! `spawn_blocking`, so tokio stays local to this module as the migration
//! plan requires. Progress notifications are bridged over a bounded, lossy
//! channel — best-effort, a dead transport never kills a call.

use std::sync::Arc;
use std::time::SystemTime;

use rmcp::ErrorData as McpError;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, Implementation, ListToolsResult, PaginatedRequestParams,
    ProgressNotificationParam, ServerCapabilities, ServerInfo, Tool,
};
use rmcp::service::RequestContext;
use rmcp::{RoleServer, ServerHandler};

use super::deps::{Notifier, ServerDeps, null_notifier};
use super::handler::handle_video_call;
use super::wire::failed;
use crate::config::{BaseDirs, load_config};
use crate::schema::{VIDEO_TOOL_DESCRIPTION, video_input_schema};

/// The MCP server: one `video` tool over injected deps.
pub struct VideoServer {
    deps: Arc<ServerDeps>,
}

impl VideoServer {
    pub fn new(deps: Arc<ServerDeps>) -> Self {
        VideoServer { deps }
    }
}

impl ServerHandler for VideoServer {
    fn get_info(&self) -> ServerInfo {
        // The rmcp model types are `#[non_exhaustive]`; build by mutation.
        let mut server_info = Implementation::default();
        server_info.name = "agent-video".to_string();
        server_info.version = env!("CARGO_PKG_VERSION").to_string();
        let mut info = ServerInfo::default();
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info.server_info = server_info;
        info
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        let serde_json::Value::Object(schema) = video_input_schema() else {
            unreachable!("video_input_schema is an object by construction");
        };
        Ok(ListToolsResult::with_all_items(vec![Tool::new(
            "video",
            VIDEO_TOOL_DESCRIPTION,
            schema,
        )]))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        if request.name.as_ref() != "video" {
            return Ok(failed(&crate::fault::Fault::bad_input(format!(
                "unknown tool \"{}\"; only \"video\" exists",
                request.name
            ))));
        }
        let args = request.arguments.unwrap_or_default();

        // Progress is best-effort; a dead transport must not kill the call.
        // The channel is bounded and lossy on purpose: progress is a
        // sampling, not a log. `try_send` drops the newest update when the
        // peer drains slowly — the backlog it does drain is already stale
        // either way, and bounding beats buffering without limit.
        let notify: Notifier = match context.meta.get_progress_token() {
            Some(token) => {
                let (tx, mut rx) = tokio::sync::mpsc::channel::<(f64, Option<f64>, String)>(64);
                let peer = context.peer.clone();
                tokio::spawn(async move {
                    while let Some((progress, total, message)) = rx.recv().await {
                        let mut param = ProgressNotificationParam::new(token.clone(), progress)
                            .with_message(message);
                        param.total = total;
                        let _ = peer.notify_progress(param).await;
                    }
                });
                Arc::new(move |p, t, m| {
                    let _ = tx.try_send((p, t, m));
                })
            }
            None => null_notifier(),
        };

        let deps = Arc::clone(&self.deps);
        tokio::task::spawn_blocking(move || handle_video_call(&args, &deps, &notify))
            .await
            .map_err(|e| McpError::internal_error(format!("handler task failed: {e}"), None))
    }
}

/// Boot the stdio MCP server with production deps; blocks until the client
/// disconnects. Owns the process's only tokio runtime.
pub fn serve_stdio() -> anyhow::Result<()> {
    // First statement on purpose: the env scrub below requires a
    // single-threaded process, so it must precede the tracing subscriber,
    // config loading, and everything else that may spawn threads.
    scrub_no_proxy_env();

    // stderr is safe: MCP transport is on stdin/stdout.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("agent_video_harness=info")),
        )
        .init();

    let dirs = BaseDirs::detect();
    let config = load_config(&dirs)?;
    let deps = Arc::new(ServerDeps::production(config));
    // Eviction sweep on boot; never blocks or fails startup.
    let _ = deps.cache.evict(SystemTime::now());
    // Route whisper.cpp/ggml logs into `log` (silent without a logger)
    // instead of stderr, keeping transcription quiet on the transport.
    #[cfg(feature = "whisper-local")]
    whisper_rs::install_logging_hooks();

    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async {
        let service =
            rmcp::serve_server(VideoServer::new(deps), rmcp::transport::io::stdio()).await?;
        service.waiting().await?;
        Ok(())
    })
}

/// Scrub `no_proxy` from the server process's environment.
///
/// The in-process libav demuxer consults `no_proxy` at connect time even
/// when the `http_proxy` option routes it through the policy relay
/// (ffmpeg's http protocol checks the env var first), so an inherited
/// value would silently bypass the per-connection egress screen. Direct
/// CLI runs (`process`, `extract`) are untouched: there the user's proxy
/// environment is their own business.
///
/// Must run while the process is still single-threaded. That is a call-site
/// obligation the type system cannot state, so debug builds enforce it with
/// a live thread count instead of trusting the caller.
fn scrub_no_proxy_env() {
    debug_assert!(
        live_thread_count().is_none_or(|threads| threads == 1),
        "scrub_no_proxy_env called after another thread exists: mutating the \
         environment now races concurrent getenv — move the call earlier in boot"
    );
    // SAFETY: the process is single-threaded here, so no concurrent getenv
    // can race the mutation — asserted above in debug builds, upheld by
    // being the first statement of `serve_stdio`, before the tracing
    // subscriber, the tokio runtime, or any library that spawns threads.
    unsafe {
        std::env::remove_var("no_proxy");
        std::env::remove_var("NO_PROXY");
    }
}

/// Live threads in this process, where the platform exposes them cheaply
/// (`/proc/self/task` on Linux); `None` elsewhere, which callers must treat
/// as "unknown", never as "one".
fn live_thread_count() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        Some(std::fs::read_dir("/proc/self/task").ok()?.count())
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    #[cfg(target_os = "linux")]
    #[test]
    fn thread_count_sees_a_live_spawned_thread() {
        let (release, held) = std::sync::mpsc::channel::<()>();
        let handle = std::thread::spawn(move || {
            let _ = held.recv();
        });
        let count = super::live_thread_count().expect("linux exposes /proc/self/task");
        assert!(count >= 2, "at least this thread and the held one: {count}");
        release.send(()).expect("held thread is receiving");
        handle.join().expect("held thread exits");
    }
}
