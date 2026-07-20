// SPDX-License-Identifier: Apache-2.0
//! The harness: everything between a model's `video` tool call and a core
//! `process`/`extract` run.
//!
//! One module per responsibility: argument schema, timestamp policy, budget
//! ladder, consent tokens, URL sources, disk cache, whisper transcription
//! and weight provisioning. The [`mcp`] module owns the protocol face (stdio
//! server, download tracker, core self-spawn) and the process's only tokio
//! runtime; everything else is synchronous and server-free.

pub mod cache;
pub mod config;
pub mod cookies;
pub mod fault;
pub mod fetch;
pub mod gate;
pub mod gpu;
pub mod identity;
pub mod install;
pub mod look;
pub mod mcp;
pub mod netguard;
pub mod offer;
pub mod plan;
pub mod probe;
pub mod proc;
pub mod raw;
pub mod relay;
pub mod schema;
#[cfg(test)]
pub(crate) mod tempdir;
pub mod time;
pub mod tokens;
pub mod transcribe;
