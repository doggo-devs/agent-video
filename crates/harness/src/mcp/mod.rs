// SPDX-License-Identifier: Apache-2.0
//! The MCP face of the harness: stdio server, background-download tracker,
//! and core self-spawn. Everything protocol-shaped lives here; the sibling
//! modules stay pure policy and I/O, callable without any server.
//!
//! One `video` call flows through the layers in order: [`server`] speaks
//! rmcp and hosts the blocking [`handler`], which executes against the
//! injected capabilities in [`deps`] and answers in the shapes of [`wire`].

pub mod core_run;
pub mod deps;
pub mod handler;
pub mod server;
pub mod tracker;
mod wire;
