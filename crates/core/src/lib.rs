// SPDX-License-Identifier: Apache-2.0
//! agent-video core: turn a video into VLM-readable grid images.
//!
//! The media kernel — decode, sample, score, compose, output — with no
//! policy, no network, no server. The public surface is what the CLI and
//! the harness crate consume: [`pipeline`] to run, [`config`] to configure,
//! [`media`]/[`probe`] to inspect, [`wire`] as the output contract, and
//! [`source`]/[`types`]/[`error`] as shared vocabulary. Everything else is
//! implementation detail.

pub mod config;
pub mod error;
pub mod media;
pub mod pipeline;
pub mod probe;
pub mod source;
pub mod subtitle;
pub mod types;
pub mod wire;

mod compose;
mod dedup;
mod extract;
mod layout;
mod score;
