// SPDX-License-Identifier: Apache-2.0
//! agent-video core: turn a video into VLM-readable grid images.
//!
//! The media kernel — decode, sample, score, compose, output — with no
//! network and no server. One piece of policy lives here deliberately: the
//! tuning-profile table in [`config`], because the harness predicts grid
//! budgets through the same table the pipeline resolves against — splitting
//! them would let prediction and execution drift. The CLI argument surface
//! is opt-in via the `clap` feature; without it core is a plain library.
//! The public surface is what the CLI and
//! the harness crate consume: [`pipeline`] to run, [`config`] to configure,
//! [`media`]/[`probe`] to inspect, [`wire`] as the output contract,
//! [`subtitle`] as the SRT contract (one parser, one serializer — every
//! SRT the workspace writes must parse back here),
//! [`event`] as the observability contract, and
//! [`source`]/[`types`]/[`error`] as shared vocabulary. Everything else is
//! implementation detail.

pub mod config;
pub mod error;
pub mod event;
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
