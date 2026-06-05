//! `spynaltap` — fast AI model analyzer and transformer-block pruner.
//!
//! Supports GGUF (v1–v3) and safetensors. Per-block "removability" is
//! scored heuristically by default; the `calibrate` feature uses `candle`
//! to run a forward pass and rank blocks by activation-delta instead.
//!
//! Quick start:
//! ```no_run
//! use spynaltap::{Analyzer, formats::gguf::GgufFile};
//!
//! let model = GgufFile::open("model.gguf")?;
//! let analysis = Analyzer::with_sample_per_tensor(200_000).analyze(&model)?;
//! println!("recommended: {:?}", analysis.recommendation);
//! # Ok::<(), spynaltap::Error>(())
//! ```
//!
//! The CLI binary is `spynaltape`.

#![allow(clippy::needless_range_loop)]

pub mod analysis;
#[cfg(feature = "calibrate")]
pub mod calibrate;
pub mod error;
pub mod formats;
pub mod model;
pub mod prune;
pub mod svd;

pub use analysis::{Analysis, Analyzer, BlockAnalysis, TensorAnalysis};
pub use error::{Error, Result};
pub use model::{BlockRef, MetadataValue, Model, ModelFormat, Tensor, TensorDtype};
pub use prune::{build_plan, parse_selection, PrunePlan, PruneReport, Selection};
pub use svd::{
    apply_to_gguf as svd_apply_to_gguf, apply_to_safetensors as svd_apply_to_safetensors,
    build_plan as build_svd_plan, LayerSelection, OutputDtype, RankClamps, RankSpec,
    RankSpecWithClamps, SvdApplied, SvdConfig, SvdPlan, SvdReport, SvdTarget, TensorSelection,
};
