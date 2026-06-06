//! Calibration-based block importance scoring.
//!
//! Gated behind the `calibrate` cargo feature. The full forward-pass
//! implementation is the next milestone; this module currently exposes the
//! configuration struct and a stub that returns a clear "not yet wired" error
//! when `--calibrate` is requested from the CLI.
//!
//! To finish this feature:
//! 1. Use `candle-transformers::models::llama::Llama` (and friends) to load
//!    a small reference model from `hf-hub`.
//! 2. Tokenize the calibration text with `tokenizers`.
//! 3. Run N forward passes; for each block, capture the residual stream
//!    `h_in` and `h_out`, compute `delta = ||h_out - h_in||_2 / ||h_in||_2`.
//! 4. Average the deltas across passes; produce a `BlockAnalysis` list where
//!    `removable = 1 - normalized_delta`. Plug into the existing prune
//!    pipeline via `build_plan(..., Some(&analyses))`.

use crate::analysis::score::BlockAnalysis;
use crate::error::{Error, Result};

/// Configuration for calibration-based scoring.
pub struct CalibrationConfig {
    /// Path to a text file with calibration prompts (one per line, or the
    /// whole file as a single prompt).
    pub text_path: String,
    /// HuggingFace repo or local path to a reference model in safetensors
    /// (e.g. `meta-llama/Llama-2-7b-hf`).
    pub model_ref: String,
    /// Number of forward passes (default 4).
    pub n_passes: usize,
    /// Maximum sequence length (default 512).
    pub max_seq_len: usize,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            text_path: String::new(),
            model_ref: String::new(),
            n_passes: 4,
            max_seq_len: 512,
        }
    }
}

/// Run calibration and return a vector of per-block analyses.
///
/// Currently returns an error indicating the feature isn't wired up yet.
pub fn run_calibration(_cfg: &CalibrationConfig) -> Result<Vec<BlockAnalysis>> {
    Err(Error::Calibration(
        "calibration requires the `candle` integration; not yet implemented. \
         See src/calibrate/mod.rs for the integration plan."
            .into(),
    ))
}
