//! Pipeline config: run a sequence of operations defined in a JSON file.

#![allow(dead_code)]

use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
pub struct PipelineConfig {
    /// Optional: model path to use as input for the first step.
    pub model: Option<PathBuf>,
    /// Sequence of pipeline steps to execute.
    pub steps: Vec<PipelineStep>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "action")]
pub enum PipelineStep {
    /// Analyze model, optionally write JSON/HTML report.
    Analyze {
        sample: Option<usize>,
        json: Option<bool>,
        report: Option<PathBuf>,
        template: Option<PathBuf>,
    },
    /// Quantize model to a target type.
    Quant {
        quant_type: String,
        out: PathBuf,
        verify: Option<bool>,
    },
    /// Prune blocks/layers.
    Prune {
        selection: String,
        out: PathBuf,
        verify: Option<bool>,
    },
    /// SVD-compress tensors.
    Svd {
        out: PathBuf,
        selection: Option<String>,
        rank: Option<String>,
        #[serde(rename = "dry-run")]
        dry_run: Option<bool>,
        verify: Option<bool>,
    },
    /// Merge two or more models.
    Merge {
        models: Vec<PathBuf>,
        out: PathBuf,
        weights: Option<Vec<f32>>,
        slerp: Option<bool>,
        verify: Option<bool>,
    },
    /// MoE expert operations.
    MoE {
        selection: Option<String>,
        strategy: Option<String>,
        out: Option<PathBuf>,
        dry_run: Option<bool>,
    },
}

impl PipelineConfig {
    pub fn from_json(path: &std::path::Path) -> Result<Self, crate::Error> {
        let contents = std::fs::read_to_string(path).map_err(crate::Error::Io)?;
        serde_json::from_str(&contents).map_err(|e| crate::Error::Gguf(format!("pipeline config parse error: {e}")))
    }
}
