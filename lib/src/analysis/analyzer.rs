//! Streaming analyzer: read each tensor's bytes (zero-copy via mmap), dequant
//! up to `sample_per_tensor` elements, accumulate stats in a single pass.

use crate::analysis::score::{per_block_scores, BlockAnalysis};
use crate::analysis::stats::{sparsity_eps_for, Accum, Analysis, TensorStats};
use crate::error::Result;
use crate::formats::gguf::dequant as gguf_dequant;
use crate::formats::gguf::types::GgmlType;
use crate::model::{Model, TensorDtype};
use std::collections::HashSet;

pub struct Analyzer {
    sample_per_tensor: usize,
    keep: HashSet<String>,
}

impl Analyzer {
    pub fn new() -> Self {
        Self {
            sample_per_tensor: 200_000,
            keep: HashSet::new(),
        }
    }

    pub fn with_sample_per_tensor(n: usize) -> Self {
        Self {
            sample_per_tensor: n,
            keep: HashSet::new(),
        }
    }

    pub fn keep(mut self, name: impl Into<String>) -> Self {
        self.keep.insert(name.into());
        self
    }

    pub fn sample_per_tensor(&self) -> usize {
        self.sample_per_tensor
    }

    pub fn analyze<M: Model + ?Sized>(&self, model: &M) -> Result<Analysis> {
        let tensors = model.tensors();
        let total_bytes: u64 = tensors.iter().map(|t| t.byte_size).sum();
        let sample = self.sample_per_tensor;

        let mut per_tensor: Vec<(crate::model::Tensor, TensorStats)> =
            Vec::with_capacity(tensors.len());

        for t in tensors {
            // Sample limit: if we don't need to look at the whole tensor, only
            // dequant `sample` elements. This is the key streaming optimization.
            let max_elems = if t.shape.iter().product::<u64>() <= sample as u64 {
                None
            } else {
                Some(sample)
            };

            let bytes = model.read_tensor_bytes(&t.name)?;
            let dequantized = dequant_for_dtype(t.dtype, &bytes, max_elems);

            let Some(values) = dequantized else {
                continue;
            };

            // Compute the adaptive sparsity threshold from the values' amax.
            let amax = values.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let eps = sparsity_eps_for(amax);

            let mut acc = Accum::new();
            for &v in &values {
                acc.push(v);
            }
            let sampled = max_elems.is_some();
            let stats = acc.finalize(eps, sampled);
            per_tensor.push((t.clone(), stats));
        }

        let blocks = per_block_scores(&per_tensor);
        let (recommendation, recommendation_count, estimated_bytes_after_prune) =
            recommend(&blocks, total_bytes);

        Ok(Analysis {
            blocks,
            recommendation,
            recommendation_count,
            estimated_bytes_after_prune,
            sample_per_tensor: sample,
            total_tensors: tensors.len(),
            total_bytes,
        })
    }
}

fn dequant_for_dtype(
    dtype: TensorDtype,
    bytes: &[u8],
    max_elems: Option<usize>,
) -> Option<Vec<f32>> {
    let gg_ty = match dtype {
        TensorDtype::F32 => GgmlType::F32,
        TensorDtype::F16 => GgmlType::F16,
        TensorDtype::Bf16 => GgmlType::Bf16,
        TensorDtype::F64 => GgmlType::F64,
        TensorDtype::I8 => GgmlType::I8,
        TensorDtype::I16 => GgmlType::I16,
        TensorDtype::I32 => GgmlType::I32,
        TensorDtype::I64 => GgmlType::I64,
        TensorDtype::Q4_0 => GgmlType::Q4_0,
        TensorDtype::Q4_1 => GgmlType::Q4_1,
        TensorDtype::Q5_0 => GgmlType::Q5_0,
        TensorDtype::Q5_1 => GgmlType::Q5_1,
        TensorDtype::Q8_0 => GgmlType::Q8_0,
        TensorDtype::Q8_1 => GgmlType::Q8_1,
        TensorDtype::Q2K => GgmlType::Q2K,
        TensorDtype::Q3K => GgmlType::Q3K,
        TensorDtype::Q4K => GgmlType::Q4K,
        TensorDtype::Q5K => GgmlType::Q5K,
        TensorDtype::Q6K => GgmlType::Q6K,
        TensorDtype::Q8K => GgmlType::Q8K,
        _ => return None,
    };
    gguf_dequant::dequantize(gg_ty, bytes, max_elems)
}

/// Pick the top-N prunable blocks (highest `removable`), where N is 15% of
/// prunable blocks rounded.
fn recommend(blocks: &[BlockAnalysis], total_bytes: u64) -> (Vec<i32>, usize, u64) {
    let mut prunable: Vec<&BlockAnalysis> =
        blocks.iter().filter(|b| b.role.is_prunable()).collect();
    prunable.sort_by(|a, b| {
        b.removable
            .partial_cmp(&a.removable)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let n = ((prunable.len() as f64) * 0.15).round() as usize;
    let n = n.max(1).min(prunable.len());
    let recommended: Vec<i32> = prunable.iter().take(n).map(|b| b.index).collect();
    let saved: u64 = prunable.iter().take(n).map(|b| b.total_bytes).sum();
    (recommended, n, total_bytes.saturating_sub(saved))
}
