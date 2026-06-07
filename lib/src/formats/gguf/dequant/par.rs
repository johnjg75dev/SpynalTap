//! Parallel block-level dequantization. Each per-block dequantizer is
//! re-exported from `scalar` and the chunks are processed in parallel
//! via rayon. Single-block / non-block types fall back to scalar.

use super::scalar::{
    dequant_iq1_s, dequant_iq2_xs, dequant_iq2_xxs, dequant_iq3_s, dequant_iq3_xxs,
    dequant_iq4_nl, dequant_iq4_xs, dequant_q2_k, dequant_q3_k, dequant_q4_0, dequant_q4_1,
    dequant_q4_k, dequant_q5_0, dequant_q5_1, dequant_q5_k, dequant_q6_k, dequant_q8_0,
    dequant_q8_1, dequant_q8_k, dequant_tq1_0, dequant_tq2_0, dequantize,
};
use super::truncate_to;
use crate::formats::gguf::types::GgmlType;
use rayon::prelude::*;

// -- Parallel block-level dequant -------------------------------------------

/// Parallel block-level dequantize. Splits `bytes` into per-block chunks
/// (using the type's `block_bytes()`) and dequantizes each chunk in
/// parallel via rayon. Falls back to the single-threaded `dequantize`
/// for types with no fixed block size (F32, F16, BF16, F64, I8..I64)
/// or when there's only a single block.
///
/// Returns `None` if the type is not dequantizable.
pub fn dequantize_par(ty: GgmlType, bytes: &[u8], max: usize) -> Option<Vec<f32>> {
    let block_bytes = ty.block_bytes()?;
    if block_bytes <= 1 || bytes.len() < block_bytes * 2 {
        return dequantize(ty, bytes, max);
    }
    let chunks: Vec<&[u8]> = bytes.chunks_exact(block_bytes).collect();
    let mut parts: Vec<Vec<f32>> = match ty {
        GgmlType::Q4_0 => chunks.par_iter().map(|b| dequant_q4_0(b)).collect(),
        GgmlType::Q4_1 => chunks.par_iter().map(|b| dequant_q4_1(b)).collect(),
        GgmlType::Q5_0 => chunks.par_iter().map(|b| dequant_q5_0(b)).collect(),
        GgmlType::Q5_1 => chunks.par_iter().map(|b| dequant_q5_1(b)).collect(),
        GgmlType::Q8_0 => chunks.par_iter().map(|b| dequant_q8_0(b)).collect(),
        GgmlType::Q8_1 => chunks.par_iter().map(|b| dequant_q8_1(b)).collect(),
        GgmlType::Q4K => chunks.par_iter().map(|b| dequant_q4_k(b)).collect(),
        GgmlType::Q5K => chunks.par_iter().map(|b| dequant_q5_k(b)).collect(),
        GgmlType::Q6K => chunks.par_iter().map(|b| dequant_q6_k(b)).collect(),
        GgmlType::Q8K => chunks.par_iter().map(|b| dequant_q8_k(b)).collect(),
        GgmlType::Q2K => chunks.par_iter().map(|b| dequant_q2_k(b)).collect(),
        GgmlType::Q3K => chunks.par_iter().map(|b| dequant_q3_k(b)).collect(),
        GgmlType::Iq1S => chunks.par_iter().map(|b| dequant_iq1_s(b)).collect(),
        GgmlType::Iq2Xxs => chunks.par_iter().map(|b| dequant_iq2_xxs(b)).collect(),
        GgmlType::Iq2Xs => chunks.par_iter().map(|b| dequant_iq2_xs(b)).collect(),
        GgmlType::Iq3Xxs => chunks.par_iter().map(|b| dequant_iq3_xxs(b)).collect(),
        GgmlType::Iq3S => chunks.par_iter().map(|b| dequant_iq3_s(b)).collect(),
        GgmlType::Iq4Nl => chunks.par_iter().map(|b| dequant_iq4_nl(b)).collect(),
        GgmlType::Iq4Xs => chunks.par_iter().map(|b| dequant_iq4_xs(b)).collect(),
        GgmlType::Tq1_0 => chunks.par_iter().map(|b| dequant_tq1_0(b)).collect(),
        GgmlType::Tq2_0 => chunks.par_iter().map(|b| dequant_tq2_0(b)).collect(),
        _ => return dequantize(ty, bytes, max),
    };
    let total: usize = parts.iter().map(|v| v.len()).sum();
    let mut out = Vec::with_capacity(total);
    for p in parts.drain(..) {
        out.extend(p);
    }
    Some(truncate_to(out, max))
}
