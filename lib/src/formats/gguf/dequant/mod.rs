//! Dequantization dispatch. Prefers SIMD when available (x86_64 AVX2 + F16C),
//! falls back to scalar code otherwise.

mod scalar;
#[cfg(target_arch = "x86_64")]
mod simd;

use crate::formats::gguf::types::GgmlType;

pub use scalar::{bf16_to_f32, f16_to_f32};

/// Decode (or sample) a tensor's bytes into `Vec<f32>`.
/// Returns `None` if the type is not dequantizable.
pub fn dequantize(ty: GgmlType, bytes: &[u8], max_elems: Option<usize>) -> Option<Vec<f32>> {
    let max = max_elems.unwrap_or(usize::MAX);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") {
            if let Some(v) = unsafe { simd::try_dequant(ty, bytes, max) } {
                return Some(v);
            }
        }
    }
    scalar::dequantize(ty, bytes, max)
}

#[inline]
fn truncate_to(mut v: Vec<f32>, max: usize) -> Vec<f32> {
    if v.len() > max {
        v.truncate(max);
    }
    v
}
