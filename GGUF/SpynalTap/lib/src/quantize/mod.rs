//! Pure-Rust GGML-style weight quantizers.
//!
//! The companion to `crate::formats::gguf::dequant`. Each submodule implements
//! a single block quantizer; `quantize` is the public dispatch entry point.
//!
//! Scope:
//!   - Per-block: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0
//!   - K-quants:  Q4_K, Q5_K, Q6_K (Q8_K dequant exists but no quantizer
//!                — it's only useful for round-trip and is rarely used as
//!                a storage format since it's not smaller than F16).
//!   - I-quants:  not yet implemented.
//!
//! The K-quant quantizers match the canonical llama.cpp on-disk layout
//! (12-byte scale tables with the "shared low-4" or "shared high-2" packing
//! tricks). Quantization is per-sub-block with a simple max-absolute-value
//! scale; the encoder's 6-bit values stay in [0, 63] so the coupling
//! constraints of the canonical layout are satisfied trivially.

pub mod apply;
pub mod q4_0;
pub mod q4_1;
pub mod q4_k;
pub mod q5_0;
pub mod q5_1;
pub mod q5_k;
pub mod q6_k;
pub mod q8_0;

use crate::formats::gguf::types::GgmlType;

/// Quantize a row-major `f32` buffer to the given GGML block type.
///
/// `src.len()` must be a multiple of the type's block size (32 for
/// Q4_0/Q4_1/Q5_0/Q5_1/Q8_0, 256 for K-quants). Returns the raw little-endian
/// block bytes; the caller is responsible for any alignment padding required
/// by the container (GGUF uses 32-byte alignment between tensors).
///
/// # Panics
/// Panics if `ty` is not a supported quant type, or if `src.len()` is not a
/// multiple of the block size.
pub fn quantize(src: &[f32], ty: GgmlType) -> Vec<u8> {
    let block = ty.block_size();
    if block > 1 {
        assert!(
            src.len() % block == 0,
            "input length {} is not a multiple of block size {} for {:?}",
            src.len(),
            block,
            ty,
        );
    }
    match ty {
        GgmlType::Q4_0 => q4_0::quantize(src),
        GgmlType::Q4_1 => q4_1::quantize(src),
        GgmlType::Q5_0 => q5_0::quantize(src),
        GgmlType::Q5_1 => q5_1::quantize(src),
        GgmlType::Q8_0 => q8_0::quantize(src),
        GgmlType::Q4K => q4_k::quantize(src),
        GgmlType::Q5K => q5_k::quantize(src),
        GgmlType::Q6K => q6_k::quantize(src),
        other => panic!("quantize: unsupported type {:?}", other),
    }
}

/// Returns true if `ty` is a quant type accepted by `quantize`.
pub fn is_quantizable(ty: GgmlType) -> bool {
    matches!(ty,
        GgmlType::Q4_0 | GgmlType::Q4_1
        | GgmlType::Q5_0 | GgmlType::Q5_1
        | GgmlType::Q8_0
        | GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K
    )
}

/// f32 -> binary16 bit pattern (round-to-nearest-even, handles inf/nan).
#[inline]
pub(crate) fn f32_to_f16_bits(v: f32) -> u16 {
    if v.is_nan() {
        return 0x7e00;
    }
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    if v.is_infinite() {
        return sign | 0x7c00;
    }
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = (bits >> 13) & 0x3ff;
    if exp >= 31 {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let mant = (mant | 0x400) >> (1 - exp);
        return sign | (mant as u16);
    }
    sign | ((exp as u16) << 10) | (mant as u16)
}
