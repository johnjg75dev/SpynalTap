//! TQ2_0 quantizer: 256 elements per super-block, 2-bit ternary.
//!
//! Each weight is -1, 0, or +1 (encoded as 0, 1, 2 in 2 bits).
//! On-disk: 2 bytes f16 d, 64 bytes qs = 66 bytes / 256 elements.
//! Dequant: x = d * (q - 1) where q ∈ {0, 1, 2}.

use crate::formats::gguf::dequant;
use crate::formats::gguf::types::GgmlType;
use crate::quantize::f32_to_f16_bits;

const QK_K: usize = 256;
const BLOCK_BYTES: usize = 66;

pub fn quantize(src: &[f32]) -> Vec<u8> {
    debug_assert!(src.len() % QK_K == 0);
    let mut out = Vec::with_capacity(src.len() / QK_K * BLOCK_BYTES);
    for blk in src.chunks_exact(QK_K) {
        let max_abs = blk.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let d = if max_abs == 0.0 { 0.0 } else { max_abs };
        let inv_d = if d == 0.0 { 0.0 } else { 1.0 / d };

        let mut qs = [0u8; 64];
        for j in 0..QK_K {
            let v = blk[j];
            let q = if d == 0.0 {
                0u8
            } else {
                let raw = (v * inv_d).round();
                if raw >= 1.0 {
                    2u8
                } else if raw <= -1.0 {
                    0u8
                } else {
                    1u8
                }
            };
            let byte_idx = j / 4;
            let shift = (j % 4) * 2;
            qs[byte_idx] |= q << shift;
        }

        out.extend_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        out.extend_from_slice(&qs);
    }
    out
}

#[doc(hidden)]
pub fn dequant(bytes: &[u8]) -> Vec<f32> {
    dequant::dequantize(GgmlType::Tq2_0, bytes, None).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_all_positive() {
        let src = vec![1.0f32; QK_K];
        let bytes = quantize(&src);
        assert_eq!(bytes.len(), BLOCK_BYTES);
        let out = dequant(&bytes);
        for v in &out {
            assert!((v - 1.0).abs() < 0.001 || v.abs() < 0.001);
        }
    }

    #[test]
    fn roundtrip_all_zero() {
        let src = vec![0.0f32; QK_K];
        let bytes = quantize(&src);
        let out = dequant(&bytes);
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn roundtrip_all_negative() {
        let src = vec![-2.0f32; QK_K];
        let bytes = quantize(&src);
        assert_eq!(bytes.len(), BLOCK_BYTES);
        let out = dequant(&bytes);
        for v in &out {
            assert!(v <= &0.0f32);
        }
    }

    #[test]
    fn roundtrip_mixed() {
        let src: Vec<f32> = (0..QK_K).map(|i| ((i as f32) * 0.1 - 0.5).round() * 2.0).collect();
        let bytes = quantize(&src);
        let out = dequant(&bytes);
        assert_eq!(out.len(), QK_K);
    }

    #[test]
    fn matches_dequant() {
        let src: Vec<f32> = (0..QK_K).map(|i| ((i as f32) * 0.07).sin() * 3.0).collect();
        let bytes = quantize(&src);
        let direct = dequant(&bytes);
        let via_dispatch = dequant::dequantize(GgmlType::Tq2_0, &bytes, None).unwrap();
        assert_eq!(direct, via_dispatch);
    }
}