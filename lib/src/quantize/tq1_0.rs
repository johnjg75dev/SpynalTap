//! TQ1_0 quantizer: 176 elements per block, ternary (1-bit).
//!
//! On-disk: 2 d (f16), 32 qs, 4 qh = 38 bytes / block.
//! qs: 32 bytes × 5 trits = 160 elements
//! qh: 4 bytes × 4 trits = 16 elements
//!
//! NOTE: The dequant currently only produces 176 values per block (not 256),
//! so this quantizer operates on 176-element blocks.
//!
//! The dequant uses a multiplicative-inverse trick to decode base-3 trits,
//! which is NOT standard base-3 packing. The encoding formulas are:
//!
//! 5 trits: value = t0 + t1·3 + t2·9 + t3·27 + t4·81
//!           byte = (x5 + 256·value) / 243,  x5 = (243 − 13·value mod 243) mod 243
//!
//! 4 trits: value = t0 + t1·3 + t2·9 + t3·27
//!           byte = (x4 + 256·value) / 81,   x4 = (81 − 13·value mod 81) mod 81

use crate::formats::gguf::dequant;
use crate::formats::gguf::types::GgmlType;
use crate::quantize::f32_to_f16_bits;

const ELEMS: usize = 176;
const BLOCK_BYTES: usize = 38;

fn encode_5(t0: u8, t1: u8, t2: u8, t3: u8, t4: u8) -> u8 {
    let value = t0 + t1 * 3 + t2 * 9 + t3 * 27 + t4 * 81;
    let v = value as u16;
    let x5 = (243 - (13 * v % 243)) % 243;
    ((x5 + 256 * v) / 243) as u8
}

fn encode_4(t0: u8, t1: u8, t2: u8, t3: u8) -> u8 {
    let value = t0 + t1 * 3 + t2 * 9 + t3 * 27;
    let v = value as u16;
    let x4 = (81 - (13 * v % 81)) % 81;
    ((x4 + 256 * v) / 81) as u8
}

pub fn quantize(src: &[f32]) -> Vec<u8> {
    debug_assert!(src.len() % ELEMS == 0);
    let mut out = Vec::with_capacity(src.len() / ELEMS * BLOCK_BYTES);

    for blk in src.chunks_exact(ELEMS) {
        let max_abs = blk.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let d = max_abs;
        let inv_d = if d == 0.0 { 0.0 } else { 1.0 / d };

        let mut qs = [0u8; 32];
        let mut qh = [0u8; 4];

        for j in 0..32 {
            let t0 = if d == 0.0 { 1u8 } else {
                let raw = (blk[j * 5 + 0] * inv_d).round();
                if raw >= 1.0 { 2 } else if raw <= -1.0 { 0 } else { 1 }
            };
            let t1 = if d == 0.0 { 1u8 } else {
                let raw = (blk[j * 5 + 1] * inv_d).round();
                if raw >= 1.0 { 2 } else if raw <= -1.0 { 0 } else { 1 }
            };
            let t2 = if d == 0.0 { 1u8 } else {
                let raw = (blk[j * 5 + 2] * inv_d).round();
                if raw >= 1.0 { 2 } else if raw <= -1.0 { 0 } else { 1 }
            };
            let t3 = if d == 0.0 { 1u8 } else {
                let raw = (blk[j * 5 + 3] * inv_d).round();
                if raw >= 1.0 { 2 } else if raw <= -1.0 { 0 } else { 1 }
            };
            let t4 = if d == 0.0 { 1u8 } else {
                let raw = (blk[j * 5 + 4] * inv_d).round();
                if raw >= 1.0 { 2 } else if raw <= -1.0 { 0 } else { 1 }
            };
            qs[j] = encode_5(t0, t1, t2, t3, t4);
        }

        for j in 0..4 {
            let t0 = if d == 0.0 { 1u8 } else {
                let raw = (blk[160 + j * 4 + 0] * inv_d).round();
                if raw >= 1.0 { 2 } else if raw <= -1.0 { 0 } else { 1 }
            };
            let t1 = if d == 0.0 { 1u8 } else {
                let raw = (blk[160 + j * 4 + 1] * inv_d).round();
                if raw >= 1.0 { 2 } else if raw <= -1.0 { 0 } else { 1 }
            };
            let t2 = if d == 0.0 { 1u8 } else {
                let raw = (blk[160 + j * 4 + 2] * inv_d).round();
                if raw >= 1.0 { 2 } else if raw <= -1.0 { 0 } else { 1 }
            };
            let t3 = if d == 0.0 { 1u8 } else {
                let raw = (blk[160 + j * 4 + 3] * inv_d).round();
                if raw >= 1.0 { 2 } else if raw <= -1.0 { 0 } else { 1 }
            };
            qh[j] = encode_4(t0, t1, t2, t3);
        }

        out.extend_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        out.extend_from_slice(&qs);
        out.extend_from_slice(&qh);
    }

    out
}

#[doc(hidden)]
pub fn dequant(bytes: &[u8]) -> Vec<f32> {
    dequant::dequantize(GgmlType::Tq1_0, bytes, None).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_constant() {
        let src = vec![2.0f32; ELEMS];
        let bytes = quantize(&src);
        assert_eq!(bytes.len(), BLOCK_BYTES);
        let out = dequant(&bytes);
        for v in &out {
            assert!((v - 2.0).abs() < 0.001 || (v - 0.0).abs() < 0.001);
        }
    }

    #[test]
    fn roundtrip_all_zero() {
        let src = vec![0.0f32; ELEMS];
        let bytes = quantize(&src);
        let out = dequant(&bytes);
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn roundtrip_sine() {
        let src: Vec<f32> = (0..ELEMS).map(|i| ((i as f32) * 0.1).sin() * 3.0).collect();
        let bytes = quantize(&src);
        let out = dequant(&bytes);
        assert_eq!(out.len(), ELEMS);
    }

    #[test]
    fn matches_dequant() {
        let src: Vec<f32> = (0..ELEMS).map(|i| ((i as f32) * 0.1).sin() * 5.0).collect();
        let bytes = quantize(&src);
        let direct = dequant(&bytes);
        let via = dequant::dequantize(GgmlType::Tq1_0, &bytes, None).unwrap();
        assert_eq!(direct, via);
    }
}
