//! Q5_K quantizer: 256 elements per super-block, 8 sub-blocks of 32, 5-bit.
//!
//! On-disk: 2 bytes f16 d, 2 bytes f16 dmin, 12 bytes scales, 32 bytes qh,
//! 128 bytes qs. Total 176 B / 256 el. Dequant: x = d * sc1 * q - dmin * mn1.
//!
//! ## Scale table layout (12 bytes, canonical llama.cpp Q5_K)
//!
//! For sub-block j:
//!   - sc_l = scales[j] & 0x3F          (low 6 bits of byte j)
//!   - mn_l = scales[4 + j] & 0x3F      (low 6 bits of byte 4+j)
//!   - sc_h = (scales[8 + (j >> 2)] >> ((j & 3) * 2)) & 3
//!   - mn_h = (scales[9 + (j >> 2)] >> ((j & 3) * 2)) & 3
//! So scales[8] holds 2-bit high parts of sc[0..3] packed, scales[9] holds
//! 2-bit high parts of sc[4..7] AND mn[0..3] (sharing the byte — the
//! encoder must satisfy the coupling: sc_h[4..7] == mn_h[0..3]).
//! scales[10] holds mn_h[0..3] for j=0..3, and scales[11] holds mn_h[4..7].
//!
//! Our simple per-sub-block quantizer keeps all 6-bit values in [0, 63]
//! (no overflow into the high 2 bits), so the coupling is trivially
//! satisfied with 0.
//!
//! ## qs + qh layout
//!
//! q is a 5-bit value in [0, 31]. Low 4 bits in qs (packed nibbles, same
//! as Q4_K), high bit in qh at bit position (j % 8) of byte qh[j / 8].

use crate::formats::gguf::dequant;
use crate::formats::gguf::types::GgmlType;
use crate::quantize::f32_to_f16_bits;

const QK_K: usize = 256;
const SUB: usize = 32;
const N_SUB: usize = QK_K / SUB;
const BLOCK_BYTES: usize = 176;

#[inline]
fn pack_qs(qs: &mut [u8; 128], j: usize, n: u8) {
    if j < QK_K / 2 {
        qs[j] |= n & 0x0F;
    } else {
        qs[j - QK_K / 2] |= (n & 0x0F) << 4;
    }
}

pub fn quantize(src: &[f32]) -> Vec<u8> {
    debug_assert!(src.len() % QK_K == 0);
    let mut out = Vec::with_capacity(src.len() / QK_K * BLOCK_BYTES);
    for blk in src.chunks_exact(QK_K) {
        let (block_max, block_min) =
            blk.iter().fold((f32::NEG_INFINITY, f32::INFINITY), |(mx, mn), &v| (mx.max(v), mn.min(v)));
        let d = if block_max == 0.0 { 0.0 } else { block_max / 31.0 };
        let dmin = if block_min >= 0.0 { 0.0 } else { -block_min / 31.0 };
        let inv_d = if d == 0.0 { 0.0 } else { 1.0 / d };
        let inv_dmin = if dmin == 0.0 { 0.0 } else { 1.0 / dmin };

        let mut sc = [0u8; N_SUB];
        let mut mn = [0u8; N_SUB];
        for s in 0..N_SUB {
            let sub = &blk[s * SUB..(s + 1) * SUB];
            let (smax, smin) =
                sub.iter().fold((f32::NEG_INFINITY, f32::INFINITY), |(mx, mn), &v| (mx.max(v), mn.min(v)));
            let range = smax - smin;
            let sc_f = (range * inv_d / 31.0).round();
            // If the sub-block has non-zero variation, ensure at least 1
            // grid step.
            sc[s] = if range > 0.0 { sc_f.clamp(1.0, 63.0) as u8 } else { 0 };
            mn[s] = (-smin * inv_dmin).round().clamp(0.0, 63.0) as u8;
        }

        // Pack 12-byte scale table. Our values fit in 6 bits (high 2 == 0),
        // so the coupling constraint (sc_h[4..7] == mn_h[0..3]) is satisfied
        // trivially: scales[9] low 2 bits = 0 = sc_h[4..7], and
        // scales[9] bits 2-3 = 0 = mn_h[0..3], etc.
        let mut packed = [0u8; 12];
        for s in 0..N_SUB {
            packed[s] = sc[s];
            packed[4 + s] = mn[s];
        }
        // The high-2-byte slots (scales[8..12]) stay zero.

        // Quantize qs + qh.
        let mut qs = [0u8; 128];
        let mut qh = [0u8; 32];
        for j in 0..QK_K {
            let s = j / SUB;
            let d_sc = d * sc[s] as f32;
            let m_sc = dmin * mn[s] as f32;
            let denom = d_sc;
            let n = if denom == 0.0 { 0.0 } else { ((blk[j] + m_sc) / denom).round() };
            let n = n.clamp(0.0, 31.0) as u8;
            let lo = n & 0x0F;
            let hi = (n >> 4) & 0x01;
            pack_qs(&mut qs, j, lo);
            qh[j / 8] |= hi << (j % 8);
        }

        out.extend_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        out.extend_from_slice(&f32_to_f16_bits(dmin).to_le_bytes());
        out.extend_from_slice(&packed);
        out.extend_from_slice(&qh);
        out.extend_from_slice(&qs);
    }
    out
}

#[doc(hidden)]
pub fn dequant(bytes: &[u8]) -> Vec<f32> {
    dequant::dequantize(GgmlType::Q5K, bytes, None).unwrap()
}
