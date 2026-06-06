//! Scalar dequantization fallbacks. Used on non-x86_64 and when AVX2+F16C
//! aren't both available.
//!
//! All hot loops are marked `#[inline]` for the compiler; LLVM unrolls the
//! 32-element block loops aggressively at `-O3`.

use super::truncate_to;
use crate::formats::gguf::types::GgmlType;

pub fn dequantize(ty: GgmlType, bytes: &[u8], max: usize) -> Option<Vec<f32>> {
    Some(truncate_to(
        match ty {
            GgmlType::F32 => scan_f32(bytes),
            GgmlType::F16 => scan_f16(bytes),
            GgmlType::Bf16 => scan_bf16(bytes),
            GgmlType::Q4_0 => dequant_q4_0(bytes),
            GgmlType::Q4_1 => dequant_q4_1(bytes),
            GgmlType::Q5_0 => dequant_q5_0(bytes),
            GgmlType::Q5_1 => dequant_q5_1(bytes),
            GgmlType::Q8_0 => dequant_q8_0(bytes),
            GgmlType::Q4K => dequant_q4_k(bytes),
            GgmlType::Q5K => dequant_q5_k(bytes),
            GgmlType::Q6K => dequant_q6_k(bytes),
            GgmlType::Q8K => dequant_q8_k(bytes),
            _ => return None,
        },
        max,
    ))
}

// -- scalar types -------------------------------------------------------------

#[inline]
pub fn scan_f32(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for c in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    }
    out
}

#[inline]
pub fn scan_f16(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for c in bytes.chunks_exact(2) {
        out.push(f16_to_f32(u16::from_le_bytes([c[0], c[1]])));
    }
    out
}

#[inline]
pub fn scan_bf16(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for c in bytes.chunks_exact(2) {
        out.push(bf16_to_f32(u16::from_le_bytes([c[0], c[1]])));
    }
    out
}

/// IEEE 754 binary16 -> binary32. Marked inline so the SIMD path
/// (F16C `_mm256_cvtph_ps`) can replace all callers under `target_feature`.
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;
    if exp == 0 {
        let val = (mant as f32) / 1024.0 * 2.0f32.powi(-14);
        return if sign == 1 { -val } else { val };
    }
    if exp == 31 {
        if mant == 0 {
            return f32::INFINITY * (if sign == 1 { -1.0 } else { 1.0 });
        }
        return f32::NAN;
    }
    let new_exp = (exp as i32 - 15 + 127) as u32;
    let bits32 = (sign << 31) | (new_exp << 23) | (mant << 13);
    f32::from_bits(bits32)
}

/// bfloat16 -> binary32. Pure left-shift, compiles to one instruction.
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

// -- Q4_0 ---------------------------------------------------------------------

#[inline]
fn dequant_q4_0(bytes: &[u8]) -> Vec<f32> {
    // 18 bytes / 32 elements: f16 d, 16 bytes of 4-bit quants
    let n_blocks = bytes.len() / 18;
    let mut out = Vec::with_capacity(n_blocks * 32);
    for blk in bytes.chunks_exact(18) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        for pair in 0..16usize {
            let q = blk[2 + pair];
            let x0 = (q & 0x0F) as i32 - 8;
            let x1 = ((q >> 4) & 0x0F) as i32 - 8;
            out.push(d * x0 as f32);
            out.push(d * x1 as f32);
        }
    }
    out
}

// -- Q4_1 ---------------------------------------------------------------------

#[inline]
fn dequant_q4_1(bytes: &[u8]) -> Vec<f32> {
    // 20 bytes / 32 elements: f16 d, f16 m, 16 bytes of 4-bit quants
    let n_blocks = bytes.len() / 20;
    let mut out = Vec::with_capacity(n_blocks * 32);
    for blk in bytes.chunks_exact(20) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let m = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
        for pair in 0..16usize {
            let q = blk[4 + pair];
            let x0 = (q & 0x0F) as f32;
            let x1 = ((q >> 4) & 0x0F) as f32;
            out.push(x0 * d + m);
            out.push(x1 * d + m);
        }
    }
    out
}

// -- Q5_0 ---------------------------------------------------------------------

#[inline]
fn dequant_q5_0(bytes: &[u8]) -> Vec<f32> {
    // 22 bytes / 32 elements: f16 d, u32 qh, 16 bytes qs
    let n_blocks = bytes.len() / 22;
    let mut out = Vec::with_capacity(n_blocks * 32);
    for blk in bytes.chunks_exact(22) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
        for pair in 0..16usize {
            let q = blk[6 + pair];
            let xh0 = ((qh >> (pair * 2)) & 1) << 4;
            let xh1 = ((qh >> (pair * 2 + 1)) & 1) << 4;
            let x0 = (q & 0x0F) as i32 | xh0 as i32;
            let x1 = ((q >> 4) & 0x0F) as i32 | xh1 as i32;
            out.push(d * (x0 as f32 - 16.0));
            out.push(d * (x1 as f32 - 16.0));
        }
    }
    out
}

// -- Q5_1 ---------------------------------------------------------------------

#[inline]
fn dequant_q5_1(bytes: &[u8]) -> Vec<f32> {
    // 24 bytes / 32 elements: f16 d, f16 m, u32 qh, 16 bytes qs
    let n_blocks = bytes.len() / 24;
    let mut out = Vec::with_capacity(n_blocks * 32);
    for blk in bytes.chunks_exact(24) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let m = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
        let qh = u32::from_le_bytes([blk[4], blk[5], blk[6], blk[7]]);
        for pair in 0..16usize {
            let q = blk[8 + pair];
            let xh0 = ((qh >> (pair * 2)) & 1) << 4;
            let xh1 = ((qh >> (pair * 2 + 1)) & 1) << 4;
            let x0 = (q & 0x0F) as f32 + xh0 as f32;
            let x1 = ((q >> 4) & 0x0F) as f32 + xh1 as f32;
            out.push(x0 * d + m);
            out.push(x1 * d + m);
        }
    }
    out
}

// -- Q8_0 ---------------------------------------------------------------------

#[inline]
fn dequant_q8_0(bytes: &[u8]) -> Vec<f32> {
    // 34 bytes / 32 elements: f16 d, 32 i8 qs
    let n_blocks = bytes.len() / 34;
    let mut out = Vec::with_capacity(n_blocks * 32);
    for blk in bytes.chunks_exact(34) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        for j in 0..32 {
            let q = blk[2 + j] as i8 as f32;
            out.push(d * q);
        }
    }
    out
}

// -- K-quants -----------------------------------------------------------------

const QK_K: usize = 256;

#[inline]
fn get_scale_min_k4(scales: &[u8; 12], j: usize) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 0x3F, scales[4 + j] & 0x3F)
    } else {
        // Canonical llama.cpp Q4_K layout (ggml-quants.c):
        //   sc[j] = (scales[8 + (j-4)] & 0x0F) | ((scales[j-4] >> 6) << 4)
        //   mn[j] = (scales[8 + (j-4)] & 0x0F) | ((scales[j]   >> 6) << 4)
        // The "shared low-4" trick: the low 4 bits of sc[j] and mn[j] are
        // the same byte (scales[8 + (j-4)]), while the high 2 bits come
        // from scales[j-4] (for sc) and scales[j] (for mn). This means
        // the encoder must satisfy the coupling: the 2-bit high part of
        // sc[j] is stored in the 6-bit slot of sc[j-4] and the 2-bit high
        // part of mn[j] is stored in the 6-bit slot of mn[j-4].
        let lbits_idx = 8 + (j - 4);
        let lo = scales[lbits_idx] & 0x0F;
        let sc = lo | ((scales[j - 4] >> 6) << 4);
        let mn = lo | ((scales[j] >> 6) << 4);
        (sc, mn)
    }
}

#[inline]
fn dequant_q4_k(bytes: &[u8]) -> Vec<f32> {
    let n_blocks = bytes.len() / 144;
    let mut out = Vec::with_capacity(n_blocks * QK_K);
    let mut sc = [0u8; 12];
    for blk in bytes.chunks_exact(144) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
        sc.copy_from_slice(&blk[4..16]);
        let qs = &blk[16..144];
        for j in 0..QK_K {
            let sub = j / 32;
            let (sc1, mn1) = get_scale_min_k4(&sc, sub);
            let d_sc = d * sc1 as f32;
            let m_sc = dmin * mn1 as f32;
            let q = if j < QK_K / 2 {
                (qs[j] & 0x0F) as f32
            } else {
                ((qs[j - QK_K / 2] >> 4) & 0x0F) as f32
            };
            out.push(d_sc * q - m_sc);
        }
    }
    out
}

#[inline]
fn get_scale_min_k5(scales: &[u8; 12], j: usize) -> (u8, u8, u8) {
    let sc_l = scales[j] & 0x3F;
    let mn_l = scales[4 + j] & 0x3F;
    let sc_h = (scales[8 + (j >> 2)] >> ((j & 3) * 2)) & 3;
    let mn_h = (scales[9 + (j >> 2)] >> ((j & 3) * 2)) & 3;
    (sc_l | (sc_h << 6), mn_l | (mn_h << 6), 0)
}

#[inline]
fn dequant_q5_k(bytes: &[u8]) -> Vec<f32> {
    let n_blocks = bytes.len() / 176;
    let mut out = Vec::with_capacity(n_blocks * QK_K);
    let mut sc = [0u8; 12];
    for blk in bytes.chunks_exact(176) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
        sc.copy_from_slice(&blk[4..16]);
        let qh = &blk[16..48];
        let qs = &blk[48..176];
        for j in 0..QK_K {
            let sub = j / 32;
            let (sc1, mn1, _) = get_scale_min_k5(&sc, sub);
            let d_sc = d * sc1 as f32;
            let m_sc = dmin * mn1 as f32;
            let h = ((qh[j / 8] >> (j % 8)) & 1) as u8;
            let q = if j < QK_K / 2 {
                (qs[j] & 0x0F) | (h << 4)
            } else {
                ((qs[j - QK_K / 2] >> 4) & 0x0F) | (h << 4)
            };
            out.push(d_sc * q as f32 - m_sc);
        }
    }
    out
}

#[inline]
fn dequant_q6_k(bytes: &[u8]) -> Vec<f32> {
    let n_blocks = bytes.len() / 210;
    let mut out = Vec::with_capacity(n_blocks * QK_K);
    for blk in bytes.chunks_exact(210) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let ql = &blk[2..130];
        let qh = &blk[130..194];
        // 16 i8 scales, one per 16-element sub-block of the 256-element super-block.
        let sc = &blk[194..210];
        for j in 0..QK_K {
            let ql_byte = ql[j / 2];
            let ql_val = if j % 2 == 0 {
                ql_byte & 0x0F
            } else {
                (ql_byte >> 4) & 0x0F
            };
            let qh_byte = qh[j / 4];
            let shift = (j % 4) * 2;
            let qh_val = (qh_byte >> shift) & 0x03;
            let q = (ql_val as i32 - 32) + (qh_val as i32) * 4;
            let s = sc[j / 16] as i8 as f32;
            out.push(d * s * q as f32);
        }
    }
    out
}

#[inline]
fn dequant_q8_k(bytes: &[u8]) -> Vec<f32> {
    let n_blocks = bytes.len() / 292;
    let mut out = Vec::with_capacity(n_blocks * QK_K);
    for blk in bytes.chunks_exact(292) {
        let d = f32::from_le_bytes([blk[0], blk[1], blk[2], blk[3]]);
        for j in 0..QK_K {
            let q = blk[4 + j] as i8 as f32;
            out.push(d * q);
        }
    }
    out
}
