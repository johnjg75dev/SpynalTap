//! Single-pass statistics for f32 weight arrays.
//!
//! Hot loop is `Accum::push`. Marked `#[inline(always)]` and branchless where
//! possible. The Welford mean/variance update has a hard data dependency
//! (each step depends on the running mean), so it stays scalar; the
//! surrounding L2/abs/bucket updates are the easy SIMD wins (see
//! `push_block`).
//!
//! Percentiles use a reservoir sample (≤ 65 536 floats) and are computed by
//! sorting only that sample.

use crate::analysis::score::BlockAnalysis;
use crate::model::Tensor;

/// All statistics we collect for one tensor (or one block).
#[derive(Debug, Clone, serde::Serialize)]
pub struct TensorStats {
    pub n: u64,
    pub n_sampled: bool,
    pub mean: f64,
    pub variance: f64,
    pub std: f64,
    pub abs_mean: f64,
    pub abs_max: f64,
    pub l2: f64,
    pub l1: f64,
    pub min: f32,
    pub max: f32,
    pub sparsity_abs: f64,
    pub outlier_ratio: f64,
    pub unique_bucket_count: u32,
    pub entropy_bits: f64,
    pub p01: f32, pub p50: f32, pub p99: f32, pub p999: f32,
}

pub const RESERVOIR_SIZE: usize = 65_536;
const SPARSITY_EPS_ABS: f32 = 1e-3;
const SPARSITY_EPS_REL: f32 = 1e-3;
const OUTLIER_K: f64 = 4.0;
const N_BUCKETS_LOG2: u32 = 12; // 4096 log-buckets

/// Top-level analysis result.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Analysis {
    pub blocks: Vec<BlockAnalysis>,
    pub recommendation: Vec<i32>,
    pub recommendation_count: usize,
    pub estimated_bytes_after_prune: u64,
    pub sample_per_tensor: usize,
    pub total_tensors: usize,
    pub total_bytes: u64,
}

#[inline]
pub fn empty_stats() -> TensorStats {
    TensorStats {
        n: 0, n_sampled: false,
        mean: 0.0, variance: 0.0, std: 0.0,
        abs_mean: 0.0, abs_max: 0.0, l2: 0.0, l1: 0.0,
        min: 0.0, max: 0.0,
        sparsity_abs: 0.0, outlier_ratio: 0.0,
        unique_bucket_count: 0, entropy_bits: 0.0,
        p01: 0.0, p50: 0.0, p99: 0.0, p999: 0.0,
    }
}

/// Single-pass Welford statistics. Call `push` repeatedly, then `finalize`.
#[derive(Default)]
pub struct Accum {
    n: u64,
    mean: f64,
    m2: f64,
    abs_sum: f64,
    abs_max: f64,
    l2: f64,
    l1: f64,
    min: f32,
    max: f32,
    #[allow(dead_code)]
    near_zero: u64,
    #[allow(dead_code)]
    far_outlier: u64,
    seen_buckets: u32,
    bucket_counts: Vec<u32>,
    reservoir: Vec<f32>,
    rng_state: u64,
}

impl Accum {
    pub fn new() -> Self {
        Self {
            n: 0,
            mean: 0.0, m2: 0.0,
            abs_sum: 0.0, abs_max: 0.0, l2: 0.0, l1: 0.0,
            min: f32::INFINITY, max: f32::NEG_INFINITY,
            near_zero: 0, far_outlier: 0,
            seen_buckets: 0,
            bucket_counts: vec![0u32; 1usize << N_BUCKETS_LOG2],
            reservoir: Vec::with_capacity(RESERVOIR_SIZE),
            rng_state: 0x9E37_79B9_7F4A_7C15,
        }
    }

    #[inline]
    fn rng_next(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.rng_state;
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        self.rng_state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Hot loop. `#[inline(always)]` so the caller (analyzer) folds in the
    /// per-element state into the dequant inner loop where possible.
    #[inline(always)]
    pub fn push(&mut self, x: f32) {
        self.n += 1;
        let xf = x as f64;
        let delta = xf - self.mean;
        self.mean += delta / self.n as f64;
        let delta2 = xf - self.mean;
        self.m2 += delta * delta2;

        let ax = x.abs() as f64;
        self.abs_sum += ax;
        if ax > self.abs_max { self.abs_max = ax; }
        self.l2 += ax * ax;
        self.l1 += ax;

        // Branchless min/max (uses cmovs/csel on x86_64/AArch64).
        self.min = self.min.min(x);
        self.max = self.max.max(x);

        // Reservoir sample.
        if self.reservoir.len() < RESERVOIR_SIZE {
            self.reservoir.push(x);
        } else {
            let j = (self.rng_next() % self.n) as usize;
            if j < RESERVOIR_SIZE {
                self.reservoir[j] = x;
            }
        }

        let key = bucket_key(x);
        if self.bucket_counts[key] == 0 {
            self.seen_buckets |= 1u32 << (key % 32);
        }
        self.bucket_counts[key] = self.bucket_counts[key].saturating_add(1);
    }

    pub fn finalize(&mut self, sparsity_eps: f32, sampled: bool) -> TensorStats {
        let n = self.n.max(1);
        let variance = self.m2 / n as f64;
        let std = variance.sqrt();
        let abs_mean = self.abs_sum / n as f64;
        let l2 = self.l2.sqrt();
        let l1 = self.l1;

        if !self.reservoir.is_empty() {
            let rlen = self.reservoir.len() as f64;
            let mut nz = 0u64;
            let mut fo = 0u64;
            for &v in &self.reservoir {
                if (v as f64).abs() < sparsity_eps as f64 { nz += 1; }
                if (v as f64).abs() > abs_mean + OUTLIER_K * std { fo += 1; }
            }
            let sparsity_abs = nz as f64 / rlen;
            let outlier_ratio = fo as f64 / rlen;

            self.reservoir.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let p01  = pct(&self.reservoir, 0.001);
            let p50  = pct(&self.reservoir, 0.50);
            let p99  = pct(&self.reservoir, 0.99);
            let p999 = pct(&self.reservoir, 0.999);

            let mut unique = 0u32;
            let mut entropy = 0.0f64;
            let total: u64 = self.bucket_counts.iter().map(|&b| b as u64).sum();
            if total > 0 {
                for &b in &self.bucket_counts {
                    if b == 0 { continue; }
                    unique += 1;
                    let p = b as f64 / total as f64;
                    entropy -= p * p.log2();
                }
            }

            return TensorStats {
                n, n_sampled: sampled,
                mean: self.mean, variance, std,
                abs_mean, abs_max: self.abs_max, l2, l1,
                min: if self.min == f32::INFINITY { 0.0 } else { self.min },
                max: if self.max == f32::NEG_INFINITY { 0.0 } else { self.max },
                sparsity_abs, outlier_ratio,
                unique_bucket_count: unique, entropy_bits: entropy,
                p01, p50, p99, p999,
            };
        }
        empty_stats()
    }
}

#[inline]
fn pct(sorted: &[f32], q: f64) -> f32 {
    if sorted.is_empty() { return 0.0; }
    let idx = ((sorted.len() as f64 - 1.0) * q) as usize;
    sorted[idx]
}

#[inline]
fn bucket_key(x: f32) -> usize {
    if x == 0.0 { return 0; }
    if !x.is_finite() { return (1usize << N_BUCKETS_LOG2) - 1; }
    let bits = x.to_bits();
    let exp = ((bits >> 23) & 0xFF) as i32 - 127;
    (exp + 1024).clamp(0, (1 << N_BUCKETS_LOG2) - 2) as usize
}

/// Per-tensor sparsity threshold (used by the analyzer).
#[inline]
pub fn sparsity_eps_for(amax: f32) -> f32 {
    (amax * SPARSITY_EPS_REL).max(SPARSITY_EPS_ABS)
}

/// Just a re-export so `analysis::*` has a `Tensor` symbol; the analyzer
/// operates on `&[Tensor]` from the Model trait.
#[allow(dead_code)]
pub type _TensorAlias = Tensor;
