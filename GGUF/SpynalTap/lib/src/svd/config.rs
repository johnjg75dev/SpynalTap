//! SVD-compression configuration.
//!
//! ## Layer selection grammar
//!
//! ```text
//!   all                          — every block
//!   0-23                         — inclusive range
//!   0,1,2                        — explicit list
//!   0-5,10,20-22                 — combinations
//!   regex:^blk\.(0|1|2)\.       — by regex (matched against tensor name)
//!   all-attn                     — alias for all 2D attention projections
//!   all-ffn                      — alias for all 2D FFN projections
//!   all-mlp                      — alias for all 2D attention + FFN projections
//! ```
//!
//! ## Tensor selection grammar (per selected layer)
//!
//! ```text
//!   attn                         — attn_q, attn_k, attn_v, attn_output
//!   ffn                          — ffn_up, ffn_down, ffn_gate, ffn_gate_up
//!   mlp                          — same as attn+ffn
//!   attn_q,attn_v                — explicit list of suffixes
//!   regex:^.*\.weight$           — by regex (matched against tensor name suffix after `blk.N.`)
//!   all                          — any 2D weight
//! ```
//!
//! ## Rank specification grammar
//!
//! ```text
//!   64                           — absolute rank for every selected tensor
//!   0.5                          — fraction of min(m, n) (50% in this example)
//!   energy:0.99                  — keep enough singular values to retain 99% of squared-singular-value sum
//!   abs:64,min:8,max:512         — absolute rank, with floor/ceiling clamps
//!   frac:0.5,min:8,max:512       — fractional with clamps
//! ```
//!
//! ## Output dtype
//!
//! ```text
//!   f32, f16, bf16               — element type for the packed (A, B) factors
//! ```

use crate::error::{Error, Result};
use regex::Regex;
use std::collections::BTreeMap;

/// Which layers (transformer blocks) the SVD should target.
#[derive(Debug, Clone)]
pub enum LayerSelection {
    /// Every block index found in the model.
    All,
    /// A list of explicit block indices.
    Indices(Vec<i32>),
    /// A regex matched against each tensor's full name.
    Pattern(Regex),
    /// Convenience: all attention projections (attn_q, attn_k, attn_v, attn_output) in every block.
    AllAttn,
    /// Convenience: all FFN projections (ffn_up, ffn_down, ffn_gate, ffn_gate_up) in every block.
    AllFfn,
    /// Convenience: all attention + FFN projections in every block.
    AllMlp,
}

impl LayerSelection {
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim();
        if s == "all" { return Ok(Self::All); }
        if s == "all-attn" { return Ok(Self::AllAttn); }
        if s == "all-ffn" { return Ok(Self::AllFfn); }
        if s == "all-mlp" { return Ok(Self::AllMlp); }
        if let Some(rest) = s.strip_prefix("regex:") {
            let re = Regex::new(rest).map_err(|e| Error::InvalidSvdConfig(format!("bad layer regex: {e}")))?;
            return Ok(Self::Pattern(re));
        }
        // Default: list / range of indices.
        let idx = crate::prune::selection::parse_index_list(s)?;
        if idx.is_empty() {
            return Err(Error::InvalidSvdConfig(format!("empty layer list in '{s}'")));
        }
        Ok(Self::Indices(idx))
    }
}

/// Which tensors within each selected layer should be compressed.
#[derive(Debug, Clone)]
pub enum TensorSelection {
    /// Any 2D `.weight` tensor (skips 1D vectors and quantization blocks that aren't 2D matrices).
    All,
    /// Suffix list (matched against the part after `blk.N.`).
    Named(Vec<String>),
    /// Regex matched against the full tensor name.
    Pattern(Regex),
    /// Convenience: attention projections.
    Attn,
    /// Convenience: FFN projections.
    Ffn,
    /// Convenience: attention + FFN.
    Mlp,
}

impl TensorSelection {
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim();
        if s == "all" { return Ok(Self::All); }
        if s == "attn" { return Ok(Self::Attn); }
        if s == "ffn" { return Ok(Self::Ffn); }
        if s == "mlp" { return Ok(Self::Mlp); }
        if let Some(rest) = s.strip_prefix("regex:") {
            let re = Regex::new(rest).map_err(|e| Error::InvalidSvdConfig(format!("bad tensor regex: {e}")))?;
            return Ok(Self::Pattern(re));
        }
        // Comma-separated list of suffixes.
        let mut v = Vec::new();
        for part in s.split(',') {
            let p = part.trim();
            if p.is_empty() { continue; }
            v.push(p.to_string());
        }
        if v.is_empty() {
            return Err(Error::InvalidSvdConfig(format!("empty tensor list in '{s}'")));
        }
        Ok(Self::Named(v))
    }

    /// Returns true if a tensor with full name `full` (e.g. `blk.3.attn_q.weight`)
    /// matches this selection.
    pub fn matches(&self, full: &str) -> bool {
        match self {
            Self::All => is_2d_weight(full),
            Self::Attn => suffix_in(full, ATTN_SUFFIXES),
            Self::Ffn => suffix_in(full, FFN_SUFFIXES),
            Self::Mlp => suffix_in(full, ATTN_SUFFIXES) || suffix_in(full, FFN_SUFFIXES),
            Self::Named(suffixes) => suffixes.iter().any(|s| full.contains(s)),
            Self::Pattern(re) => re.is_match(full),
        }
    }
}

/// Rank specification. May carry optional floor/ceiling clamps.
#[derive(Debug, Clone)]
pub enum RankSpec {
    /// Absolute rank `k` for every selected tensor.
    Absolute(usize),
    /// Fraction of `min(m, n)` for every selected tensor.
    Fraction(f64),
    /// Keep the smallest `k` such that `sum_{i<k} s_i^2 >= energy * total`.
    Energy(f64),
}

#[derive(Debug, Clone)]
pub struct RankClamps {
    pub min: usize,
    pub max: Option<usize>,
}

impl Default for RankClamps {
    fn default() -> Self { Self { min: 1, max: None } }
}

#[derive(Debug, Clone)]
pub struct RankSpecWithClamps {
    pub spec: RankSpec,
    pub clamps: RankClamps,
}

impl RankSpecWithClamps {
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim();
        // Parse the leading rank spec; collect trailing ,key:val clamps.
        let mut parts = s.split(',').map(str::trim).filter(|p| !p.is_empty());
        let head = parts.next().ok_or_else(|| Error::InvalidSvdConfig("empty rank spec".into()))?;
        let mut clamps = RankClamps::default();
        let spec: Option<RankSpec>;
        if let Some(rest) = head.strip_prefix("energy:") {
            let e: f64 = rest.parse().map_err(|e| Error::InvalidSvdConfig(format!("bad energy '{rest}': {e}")))?;
            if !(0.0..=1.0).contains(&e) {
                return Err(Error::InvalidSvdConfig(format!("energy must be in [0,1], got {e}")));
            }
            spec = Some(RankSpec::Energy(e));
        } else if let Some(rest) = head.strip_prefix("abs:") {
            let n: usize = rest.parse().map_err(|e| Error::InvalidSvdConfig(format!("bad abs rank '{rest}': {e}")))?;
            spec = Some(RankSpec::Absolute(n));
        } else if let Some(rest) = head.strip_prefix("frac:") {
            let n: f64 = rest.parse().map_err(|e| Error::InvalidSvdConfig(format!("bad frac rank '{rest}': {e}")))?;
            if !(0.0..=1.0).contains(&n) {
                return Err(Error::InvalidSvdConfig(format!("frac must be in [0,1], got {n}")));
            }
            spec = Some(RankSpec::Fraction(n));
        } else {
            if let Ok(n) = head.parse::<usize>() {
                spec = Some(RankSpec::Absolute(n));
            } else if let Ok(f) = head.parse::<f64>() {
                if !(0.0..=1.0).contains(&f) {
                    return Err(Error::InvalidSvdConfig(format!("rank must be int or fraction in [0,1], got {f}")));
                }
                spec = Some(RankSpec::Fraction(f));
            } else {
                return Err(Error::InvalidSvdConfig(format!("unrecognized rank '{head}'")));
            }
        }
        for p in parts {
            if let Some(rest) = p.strip_prefix("min:") {
                clamps.min = rest.parse().map_err(|e| Error::InvalidSvdConfig(format!("bad min: {e}")))?;
            } else if let Some(rest) = p.strip_prefix("max:") {
                clamps.max = Some(rest.parse().map_err(|e| Error::InvalidSvdConfig(format!("bad max: {e}")))?);
            } else {
                return Err(Error::InvalidSvdConfig(format!("unknown rank option '{p}'")));
            }
        }
        Ok(Self { spec: spec.expect("rank spec must be set above"), clamps })
    }

    /// Apply the spec to a tensor of shape `m x n`, given a precomputed
    /// spectrum `s` (required for `Energy`, ignored otherwise).
    pub fn resolve(&self, m: usize, n: usize, s: Option<&[f32]>) -> usize {
        let max_possible = m.min(n).max(1);
        let raw = match &self.spec {
            RankSpec::Absolute(k) => *k,
            RankSpec::Fraction(f) => ((max_possible as f64) * f).floor() as usize,
            RankSpec::Energy(e) => {
                let s = s.unwrap_or(&[]);
                super::linalg::rank_for_energy(s, *e, 1, max_possible)
            }
        };
        let lo = self.clamps.min.max(1);
        let hi = self.clamps.max.unwrap_or(max_possible).min(max_possible);
        raw.clamp(lo, hi.max(lo))
    }
}

/// Output element type for the packed (A, B) factors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputDtype {
    F32, F16, Bf16,
}

impl OutputDtype {
    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "f32" | "float32" => Ok(Self::F32),
            "f16" | "float16" | "fp16" | "half" => Ok(Self::F16),
            "bf16" | "bfloat16" => Ok(Self::Bf16),
            other => Err(Error::InvalidSvdConfig(format!("unknown dtype '{other}' (want f32/f16/bf16)"))),
        }
    }
    pub fn as_str(self) -> &'static str {
        match self { Self::F32 => "F32", Self::F16 => "F16", Self::Bf16 => "BF16" }
    }
    pub fn is_supported_for_ggml(self) -> bool {
        matches!(self, Self::F32 | Self::F16 | Self::Bf16)
    }
}

/// Top-level SVD compression configuration.
#[derive(Debug, Clone)]
pub struct SvdConfig {
    pub layers: LayerSelection,
    pub tensors: TensorSelection,
    pub rank: RankSpecWithClamps,
    pub dtype: OutputDtype,
    /// Minimum size of a tensor (min(m, n)) to be eligible. Smaller tensors are skipped.
    pub min_dim: usize,
    /// Random-seeded randomized SVD for large matrices.
    pub randomized: bool,
    /// Randomized SVD oversampling (extra columns in the test matrix).
    pub randomized_oversample: usize,
    /// Randomized SVD power iterations.
    pub randomized_power_iters: usize,
    /// Threshold (in elements) above which randomized SVD is used (when enabled).
    pub randomized_min_elems: usize,
    /// Suffix appended to the original name to form the "A" (tall) factor.
    pub suffix_a: String,
    /// Suffix appended to the original name to form the "B" (wide) factor.
    pub suffix_b: String,
    /// Per-layer rank overrides (block index -> rank spec).
    pub per_layer: BTreeMap<i32, RankSpecWithClamps>,
    /// Per-tensor-suffix rank overrides (matched substring -> rank spec).
    pub per_tensor: Vec<(String, RankSpecWithClamps)>,
}

impl Default for SvdConfig {
    fn default() -> Self {
        Self {
            layers: LayerSelection::All,
            tensors: TensorSelection::Mlp,
            rank: RankSpecWithClamps {
                spec: RankSpec::Fraction(0.5),
                clamps: RankClamps { min: 4, max: None },
            },
            dtype: OutputDtype::F16,
            min_dim: 16,
            randomized: true,
            randomized_oversample: 8,
            randomized_power_iters: 2,
            randomized_min_elems: 1 << 18, // 256K elems
            suffix_a: ".svd_a".into(),
            suffix_b: ".svd_b".into(),
            per_layer: BTreeMap::new(),
            per_tensor: Vec::new(),
        }
    }
}

impl SvdConfig {
    /// Resolve the effective rank for a single tensor.
    pub fn resolve_rank(&self, name: &str, block_idx: i32, m: usize, n: usize, s: Option<&[f32]>) -> usize {
        // 1) per-tensor override (first match wins)
        for (needle, spec) in &self.per_tensor {
            if name.contains(needle) {
                return spec.resolve(m, n, s);
            }
        }
        // 2) per-layer override
        if let Some(spec) = self.per_layer.get(&block_idx) {
            return spec.resolve(m, n, s);
        }
        // 3) global spec
        self.rank.resolve(m, n, s)
    }

    /// Build the `(a, b)` factor names from an original tensor name.
    pub fn factor_names(&self, original: &str) -> (String, String) {
        (format!("{original}{}", self.suffix_a), format!("{original}{}", self.suffix_b))
    }
}

// -- Helpers used by TensorSelection / apply --------------------------------

/// Common attention projection suffixes (lowercase, after `blk.N.`).
pub const ATTN_SUFFIXES: &[&str] = &[
    ".attn_q.weight",
    ".attn_k.weight",
    ".attn_v.weight",
    ".attn_output.weight",
    ".attn_qkv.weight",
];

/// Common FFN / MLP projection suffixes.
pub const FFN_SUFFIXES: &[&str] = &[
    ".ffn_up.weight",
    ".ffn_down.weight",
    ".ffn_gate.weight",
    ".ffn_gate_up.weight",
    ".ffn_up_exps.weight",
    ".ffn_down_exps.weight",
    ".ffn_gate_exps.weight",
];

#[inline]
pub fn is_2d_weight(name: &str) -> bool {
    name.ends_with(".weight") && !name.contains("norm") && !name.contains("rope")
}

#[inline]
pub(crate) fn suffix_in(name: &str, suffixes: &[&str]) -> bool {
    suffixes.iter().any(|s| name.ends_with(s))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_layers_all() { assert!(matches!(LayerSelection::parse("all").unwrap(), LayerSelection::All)); }
    #[test]
    fn parse_layers_range() {
        match LayerSelection::parse("0-3,7").unwrap() {
            LayerSelection::Indices(v) => assert_eq!(v, vec![0, 1, 2, 3, 7]),
            _ => panic!(),
        }
    }
    #[test]
    fn parse_layers_alias() {
        assert!(matches!(LayerSelection::parse("all-mlp").unwrap(), LayerSelection::AllMlp));
    }
    #[test]
    fn parse_tensors_attn() {
        assert!(TensorSelection::parse("attn").unwrap().matches("blk.0.attn_q.weight"));
        assert!(!TensorSelection::parse("attn").unwrap().matches("blk.0.ffn_up.weight"));
    }
    #[test]
    fn parse_rank_int() {
        let r = RankSpecWithClamps::parse("64").unwrap();
        assert_eq!(r.resolve(100, 100, None), 64);
    }
    #[test]
    fn parse_rank_frac() {
        let r = RankSpecWithClamps::parse("0.5,min:4,max:200").unwrap();
        assert_eq!(r.resolve(100, 100, None), 50);
        assert_eq!(r.resolve(8, 8, None), 4);
    }
    #[test]
    fn parse_rank_energy() {
        let s = vec![10.0, 9.0, 1.0, 0.1];
        let r = RankSpecWithClamps::parse("energy:0.99").unwrap();
        // squared s: 100, 81, 1, 0.01 -> total 182.01. 99% threshold = 180.19.
        // First two (181) already exceed 180.19, so k=2.
        assert_eq!(r.resolve(10, 10, Some(&s)), 2);

        // 0.9999: 99.99% of 182.01 = 181.998 -> k=3 (sum 182 > 181.998).
        let r2 = RankSpecWithClamps::parse("energy:0.9999").unwrap();
        assert_eq!(r2.resolve(10, 10, Some(&s)), 3);

        // 0.5 needs only the dominant singular value.
        let r3 = RankSpecWithClamps::parse("energy:0.5").unwrap();
        assert_eq!(r3.resolve(10, 10, Some(&s)), 1);
    }
    #[test]
    fn parse_dtype() {
        assert_eq!(OutputDtype::parse("f16").unwrap(), OutputDtype::F16);
        assert_eq!(OutputDtype::parse("bf16").unwrap(), OutputDtype::Bf16);
        assert!(OutputDtype::parse("garbage").is_err());
    }
    #[test]
    fn factor_names() {
        let mut cfg = SvdConfig::default();
        cfg.suffix_a = ".lora_a".into();
        cfg.suffix_b = ".lora_b".into();
        let (a, b) = cfg.factor_names("blk.5.attn_q.weight");
        assert_eq!(a, "blk.5.attn_q.weight.lora_a");
        assert_eq!(b, "blk.5.attn_q.weight.lora_b");
    }
}
