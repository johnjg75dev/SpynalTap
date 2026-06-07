//! Build an `SvdPlan` from an `SvdConfig` and a model.
//!
//! The plan is the data structure the writer consumes. For each eligible
//! tensor we record the original name, the resolved rank, and the names of
//! the two new factors (A: m x k, B: k x n).

use crate::analysis::score::{classify, BlockRole};
use crate::error::Result;
use crate::model::Model;
use crate::svd::config::{is_2d_weight, LayerSelection, SvdConfig};
use std::collections::HashSet;

/// One tensor to be replaced by an (A, B) low-rank factorization.
#[derive(Debug, Clone)]
pub struct SvdTarget {
    pub name: String,
    pub name_a: String,
    pub name_b: String,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub orig_bytes: u64,
    pub new_bytes: u64,
}

/// Concrete plan describing every compression to perform.
#[derive(Debug, Clone)]
pub struct SvdPlan {
    pub targets: Vec<SvdTarget>,
    pub skipped: Vec<SkippedTensor>,
    pub config: SvdConfig,
    pub original_block_count: i32,
}

#[derive(Debug, Clone)]
pub struct SkippedTensor {
    pub name: String,
    pub reason: String,
}

pub fn build_plan<M: Model + ?Sized>(model: &M, cfg: &SvdConfig) -> Result<SvdPlan> {
    // 1) Discover all block indices present in the model.
    let mut all_blocks: HashSet<i32> = HashSet::new();
    for t in model.tensors() {
        let (role, idx, _) = classify(&t.name);
        if role == BlockRole::Block {
            all_blocks.insert(idx);
        }
    }
    let mut sorted_blocks: Vec<i32> = all_blocks.into_iter().collect();
    sorted_blocks.sort();

    // 2) Build the set of allowed block indices from the layer selection.
    let allowed_blocks: Option<HashSet<i32>> = match &cfg.layers {
        LayerSelection::All
        | LayerSelection::AllAttn
        | LayerSelection::AllFfn
        | LayerSelection::AllMlp => None,
        LayerSelection::Indices(v) => Some(v.iter().copied().collect()),
        LayerSelection::Pattern(_) => None, // matched per-tensor below
    };

    let mut targets = Vec::new();
    let mut skipped = Vec::new();

    for t in model.tensors() {
        // Layer filter
        let (role, idx, _) = classify(&t.name);
        if role != BlockRole::Block {
            continue;
        }
        if let Some(set) = &allowed_blocks {
            if !set.contains(&idx) {
                continue;
            }
        }
        if let LayerSelection::Pattern(re) = &cfg.layers {
            if !re.is_match(&t.name) {
                continue;
            }
        }
        // Convenience aliases: only act on the relevant suffix family.
        match &cfg.layers {
            LayerSelection::AllAttn | LayerSelection::AllFfn | LayerSelection::AllMlp => {
                let ok = match &cfg.layers {
                    LayerSelection::AllAttn => {
                        cfg.tensors.matches(&t.name)
                            && crate::svd::config::suffix_in(
                                &t.name,
                                crate::svd::config::ATTN_SUFFIXES,
                            )
                    }
                    LayerSelection::AllFfn => {
                        crate::svd::config::suffix_in(&t.name, crate::svd::config::FFN_SUFFIXES)
                    }
                    LayerSelection::AllMlp => {
                        crate::svd::config::suffix_in(&t.name, crate::svd::config::ATTN_SUFFIXES)
                            || crate::svd::config::suffix_in(
                                &t.name,
                                crate::svd::config::FFN_SUFFIXES,
                            )
                    }
                    _ => unreachable!(),
                };
                if !ok {
                    continue;
                }
            }
            _ => {}
        }

        // Tensor filter
        if !cfg.tensors.matches(&t.name) {
            continue;
        }
        if !is_2d_weight(&t.name) {
            continue;
        }

        // Shape filter (2D weight matrix m x n)
        let shape: Vec<u64> = t.shape.iter().copied().filter(|&d| d > 0).collect();
        if shape.len() != 2 {
            skipped.push(SkippedTensor {
                name: t.name.clone(),
                reason: format!("non-2D shape {:?}", t.shape),
            });
            continue;
        }
        let m = shape[0] as usize;
        let n = shape[1] as usize;

        if m.min(n) < cfg.min_dim {
            skipped.push(SkippedTensor {
                name: t.name.clone(),
                reason: format!("min dim {} < {}", m.min(n), cfg.min_dim),
            });
            continue;
        }

        // For Energy rank specs we have to read the full tensor to compute S.
        // For other specs we can resolve the rank from shape alone.
        let (name_a, name_b) = cfg.factor_names(&t.name);
        let k = match &cfg.rank.spec {
            crate::svd::config::RankSpec::Energy(_) => {
                // Defer rank resolution to apply time (where we already have the bytes).
                // Plan stores k = 0 as a sentinel; apply replaces it.
                0
            }
            _ => cfg.resolve_rank(&t.name, idx, m, n, None),
        };
        // Compute output element size for byte estimate.
        let esz = match cfg.dtype {
            crate::svd::config::OutputDtype::F32 => 4,
            crate::svd::config::OutputDtype::F16
            | crate::svd::config::OutputDtype::Bf16
            | crate::svd::config::OutputDtype::AutoQuant => 2,
            crate::svd::config::OutputDtype::Ggml(t) => {
                // Quantized: 1 byte per 32 values is a safe lower-bound estimate
                // for the compression_ratio reporting. Real byte size is
                // computed precisely in apply.rs.
                t.block_bytes().unwrap_or(34) as u64 / 32
            }
        };
        let new_bytes = ((m as u64 * k as u64) + (k as u64 * n as u64)) * esz;
        targets.push(SvdTarget {
            name: t.name.clone(),
            name_a,
            name_b,
            m,
            n,
            k,
            orig_bytes: t.byte_size,
            new_bytes,
        });
    }

    targets.sort_by(|a, b| a.name.cmp(&b.name));

    // 3) Adjacent pass: for every primary target, add targets for each
    //    (block_idx + offset, role) pair from cfg.adjacent. Adjacent
    //    targets bypass the layer + tensor selection filters; the user is
    //    explicitly saying "also compress this". Out-of-range offsets are
    //    recorded as SkippedTensor rather than failing the whole plan.
    if let Some(adj) = &cfg.adjacent {
        let (min_block, max_block) = match (sorted_blocks.first(), sorted_blocks.last()) {
            (Some(lo), Some(hi)) => (*lo, *hi),
            _ => (0, -1), // no blocks in the model
        };
        let mut existing: HashSet<String> = targets.iter().map(|t| t.name.clone()).collect();
        let mut added: Vec<SvdTarget> = Vec::new();
        for primary in &targets {
            let (role, primary_idx, _) = classify(&primary.name);
            if role != BlockRole::Block {
                continue;
            }
            for entry in &adj.entries {
                let adj_idx = primary_idx + entry.offset;
                let adj_name =
                    format!("blk.{adj_idx}.{}.weight", entry.role.as_str());

                if adj_idx < min_block || adj_idx > max_block {
                    skipped.push(SkippedTensor {
                        name: adj_name,
                        reason: "out-of-range block offset".into(),
                    });
                    continue;
                }
                if existing.contains(&adj_name) {
                    // Either already a primary, or already added by a
                    // previous iteration. Keep one copy.
                    continue;
                }
                let t = match model.tensor(&adj_name) {
                    Some(t) => t,
                    None => {
                        skipped.push(SkippedTensor {
                            name: adj_name,
                            reason: "adjacent tensor not found in model".into(),
                        });
                        continue;
                    }
                };
                let shape: Vec<u64> = t.shape.iter().copied().filter(|&d| d > 0).collect();
                if shape.len() != 2 {
                    skipped.push(SkippedTensor {
                        name: adj_name,
                        reason: format!("non-2D shape {:?}", t.shape),
                    });
                    continue;
                }
                let m = shape[0] as usize;
                let n = shape[1] as usize;
                if m.min(n) < cfg.min_dim {
                    skipped.push(SkippedTensor {
                        name: adj_name,
                        reason: format!("min dim {} < {}", m.min(n), cfg.min_dim),
                    });
                    continue;
                }
                if !is_2d_weight(&t.name) {
                    skipped.push(SkippedTensor {
                        name: adj_name,
                        reason: "not a 2D weight".into(),
                    });
                    continue;
                }
                let (name_a, name_b) = cfg.factor_names(&t.name);
                let k = match &cfg.rank.spec {
                    crate::svd::config::RankSpec::Energy(_) => 0,
                    _ => cfg.resolve_rank(&t.name, adj_idx, m, n, None),
                };
                let esz = match cfg.dtype {
                    crate::svd::config::OutputDtype::F32 => 4,
                    crate::svd::config::OutputDtype::F16
                    | crate::svd::config::OutputDtype::Bf16
                    | crate::svd::config::OutputDtype::AutoQuant => 2,
                    crate::svd::config::OutputDtype::Ggml(t) => {
                        t.block_bytes().unwrap_or(34) as u64 / 32
                    }
                };
                let new_bytes = ((m as u64 * k as u64) + (k as u64 * n as u64)) * esz;
                existing.insert(adj_name.clone());
                added.push(SvdTarget {
                    name: adj_name,
                    name_a,
                    name_b,
                    m,
                    n,
                    k,
                    orig_bytes: t.byte_size,
                    new_bytes,
                });
            }
        }
        targets.extend(added);
        targets.sort_by(|a, b| a.name.cmp(&b.name));
    }

    Ok(SvdPlan {
        targets,
        skipped,
        config: cfg.clone(),
        original_block_count: sorted_blocks.len() as i32,
    })
}

impl SvdPlan {
    pub fn orig_bytes(&self) -> u64 {
        self.targets.iter().map(|t| t.orig_bytes).sum()
    }
    pub fn new_bytes(&self) -> u64 {
        self.targets.iter().map(|t| t.new_bytes).sum()
    }
    pub fn compression_ratio(&self) -> f64 {
        let o = self.orig_bytes() as f64;
        if o == 0.0 {
            0.0
        } else {
            1.0 - (self.new_bytes() as f64 / o)
        }
    }
    /// Names of tensors that the writer should drop from the source.
    pub fn dropped_names(&self) -> HashSet<&str> {
        self.targets.iter().map(|t| t.name.as_str()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{MetadataValue, Model, ModelFormat, Tensor, TensorDtype};
    use crate::svd::config::TensorSelection;
    use std::borrow::Cow;

    struct FakeModel {
        tensors: Vec<Tensor>,
    }
    impl Model for FakeModel {
        fn format(&self) -> ModelFormat {
            ModelFormat::Gguf
        }
        fn name(&self) -> Option<&str> {
            Some("fake")
        }
        fn architecture(&self) -> Option<&str> {
            Some("llama")
        }
        fn block_count(&self) -> Option<usize> {
            Some(2)
        }
        fn tensors(&self) -> &[Tensor] {
            &self.tensors
        }
        fn tensor(&self, name: &str) -> Option<&Tensor> {
            self.tensors.iter().find(|t| t.name == name)
        }
        fn metadata(&self, _: &str) -> Option<MetadataValue<'_>> {
            None
        }
        fn read_tensor_bytes(&self, _: &str) -> Result<Cow<'_, [u8]>> {
            Ok(Cow::Borrowed(&[]))
        }
    }

    fn t(name: &str, m: u64, n: u64) -> Tensor {
        Tensor {
            name: name.into(),
            dtype: TensorDtype::F32,
            shape: vec![m, n],
            byte_size: m * n * 4,
            data_offset: 0,
        }
    }

    #[test]
    fn plan_attn_ffn_frac() {
        let m = FakeModel {
            tensors: vec![
                t("blk.0.attn_q.weight", 64, 64),
                t("blk.0.attn_k.weight", 64, 64),
                t("blk.0.attn_v.weight", 64, 64),
                t("blk.0.attn_output.weight", 64, 64),
                t("blk.0.ffn_up.weight", 64, 128),
                t("blk.0.ffn_down.weight", 128, 64),
                t("blk.0.ffn_gate.weight", 64, 128),
                t("blk.0.attn_norm.weight", 64, 1), // 1D, skip
                t("blk.0.token_embd", 64, 1),       // 1D, skip
                t("output.weight", 32, 64),         // not a block, skip
            ],
        };
        let mut cfg = SvdConfig::default();
        cfg.layers = LayerSelection::All;
        cfg.tensors = TensorSelection::Mlp;
        cfg.rank = crate::svd::config::RankSpecWithClamps {
            spec: crate::svd::config::RankSpec::Fraction(0.5),
            clamps: crate::svd::config::RankClamps { min: 1, max: None },
        };
        let plan = build_plan(&m, &cfg).unwrap();
        // 4 attn + 3 ffn = 7 targets
        assert_eq!(plan.targets.len(), 7);
        // attn_q: 64x64, frac 0.5 -> k=32
        let q = plan
            .targets
            .iter()
            .find(|t| t.name == "blk.0.attn_q.weight")
            .unwrap();
        assert_eq!(q.k, 32);
        // ffn_up: 64x128, frac 0.5 -> k=32
        let u = plan
            .targets
            .iter()
            .find(|t| t.name == "blk.0.ffn_up.weight")
            .unwrap();
        assert_eq!(u.k, 32);
        // output.weight is a non-block, skipped
        assert!(plan.targets.iter().all(|t| !t.name.starts_with("output")));
    }

    #[test]
    fn plan_per_layer_override() {
        let m = FakeModel {
            tensors: vec![
                t("blk.0.attn_q.weight", 64, 64),
                t("blk.1.attn_q.weight", 64, 64),
            ],
        };
        let mut cfg = SvdConfig::default();
        cfg.layers = LayerSelection::All;
        cfg.tensors = TensorSelection::Attn;
        cfg.rank = crate::svd::config::RankSpecWithClamps {
            spec: crate::svd::config::RankSpec::Absolute(8),
            clamps: crate::svd::config::RankClamps { min: 1, max: None },
        };
        cfg.per_layer.insert(
            1,
            crate::svd::config::RankSpecWithClamps {
                spec: crate::svd::config::RankSpec::Absolute(16),
                clamps: crate::svd::config::RankClamps { min: 1, max: None },
            },
        );
        let plan = build_plan(&m, &cfg).unwrap();
        let t0 = plan
            .targets
            .iter()
            .find(|t| t.name == "blk.0.attn_q.weight")
            .unwrap();
        let t1 = plan
            .targets
            .iter()
            .find(|t| t.name == "blk.1.attn_q.weight")
            .unwrap();
        assert_eq!(t0.k, 8);
        assert_eq!(t1.k, 16);
    }

    #[test]
    fn plan_skips_small_min_dim() {
        let m = FakeModel {
            tensors: vec![t("blk.0.attn_q.weight", 8, 8)],
        };
        let mut cfg = SvdConfig::default();
        cfg.layers = LayerSelection::All;
        cfg.tensors = TensorSelection::Attn;
        cfg.min_dim = 16;
        let plan = build_plan(&m, &cfg).unwrap();
        assert_eq!(plan.targets.len(), 0);
        assert_eq!(plan.skipped.len(), 1);
    }

    #[test]
    fn plan_layer_pattern() {
        let m = FakeModel {
            tensors: vec![
                t("blk.0.attn_q.weight", 64, 64),
                t("blk.5.attn_q.weight", 64, 64),
            ],
        };
        let mut cfg = SvdConfig::default();
        cfg.layers = LayerSelection::parse(r"regex:^blk\.0\.").unwrap();
        cfg.tensors = TensorSelection::Attn;
        let plan = build_plan(&m, &cfg).unwrap();
        assert_eq!(plan.targets.len(), 1);
        assert_eq!(plan.targets[0].name, "blk.0.attn_q.weight");
    }

    // ---- AdjacentSelection parser ----------------------------------------

    use crate::svd::config::AdjacentSelection;

    fn adj(s: &str) -> AdjacentSelection {
        AdjacentSelection::parse(s)
            .unwrap()
            .expect("expected Some(AdjacentSelection) for non-empty input")
    }

    #[test]
    fn adjacent_parse_empty_returns_none() {
        assert!(AdjacentSelection::parse("").unwrap().is_none());
        assert!(AdjacentSelection::parse("   ").unwrap().is_none());
    }

    #[test]
    fn adjacent_parse_single_role_default_offset() {
        let a = adj("attn_v");
        assert_eq!(a.entries.len(), 1);
        assert_eq!(a.entries[0].role, crate::svd::config::AdjacentRole::AttnV);
        assert_eq!(a.entries[0].offset, 0);
    }

    #[test]
    fn adjacent_parse_single_role_positive_offset() {
        let a = adj("attn_v+1");
        assert_eq!(a.entries.len(), 1);
        assert_eq!(a.entries[0].role, crate::svd::config::AdjacentRole::AttnV);
        assert_eq!(a.entries[0].offset, 1);
    }

    #[test]
    fn adjacent_parse_single_role_negative_offset() {
        let a = adj("attn_v-1");
        assert_eq!(a.entries.len(), 1);
        assert_eq!(a.entries[0].role, crate::svd::config::AdjacentRole::AttnV);
        assert_eq!(a.entries[0].offset, -1);
    }

    #[test]
    fn adjacent_parse_multiple_roles_and_offsets() {
        // attn_v  +  ffn_gate-1  +  1   -> two entries:
        //   (AttnV, 0), (FfnGate, 1)
        let a = adj("attn_v+ffn_gate-1+1");
        assert_eq!(a.entries.len(), 2);
        assert_eq!(a.entries[0].role, crate::svd::config::AdjacentRole::AttnV);
        assert_eq!(a.entries[0].offset, 0);
        assert_eq!(a.entries[1].role, crate::svd::config::AdjacentRole::FfnGate);
        assert_eq!(a.entries[1].offset, 1);
    }

    #[test]
    fn adjacent_parse_attn_v_plus_ffn_gate_plus_zero() {
        // Each role gets its own offset; trailing +0 is a no-op for the
        // last role.
        let a = adj("attn_v+ffn_gate+0");
        assert_eq!(a.entries.len(), 2);
        assert_eq!(a.entries[0].role, crate::svd::config::AdjacentRole::AttnV);
        assert_eq!(a.entries[0].offset, 0);
        assert_eq!(a.entries[1].role, crate::svd::config::AdjacentRole::FfnGate);
        assert_eq!(a.entries[1].offset, 0);
    }

    #[test]
    fn adjacent_parse_ffn_gate_up_not_confused_with_ffn_gate() {
        // Longest role must match first so ffn_gate_up doesn't get parsed
        // as ffn_gate with offset "_up".
        let a = adj("ffn_gate_up+1");
        assert_eq!(a.entries.len(), 1);
        assert_eq!(
            a.entries[0].role,
            crate::svd::config::AdjacentRole::FfnGateUp
        );
        assert_eq!(a.entries[0].offset, 1);
    }

    #[test]
    fn adjacent_parse_errors() {
        assert!(AdjacentSelection::parse("+attn_v").is_err());
        assert!(AdjacentSelection::parse("attn_v+").is_err());
        assert!(AdjacentSelection::parse("attn_v+abc").is_err());
        assert!(AdjacentSelection::parse("attn_v+1+").is_err());
        assert!(AdjacentSelection::parse("nope").is_err());
        assert!(AdjacentSelection::parse("attn_v-abc").is_err());
        assert!(AdjacentSelection::parse("1").is_err(), "offset with no role");
    }

    // ---- build_plan with adjacent -----------------------------------------

    /// Three-block model with attn_q, attn_v, and ffn_gate on each block.
    fn three_block_model() -> FakeModel {
        let mut tensors = Vec::new();
        for i in 0..3 {
            tensors.push(t(&format!("blk.{i}.attn_q.weight"), 64, 64));
            tensors.push(t(&format!("blk.{i}.attn_v.weight"), 64, 64));
            tensors.push(t(&format!("blk.{i}.ffn_gate.weight"), 64, 64));
        }
        FakeModel { tensors }
    }

    fn cfg_attn_q() -> SvdConfig {
        let mut cfg = SvdConfig::default();
        cfg.layers = LayerSelection::All;
        cfg.tensors = TensorSelection::Named(vec!["attn_q".into()]);
        cfg
    }

    #[test]
    fn plan_adjacent_none_default_unchanged() {
        let m = three_block_model();
        let cfg = cfg_attn_q();
        let baseline = build_plan(&m, &cfg).unwrap();
        let plan = build_plan(&m, &cfg).unwrap();
        assert_eq!(plan.targets.len(), baseline.targets.len());
        assert_eq!(plan.targets.len(), 3);
        assert_eq!(plan.skipped.len(), baseline.skipped.len());
    }

    #[test]
    fn plan_adjacent_single_role_same_block() {
        let m = three_block_model();
        let mut cfg = cfg_attn_q();
        cfg.adjacent = AdjacentSelection::parse("attn_v").ok().flatten();
        let plan = build_plan(&m, &cfg).unwrap();
        // 3 primary attn_q + 3 adjacent attn_v (one per block) = 6 targets.
        assert_eq!(plan.targets.len(), 6);
        for i in 0..3 {
            assert!(
                plan.targets
                    .iter()
                    .any(|t| t.name == format!("blk.{i}.attn_v.weight")),
                "missing blk.{i}.attn_v.weight"
            );
        }
    }

    #[test]
    fn plan_adjacent_positive_offset() {
        let m = three_block_model();
        let mut cfg = cfg_attn_q();
        cfg.adjacent = AdjacentSelection::parse("attn_v+1").ok().flatten();
        let plan = build_plan(&m, &cfg).unwrap();
        // 3 primary attn_q. Adjacent attn_v: blk.0+1=1 (NEW),
        // blk.1+1=2 (NEW), blk.2+1=3 (OOR). New: 2.
        assert_eq!(plan.targets.len(), 5);
        assert!(plan
            .targets
            .iter()
            .any(|t| t.name == "blk.1.attn_v.weight"));
        assert!(plan
            .targets
            .iter()
            .any(|t| t.name == "blk.2.attn_v.weight"));
        let oor = plan
            .skipped
            .iter()
            .find(|s| s.name == "blk.3.attn_v.weight")
            .expect("blk.3.attn_v.weight should be skipped");
        assert_eq!(oor.reason, "out-of-range block offset");
    }

    #[test]
    fn plan_adjacent_negative_offset() {
        let m = three_block_model();
        let mut cfg = cfg_attn_q();
        cfg.adjacent = AdjacentSelection::parse("attn_v-1").ok().flatten();
        let plan = build_plan(&m, &cfg).unwrap();
        // 3 primary attn_q. Adjacent attn_v: blk.0-1=-1 (OOR),
        // blk.1-1=0 (NEW), blk.2-1=1 (NEW). New: 2.
        assert_eq!(plan.targets.len(), 5);
        assert!(plan
            .targets
            .iter()
            .any(|t| t.name == "blk.0.attn_v.weight"));
        assert!(plan
            .targets
            .iter()
            .any(|t| t.name == "blk.1.attn_v.weight"));
        let oor = plan
            .skipped
            .iter()
            .find(|s| s.name == "blk.-1.attn_v.weight")
            .expect("blk.-1.attn_v.weight should be skipped");
        assert_eq!(oor.reason, "out-of-range block offset");
    }

    #[test]
    fn plan_adjacent_out_of_range() {
        let m = three_block_model();
        let mut cfg = SvdConfig::default();
        cfg.layers = LayerSelection::Indices(vec![0]);
        cfg.tensors = TensorSelection::Named(vec!["attn_q".into()]);
        cfg.adjacent = AdjacentSelection::parse("attn_v+5").ok().flatten();
        let plan = build_plan(&m, &cfg).unwrap();
        // 1 primary (blk.0.attn_q) + 1 adjacent (blk.0.attn_v@5, OOR) =
        // 1 target, 1 SkippedTensor.
        assert_eq!(plan.targets.len(), 1);
        let oor: Vec<&SkippedTensor> = plan
            .skipped
            .iter()
            .filter(|s| s.reason == "out-of-range block offset")
            .collect();
        assert_eq!(oor.len(), 1);
        assert_eq!(oor[0].name, "blk.5.attn_v.weight");
    }

    #[test]
    fn plan_adjacent_multi_roles_offsets() {
        let m = three_block_model();
        let mut cfg = SvdConfig::default();
        cfg.layers = LayerSelection::Indices(vec![0]);
        cfg.tensors = TensorSelection::Named(vec!["attn_q".into()]);
        // attn_v (offset 0, NEW since blk.0.attn_v != blk.0.attn_q) and
        // ffn_gate-1+1 -> ffn_gate offset 1 (blk.1.ffn_gate, NEW).
        cfg.adjacent = AdjacentSelection::parse("attn_v+ffn_gate-1+1").ok().flatten();
        let plan = build_plan(&m, &cfg).unwrap();
        // 1 primary blk.0.attn_q. Adjacent: blk.0.attn_v (NEW) +
        // blk.1.ffn_gate (NEW). Total: 3.
        assert_eq!(plan.targets.len(), 3);
        assert!(plan
            .targets
            .iter()
            .any(|t| t.name == "blk.0.attn_v.weight"));
        assert!(plan
            .targets
            .iter()
            .any(|t| t.name == "blk.1.ffn_gate.weight"));
    }

    #[test]
    fn plan_adjacent_duplicate_suppression() {
        // Primary = TensorSelection::Attn (matches attn_q, attn_k,
        // attn_v, attn_output). The three-block model has attn_q and
        // attn_v on each block, so 6 primaries. Adjacent "attn_q"
        // (offset 0) targets the same attn_q tensors — must be deduped.
        let m = three_block_model();
        let mut cfg = SvdConfig::default();
        cfg.layers = LayerSelection::All;
        cfg.tensors = TensorSelection::Attn;
        cfg.adjacent = AdjacentSelection::parse("attn_q").ok().flatten();
        let plan = build_plan(&m, &cfg).unwrap();
        // Still 6 targets, no duplicates.
        assert_eq!(plan.targets.len(), 6);
        let names: std::collections::HashSet<&str> =
            plan.targets.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names.len(), 6);
    }

    #[test]
    fn plan_adjacent_bypasses_layer_filter() {
        // Primary is restricted to blk.0 only; adjacent `attn_v+1` should
        // still produce a target at blk.1 even though blk.1 was excluded
        // by the layer filter.
        let m = three_block_model();
        let mut cfg = SvdConfig::default();
        cfg.layers = LayerSelection::Indices(vec![0]);
        cfg.tensors = TensorSelection::Named(vec!["attn_q".into()]);
        cfg.adjacent = AdjacentSelection::parse("attn_v+1").ok().flatten();
        let plan = build_plan(&m, &cfg).unwrap();
        // Primary: blk.0.attn_q (1). Adjacent: blk.0+1=1 -> blk.1.attn_v
        // (1, NEW — blk.1 was excluded by the primary layer filter).
        // Total: 2.
        assert_eq!(plan.targets.len(), 2);
        assert!(plan
            .targets
            .iter()
            .any(|t| t.name == "blk.0.attn_q.weight"));
        assert!(plan
            .targets
            .iter()
            .any(|t| t.name == "blk.1.attn_v.weight"));
        // ffn_gate is not in the adjacent list, so it should not appear.
        assert!(plan
            .targets
            .iter()
            .all(|t| !t.name.ends_with(".ffn_gate.weight")));
        // blk.0.attn_v is not in the adjacent list either, so it should
        // not appear (primary is attn_q, not attn_v).
        assert!(plan
            .targets
            .iter()
            .all(|t| t.name != "blk.0.attn_v.weight"));
    }
}
