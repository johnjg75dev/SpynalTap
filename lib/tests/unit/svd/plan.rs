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
    // (offset 0) targets the same attn_q tensors â€” must be deduped.
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
    // (1, NEW â€” blk.1 was excluded by the primary layer filter).
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
