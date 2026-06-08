use super::*;
use crate::model::{MetadataValue, Model, ModelFormat, Tensor, TensorDtype};
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
        None
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
fn plan_tying_detects_default_pair() {
    let m = FakeModel {
        tensors: vec![t("token_embd.weight", 32, 64), t("output.weight", 32, 64)],
    };
    let p = plan_tying(&m).expect("plan should be detected");
    assert_eq!(p.embed_name, "token_embd.weight");
    assert_eq!(p.output_name, "output.weight");
}

#[test]
fn plan_tying_detects_alt_pair() {
    let m = FakeModel {
        tensors: vec![
            t("tok_embeddings.weight", 32, 64),
            t("lm_head.weight", 32, 64),
        ],
    };
    let p = plan_tying(&m).expect("plan should be detected");
    assert_eq!(p.embed_name, "tok_embeddings.weight");
    assert_eq!(p.output_name, "lm_head.weight");
}

#[test]
fn plan_tying_returns_none_without_embed() {
    let m = FakeModel {
        tensors: vec![t("output.weight", 32, 64)],
    };
    assert!(plan_tying(&m).is_none());
}

#[test]
fn plan_tying_returns_none_without_output() {
    let m = FakeModel {
        tensors: vec![t("token_embd.weight", 32, 64)],
    };
    assert!(plan_tying(&m).is_none());
}

#[test]
fn apply_tying_returns_alias() {
    let plan = TyingPlan {
        embed_name: "token_embd.weight".into(),
        output_name: "output.weight".into(),
    };
    let r = apply_tying(&plan);
    assert_eq!(r.embed_name, "token_embd.weight");
    assert_eq!(r.output_name, "output.weight");
    assert!(r.tied);
}

#[test]
fn verify_tying_compatible_accepts_matching() {
    let m = FakeModel {
        tensors: vec![t("token_embd.weight", 32, 64), t("output.weight", 64, 32)],
    };
    let plan = TyingPlan {
        embed_name: "token_embd.weight".into(),
        output_name: "output.weight".into(),
    };
    assert!(verify_tying_compatible(&m, &plan).is_ok());
}

#[test]
fn verify_tying_compatible_rejects_mismatch() {
    let m = FakeModel {
        tensors: vec![t("token_embd.weight", 32, 64), t("output.weight", 16, 32)],
    };
    let plan = TyingPlan {
        embed_name: "token_embd.weight".into(),
        output_name: "output.weight".into(),
    };
    assert!(verify_tying_compatible(&m, &plan).is_err());
}
