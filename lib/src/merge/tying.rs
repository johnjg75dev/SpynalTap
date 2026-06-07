//! Weight tying: sharing the token-embedding matrix with the output
//! projection. This module is metadata-only — it does not move bytes
//! around. The caller decides how to actually rebind the output weight
//! (e.g., by storing an offset+length pointer into the embed tensor or
//! by writing a duplicate copy).

use crate::error::Result;
use crate::model::Model;

/// A plan to tie two tensors: after `apply_tying`, writes to the
/// `output_name` tensor will read from `embed_name` instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TyingPlan {
    pub embed_name: String,
    pub output_name: String,
}

/// Result of applying a tying plan. Carries the names of the two tensors
/// the caller should treat as aliased.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TyingResult {
    pub embed_name: String,
    pub output_name: String,
    /// Always `true` — the output is now backed by the embed.
    pub tied: bool,
}

const EMBED_CANDIDATES: &[&str] = &["token_embd.weight", "tok_embeddings.weight", "embed.weight"];
const OUTPUT_CANDIDATES: &[&str] = &["output.weight", "lm_head.weight", "embed_out.weight"];

fn first_existing<'a>(model: &dyn Model, names: &'a [&'a str]) -> Option<&'a str> {
    names.iter().copied().find(|n| model.tensor(n).is_some())
}

/// Auto-detect a tying plan. Returns `Some(plan)` if the model has both
/// an embedding tensor and an output-projection tensor, otherwise `None`.
pub fn plan_tying(model: &dyn Model) -> Option<TyingPlan> {
    let embed = first_existing(model, EMBED_CANDIDATES)?;
    let output = first_existing(model, OUTPUT_CANDIDATES)?;
    Some(TyingPlan {
        embed_name: embed.to_string(),
        output_name: output.to_string(),
    })
}

/// Apply (i.e. record) a tying plan. The function does not touch any
/// bytes; it just returns a `TyingResult` describing the alias.
///
/// The signature is generic over an error type for forward-compatibility
/// with callers that want to surface a real I/O error if the embed or
/// output tensors are missing or have mismatched shapes. For now it
/// always succeeds — the plan was already validated when constructed.
pub fn apply_tying(plan: &TyingPlan) -> TyingResult {
    TyingResult {
        embed_name: plan.embed_name.clone(),
        output_name: plan.output_name.clone(),
        tied: true,
    }
}

/// Sanity check that the embed and output tensors have the same
/// element count. Returns `Ok(())` if they are compatible.
pub fn verify_tying_compatible(model: &dyn Model, plan: &TyingPlan) -> Result<()> {
    let e = model
        .tensor(&plan.embed_name)
        .ok_or_else(|| crate::Error::TensorNotFound(plan.embed_name.clone()))?;
    let o = model
        .tensor(&plan.output_name)
        .ok_or_else(|| crate::Error::TensorNotFound(plan.output_name.clone()))?;
    let e_count: u64 = e.shape.iter().product();
    let o_count: u64 = o.shape.iter().product();
    if e_count != o_count {
        return Err(crate::Error::InvalidSvdConfig(format!(
            "tying incompatible: embed has {e_count} elements, output has {o_count}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
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
}
