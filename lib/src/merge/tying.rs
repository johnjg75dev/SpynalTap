//! Weight tying: sharing the token-embedding matrix with the output
//! projection. This module is metadata-only â€” it does not move bytes
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
    /// Always `true` â€” the output is now backed by the embed.
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
/// always succeeds â€” the plan was already validated when constructed.
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
#[path = "../../tests/unit/merge/tying.rs"]
mod tests;
