//! Depth expansion: insert a new transformer block into a model.
//!
//! The actual file is **not** modified. The function walks the model's
//! tensor list, copies/zero-fills the new block's tensors under their
//! final names, and copies the bytes of any existing blocks that need to
//! be renumbered (every block with `block_idx >= new_block_index` is
//! shifted up by one). The caller is responsible for splicing these
//! `(name, bytes)` pairs into the on-disk representation.

use crate::error::{Error, Result};
use crate::model::Model;

/// Where the new block's weights come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertSource {
    /// Copy weights from the model block at `block_index`.
    Duplicate { block_index: i32 },
    /// Fill all 2D weights with zeros. Requires `explicit_shape` on the
    /// `InsertPlan`, otherwise the shape set is unknown.
    Zeros,
}

/// Plan for inserting a new block.
#[derive(Debug, Clone)]
pub struct InsertPlan {
    /// Insert **after** this block index. The new block lives at
    /// `after_block + 1`.
    pub after_block: i32,
    /// Source of the new block's weights.
    pub source: InsertSource,
    /// Optional override for the (rows, cols) of 2-D weight tensors.
    /// If `Some`, every 2-D tensor in the new block uses this shape
    /// regardless of the source block's shape.
    pub explicit_shape: Option<(usize, usize)>,
    /// Optional human-readable name for the new block. Currently
    /// informational only (the on-disk name is always `blk.<N>`).
    pub new_block_name: Option<String>,
}

/// Result of an `insert_block` call. Contains the new tensors to write
/// to disk (either the inserted block, the renumbered original blocks, or
/// both). The model file itself is not modified.
#[derive(Debug, Clone)]
pub struct InsertResult {
    pub new_block_index: i32,
    pub tensors: Vec<(String, Vec<u8>)>,
    pub skipped: Vec<String>,
}

/// Insert a block into a model.
///
/// `model` is only borrowed for its tensor metadata and (for
/// `Duplicate`) its raw bytes. The function never writes the model; it
/// returns the new tensors as `(name, f32 bytes)` pairs so the caller
/// can write them through the format writer.
pub fn insert_block<M: Model + ?Sized>(
    model: &M,
    plan: &InsertPlan,
) -> Result<InsertResult> {
    let new_block_index = plan.after_block + 1;

    // 1. Discover the source block's tensor list (for shape lookup).
    //
    // For `Duplicate`, the source is the explicit `block_index`. For
    // `Zeros`, the source is implicitly the block we're inserting after
    // (`after_block`); this gives the new block the same tensor name
    // layout as its neighbour, with values replaced by zeros.
    let (source_tensors, source_block_index): (Vec<&crate::model::Tensor>, i32) = match plan.source
    {
        InsertSource::Duplicate { block_index } => {
            let prefix = format!("blk.{block_index}.");
            let collected: Vec<&crate::model::Tensor> = model
                .tensors()
                .iter()
                .filter(|t| t.name.starts_with(&prefix))
                .collect();
            if collected.is_empty() {
                return Err(Error::Gguf(format!(
                    "insert_block: source block blk.{block_index} has no tensors"
                )));
            }
            (collected, block_index)
        }
        InsertSource::Zeros => {
            let prefix = format!("blk.{}.", plan.after_block);
            let collected: Vec<&crate::model::Tensor> = model
                .tensors()
                .iter()
                .filter(|t| t.name.starts_with(&prefix))
                .collect();
            if collected.is_empty() {
                return Err(Error::Gguf(format!(
                    "insert_block: InsertSource::Zeros needs a source block (blk.{}.*) with at least one tensor",
                    plan.after_block
                )));
            }
            (collected, plan.after_block)
        }
    };

    // 2. Build the new block's tensors.
    let mut out: Vec<(String, Vec<u8>)> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();

    match plan.source {
        InsertSource::Duplicate { .. } => {
            for t in &source_tensors {
                let new_name = rename_block_prefix(&t.name, source_block_index, new_block_index);
                let bytes = model.read_tensor_bytes(&t.name)?.into_owned();
                out.push((new_name, bytes));
            }
        }
        InsertSource::Zeros => {
            for t in &source_tensors {
                let new_name = rename_block_prefix(&t.name, source_block_index, new_block_index);
                let shape = if t.shape.len() == 2 {
                    plan.explicit_shape
                        .map(|(r, c)| (r as u64, c as u64))
                        .unwrap_or((t.shape[0], t.shape[1]))
                } else {
                    (t.shape[0], 1)
                };
                let n_elem = (shape.0 * shape.1) as usize;
                let bytes = vec![0u8; n_elem * std::mem::size_of::<f32>()];
                out.push((new_name, bytes));
            }
        }
    }

    // 3. Renumber subsequent blocks: every tensor in the model with
    //    `block_idx >= new_block_index` (including those that lived at
    //    the new block's slot before the insert) gets shifted up by 1.
    //    The new block's own tensors were emitted in step 2 with the
    //    source block's bytes; here we emit copies of the *original*
    //    tensors with their final, post-shift names.
    for t in model.tensors() {
        let blk_idx = match block_index_of(&t.name) {
            Some(i) => i,
            None => continue,
        };
        if blk_idx < new_block_index {
            continue;
        }
        let new_name = rename_block_prefix(&t.name, blk_idx, blk_idx + 1);
        match model.read_tensor_bytes(&t.name) {
            Ok(bytes) => out.push((new_name, bytes.into_owned())),
            Err(_) => skipped.push(t.name.clone()),
        }
    }

    Ok(InsertResult {
        new_block_index,
        tensors: out,
        skipped,
    })
}

/// If `name` has the form `blk.<idx>.<rest>`, returns `Some(idx)`.
fn block_index_of(name: &str) -> Option<i32> {
    let rest = name.strip_prefix("blk.")?;
    let mut parts = rest.splitn(2, '.');
    let idx_str = parts.next()?;
    idx_str.parse().ok()
}

/// Replace the `blk.<old_idx>.` prefix with `blk.<new_idx>.`. If the
/// name does not start with `blk.<old_idx>.`, it is returned unchanged.
fn rename_block_prefix(name: &str, old_idx: i32, new_idx: i32) -> String {
    let prefix = format!("blk.{old_idx}.");
    if let Some(rest) = name.strip_prefix(&prefix) {
        format!("blk.{new_idx}.{rest}")
    } else {
        name.to_string()
    }
}

#[cfg(test)]
#[path = "../../tests/unit/merge/depth.rs"]
mod tests;
