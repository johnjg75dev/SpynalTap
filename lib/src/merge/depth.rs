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
mod tests {
    use super::*;
    use crate::model::{MetadataValue, Model, ModelFormat, Tensor, TensorDtype};
    use std::borrow::Cow;

    struct FakeModel {
        tensors: Vec<Tensor>,
        bytes: std::collections::HashMap<String, Vec<u8>>,
    }

    impl FakeModel {
        fn new(tensors: Vec<(String, Vec<u64>, Vec<u8>)>) -> Self {
            let mut ts = Vec::new();
            let mut bytes = std::collections::HashMap::new();
            for (name, shape, data) in tensors {
                let byte_size: u64 = shape.iter().product::<u64>() * 4;
                ts.push(Tensor {
                    name: name.clone(),
                    dtype: TensorDtype::F32,
                    shape,
                    byte_size,
                    data_offset: 0,
                });
                bytes.insert(name, data);
            }
            Self { tensors: ts, bytes }
        }
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
        fn read_tensor_bytes(&self, name: &str) -> Result<Cow<'_, [u8]>> {
            self.bytes
                .get(name)
                .map(|v| Cow::Borrowed(v.as_slice()))
                .ok_or_else(|| Error::TensorNotFound(name.to_string()))
        }
    }

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        let mut b = Vec::with_capacity(v.len() * 4);
        for x in v {
            b.extend_from_slice(&x.to_le_bytes());
        }
        b
    }

    #[test]
    fn duplicate_insert_renumbers_subsequent() {
        // Model has blk.0 and blk.1, each with a 2x2 weight.
        let model = FakeModel::new(vec![
            ("blk.0.attn_q.weight".into(), vec![2, 2], f32_bytes(&[1.0, 2.0, 3.0, 4.0])),
            ("blk.0.attn_norm.weight".into(), vec![2, 1], f32_bytes(&[0.5, 0.25])),
            ("blk.1.attn_q.weight".into(), vec![2, 2], f32_bytes(&[9.0, 8.0, 7.0, 6.0])),
            ("blk.1.attn_norm.weight".into(), vec![2, 1], f32_bytes(&[0.1, 0.2])),
        ]);
        let plan = InsertPlan {
            after_block: 0,
            source: InsertSource::Duplicate { block_index: 0 },
            explicit_shape: None,
            new_block_name: None,
        };
        let res = insert_block(&model, &plan).unwrap();
        assert_eq!(res.new_block_index, 1);

        // We expect 4 returned tensors: blk.1.* (the duplicate) and
        // blk.2.* (the renumbered original blk.1).
        let names: Vec<&str> = res.tensors.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"blk.1.attn_q.weight"), "got {:?}", names);
        assert!(names.contains(&"blk.1.attn_norm.weight"));
        assert!(names.contains(&"blk.2.attn_q.weight"));
        assert!(names.contains(&"blk.2.attn_norm.weight"));

        // Verify the duplicate's bytes match blk.0's.
        let dup_q = res
            .tensors
            .iter()
            .find(|(n, _)| n == "blk.1.attn_q.weight")
            .unwrap();
        assert_eq!(dup_q.1, f32_bytes(&[1.0, 2.0, 3.0, 4.0]));

        // Verify the renumbered bytes match the original blk.1.
        let renum_q = res
            .tensors
            .iter()
            .find(|(n, _)| n == "blk.2.attn_q.weight")
            .unwrap();
        assert_eq!(renum_q.1, f32_bytes(&[9.0, 8.0, 7.0, 6.0]));
    }

    #[test]
    fn zeros_insert_uses_explicit_shape() {
        // Model has blk.0 with a 2x2 weight.
        let model = FakeModel::new(vec![(
            "blk.0.attn_q.weight".into(),
            vec![2, 2],
            f32_bytes(&[1.0, 2.0, 3.0, 4.0]),
        )]);
        let plan = InsertPlan {
            after_block: 0,
            source: InsertSource::Zeros,
            explicit_shape: Some((4, 4)),
            new_block_name: None,
        };
        let res = insert_block(&model, &plan).unwrap();
        assert_eq!(res.new_block_index, 1);

        // The new block should be blk.1.* with 4x4 = 16 f32 zeros.
        let dup = res
            .tensors
            .iter()
            .find(|(n, _)| n == "blk.1.attn_q.weight")
            .unwrap();
        assert_eq!(dup.1.len(), 16 * 4);
        assert!(dup.1.iter().all(|&b| b == 0));

        // No renumbering needed (no blk.1 originally).
        let renum_count = res
            .tensors
            .iter()
            .filter(|(n, _)| n.starts_with("blk.2."))
            .count();
        assert_eq!(renum_count, 0);
    }

    #[test]
    fn zeros_insert_without_source_block_errors() {
        // The Zeros path uses the block at after_block as the shape
        // source. If there is no such block in the model, it must error.
        let model = FakeModel::new(vec![(
            "blk.5.attn_q.weight".into(),
            vec![2, 2],
            f32_bytes(&[1.0, 2.0, 3.0, 4.0]),
        )]);
        let plan = InsertPlan {
            after_block: 0,
            source: InsertSource::Zeros,
            explicit_shape: None,
            new_block_name: None,
        };
        assert!(insert_block(&model, &plan).is_err());
    }

    #[test]
    fn duplicate_insert_at_end_with_no_subsequent_blocks() {
        // Model has only blk.0. Inserting after blk.0 with no blk.1
        // means no renumbering.
        let model = FakeModel::new(vec![(
            "blk.0.attn_q.weight".into(),
            vec![2, 2],
            f32_bytes(&[1.0, 2.0, 3.0, 4.0]),
        )]);
        let plan = InsertPlan {
            after_block: 0,
            source: InsertSource::Duplicate { block_index: 0 },
            explicit_shape: None,
            new_block_name: None,
        };
        let res = insert_block(&model, &plan).unwrap();
        assert_eq!(res.new_block_index, 1);
        // Only the duplicate should be present; no blk.2.
        assert_eq!(res.tensors.len(), 1);
        assert_eq!(res.tensors[0].0, "blk.1.attn_q.weight");
    }

    #[test]
    fn explicit_shape_overrides_2d_on_duplicate() {
        // attn_q is a 2x2 weight (2D); attn_norm is a 2-element
        // vector (truly 1D — shape [2], not [2, 1]).
        let model = FakeModel::new(vec![
            ("blk.0.attn_q.weight".into(), vec![2, 2], f32_bytes(&[1.0, 2.0, 3.0, 4.0])),
            ("blk.0.attn_norm.weight".into(), vec![2], f32_bytes(&[0.5, 0.25])),
        ]);
        let plan = InsertPlan {
            after_block: 0,
            source: InsertSource::Zeros,
            explicit_shape: Some((8, 8)),
            new_block_name: None,
        };
        let res = insert_block(&model, &plan).unwrap();
        // The 2D attn_q should be 8*8 = 64 f32 zeros.
        let dup_q = res
            .tensors
            .iter()
            .find(|(n, _)| n == "blk.1.attn_q.weight")
            .unwrap();
        assert_eq!(dup_q.1.len(), 64 * 4);
        // The 1D attn_norm should keep its source shape (2 elements).
        let dup_n = res
            .tensors
            .iter()
            .find(|(n, _)| n == "blk.1.attn_norm.weight")
            .unwrap();
        assert_eq!(dup_n.1.len(), 2 * 4);
    }
}
