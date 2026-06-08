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
    // vector (truly 1D â€” shape [2], not [2, 1]).
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
