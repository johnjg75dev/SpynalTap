use super::*;
use crate::error::Error;
use crate::model::{MetadataValue, Model, ModelFormat};
use std::borrow::Cow;

struct FakeModel {
    tensors: Vec<Tensor>,
    bytes: HashMap<String, Vec<u8>>,
}

impl FakeModel {
    fn new(tensors: Vec<(String, Vec<u64>, Vec<f32>)>) -> Self {
        let mut ts = Vec::with_capacity(tensors.len());
        let mut bytes = HashMap::new();
        for (name, shape, data) in tensors {
            let elems: u64 = shape.iter().product();
            let byte_size = elems * 4;
            let mut buf = Vec::with_capacity(data.len() * 4);
            for v in &data {
                buf.extend_from_slice(&v.to_le_bytes());
            }
            ts.push(Tensor {
                name: name.clone(),
                dtype: TensorDtype::F32,
                shape,
                byte_size,
                data_offset: 0,
            });
            bytes.insert(name, buf);
        }
        Self {
            tensors: ts,
            bytes,
        }
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

fn f32_data(rows: usize, cols: usize, seed: u32) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * cols);
    for i in 0..(rows * cols) {
        let v = (((i as u32).wrapping_add(seed).wrapping_mul(2654435761)) as f32)
            / (u32::MAX as f32);
        out.push(v * 2.0 - 1.0);
    }
    out
}

#[test]
fn analyzer_populates_spectra() {
    // Build a model with several prunable block tensors plus a few
    // non-prunable ones (embed, output). All F32, all 2-D.
    let mut entries: Vec<(String, Vec<u64>, Vec<f32>)> = Vec::new();
    for blk in 0..3 {
        entries.push((
            format!("blk.{blk}.attn_q.weight"),
            vec![8, 8],
            f32_data(8, 8, blk * 100),
        ));
        entries.push((
            format!("blk.{blk}.attn_v.weight"),
            vec![8, 8],
            f32_data(8, 8, blk * 100 + 1),
        ));
    }
    entries.push((
        "token_embd.weight".into(),
        vec![8, 8],
        f32_data(8, 8, 999),
    ));
    entries.push((
        "output.weight".into(),
        vec![8, 8],
        f32_data(8, 8, 998),
    ));

    let m = FakeModel::new(entries);
    let analysis = Analyzer::with_sample_per_tensor(200_000).analyze(&m).unwrap();

    // Spectra must exist and the total count must be <= 12.
    let total: usize = analysis.blocks.iter().map(|b| b.spectra.len()).sum();
    assert!(total <= MAX_SPECTRA_PER_ANALYSIS, "got {total} spectra");
    // With 6 prunable + 2 non-prunable we expect at least 2 spectra
    // (prunable cap is 8, non-prunable cap is 4) -- but we don't pin
    // the exact number, only the upper bound.
    assert!(total > 0, "expected at least one spectrum");
}

#[test]
fn analyzer_spectra_bounded_when_many_blocks() {
    // 20 prunable tensors -> still <= 8 prunable + 4 non-prunable = 12.
    let mut entries: Vec<(String, Vec<u64>, Vec<f32>)> = Vec::new();
    for blk in 0..10 {
        entries.push((
            format!("blk.{blk}.attn_q.weight"),
            vec![6, 6],
            f32_data(6, 6, blk * 11),
        ));
    }
    entries.push((
        "token_embd.weight".into(),
        vec![6, 6],
        f32_data(6, 6, 7),
    ));
    entries.push((
        "output.weight".into(),
        vec![6, 6],
        f32_data(6, 6, 9),
    ));
    entries.push((
        "norm.weight".into(),
        vec![6, 6],
        f32_data(6, 6, 11),
    ));
    let m = FakeModel::new(entries);
    let analysis = Analyzer::with_sample_per_tensor(200_000).analyze(&m).unwrap();
    let total: usize = analysis.blocks.iter().map(|b| b.spectra.len()).sum();
    assert!(total <= MAX_SPECTRA_PER_ANALYSIS, "got {total}");
}
