//! Apply a `PrunePlan` to a model and write the pruned file to disk.

use crate::error::{Error, Result};
use crate::formats::gguf::reader::GgufFile;
use crate::formats::gguf::types::MetaValue;
use crate::formats::gguf::writer::GgufWriter;
use crate::formats::safetensors::reader::SafetensorsFile;
use crate::formats::safetensors::writer::SafetensorsWriter;
use crate::model::Model;
use crate::prune::plan::PrunePlan;
use std::collections::HashSet;
use std::io::Write;
use std::path::Path;

#[derive(Debug, serde::Serialize)]
pub struct PruneReport {
    pub bytes_in: u64,
    pub bytes_out: u64,
    pub tensors_kept: usize,
    pub tensors_dropped: usize,
    pub blocks_dropped: Vec<i32>,
    pub original_block_count: i32,
    pub new_block_count: i32,
    pub output_path: String,
}

pub fn apply_to_gguf(gg: &GgufFile, plan: &PrunePlan, dst: &Path) -> Result<PruneReport> {
    let mut writer = GgufWriter::new(gg.version, gg.alignment);
    let block_count_keys: HashSet<&str> = [
        "llama.block_count", "qwen2.block_count", "gemma2.block_count",
        "phi2.block_count", "mistral.block_count", "general.block_count",
        "block_count",
    ].iter().copied().collect();

    for kv in &gg.metadata {
        let mut new_kv = kv.clone();
        if block_count_keys.contains(kv.key.as_str()) {
            new_kv.value = match kv.value {
                MetaValue::U32(_) => MetaValue::U32(plan.new_block_count as u32),
                MetaValue::U64(_) => MetaValue::U64(plan.new_block_count as u64),
                _ => kv.value.clone(),
            };
        }
        writer.add_kv(new_kv);
    }

    let name_to_idx: std::collections::HashMap<&str, &crate::formats::gguf::types::TensorInfo> =
        gg.tensors.iter().map(|t| (t.name.as_str(), t)).collect();

    let mut kept = 0usize;
    let mut dropped = 0usize;
    for (name, k) in &plan.keep {
        if !*k { dropped += 1; continue; }
        let ti = name_to_idx.get(name.as_str()).copied()
            .ok_or_else(|| Error::TensorNotFound(name.clone()))?;
        let bytes = gg.tensor_slice(ti)
            .ok_or_else(|| Error::Gguf("tensor not in mmap".into()))?;
        let new_name = rename_block(name, &plan.remap);
        writer.add_tensor(new_name, ti.n_dims, ti.dims, ti.ggml_type, bytes);
        kept += 1;
    }

    let bytes_in: u64 = gg.tensors.iter().map(|t| t.byte_size).sum();
    let bytes_out: u64 = writer.tensors.iter().map(|t| t.byte_size).sum();

    let out_bytes = writer.into_bytes()?;
    let mut out_file = std::fs::File::create(dst)?;
    out_file.write_all(&out_bytes)?;
    out_file.sync_all()?;

    Ok(PruneReport {
        bytes_in, bytes_out,
        tensors_kept: kept, tensors_dropped: dropped,
        blocks_dropped: plan.dropped_blocks.clone(),
        original_block_count: plan.original_block_count,
        new_block_count: plan.new_block_count,
        output_path: dst.display().to_string(),
    })
}

pub fn apply_to_safetensors(st: &SafetensorsFile, plan: &PrunePlan, dst: &Path) -> Result<PruneReport> {
    let mut writer = SafetensorsWriter::new();
    let mut kept = 0usize;
    let mut dropped = 0usize;
    for (name, k) in &plan.keep {
        if !*k { dropped += 1; continue; }
        let t = st.tensor(name).ok_or_else(|| Error::TensorNotFound(name.clone()))?;
        let bytes = st.read_tensor_bytes(name)?;
        let new_name = rename_block(name, &plan.remap);
        writer.add_raw(new_name, t.dtype, t.shape.clone(), &bytes);
        kept += 1;
    }
    let bytes_in: u64 = st.tensors.iter().map(|t| t.byte_size).sum();
    let out_file = std::fs::File::create(dst)?;
    writer.write_to(&out_file)?;
    out_file.sync_all()?;

    let bytes_out = std::fs::metadata(dst)?.len();
    Ok(PruneReport {
        bytes_in, bytes_out,
        tensors_kept: kept, tensors_dropped: dropped,
        blocks_dropped: plan.dropped_blocks.clone(),
        original_block_count: plan.original_block_count,
        new_block_count: plan.new_block_count,
        output_path: dst.display().to_string(),
    })
}

pub fn rename_block(name: &str, remap: &std::collections::HashMap<i32, i32>) -> String {
    if !name.starts_with("blk.") { return name.to_string(); }
    let rest = &name[4..];
    let mut parts = rest.splitn(2, '.');
    let idx_str = match parts.next() { Some(s) => s, None => return name.to_string() };
    let rest2 = parts.next().unwrap_or("");
    let idx: i32 = match idx_str.parse() { Ok(i) => i, Err(_) => return name.to_string() };
    match remap.get(&idx) {
        Some(new_idx) => format!("blk.{new_idx}.{rest2}"),
        None => name.to_string(),
    }
}
