//! GGUF writer used for pruned-output generation.
//!
//! The writer builds the whole file in a single `Vec<u8>` (with
//! `Vec::with_capacity` to avoid reallocations) and then dumps it to disk.
//! This is the fastest path for our use case (one big output, not streaming).

use crate::formats::gguf::types::{
    byte_size_for, dims_product, GgmlType, MetaValue, MetadataKv, GGUF_MAGIC,
};

#[derive(Clone)]
pub struct GgufWriter {
    pub version: u32,
    pub alignment: usize,
    pub metadata: Vec<MetadataKv>,
    pub tensors: Vec<WriterTensor>,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct WriterTensor {
    pub name: String,
    pub n_dims: u32,
    pub dims: [u64; 4],
    pub ggml_type: GgmlType,
    pub offset: u64,
    pub n_elements: u64,
    pub byte_size: u64,
}

impl GgufWriter {
    pub fn new(version: u32, alignment: usize) -> Self {
        Self {
            version,
            metadata: Vec::new(),
            alignment,
            tensors: Vec::new(),
            data: Vec::new(),
        }
    }

    #[inline]
    pub fn add_kv(&mut self, kv: MetadataKv) {
        self.metadata.push(kv);
    }

    /// Append a tensor whose raw bytes (already aligned to block size) are `bytes`.
    /// The tensor is recorded with a fresh offset = current `data.len()`.
    pub fn add_tensor(
        &mut self,
        name: String,
        n_dims: u32,
        dims: [u64; 4],
        ty: GgmlType,
        bytes: &[u8],
    ) {
        let offset = self.data.len() as u64;
        let n_elements = dims_product(&dims, n_dims);
        let expected = byte_size_for(n_elements, ty);
        let byte_size = bytes.len() as u64;
        debug_assert_eq!(byte_size, expected, "tensor '{name}': byte count mismatch");
        self.tensors.push(WriterTensor {
            name,
            n_dims,
            dims,
            ggml_type: ty,
            offset,
            n_elements,
            byte_size,
        });
        self.data.extend_from_slice(bytes);
    }

    /// Build the complete GGUF bytes in memory, then write to `w`.
    pub fn write_to<W: std::io::Write>(&self, mut w: W) -> std::io::Result<()> {
        let bytes = self.clone().into_bytes()?;
        w.write_all(&bytes)?;
        Ok(())
    }

    /// Move the writer into the final byte buffer (frees the rest).
    pub fn into_bytes(self) -> std::io::Result<Vec<u8>> {
        // Pre-size: rough estimate from current metadata + tensors + data.
        let mut cap = 24; // header
        for kv in &self.metadata {
            cap += 8 + kv.key.len() + 4 + meta_value_size_estimate(&kv.value);
        }
        for t in &self.tensors {
            cap += 8 + t.name.len() + 4 + (t.n_dims as usize) * 8 + 4 + 8;
        }
        cap += self.data.len() + self.alignment;

        let mut out: Vec<u8> = Vec::with_capacity(cap);
        out.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.metadata.len() as u64).to_le_bytes());

        for kv in &self.metadata {
            push_string(&mut out, &kv.key);
            out.extend_from_slice(&kv.value_type.to_le_bytes());
            push_meta_value(&mut out, &kv.value);
        }
        for t in &self.tensors {
            push_string(&mut out, &t.name);
            out.extend_from_slice(&t.n_dims.to_le_bytes());
            for i in 0..t.n_dims as usize {
                out.extend_from_slice(&t.dims[i].to_le_bytes());
            }
            out.extend_from_slice(&t.ggml_type.to_u32().to_le_bytes());
            out.extend_from_slice(&t.offset.to_le_bytes());
        }
        let pos = out.len();
        let pad = (self.alignment - (pos % self.alignment)) % self.alignment;
        out.resize(out.len() + pad, 0);
        out.extend_from_slice(&self.data);
        Ok(out)
    }
}

fn push_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn push_meta_value(out: &mut Vec<u8>, v: &MetaValue) {
    match v {
        MetaValue::U8(x) => out.extend_from_slice(&x.to_le_bytes()),
        MetaValue::I8(x) => out.extend_from_slice(&x.to_le_bytes()),
        MetaValue::U16(x) => out.extend_from_slice(&x.to_le_bytes()),
        MetaValue::I16(x) => out.extend_from_slice(&x.to_le_bytes()),
        MetaValue::U32(x) => out.extend_from_slice(&x.to_le_bytes()),
        MetaValue::I32(x) => out.extend_from_slice(&x.to_le_bytes()),
        MetaValue::F32(x) => out.extend_from_slice(&x.to_le_bytes()),
        MetaValue::Bool(b) => out.push(*b as u8),
        MetaValue::String(s) => push_string(out, s),
        MetaValue::U64(x) => out.extend_from_slice(&x.to_le_bytes()),
        MetaValue::I64(x) => out.extend_from_slice(&x.to_le_bytes()),
        MetaValue::F64(x) => out.extend_from_slice(&x.to_le_bytes()),
        MetaValue::Array(a) => {
            out.extend_from_slice(&a.elem_type.to_le_bytes());
            out.extend_from_slice(&(a.elements.len() as u64).to_le_bytes());
            for e in &a.elements {
                push_meta_value(out, e);
            }
        }
    }
}

fn meta_value_size_estimate(v: &MetaValue) -> usize {
    match v {
        MetaValue::String(s) => 8 + s.len(),
        MetaValue::Array(a) => {
            12 + a
                .elements
                .iter()
                .map(meta_value_size_estimate)
                .sum::<usize>()
        }
        _ => 8,
    }
}
