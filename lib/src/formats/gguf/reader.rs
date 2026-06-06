//! GGUF v1–v3 reader.
//!
//! On-disk layout:
//! ```text
//!   Header       { magic:u32, version:u32, tensor_count:u64, kv_count:u64 }  (24 B)
//!   KV pairs     [ MetadataKv; kv_count ]
//!   Tensor infos [ TensorInfo; tensor_count ]
//!   Padding      to `general.alignment` (default 32) bytes
//!   Tensor data  raw bytes, each tensor at file_offset = data_offset + tensor.offset
//! ```
//!
//! All multi-byte values are little-endian.
//!
//! `GgufFile::open(path)` mmaps the file for zero-copy tensor reads.
//! `GgufFile::read_from(reader)` parses the metadata from any `Read+Seek`
//! but cannot offer mmap-backed reads.

use crate::error::{Error, Result};
use crate::formats::gguf::types::{
    byte_size_for, dims_product, ArrayValue, GgmlType, MetaValue, MetadataKv, TensorInfo,
    DEFAULT_ALIGNMENT, GGUF_MAGIC,
};
use crate::model::{MetadataValue, Model, ModelFormat, Tensor};
use memmap2::Mmap;
use std::borrow::Cow;
use std::fs::File;
use std::io::{Read, Seek};
use std::path::Path;

pub struct GgufFile {
    pub version: u32,
    pub tensor_count: u64,
    pub kv_count: u64,
    pub metadata: Vec<MetadataKv>,
    pub tensors: Vec<TensorInfo>,
    /// File offset where the tensor data section begins (after padding).
    pub data_section_offset: u64,
    pub alignment: usize,
    /// Optional mmap of the whole file. Present when opened via `open()`.
    mmap: Option<Mmap>,
    /// Cached model-agnostic view of every tensor.
    model_tensors: Vec<Tensor>,
    /// Name -> index into `tensors` for O(1) lookup.
    name_to_idx: std::collections::HashMap<String, usize>,
}

impl std::fmt::Debug for GgufFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufFile")
            .field("version", &self.version)
            .field("tensor_count", &self.tensor_count)
            .field("kv_count", &self.kv_count)
            .field("metadata", &self.metadata.len())
            .field("tensors", &self.tensors.len())
            .field("data_section_offset", &self.data_section_offset)
            .field("alignment", &self.alignment)
            .field("mmap_bytes", &self.mmap.as_ref().map(|m| m.len()))
            .finish()
    }
}

impl GgufFile {
    /// Open a GGUF file from disk, mmapping it for zero-copy tensor reads.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();
        let file = File::open(path_ref)?;
        // SAFETY: We hold `file` open for the lifetime of the mmap, and we
        // only read from the mmap. The file is opened read-only via `File::open`.
        let mmap =
            unsafe { Mmap::map(&file) }.map_err(|e| Error::Gguf(format!("mmap failed: {e}")))?;
        let mut f = File::open(path_ref)?;
        Self::parse(&mut f, Some(mmap))
    }

    /// Parse the header + metadata from any `Read + Seek` source.
    /// Tensor bytes are not loaded; `read_tensor_bytes` will error.
    pub fn read_from<R: Read + Seek>(r: &mut R) -> Result<Self> {
        Self::parse(r, None)
    }

    fn parse<R: Read + Seek>(r: &mut R, mmap: Option<Mmap>) -> Result<Self> {
        // ---- Header (batched: one read_exact, four from_le_bytes) ----------
        let mut hdr = [0u8; 24];
        r.read_exact(&mut hdr)?;
        let magic = u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]);
        if magic != GGUF_MAGIC {
            return Err(Error::Gguf(format!("bad magic: 0x{magic:08x}")));
        }
        let version = u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]);
        if !(1..=3).contains(&version) {
            return Err(Error::Gguf(format!("unsupported GGUF version {version}")));
        }
        let tensor_count = u64::from_le_bytes(hdr[8..16].try_into().unwrap());
        let kv_count = u64::from_le_bytes(hdr[16..24].try_into().unwrap());

        // ---- KV pairs --------------------------------------------------------
        let mut metadata = Vec::with_capacity(kv_count as usize);
        for _ in 0..kv_count {
            metadata.push(read_kv(r)?);
        }

        let mut alignment = DEFAULT_ALIGNMENT;
        for kv in &metadata {
            if kv.key == "general.alignment" {
                alignment = match &kv.value {
                    MetaValue::U32(v) => *v as usize,
                    MetaValue::U64(v) => *v as usize,
                    _ => alignment,
                };
            }
        }

        // ---- Tensor infos (batched tail read) --------------------------------
        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            tensors.push(read_tensor_info(r)?);
        }

        for t in tensors.iter_mut() {
            t.n_elements = dims_product(&t.dims, t.n_dims);
            t.byte_size = byte_size_for(t.n_elements, t.ggml_type);
        }

        let pos_after_infos = r.stream_position()?;
        let pad = ((alignment - (pos_after_infos as usize % alignment)) % alignment) as u64;
        let data_section_offset = pos_after_infos + pad;

        // Build the cached model-agnostic view and lookup map.
        let mut model_tensors = Vec::with_capacity(tensors.len());
        let mut name_to_idx = std::collections::HashMap::with_capacity(tensors.len());
        for (i, t) in tensors.iter().enumerate() {
            name_to_idx.insert(t.name.clone(), i);
            model_tensors.push(Tensor {
                name: t.name.clone(),
                dtype: t.ggml_type.to_tensor_dtype(),
                shape: (0..t.n_dims as usize).map(|i| t.dims[i]).collect(),
                byte_size: t.byte_size,
                data_offset: t.offset,
            });
        }

        Ok(GgufFile {
            version,
            tensor_count,
            kv_count,
            metadata,
            tensors,
            data_section_offset,
            alignment,
            mmap,
            model_tensors,
            name_to_idx,
        })
    }

    /// Zero-copy slice of one tensor's bytes (requires `open()`, not `read_from`).
    #[inline]
    pub fn tensor_slice(&self, t: &TensorInfo) -> Option<&[u8]> {
        let mmap = self.mmap.as_ref()?;
        let start = (self.data_section_offset + t.offset) as usize;
        let end = start + t.byte_size as usize;
        if end > mmap.len() {
            return None;
        }
        Some(&mmap[start..end])
    }

    /// Look up a tensor by name (returns the GGUF-native view).
    #[inline]
    pub fn get_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.name_to_idx.get(name).map(|&i| &self.tensors[i])
    }

    pub fn metadata_str(&self, key: &str) -> Option<&str> {
        self.metadata.iter().find(|k| k.key == key).and_then(|kv| {
            if let MetaValue::String(s) = &kv.value {
                Some(s.as_str())
            } else {
                None
            }
        })
    }

    pub fn metadata_u32(&self, key: &str) -> Option<u32> {
        self.metadata
            .iter()
            .find(|k| k.key == key)
            .and_then(|kv| match &kv.value {
                MetaValue::U32(v) => Some(*v),
                MetaValue::U64(v) => Some(*v as u32),
                _ => None,
            })
    }
}

impl Model for GgufFile {
    fn format(&self) -> ModelFormat {
        ModelFormat::Gguf
    }
    fn name(&self) -> Option<&str> {
        self.metadata_str("general.name")
    }
    fn architecture(&self) -> Option<&str> {
        self.metadata_str("general.architecture")
    }
    fn block_count(&self) -> Option<usize> {
        let arch = self.architecture()?;
        self.metadata_u32(&format!("{arch}.block_count"))
            .map(|v| v as usize)
    }
    fn tensors(&self) -> &[Tensor] {
        &self.model_tensors
    }
    fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.name_to_idx.get(name).map(|&i| &self.model_tensors[i])
    }
    fn metadata(&self, key: &str) -> Option<MetadataValue<'_>> {
        let kv = self.metadata.iter().find(|k| k.key == key)?;
        Some(match &kv.value {
            MetaValue::U8(v) => MetadataValue::U8(*v),
            MetaValue::I8(v) => MetadataValue::I8(*v),
            MetaValue::U16(v) => MetadataValue::U16(*v),
            MetaValue::I16(v) => MetadataValue::I16(*v),
            MetaValue::U32(v) => MetadataValue::U32(*v),
            MetaValue::I32(v) => MetadataValue::I32(*v),
            MetaValue::F32(v) => MetadataValue::F32(*v),
            MetaValue::Bool(v) => MetadataValue::Bool(*v),
            MetaValue::String(v) => MetadataValue::String(v),
            MetaValue::U64(v) => MetadataValue::U64(*v),
            MetaValue::I64(v) => MetadataValue::I64(*v),
            MetaValue::F64(v) => MetadataValue::F64(*v),
            MetaValue::Array(_) => return None,
        })
    }
    fn read_tensor_bytes(&self, name: &str) -> Result<Cow<'_, [u8]>> {
        let t = self
            .get_tensor(name)
            .ok_or_else(|| Error::TensorNotFound(name.to_string()))?;
        match self.tensor_slice(t) {
            Some(slice) => Ok(Cow::Borrowed(slice)),
            None => Err(Error::Gguf(
                "tensor bytes unavailable: file was parsed from a stream, not opened with mmap"
                    .into(),
            )),
        }
    }
}

// -- raw read helpers --------------------------------------------------------

trait ReadExt: Read {
    #[inline]
    fn read_u8_le(&mut self) -> std::io::Result<u8> {
        let mut b = [0u8; 1];
        self.read_exact(&mut b)?;
        Ok(b[0])
    }
    #[inline]
    fn read_u16_le(&mut self) -> std::io::Result<u16> {
        let mut b = [0u8; 2];
        self.read_exact(&mut b)?;
        Ok(u16::from_le_bytes(b))
    }
    #[inline]
    fn read_i16_le(&mut self) -> std::io::Result<i16> {
        Ok(self.read_u16_le()? as i16)
    }
    #[inline]
    fn read_u32_le(&mut self) -> std::io::Result<u32> {
        let mut b = [0u8; 4];
        self.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }
    #[inline]
    fn read_i32_le(&mut self) -> std::io::Result<i32> {
        Ok(self.read_u32_le()? as i32)
    }
    #[inline]
    fn read_u64_le(&mut self) -> std::io::Result<u64> {
        let mut b = [0u8; 8];
        self.read_exact(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }
    #[inline]
    fn read_i64_le(&mut self) -> std::io::Result<i64> {
        Ok(self.read_u64_le()? as i64)
    }
    #[inline]
    fn read_f32_le(&mut self) -> std::io::Result<f32> {
        Ok(f32::from_le_bytes(self.read_u32_le()?.to_le_bytes()))
    }
    #[inline]
    fn read_f64_le(&mut self) -> std::io::Result<f64> {
        Ok(f64::from_le_bytes(self.read_u64_le()?.to_le_bytes()))
    }
    #[inline]
    fn read_string(&mut self) -> std::io::Result<String> {
        let len = self.read_u64_le()? as usize;
        let mut buf = vec![0u8; len];
        self.read_exact(&mut buf)?;
        String::from_utf8(buf).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}
impl<R: Read> ReadExt for R {}

#[inline]
fn read_kv<R: Read>(r: &mut R) -> Result<MetadataKv> {
    let key = r.read_string()?;
    let value_type = r.read_u32_le()?;
    let value = read_meta_value(r, value_type)
        .map_err(|e| Error::Gguf(format!("metadata '{key}': {e}")))?;
    Ok(MetadataKv {
        key,
        value_type,
        value,
    })
}

fn read_meta_value<R: Read>(r: &mut R, ty: u32) -> std::io::Result<MetaValue> {
    Ok(match ty {
        0 => MetaValue::U8(r.read_u8_le()?),
        1 => MetaValue::I8(r.read_u8_le()? as i8),
        2 => MetaValue::U16(r.read_u16_le()?),
        3 => MetaValue::I16(r.read_i16_le()?),
        4 => MetaValue::U32(r.read_u32_le()?),
        5 => MetaValue::I32(r.read_i32_le()?),
        6 => MetaValue::F32(r.read_f32_le()?),
        7 => MetaValue::Bool(r.read_u8_le()? != 0),
        8 => MetaValue::String(r.read_string()?),
        9 => {
            let elem_type = r.read_u32_le()?;
            let len = r.read_u64_le()? as usize;
            let mut elems = Vec::with_capacity(len);
            for _ in 0..len {
                elems.push(read_meta_value(r, elem_type)?);
            }
            MetaValue::Array(ArrayValue {
                elem_type,
                elements: elems,
            })
        }
        10 => MetaValue::U64(r.read_u64_le()?),
        11 => MetaValue::I64(r.read_i64_le()?),
        12 => MetaValue::F64(r.read_f64_le()?),
        other => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unknown metadata value type {other}"),
            ))
        }
    })
}

/// Batched tensor info read: read the variable-length name, then a tail of
/// `4 + n_dims*8 + 4 + 8` bytes in a single read_exact.
///
/// The tail is sized by `n_dims` (read first), not by the maximum 4 dims, so
/// that tensors with `n_dims < 4` don't bleed into the next tensor's header.
fn read_tensor_info<R: Read>(r: &mut R) -> Result<TensorInfo> {
    let name = r.read_string()?;
    let n_dims = r.read_u32_le()?;
    if n_dims > 4 {
        return Err(Error::Gguf(format!(
            "tensor '{name}' has {n_dims} dims (max 4)"
        )));
    }
    let mut dims_tail = [0u8; 4 * 8];
    r.read_exact(&mut dims_tail[..(n_dims as usize) * 8])?;
    let mut dims = [1u64; 4];
    for i in 0..n_dims as usize {
        let off = i * 8;
        dims[i] = u64::from_le_bytes(dims_tail[off..off + 8].try_into().unwrap());
    }
    let ty_raw = r.read_u32_le()?;
    let offset = r.read_u64_le()?;
    Ok(TensorInfo {
        name,
        n_dims,
        dims,
        ggml_type: GgmlType::from_u32(ty_raw),
        offset,
        n_elements: 0,
        byte_size: 0,
    })
}
