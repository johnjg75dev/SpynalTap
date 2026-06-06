use spynaltap::formats::gguf::dequant;
use spynaltap::formats::gguf::types::GgmlType;

#[test]
fn dequant_f16_roundtrip() {
    // Build input: 8 f16 values encoded as bytes.
    let inputs: [u16; 8] = [
        0x3c00, // 1.0
        0x4000, // 2.0
        0xbc00, // -1.0
        0x0000, // 0.0
        0x7c00, // inf
        0x7e00, // nan
        0x3a00, // ~0.000488
        0x3555, // small
    ];
    let mut bytes = Vec::with_capacity(16);
    for w in inputs {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    let v = dequant::dequantize(GgmlType::F16, &bytes, None).unwrap();
    assert_eq!(v.len(), 8);
    assert!((v[0] - 1.0).abs() < 1e-6);
    assert!((v[1] - 2.0).abs() < 1e-6);
    assert!((v[2] + 1.0).abs() < 1e-6);
    assert_eq!(v[3], 0.0);
    assert!(v[4].is_infinite() && v[4] > 0.0);
    assert!(v[5].is_nan());
}

#[test]
fn dequant_f32_basic() {
    let mut bytes = [0u8; 16];
    let vals = [1.0f32, -2.5, 0.0, 1e6];
    for (i, v) in vals.iter().enumerate() {
        bytes[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
    }
    let out = dequant::dequantize(GgmlType::F32, &bytes, None).unwrap();
    assert_eq!(out, vals.to_vec());
}

#[test]
fn dequant_q4_0_block() {
    // 1 Q4_0 block: 18 bytes = f16 d (2) + 16 bytes of 4-bit quants = 32 elements.
    let mut bytes = vec![0u8; 18];
    // d = 1.0 (f16 = 0x3c00).
    bytes[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
    // Set all nibbles to 8 (i.e. quantized value 0 after offset by 8).
    for i in 0..16 {
        bytes[2 + i] = 0x88;
    }
    let v = dequant::dequantize(GgmlType::Q4_0, &bytes, None).unwrap();
    assert_eq!(v.len(), 32);
    for &x in &v {
        assert_eq!(x, 0.0);
    }
}

#[test]
fn dequant_q8_0_block() {
    // 1 Q8_0 block: 34 bytes = f16 d (2) + 32 i8 qs = 32 elements.
    let mut bytes = vec![0u8; 34];
    // d = 1.0
    bytes[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
    // qs = 4 for all elements
    for i in 0..32 {
        bytes[2 + i] = 4;
    }
    let v = dequant::dequantize(GgmlType::Q8_0, &bytes, None).unwrap();
    assert_eq!(v.len(), 32);
    for &x in &v {
        assert_eq!(x, 4.0);
    }
}

#[test]
fn dequant_max_elems_truncates() {
    let bytes = vec![0u8; 32]; // 8 f32 values
    let v = dequant::dequantize(GgmlType::F32, &bytes, Some(3)).unwrap();
    assert_eq!(v.len(), 3);
}

#[test]
fn dequant_unsupported_returns_none() {
    let bytes = vec![0u8; 16];
    let v = dequant::dequantize(GgmlType::Iq1S, &bytes, None);
    assert!(v.is_none());
}
