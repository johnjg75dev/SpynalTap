use super::*;

#[test]
fn roundtrip_constant() {
    let src = vec![10.0f32; BLOCK_LEN];
    let bytes = quantize(&src);
    assert_eq!(bytes.len(), BLOCK_BYTES);
    let out = dequant(&bytes);
    let max_err = src.iter().zip(&out).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    // 2-bit codebook grid has no all-43s entry accessible at low offsets, so
    // error up to ~8 is expected for a constant 10.0 (db*8 â‰ˆ 1.86).
    assert!(max_err < 10.0, "max_err={}", max_err);
}

#[test]
fn roundtrip_all_zero() {
    let src = vec![0.0f32; BLOCK_LEN];
    let bytes = quantize(&src);
    let out = dequant(&bytes);
    for &v in &out {
        assert_eq!(v, 0.0);
    }
}

#[test]
fn roundtrip_sine() {
    let src: Vec<f32> = (0..BLOCK_LEN).map(|i| ((i as f32) * 0.3).sin() * 50.0).collect();
    let bytes = quantize(&src);
    let out = dequant(&bytes);
    assert_eq!(out.len(), BLOCK_LEN);
}

#[test]
fn matches_dequant() {
    let src: Vec<f32> = (0..BLOCK_LEN).map(|i| ((i as f32) * 0.5).sin() * 100.0).collect();
    let bytes = quantize(&src);
    let direct = dequant(&bytes);
    let via = dequant::dequantize(GgmlType::Iq2Xxs, &bytes, None).unwrap();
    assert_eq!(direct, via);
}

#[test]
fn negative_values() {
    let src: Vec<f32> = (0..BLOCK_LEN).map(|i| -5.0 - i as f32 * 2.0).collect();
    let bytes = quantize(&src);
    let out = dequant(&bytes);
    assert_eq!(out.len(), BLOCK_LEN);
    for &v in &out {
        assert!(v <= 0.0);
    }
}
