use super::*;

#[test]
fn roundtrip_all_positive() {
    let src = vec![1.0f32; QK_K];
    let bytes = quantize(&src);
    assert_eq!(bytes.len(), BLOCK_BYTES);
    let out = dequant(&bytes);
    for v in &out {
        assert!((v - 1.0).abs() < 0.001 || v.abs() < 0.001);
    }
}

#[test]
fn roundtrip_all_zero() {
    let src = vec![0.0f32; QK_K];
    let bytes = quantize(&src);
    let out = dequant(&bytes);
    for &v in &out {
        assert_eq!(v, 0.0);
    }
}

#[test]
fn roundtrip_all_negative() {
    let src = vec![-2.0f32; QK_K];
    let bytes = quantize(&src);
    assert_eq!(bytes.len(), BLOCK_BYTES);
    let out = dequant(&bytes);
    for v in &out {
        assert!(v <= &0.0f32);
    }
}

#[test]
fn roundtrip_mixed() {
    let src: Vec<f32> = (0..QK_K).map(|i| ((i as f32) * 0.1 - 0.5).round() * 2.0).collect();
    let bytes = quantize(&src);
    let out = dequant(&bytes);
    assert_eq!(out.len(), QK_K);
}

#[test]
fn matches_dequant() {
    let src: Vec<f32> = (0..QK_K).map(|i| ((i as f32) * 0.07).sin() * 3.0).collect();
    let bytes = quantize(&src);
    let direct = dequant(&bytes);
    let via_dispatch = dequant::dequantize(GgmlType::Tq2_0, &bytes, None).unwrap();
    assert_eq!(direct, via_dispatch);
}
