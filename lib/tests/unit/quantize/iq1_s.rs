use super::*;

#[test]
fn roundtrip_constant() {
    let src = vec![1.0f32; QK_K];
    let bytes = quantize(&src);
    assert_eq!(bytes.len(), BLOCK_BYTES);
    let out = dequant(&bytes);
    assert_eq!(out.len(), QK_K);
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
fn roundtrip_sine() {
    let src: Vec<f32> = (0..QK_K).map(|i| ((i as f32) * 0.07).sin() * 500.0).collect();
    let bytes = quantize(&src);
    let out = dequant(&bytes);
    assert_eq!(out.len(), QK_K);
}

#[test]
fn matches_dequant() {
    let src: Vec<f32> = (0..QK_K).map(|i| ((i as f32) * 0.05).sin() * 300.0).collect();
    let bytes = quantize(&src);
    let direct = dequant(&bytes);
    let via = dequant::dequantize(GgmlType::Iq1S, &bytes, None).unwrap();
    assert_eq!(direct, via);
}
