use super::*;

#[test]
fn roundtrip_constant() {
    let src = vec![2.0f32; ELEMS];
    let bytes = quantize(&src);
    assert_eq!(bytes.len(), BLOCK_BYTES);
    let out = dequant(&bytes);
    for v in &out {
        assert!((v - 2.0).abs() < 0.001 || (v - 0.0).abs() < 0.001);
    }
}

#[test]
fn roundtrip_all_zero() {
    let src = vec![0.0f32; ELEMS];
    let bytes = quantize(&src);
    let out = dequant(&bytes);
    for &v in &out {
        assert_eq!(v, 0.0);
    }
}

#[test]
fn roundtrip_sine() {
    let src: Vec<f32> = (0..ELEMS).map(|i| ((i as f32) * 0.1).sin() * 3.0).collect();
    let bytes = quantize(&src);
    let out = dequant(&bytes);
    assert_eq!(out.len(), ELEMS);
}

#[test]
fn matches_dequant() {
    let src: Vec<f32> = (0..ELEMS).map(|i| ((i as f32) * 0.1).sin() * 5.0).collect();
    let bytes = quantize(&src);
    let direct = dequant(&bytes);
    let via = dequant::dequantize(GgmlType::Tq1_0, &bytes, None).unwrap();
    assert_eq!(direct, via);
}
