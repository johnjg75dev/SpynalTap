use super::*;

fn build_matrix(values: &[f32], m: usize, n: usize) -> Vec<f32> {
    assert_eq!(values.len(), m * n);
    values.to_vec()
}

#[test]
fn spectrum_identity() {
    let m = 4usize;
    let n = 4usize;
    let mut data = vec![0.0f32; m * n];
    for i in 0..m {
        data[i * n + i] = 1.0;
    }
    let s = tensor_spectrum(&data, m, n, None).expect("spectrum");
    assert_eq!(s.len(), n);
    for &v in &s {
        assert!((v - 1.0).abs() < 1e-3, "got {v}");
    }
}

#[test]
fn spectrum_diagonal() {
    let m = 3usize;
    let n = 3usize;
    let mut data = vec![0.0f32; m * n];
    data[0 * n + 0] = 2.0;
    data[1 * n + 1] = 3.0;
    data[2 * n + 2] = 5.0;
    let s = tensor_spectrum(&data, m, n, None).expect("spectrum");
    assert_eq!(s.len(), 3);
    // Jacobi SVD returns values sorted descending.
    assert!((s[0] - 5.0).abs() < 1e-3, "got {}", s[0]);
    assert!((s[1] - 3.0).abs() < 1e-3, "got {}", s[1]);
    assert!((s[2] - 2.0).abs() < 1e-3, "got {}", s[2]);
}

#[test]
fn spectrum_too_small() {
    let data = vec![1.0f32];
    assert!(tensor_spectrum(&data, 1, 1, None).is_none());
    // 1x5 is also degenerate (n >= 2 holds but m < 2).
    let data2 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    assert!(tensor_spectrum(&data2, 1, 5, None).is_none());
}

#[test]
fn spectrum_sampling() {
    let m = 100usize;
    let n = 100usize;
    // Fill with a deterministic pseudo-random pattern.
    let mut data = vec![0.0f32; m * n];
    for i in 0..m * n {
        data[i] = (((i as u32).wrapping_mul(2654435761)) as f32 / (u32::MAX as f32)) * 2.0
            - 1.0;
    }
    let max = 50usize;
    let s = tensor_spectrum(&data, m, n, Some(max)).expect("spectrum");
    // With max_rows=50 and m=100, we keep 50 rows (stride=2).
    assert_eq!(s.len(), max);
}

#[test]
fn spectrum_full_size() {
    // 4x4 full matrix (no sampling) returns 4 singular values.
    let m = 4usize;
    let n = 4usize;
    let data = build_matrix(
        &[1.0, 0.5, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.5, 1.0],
        m,
        n,
    );
    let s = tensor_spectrum(&data, m, n, None).expect("spectrum");
    assert_eq!(s.len(), n);
    for &v in &s {
        assert!(v.is_finite() && v >= 0.0);
    }
}
