//! Singular-value spectrum of a single tensor.
//!
//! Used by the analyzer to populate `BlockAnalysis::spectra` for a small
//! selection of high-signal tensors. The selection logic lives in
//! `analyzer.rs`; this module just computes the spectrum.
//!
//! The implementation calls the one-sided Jacobi SVD from `svd::linalg`.
//! Matrices larger than `max_elems` are row-subsampled first so the SVD
//! cost stays bounded.

use crate::svd::linalg::{svd_jacobi, Mat};

/// Compute the singular values of an `m x n` row-major matrix.
///
/// If `m > max_elems.unwrap_or(m)`, the first `max_elems` rows (in a
/// strided pattern) are kept and the rest dropped before the SVD runs.
/// Returns `None` for matrices with `m < 2 || n < 2`, or if the input
/// slice length doesn't match `m * n`, or if the SVD itself errors.
pub fn tensor_spectrum(
    values: &[f32],
    m: usize,
    n: usize,
    max_elems: Option<usize>,
) -> Option<Vec<f32>> {
    if m < 2 || n < 2 {
        return None;
    }
    if values.len() != m * n {
        return None;
    }

    let max_rows = max_elems.unwrap_or(m);
    let (rows_to_take, stride) = if m > max_rows && max_rows >= 2 {
        // Stride-sampled so we cover the whole matrix rather than the top
        // rows. We also force at least 2 rows so the SVD has something to
        // work with.
        let stride = m.div_ceil(max_rows).max(1);
        let take = m / stride;
        let take = take.max(2).min(m);
        (take, stride)
    } else {
        (m, 1)
    };

    let data: Vec<f32> = if rows_to_take == m && stride == 1 {
        values.to_vec()
    } else {
        let mut buf = Vec::with_capacity(rows_to_take * n);
        for r_idx in 0..rows_to_take {
            let src_row = r_idx * stride;
            let start = src_row * n;
            buf.extend_from_slice(&values[start..start + n]);
        }
        buf
    };

    let mat = Mat::from_vec(rows_to_take, n, data);
    let svd = svd_jacobi(&mat, 100, 1e-10).ok()?;
    // `svd_jacobi` always reports `n` singular values; for a tall (m < n)
    // matrix only the first `m` are meaningful. Truncate to the actual
    // rank so callers see a consistent answer.
    let k = rows_to_take.min(n);
    Some(svd.s.into_iter().take(k).collect())
}

#[cfg(test)]
mod tests {
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
}
