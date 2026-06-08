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
#[path = "../../tests/unit/analysis/spectrum.rs"]
mod tests;
