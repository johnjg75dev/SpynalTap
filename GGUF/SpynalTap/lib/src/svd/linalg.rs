//! Pure-Rust linear-algebra primitives used by the SVD compressor.
//!
//! We deliberately avoid `ndarray` / `nalgebra` / LAPACK to keep the binary
//! lean. The two algorithms implemented here are:
//!
//! * **One-sided Jacobi SVD** — accurate, simple, fast for matrices up to a
//!   few thousand rows/columns (the size of typical transformer attention /
//!   FFN projections).
//! * **Randomized SVD** (Halko et al., 2011) — used for matrices where Jacobi
//!   would be too slow. Trades some accuracy for an O(m n log k) cost.
//!
//! All matrices are stored in row-major `Vec<f32>` with `n_rows` and `n_cols`
//! passed alongside. Element `(i, j)` is at index `i * n_cols + j`.

use crate::error::{Error, Result};

/// Row-major dense matrix in `f32`.
#[derive(Debug, Clone)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Mat {
    #[inline]
    pub fn new(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![0.0; rows * cols] }
    }
    #[inline]
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        debug_assert_eq!(data.len(), rows * cols);
        Self { rows, cols, data }
    }
    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f32 { self.data[r * self.cols + c] }
    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: f32) { self.data[r * self.cols + c] = v; }

    /// Frobenius norm.
    pub fn norm_fro(&self) -> f64 {
        let mut s = 0.0f64;
        for &x in &self.data { s += (x as f64) * (x as f64); }
        s.sqrt()
    }

    /// `out = a * b`. `a` is m x k, `b` is k x n, `out` is m x n.
    pub fn matmul_into(a: &Mat, b: &Mat, out: &mut Mat) {
        assert_eq!(a.cols, b.rows, "matmul: inner dims must match");
        assert_eq!(out.rows, a.rows, "matmul: out rows must match a rows");
        assert_eq!(out.cols, b.cols, "matmul: out cols must match b cols");
        let m = a.rows; let k = a.cols; let n = b.cols;
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..k { s += a.data[i * k + p] * b.data[p * n + j]; }
                out.data[i * n + j] = s;
            }
        }
    }
}

/// One-sided Jacobi SVD result. `s.len() == u.cols == vt.rows`.
///
/// `A == U * diag(s) * Vt` (within numerical tolerance). The decomposition is
/// economy-size: `u.rows == a.rows`, `u.cols == vt.rows == s.len() == rank`
/// where `rank = min(a.rows, a.cols)`.
#[derive(Debug, Clone)]
pub struct Svd {
    pub u: Mat,   // m x k
    pub s: Vec<f32>, // k
    pub vt: Mat,  // k x n
}

/// Run one-sided Jacobi SVD on a row-major matrix `a` (m x n).
///
/// `max_sweeps` caps the number of full sweeps (one sweep = n*(n-1)/2 pair
/// rotations). `tol` is the convergence threshold on the off-diagonal
/// Frobenius norm of `A^T A`, relative to the diagonal norm.
pub fn svd_jacobi(a: &Mat, max_sweeps: usize, tol: f64) -> Result<Svd> {
    let m = a.rows;
    let n = a.cols;
    if m == 0 || n == 0 {
        return Err(Error::Svd("empty matrix".into()));
    }

    // Work copy of A; V starts as identity.
    let mut work = a.clone();
    let mut v = Mat::new(n, n);
    for i in 0..n { v.set(i, i, 1.0); }

    let mut sweep = 0;
    let target_off = (a.norm_fro() * a.norm_fro()) * tol * tol;

    while sweep < max_sweeps {
        sweep += 1;
        // Compute A^T A diagonal (squared column norms) and off-diagonal norm.
        let mut col_norms_sq = vec![0.0f64; n];
        for c in 0..n {
            let mut s = 0.0f64;
            for r in 0..m { let x = work.data[r * n + c] as f64; s += x * x; }
            col_norms_sq[c] = s;
        }
        let mut off_sq = 0.0f64;
        for i in 0..n { for j in (i + 1)..n {
            let mut s = 0.0f64;
            for r in 0..m {
                let x = work.data[r * n + i] as f64;
                let y = work.data[r * n + j] as f64;
                s += x * y;
            }
            off_sq += s * s;
        }}
        if off_sq <= target_off { break; }

        // One sweep: for each pair (i, j) with i < j, apply a 2x2 Jacobi
        // rotation to the columns of `work` and to `v`.
        let mut any_change = false;
        for i in 0..n {
            for j in (i + 1)..n {
                // Reuse the cached diagonal; recompute the (i, j) element of A^T A.
                let mut alpha = 0.0f64; let mut beta = 0.0f64; let mut gamma = 0.0f64;
                for r in 0..m {
                    let x = work.data[r * n + i] as f64;
                    let y = work.data[r * n + j] as f64;
                    alpha += x * x;
                    beta  += y * y;
                    gamma += x * y;
                }
                if gamma == 0.0 { continue; }

                // Closed-form Jacobi rotation that diagonalizes [[alpha, gamma], [gamma, beta]].
                let (c_rot, s_rot) = jacobi_2x2(alpha, beta, gamma);
                if s_rot.abs() < 1e-30 { continue; }
                any_change = true;

                // Apply to `work` columns i, j in place.
                for r in 0..m {
                    let xi = work.data[r * n + i];
                    let xj = work.data[r * n + j];
                    work.data[r * n + i] = (c_rot * xi + s_rot * xj) as f32;
                    work.data[r * n + j] = (-s_rot * xi + c_rot * xj) as f32;
                }
                // Apply the same rotation to V's columns i, j.
                for r in 0..n {
                    let vi = v.data[r * n + i];
                    let vj = v.data[r * n + j];
                    v.data[r * n + i] = (c_rot * vi + s_rot * vj) as f32;
                    v.data[r * n + j] = (-s_rot * vi + c_rot * vj) as f32;
                }
                // Update cached norms.
                col_norms_sq[i] = alpha;
                col_norms_sq[j] = beta;
                let _ = gamma; // gamma -> 0 after rotation; ignored.
            }
        }
        if !any_change { break; }
    }

    // Singular values = column norms of `work`. U = normalized columns of `work`.
    let mut s = Vec::with_capacity(n);
    let mut u = Mat::new(m, n);
    for c in 0..n {
        let mut nrm = 0.0f64;
        for r in 0..m { let x = work.data[r * n + c] as f64; nrm += x * x; }
        let nrm = nrm.sqrt();
        s.push(nrm as f32);
        let inv = if nrm > 0.0 { 1.0 / nrm as f32 } else { 0.0 };
        for r in 0..m {
            u.data[r * n + c] = work.data[r * n + c] * inv;
        }
    }

    // Sort descending by singular value, propagating the same permutation to U and V^T.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| s[b].partial_cmp(&s[a]).unwrap_or(std::cmp::Ordering::Equal));
    let s_sorted: Vec<f32> = order.iter().map(|&i| s[i]).collect();
    let mut u_sorted = Mat::new(m, n);
    for (new_c, &old_c) in order.iter().enumerate() {
        for r in 0..m {
            u_sorted.data[r * n + new_c] = u.data[r * n + old_c];
        }
    }
    // V is the accumulated rotation; V^T is what we need. Apply the same perm to V's columns.
    let mut v_perm = Mat::new(n, n);
    for (new_c, &old_c) in order.iter().enumerate() {
        for r in 0..n {
            v_perm.data[r * n + new_c] = v.data[r * n + old_c];
        }
    }
    // Transpose to get V^T.
    let mut vt = Mat::new(n, n);
    for i in 0..n { for j in 0..n { vt.data[j * n + i] = v_perm.data[i * n + j]; } }

    Ok(Svd { u: u_sorted, s: s_sorted, vt })
}

/// Closed-form Jacobi rotation for the symmetric 2x2 `[[a, g], [g, b]]`.
/// Returns `(cos, sin)` such that the rotation `R = [[c, s], [-s, c]]`
/// satisfies `R^T * M * R = diag(a', b')` with `a' >= b'`.
#[inline]
fn jacobi_2x2(a: f64, b: f64, g: f64) -> (f32, f32) {
    if g.abs() < 1e-30 { return (1.0, 0.0); }
    // tan(2θ) = 2g / (a - b); pick θ so the larger eigenvalue lands at (0, 0).
    let tau = (a - b) / (2.0 * g);
    let t = if tau >= 0.0 {
        1.0 / (tau + (1.0 + tau * tau).sqrt())
    } else {
        -1.0 / (-tau + (1.0 + tau * tau).sqrt())
    };
    let c = 1.0 / (1.0 + t * t).sqrt();
    let s = t * c;
    (c as f32, s as f32)
}

/// Compute the rank `k` truncated SVD using the randomized algorithm
/// (Halko, Martinsson, Tropp, 2011, "Finding structure with randomness").
///
/// 1. Draw an n x (k + p) Gaussian test matrix Omega.
/// 2. Form Y = (A A^T)^q A Omega via q power iterations.
/// 3. Orthonormalize Y's columns to obtain an approximate column-space basis Q.
/// 4. Form B = Q^T A and compute its small SVD; lift back to U of A.
pub fn svd_randomized(
    a: &Mat,
    target_rank: usize,
    oversample: usize,
    power_iters: usize,
    seed: u64,
) -> Result<Svd> {
    let m = a.rows;
    let n = a.cols;
    if m == 0 || n == 0 {
        return Err(Error::Svd("empty matrix".into()));
    }
    let k = target_rank.min(m).min(n);
    if k == 0 {
        return Err(Error::Svd("target rank must be > 0".into()));
    }
    let l = (k + oversample).min(n).min(m);

    // 1) Draw Omega: n x l Gaussian (deterministic via xorshift).
    let mut rng = XorShift::new(seed);
    let mut omega = Mat::new(n, l);
    for j in 0..l { for i in 0..n { omega.data[i * l + j] = rng.gauss(); } }

    // 2) Y = A * Omega  (m x l)
    let mut y = Mat::new(m, l);
    Mat::matmul_into(a, &omega, &mut y);

    // Power iterations: Y <- (A A^T)^q * Y
    let at = transpose(a);
    for _ in 0..power_iters {
        // Z = A^T * Y  (n x l)
        let mut z = Mat::new(n, l);
        Mat::matmul_into(&at, &y, &mut z);
        // Y = A * Z
        y = Mat::new(m, l);
        Mat::matmul_into(a, &z, &mut y);
    }

    // 3) Orthonormalize Y's columns (modified Gram-Schmidt).
    let q = orthonormalize_cols(&y);

    // 4) B = Q^T * A  (l x n), then SVD of B.
    let qt = transpose(&q);
    let mut b = Mat::new(l, n);
    Mat::matmul_into(&qt, a, &mut b);
    let b_svd = svd_jacobi(&b, 100, 1e-12)?;
    // Truncate to k.
    let ks = k.min(b_svd.s.len());
    let u_tilde = slice_cols(&b_svd.u, 0, ks);
    let s_k: Vec<f32> = b_svd.s.iter().take(ks).copied().collect();
    let vt = slice_rows(&b_svd.vt, 0, ks);

    // U = Q * U_tilde (m x ks)
    let mut u = Mat::new(m, ks);
    Mat::matmul_into(&q, &u_tilde, &mut u);
    Ok(Svd { u, s: s_k, vt })
}

fn transpose(a: &Mat) -> Mat {
    let mut t = Mat::new(a.cols, a.rows);
    for i in 0..a.rows { for j in 0..a.cols { t.data[j * a.rows + i] = a.data[i * a.cols + j]; } }
    t
}

pub(crate) fn slice_cols(a: &Mat, start: usize, count: usize) -> Mat {
    let mut out = Mat::new(a.rows, count);
    for r in 0..a.rows { for c in 0..count { out.data[r * count + c] = a.data[r * a.cols + start + c]; } }
    out
}

pub(crate) fn slice_rows(a: &Mat, start: usize, count: usize) -> Mat {
    let mut out = Mat::new(count, a.cols);
    for r in 0..count { for c in 0..a.cols { out.data[r * a.cols + c] = a.data[(start + r) * a.cols + c]; } }
    out
}

fn orthonormalize_cols(a: &Mat) -> Mat {
    let m = a.rows;
    let n = a.cols;
    // Use classical Gram-Schmidt (CGS) with two passes of re-orthogonalization.
    // The first pass uses the original input for the inner product; the second
    // pass uses the already-orthogonalized q for the inner product, which
    // stabilizes the result for near-degenerate inputs (a common situation
    // for the randomized SVD's `Y` matrix, whose rank can be much smaller
    // than its column count).
    let mut q = a.clone();
    for _pass in 0..2 {
        for j in 0..n {
            for k in 0..j {
                let mut dot = 0.0f64;
                let src: &[f32] = if _pass == 0 { &a.data } else { &q.data };
                for r in 0..m { dot += (q.data[r * n + k] as f64) * (src[r * n + j] as f64); }
                for r in 0..m { q.data[r * n + j] -= (dot as f32) * q.data[r * n + k]; }
            }
            let mut nrm = 0.0f64;
            for r in 0..m { let x = q.data[r * n + j] as f64; nrm += x * x; }
            let nrm = nrm.sqrt();
            if nrm > 0.0 {
                let inv = 1.0 / nrm as f32;
                for r in 0..m { q.data[r * n + j] *= inv; }
            } else {
                // Column is in the span of previous ones; zero it.
                for r in 0..m { q.data[r * n + j] = 0.0; }
            }
        }
    }
    q
}

/// Tiny deterministic xorshift64* PRNG.
struct XorShift(u64);
impl XorShift {
    fn new(seed: u64) -> Self { Self(seed.max(1)) }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Box-Muller standard normal.
    fn gauss(&mut self) -> f32 {
        loop {
            let u1 = ((self.next_u64() >> 11) as f64 + 1.0) / ((1u64 << 53) as f64);
            let u2 = (self.next_u64() as f64) / (u64::MAX as f64);
            if u1 > 0.0 {
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = 2.0 * std::f64::consts::PI * u2;
                return (r * theta.cos()) as f32;
            }
        }
    }
}

/// Pick the smallest rank `k` such that `sum_{i<k} s_i^2 >= energy * total`.
/// Returns at least 1 (or `min_k` if provided), and at most `max_k`.
pub fn rank_for_energy(s: &[f32], energy: f64, min_k: usize, max_k: usize) -> usize {
    if s.is_empty() { return min_k.max(1); }
    let total: f64 = s.iter().map(|x| (*x as f64) * (*x as f64)).sum();
    if total <= 0.0 { return min_k.max(1); }
    let mut acc = 0.0f64;
    for (i, &v) in s.iter().enumerate() {
        acc += (v as f64) * (v as f64);
        if acc / total >= energy {
            return (i + 1).clamp(min_k.max(1), max_k.max(1));
        }
    }
    s.len().clamp(min_k.max(1), max_k.max(1))
}

/// Compose two factors back into a single matrix. `a` is m x k, `b` is k x n.
pub fn reconstruct(a: &Mat, b: &Mat) -> Mat {
    let mut out = Mat::new(a.rows, b.cols);
    Mat::matmul_into(a, b, &mut out);
    out
}

/// Pack an SVD into a low-rank pair `(a, b)` such that `a * b ~ U * diag(s) * V^T`.
/// `a` is m x k, `b` is k x n. Uses the symmetric "square-root" scaling so that
/// `||a||_2 ~ ||b||_2 ~ sqrt(max singular value)`.
pub fn pack_lowrank(svd: &Svd) -> (Mat, Mat) {
    let m = svd.u.rows;
    let n = svd.vt.cols;
    let k = svd.s.len();
    let mut a = Mat::new(m, k);
    let mut b = Mat::new(k, n);
    for j in 0..k {
        let s = svd.s[j].max(0.0).sqrt();
        for i in 0..m { a.data[i * k + j] = svd.u.data[i * k + j] * s; }
        for jj in 0..n { b.data[j * n + jj] = svd.vt.data[j * n + jj] * s; }
    }
    (a, b)
}
