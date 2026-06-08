//! Spherical linear interpolation (SLERP) of two equal-length tensors.
//!
//! Per element `(a, b)`, we treat the pair as a 2D point `(a, b)` and
//! interpolate its **angle** by `t`. The closed-form (non-degenerate case)
//! is
//!
//! ```text
//!     result = (sin((1-t)Â·Î¸) / sin(Î¸)) Â· a
//!            + (sin(tÂ·Î¸)     / sin(Î¸)) Â· b
//!     where Î¸ = atan2(b, a)
//! ```
//!
//! When `Î¸` is near zero (the 2D point lies on the positive x-axis), the
//! ratio degenerates and we fall back to plain LERP: `(1-t)Â·a + tÂ·b`.

/// Newtype wrapper that lets you carry an SLERP `t` value in contexts that
/// expect a single value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlerpT(pub f32);

/// Returns the elementwise SLERP of `a` and `b` at parameter `t` âˆˆ [0, 1].
///
/// # Panics
/// Panics if `a.len() != b.len()` or if `t` is non-finite.
pub fn slerp_tensors(a: &[f32], b: &[f32], t: f32) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "slerp_tensors: length mismatch");
    assert!(t.is_finite(), "slerp_tensors: t must be finite");
    let mut out = Vec::with_capacity(a.len());
    for (x, y) in a.iter().zip(b.iter()) {
        out.push(slerp_scalar(*x, *y, t));
    }
    out
}

#[inline]
fn slerp_scalar(a: f32, b: f32, t: f32) -> f32 {
    let theta = b.atan2(a);
    // Threshold for "degenerate" angle. |sin(theta)| ~ |theta| near zero.
    if theta.abs() < 1e-6 {
        // Fall back to linear interpolation.
        (1.0 - t) * a + t * b
    } else {
        let s = theta.sin();
        let w0 = ((1.0 - t) * theta).sin() / s;
        let w1 = (t * theta).sin() / s;
        w0 * a + w1 * b
    }
}

#[cfg(test)]
#[path = "../../tests/unit/merge/slerp.rs"]
mod tests;
