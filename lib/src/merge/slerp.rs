//! Spherical linear interpolation (SLERP) of two equal-length tensors.
//!
//! Per element `(a, b)`, we treat the pair as a 2D point `(a, b)` and
//! interpolate its **angle** by `t`. The closed-form (non-degenerate case)
//! is
//!
//! ```text
//!     result = (sin((1-t)·θ) / sin(θ)) · a
//!            + (sin(t·θ)     / sin(θ)) · b
//!     where θ = atan2(b, a)
//! ```
//!
//! When `θ` is near zero (the 2D point lies on the positive x-axis), the
//! ratio degenerates and we fall back to plain LERP: `(1-t)·a + t·b`.

/// Newtype wrapper that lets you carry an SLERP `t` value in contexts that
/// expect a single value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlerpT(pub f32);

/// Returns the elementwise SLERP of `a` and `b` at parameter `t` ∈ [0, 1].
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
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn slerp_t_zero_returns_a() {
        let a = [1.0f32, 2.0, 3.0, -1.0];
        let b = [0.0f32, 5.0, -2.0, 4.0];
        let r = slerp_tensors(&a, &b, 0.0);
        for (got, want) in r.iter().zip(a.iter()) {
            assert!(approx(*got, *want), "t=0: got {got} want {want}");
        }
    }

    #[test]
    fn slerp_t_one_returns_b() {
        let a = [1.0f32, 2.0, 3.0, -1.0];
        let b = [0.0f32, 5.0, -2.0, 4.0];
        let r = slerp_tensors(&a, &b, 1.0);
        for (got, want) in r.iter().zip(b.iter()) {
            assert!(approx(*got, *want), "t=1: got {got} want {want}");
        }
    }

    #[test]
    fn slerp_t_half_orthogonal_pair() {
        // a=0, b=1: 2D point lies on the +y axis, θ = π/2.
        // SLERP at t=0.5 gives the angular midpoint, which on a unit
        // circle is (cos(π/4), sin(π/4)) = (√2/2, √2/2). The scalar
        // form returns √2/2.
        let r = slerp_tensors(&[0.0], &[1.0], 0.5);
        let expected = (std::f32::consts::FRAC_PI_4).sin();
        assert!(
            approx(r[0], expected),
            "t=0.5 (0→1): got {} want {}",
            r[0],
            expected
        );
    }

    #[test]
    fn slerp_t_half_general_pair() {
        // a=3, b=4: r=5, θ=atan2(4, 3).
        // At t=0.5, weights = sin(0.5θ)/sin(θ).
        let a = 3.0f32;
        let b = 4.0f32;
        let theta = b.atan2(a);
        let w = (0.5f32 * theta).sin() / theta.sin();
        let expected = w * a + w * b;
        let r = slerp_tensors(&[a], &[b], 0.5);
        assert!(approx(r[0], expected), "got {} want {}", r[0], expected);
        // And it should NOT equal the linear midpoint.
        assert!((r[0] - 3.5).abs() > 1e-3);
    }

    #[test]
    fn slerp_degenerate_axis_falls_back_to_lerp() {
        // a>0, b=0: 2D point lies on the +x axis, θ ≈ 0.
        let a = 2.0f32;
        let b = 0.0f32;
        let r0 = slerp_tensors(&[a], &[b], 0.0);
        let r1 = slerp_tensors(&[a], &[b], 1.0);
        let r5 = slerp_tensors(&[a], &[b], 0.5);
        assert!(approx(r0[0], a));
        assert!(approx(r1[0], b));
        assert!(approx(r5[0], 0.5 * a + 0.5 * b));
    }

    #[test]
    fn slerp_t_one_quarter_skew() {
        // a=1, b=2: not orthogonal. At t=0.25, result should lie between
        // a and b and be finite.
        let r = slerp_tensors(&[1.0], &[2.0], 0.25);
        assert!(r[0].is_finite());
        assert!(r[0] > 1.0 && r[0] < 2.0, "got {}", r[0]);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn slerp_panics_on_mismatch() {
        let _ = slerp_tensors(&[1.0, 2.0], &[1.0], 0.5);
    }

    #[test]
    fn slerpt_newtype() {
        let s = SlerpT(0.3);
        assert!(approx(s.0, 0.3));
    }
}
