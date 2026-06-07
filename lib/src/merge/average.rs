//! Elementwise average (mean) of two tensors.

/// Returns the elementwise mean of `a` and `b`.
///
/// # Panics
/// Panics if `a.len() != b.len()`.
pub fn average_tensors(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "average_tensors: length mismatch");
    let mut out = Vec::with_capacity(a.len());
    for (x, y) in a.iter().zip(b.iter()) {
        out.push((x + y) * 0.5);
    }
    out
}

/// Writes the elementwise mean of `a` and `b` into `out`.
///
/// # Panics
/// Panics if `out.len() != a.len()` or `a.len() != b.len()`.
pub fn average_into(out: &mut [f32], a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "average_into: length mismatch");
    assert_eq!(out.len(), a.len(), "average_into: out length mismatch");
    for ((o, x), y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
        *o = (x + y) * 0.5;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn average_basic() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [3.0f32, 2.0, 1.0];
        let r = average_tensors(&a, &b);
        assert_eq!(r, vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn average_with_negatives() {
        let a = [-1.0f32, 0.5, 2.0];
        let b = [1.0f32, -0.5, -2.0];
        let r = average_tensors(&a, &b);
        assert_eq!(r, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn average_into_reuses_buffer() {
        let a = [1.0f32, 4.0];
        let b = [3.0f32, 0.0];
        let mut out = [99.0f32, 99.0];
        average_into(&mut out, &a, &b);
        assert_eq!(out, [2.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn average_panics_on_mismatch() {
        let _ = average_tensors(&[1.0, 2.0], &[1.0, 2.0, 3.0]);
    }
}
