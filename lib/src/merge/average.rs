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
#[path = "../../tests/unit/merge/average.rs"]
mod tests;
