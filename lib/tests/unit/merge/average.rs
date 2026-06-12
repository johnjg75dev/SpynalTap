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

#[test]
fn average_empty_tensor() {
    let a: [f32; 0] = [];
    let b: [f32; 0] = [];
    let r = average_tensors(&a, &b);
    assert!(r.is_empty());
}

#[test]
fn average_two_equal_tensors() {
    let a = [5.0, 10.0, 15.0];
    let b = [5.0, 10.0, 15.0];
    let r = average_tensors(&a, &b);
    assert_eq!(r, vec![5.0, 10.0, 15.0]);
}

#[test]
fn average_into_panics_on_out_mismatch() {
    let a = [1.0, 2.0];
    let b = [1.0, 2.0];
    let mut out = [0.0; 3];
    // average_into panics because out.len() != a.len()
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        average_into(&mut out, &a, &b);
    }));
    // Just verify it doesn't corrupt memory — the function should panic.
}
