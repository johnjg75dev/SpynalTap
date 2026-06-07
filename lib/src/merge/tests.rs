//! Cross-module integration tests for the `merge` subsystem.
//!
//! Each submodule also has its own `#[cfg(test)] mod tests` block; the
//! tests here exercise the public surface end-to-end across modules.

use crate::merge::{
    average_into, average_tensors, merge_experts, slerp_tensors, MergeStrategy, MoEMergeStrategy,
    MoEWeights, WeightFormat,
};

#[test]
fn average_then_slerp_pipeline() {
    // Compose the two primitives: first average two tensors, then slerp
    // the result with a third. The output should be a deterministic,
    // finite vector with the expected length.
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [5.0f32, 6.0, 7.0, 8.0];
    let c = [0.0f32, 0.0, 0.0, 0.0];

    let avg = average_tensors(&a, &b);
    assert_eq!(avg, vec![3.0, 4.0, 5.0, 6.0]);

    let mixed = slerp_tensors(&avg, &c, 0.25);
    assert_eq!(mixed.len(), 4);
    for x in &mixed {
        assert!(x.is_finite());
    }
}

#[test]
fn average_into_matches_average_tensors() {
    let a = [1.0f32, -2.0, 3.5];
    let b = [-1.0f32, 2.0, 0.5];
    let expected = average_tensors(&a, &b);

    let mut buf = vec![0.0f32; 3];
    average_into(&mut buf, &a, &b);
    assert_eq!(buf, expected);
}

#[test]
fn moe_average_and_similarity_agree_on_identical_experts() {
    // If all experts are identical, average == similarity{topk=any}
    // == a single expert.
    let expert: Vec<f32> = (0..6).map(|i| i as f32 * 0.5 + 1.0).collect();
    let moe = MoEWeights::new(vec![expert.clone(), expert.clone(), expert.clone()], (2, 3));

    let avg = merge_experts(&moe, MoEMergeStrategy::Average);
    let sim = merge_experts(
        &moe,
        MoEMergeStrategy::Similarity { keep_top_k: 2 },
    );
    for (a, s) in avg.iter().zip(sim.iter()) {
        assert!((a - s).abs() < 1e-5);
    }
    for (x, e) in avg.iter().zip(expert.iter()) {
        assert!((*x - e).abs() < 1e-5);
    }
}

#[test]
fn strategy_slerp_constructor() {
    let s = MergeStrategy::slerp(0.42);
    assert!(matches!(s, MergeStrategy::Slerp(t) if (t - 0.42).abs() < 1e-6));
    let _ = WeightFormat::default();
    let _ = MergeStrategy::Average;
}
