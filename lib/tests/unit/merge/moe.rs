use super::*;

fn experts() -> MoEWeights {
    // Three experts, each a 2x3 matrix.
    let e0: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let e1: Vec<f32> = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let e2: Vec<f32> = vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0];
    MoEWeights::new(vec![e0, e1, e2], (2, 3))
}

#[test]
fn merge_average_basic() {
    let moe = experts();
    let m = merge_experts(&moe, MoEMergeStrategy::Average);
    // (1+2+10)/3, (2+3+10)/3, ... -> [13/3, 15/3, 17/3, 19/3, 21/3, 23/3]
    let want: Vec<f32> = vec![13.0, 15.0, 17.0, 19.0, 21.0, 23.0]
        .into_iter()
        .map(|x| x / 3.0)
        .collect();
    for (g, w) in m.iter().zip(want.iter()) {
        assert!((g - w).abs() < 1e-5, "got {g} want {w}");
    }
    assert_eq!(m.len(), 6);
}

#[test]
fn merge_similarity_keeps_top_k() {
    // e0 and e1 are similar (close numbers), e2 is a constant
    // constant vector that is far from both. The "mean similarity to
    // others" should rank e0 and e1 highest.
    let moe = experts();
    let m = merge_experts(&moe, MoEMergeStrategy::Similarity { keep_top_k: 2 });
    // Result should be the average of e0 and e1.
    let want: Vec<f32> = vec![
        (1.0 + 2.0) / 2.0,
        (2.0 + 3.0) / 2.0,
        (3.0 + 4.0) / 2.0,
        (4.0 + 5.0) / 2.0,
        (5.0 + 6.0) / 2.0,
        (6.0 + 7.0) / 2.0,
    ];
    for (g, w) in m.iter().zip(want.iter()) {
        assert!((g - w).abs() < 1e-5, "got {g} want {w}");
    }
}

#[test]
fn merge_similarity_top_k_one_picks_most_central() {
    let moe = experts();
    let m = merge_experts(&moe, MoEMergeStrategy::Similarity { keep_top_k: 1 });
    // The most-similar expert on average should be one of e0 or e1
    // (they are mutually closest); either way, the result is one of
    // those two weight matrices exactly.
    let is_e0 = m.iter().zip(moe.experts[0].iter()).all(|(a, b)| (a - b).abs() < 1e-6);
    let is_e1 = m.iter().zip(moe.experts[1].iter()).all(|(a, b)| (a - b).abs() < 1e-6);
    assert!(is_e0 || is_e1, "expected e0 or e1, got {:?}", m);
}

#[test]
#[should_panic(expected = "keep_top_k must be > 0")]
fn similarity_rejects_zero_top_k() {
    let moe = experts();
    let _ = merge_experts(&moe, MoEMergeStrategy::Similarity { keep_top_k: 0 });
}
