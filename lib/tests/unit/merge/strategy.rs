use super::*;

#[test]
fn average_strategy_default() {
    let s = MergeStrategy::Average;
    assert_eq!(s, MergeStrategy::Average);
}

#[test]
fn slerp_strategy_carries_t() {
    let s = MergeStrategy::Slerp(0.25);
    match s {
        MergeStrategy::Slerp(t) => assert!((t - 0.25).abs() < 1e-6),
        _ => panic!("expected slerp"),
    }
}

#[test]
fn slerp_constructor() {
    let s = MergeStrategy::slerp(0.75);
    assert_eq!(s, MergeStrategy::Slerp(0.75));
}

#[test]
fn weight_format_default_is_row_major() {
    assert_eq!(WeightFormat::default(), WeightFormat::RowMajor);
}
