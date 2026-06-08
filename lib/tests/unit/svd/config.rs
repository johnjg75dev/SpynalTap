use super::*;

#[test]
fn parse_layers_all() {
    assert!(matches!(
        LayerSelection::parse("all").unwrap(),
        LayerSelection::All
    ));
}
#[test]
fn parse_layers_range() {
    match LayerSelection::parse("0-3,7").unwrap() {
        LayerSelection::Indices(v) => assert_eq!(v, vec![0, 1, 2, 3, 7]),
        _ => panic!(),
    }
}
#[test]
fn parse_layers_alias() {
    assert!(matches!(
        LayerSelection::parse("all-mlp").unwrap(),
        LayerSelection::AllMlp
    ));
}
#[test]
fn parse_tensors_attn() {
    assert!(TensorSelection::parse("attn")
        .unwrap()
        .matches("blk.0.attn_q.weight"));
    assert!(!TensorSelection::parse("attn")
        .unwrap()
        .matches("blk.0.ffn_up.weight"));
}
#[test]
fn parse_rank_int() {
    let r = RankSpecWithClamps::parse("64").unwrap();
    assert_eq!(r.resolve(100, 100, None), 64);
}
#[test]
fn parse_rank_frac() {
    let r = RankSpecWithClamps::parse("0.5,min:4,max:200").unwrap();
    assert_eq!(r.resolve(100, 100, None), 50);
    assert_eq!(r.resolve(8, 8, None), 4);
}
#[test]
fn parse_rank_energy() {
    let s = vec![10.0, 9.0, 1.0, 0.1];
    let r = RankSpecWithClamps::parse("energy:0.99").unwrap();
    // squared s: 100, 81, 1, 0.01 -> total 182.01. 99% threshold = 180.19.
    // First two (181) already exceed 180.19, so k=2.
    assert_eq!(r.resolve(10, 10, Some(&s)), 2);

    // 0.9999: 99.99% of 182.01 = 181.998 -> k=3 (sum 182 > 181.998).
    let r2 = RankSpecWithClamps::parse("energy:0.9999").unwrap();
    assert_eq!(r2.resolve(10, 10, Some(&s)), 3);

    // 0.5 needs only the dominant singular value.
    let r3 = RankSpecWithClamps::parse("energy:0.5").unwrap();
    assert_eq!(r3.resolve(10, 10, Some(&s)), 1);
}
#[test]
fn parse_dtype() {
    assert_eq!(OutputDtype::parse("f16").unwrap(), OutputDtype::F16);
    assert_eq!(OutputDtype::parse("bf16").unwrap(), OutputDtype::Bf16);
    assert_eq!(
        OutputDtype::parse("auto").unwrap(),
        OutputDtype::AutoQuant
    );
    assert_eq!(
        OutputDtype::parse("q8_0").unwrap(),
        OutputDtype::Ggml(crate::formats::gguf::types::GgmlType::Q8_0)
    );
    assert!(OutputDtype::parse("garbage").is_err());
}
#[test]
fn factor_names() {
    let mut cfg = SvdConfig::default();
    cfg.suffix_a = ".lora_a".into();
    cfg.suffix_b = ".lora_b".into();
    let (a, b) = cfg.factor_names("blk.5.attn_q.weight");
    assert_eq!(a, "blk.5.attn_q.weight.lora_a");
    assert_eq!(b, "blk.5.attn_q.weight.lora_b");
}
