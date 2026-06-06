use spynaltap::prune::selection::{parse_index_list, parse_selection, Selection};

#[test]
fn selection_all() {
    let s = parse_selection("all").unwrap();
    assert!(matches!(s, Selection::All));
}

#[test]
fn selection_keep_range() {
    let s = parse_selection("keep:0-23").unwrap();
    match s {
        Selection::Keep(v) => {
            assert_eq!(v.len(), 24);
            assert_eq!(v[0], 0);
            assert_eq!(v[23], 23);
        }
        _ => panic!("expected Keep"),
    }
}

#[test]
fn selection_drop_list() {
    let s = parse_selection("drop:5,6,7").unwrap();
    match s {
        Selection::Drop(v) => assert_eq!(v, vec![5, 6, 7]),
        _ => panic!("expected Drop"),
    }
}

#[test]
fn selection_drop_range() {
    let s = parse_selection("drop:5-7").unwrap();
    match s {
        Selection::Drop(v) => assert_eq!(v, vec![5, 6, 7]),
        _ => panic!("expected Drop"),
    }
}

#[test]
fn selection_auto() {
    let s = parse_selection("auto:4").unwrap();
    assert!(matches!(s, Selection::Auto(4)));
}

#[test]
fn selection_pattern() {
    let s = parse_selection(r"pattern:blk\.(5|6|7)\..*").unwrap();
    assert!(matches!(s, Selection::Pattern(_)));
}

#[test]
fn selection_unknown() {
    assert!(parse_selection("nonsense").is_err());
}

#[test]
fn index_list_dedup_and_sort() {
    let v = parse_index_list("3,1,2,2,1,5-7,3").unwrap();
    assert_eq!(v, vec![1, 2, 3, 5, 6, 7]);
}

#[test]
fn index_list_reverse_range() {
    let v = parse_index_list("7-5").unwrap();
    assert_eq!(v, vec![5, 6, 7]);
}
