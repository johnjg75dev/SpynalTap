use super::*;
use crate::analysis::stats::Analysis;
fn empty_analysis() -> Analysis {
    Analysis {
        sample_per_tensor: 1000,
        blocks: vec![],
        recommendation: vec![],
        recommendation_count: 0,
        estimated_bytes_after_prune: 0,
        total_tensors: 0,
        total_bytes: 0,
        model_name: None,
        architecture: None,
        file_size: None,
    }
}

#[test]
fn test_render_minimal() {
    let analysis = empty_analysis();
    let html = render_html_report(&analysis, None).unwrap();
    assert!(html.contains("<!DOCTYPE html>"));
    assert!(html.contains("TensorKit"));
    assert!(html.contains("chart.js"));
    assert!(html.contains("CHART_DATA"));
}

#[test]
fn test_render_with_model_metadata() {
    let mut analysis = empty_analysis();
    analysis.model_name = Some("test-model".into());
    analysis.architecture = Some("llama".into());
    analysis.total_bytes = 1_048_576;
    let html = render_html_report(&analysis, None).unwrap();
    assert!(html.contains("test-model"));
    assert!(html.contains("llama"));
    assert!(html.contains("1.00 MB"));
}

#[test]
fn test_chart_json_contains_expected_keys() {
    let analysis = empty_analysis();
    let json = build_chart_json(&analysis);
    assert!(json.contains("amax"));
    assert!(json.contains("roleCounts"));
    assert!(json.contains("spectra"));
}
