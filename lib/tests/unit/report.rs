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

#[test]
fn date_format_is_valid() {
    let d = super::chrono_lite_date();
    assert_eq!(d.len(), 10);
    assert_eq!(&d[4..5], "-");
    assert_eq!(&d[7..8], "-");
    // Verify it parses as a real date (year >= 2024, month 1-12, day 1-31)
    let parts: Vec<u32> = d.split('-').filter_map(|p| p.parse().ok()).collect();
    assert_eq!(parts.len(), 3);
    assert!(parts[0] >= 2024, "year too small: {}", parts[0]);
    assert!(parts[1] >= 1 && parts[1] <= 12, "month out of range: {}", parts[1]);
    assert!(parts[2] >= 1 && parts[2] <= 31, "day out of range: {}", parts[2]);
}
