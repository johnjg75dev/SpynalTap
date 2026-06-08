use super::*;
use crate::analysis::score::BlockRole;
use crate::analysis::stats::{Analysis, TensorStats};
use std::collections::HashMap;

fn dummy_stats(n: u64, abs_max: f32) -> TensorStats {
    let mut s = crate::analysis::stats::empty_stats();
    s.n = n;
    s.abs_max = abs_max as f64;
    s
}

fn tiny_analysis() -> Analysis {
    use crate::analysis::score::BlockAnalysis;
    Analysis {
        blocks: vec![
            BlockAnalysis {
                index: 0,
                label: "blk.0".into(),
                role: BlockRole::Block,
                removable: 0.5,
                total_bytes: 1024,
                tensor_count: 2,
                neighbor_similarity: None,
                tensors: vec![],
                spectra: HashMap::new(),
            },
            BlockAnalysis {
                index: -1,
                label: "embed".into(),
                role: BlockRole::Embedding,
                removable: 0.0,
                total_bytes: 256,
                tensor_count: 1,
                neighbor_similarity: None,
                tensors: vec![],
                spectra: HashMap::new(),
            },
        ],
        recommendation: vec![0],
        recommendation_count: 1,
        estimated_bytes_after_prune: 256,
        sample_per_tensor: 200_000,
        total_tensors: 2,
        total_bytes: 1280,
        model_name: None,
        architecture: None,
        file_size: None,
    }
}

#[test]
fn svg_renders_chart_with_one_series() {
    let chart = Chart {
        title: "test".into(),
        width: 400,
        height: 200,
        series: vec![ChartSeries {
            label: "s".into(),
            points: (0..10).map(|i| i as f32).collect(),
            color: "#ffffff".into(),
        }],
    };
    let svg = render_svg(&chart);
    assert!(svg.contains("<svg"), "missing <svg");
    assert!(svg.contains("</svg>"), "missing </svg>");
    // The polyline encodes each point as an "L" command; we should
    // have at least 5 of those after the first "M".
    let line_count = svg.matches(" L").count();
    assert!(line_count >= 5, "expected at least 5 line segments, got {line_count}");
}

#[test]
fn html_renders_with_title_and_charts() {
    let chart = Chart {
        title: "c1".into(),
        width: 300,
        height: 120,
        series: vec![ChartSeries {
            label: "a".into(),
            points: vec![1.0, 2.0, 3.0],
            color: "#fff".into(),
        }],
    };
    let chart2 = Chart {
        title: "c2".into(),
        width: 300,
        height: 120,
        series: vec![ChartSeries {
            label: "b".into(),
            points: vec![3.0, 2.0, 1.0],
            color: "#0f0".into(),
        }],
    };
    let section = ReportSection {
        heading: "h".into(),
        body: "body".into(),
    };
    let html = render_html("My Report", &[section], &[chart, chart2]);
    assert!(html.contains("<title>My Report</title>"));
    // Two <svg> elements.
    let svg_count = html.matches("<svg").count();
    assert_eq!(svg_count, 2);
    assert!(html.contains("</html>"));
}

#[test]
fn analysis_to_charts_produces_at_least_one() {
    let a = tiny_analysis();
    let charts = analysis_to_charts(&a);
    // We have 0 amax in the fake (no tensors), so the per-tensor amax
    // chart is skipped. The role-distribution chart should still be
    // present (we have one block + one embed).
    assert!(!charts.is_empty(), "expected at least one chart");
    let titles: Vec<&str> = charts.iter().map(|c| c.title.as_str()).collect();
    assert!(
        titles.iter().any(|t| t.contains("block role")),
        "titles = {:?}",
        titles
    );
}

#[test]
fn analysis_to_charts_includes_amax_when_tensors_present() {
    use crate::analysis::score::{BlockAnalysis, TensorAnalysis};
    let a = Analysis {
        blocks: vec![BlockAnalysis {
            index: 0,
            label: "blk.0".into(),
            role: BlockRole::Block,
            removable: 0.4,
            total_bytes: 64,
            tensor_count: 2,
            neighbor_similarity: None,
            tensors: vec![
                TensorAnalysis {
                    name: "blk.0.w1".into(),
                    removable: 0.4,
                    stats: dummy_stats(10, 1.5),
                },
                TensorAnalysis {
                    name: "blk.0.w2".into(),
                    removable: 0.3,
                    stats: dummy_stats(8, 0.7),
                },
            ],
            spectra: HashMap::new(),
        }],
        recommendation: vec![],
        recommendation_count: 0,
        estimated_bytes_after_prune: 0,
        sample_per_tensor: 100,
        total_tensors: 2,
        total_bytes: 64,
        model_name: None,
        architecture: None,
        file_size: None,
    };
    let charts = analysis_to_charts(&a);
    let titles: Vec<&str> = charts.iter().map(|c| c.title.as_str()).collect();
    assert!(
        titles.iter().any(|t| t.contains("amax")),
        "titles = {:?}",
        titles
    );
}

#[test]
fn analysis_to_html_is_valid_structure() {
    let a = tiny_analysis();
    let html = analysis_to_html(&a);
    assert!(html.contains("<!DOCTYPE html>"), "missing doctype");
    assert!(html.contains("</html>"), "missing </html>");
    assert!(html.contains("<title>"));
}

#[test]
fn chart_handles_empty_series() {
    let chart = Chart {
        title: "empty".into(),
        width: 200,
        height: 100,
        series: vec![],
    };
    let svg = render_svg(&chart);
    assert!(svg.contains("<svg"));
    assert!(svg.contains("</svg>"));
}

#[test]
fn chart_handles_constant_series() {
    // y_min == y_max case should not divide by zero.
    let chart = Chart {
        title: "const".into(),
        width: 240,
        height: 120,
        series: vec![ChartSeries {
            label: "s".into(),
            points: vec![1.0; 5],
            color: "#fff".into(),
        }],
    };
    let svg = render_svg(&chart);
    assert!(svg.contains("<svg"));
    assert!(svg.contains("</svg>"));
}
