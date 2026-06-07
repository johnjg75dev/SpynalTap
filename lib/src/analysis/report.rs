//! SVG chart and HTML report rendering for `Analysis`.
//!
//! Pure-string builders; no external SVG / HTML library. The output is
//! self-contained HTML5 with embedded SVG; opening the file in a browser
//! shows the analysis without any further asset fetches.

use crate::analysis::score::BlockRole;
use crate::analysis::stats::Analysis;

/// One line series on a chart. `points[i]` is the y-value at x = i.
#[derive(Debug, Clone)]
pub struct ChartSeries {
    pub label: String,
    pub points: Vec<f32>,
    pub color: String,
}

/// A single chart: a title, a size, and a list of overlaid series.
#[derive(Debug, Clone)]
pub struct Chart {
    pub title: String,
    pub width: u32,
    pub height: u32,
    pub series: Vec<ChartSeries>,
}

/// A piece of the report's body text.
#[derive(Debug, Clone)]
pub struct ReportSection {
    pub heading: String,
    pub body: String,
}

/// Render a `Chart` as a self-contained SVG document.
///
/// Layout: a title strip, a plotting area bounded by a y-axis tick on the
/// left and a baseline, a polyline per series, and a legend on the right.
pub fn render_svg(chart: &Chart) -> String {
    let w = chart.width.max(120);
    let h = chart.height.max(80);
    let margin_left = 56.0_f32;
    let margin_right = 180.0_f32;
    let margin_top = 32.0_f32;
    let margin_bottom = 32.0_f32;
    let plot_w = (w as f32 - margin_left - margin_right).max(10.0);
    let plot_h = (h as f32 - margin_top - margin_bottom).max(10.0);

    // Compute global y-range across all series.
    let mut y_min = f32::INFINITY;
    let mut y_max = f32::NEG_INFINITY;
    let mut x_max_len = 0usize;
    for s in &chart.series {
        x_max_len = x_max_len.max(s.points.len());
        for &v in &s.points {
            if v.is_finite() {
                if v < y_min {
                    y_min = v;
                }
                if v > y_max {
                    y_max = v;
                }
            }
        }
    }
    if !y_min.is_finite() || !y_max.is_finite() {
        y_min = 0.0;
        y_max = 1.0;
    }
    if (y_max - y_min).abs() < 1e-9 {
        y_max = y_min + 1.0;
    }
    let x_max_len = x_max_len.max(1) as f32;

    let x_to_px = |i: f32| margin_left + (i / (x_max_len - 1.0).max(1.0)) * plot_w;
    let y_to_px = |v: f32| margin_top + (1.0 - (v - y_min) / (y_max - y_min)) * plot_h;

    let mut out = String::with_capacity(4096);
    out.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {w} {h}\" \
         width=\"{w}\" height=\"{h}\" font-family=\"monospace\" font-size=\"11\">"
    ));
    out.push_str(&format!(
        "<rect x=\"0\" y=\"0\" width=\"{w}\" height=\"{h}\" fill=\"#0e0e0e\"/>"
    ));
    // Title.
    out.push_str(&format!(
        "<text x=\"{}\" y=\"18\" fill=\"#dcdcdc\">{}</text>",
        margin_left,
        escape_text(&chart.title)
    ));

    // Plot area background.
    out.push_str(&format!(
        "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"#161616\" \
         stroke=\"#2a2a2a\"/>",
        margin_left, margin_top, plot_w, plot_h
    ));

    // Y-axis ticks (5 evenly spaced).
    for i in 0..=4 {
        let frac = i as f32 / 4.0;
        let v = y_min + frac * (y_max - y_min);
        let yp = y_to_px(v);
        out.push_str(&format!(
            "<line x1=\"{}\" y1=\"{:.1}\" x2=\"{}\" y2=\"{:.1}\" stroke=\"#2a2a2a\"/>",
            margin_left,
            yp,
            margin_left + plot_w,
            yp
        ));
        out.push_str(&format!(
            "<text x=\"{}\" y=\"{:.1}\" text-anchor=\"end\" fill=\"#8a8a8a\">{:.3}</text>",
            margin_left - 4.0,
            yp + 3.0,
            v
        ));
    }

    // Axis labels.
    out.push_str(&format!(
        "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" fill=\"#8a8a8a\">index</text>",
        margin_left + plot_w / 2.0,
        h as f32 - 6.0
    ));
    out.push_str(&format!(
        "<text x=\"12\" y=\"{}\" text-anchor=\"middle\" fill=\"#8a8a8a\" \
         transform=\"rotate(-90 12 {})\">value</text>",
        margin_top + plot_h / 2.0,
        margin_top + plot_h / 2.0
    ));

    // Series.
    for s in &chart.series {
        if s.points.is_empty() {
            continue;
        }
        let mut path = String::new();
        for (i, &v) in s.points.iter().enumerate() {
            let x = x_to_px(i as f32);
            let y = if v.is_finite() {
                y_to_px(v)
            } else {
                y_to_px(y_min)
            };
            if i == 0 {
                path.push_str(&format!("M{:.1},{:.1}", x, y));
            } else {
                path.push_str(&format!(" L{:.1},{:.1}", x, y));
            }
        }
        out.push_str(&format!(
            "<path d=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"1.4\"/>",
            path,
            escape_text(&s.color)
        ));
    }

    // Legend.
    let legend_x = margin_left + plot_w + 12.0;
    let mut legend_y = margin_top;
    for s in &chart.series {
        out.push_str(&format!(
            "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" \
             stroke=\"{}\" stroke-width=\"2\"/>",
            legend_x,
            legend_y + 6.0,
            legend_x + 16.0,
            legend_y + 6.0,
            escape_text(&s.color)
        ));
        out.push_str(&format!(
            "<text x=\"{:.1}\" y=\"{:.1}\" fill=\"#dcdcdc\">{}</text>",
            legend_x + 22.0,
            legend_y + 10.0,
            escape_text(&truncate(&s.label, 28))
        ));
        legend_y += 18.0;
    }

    out.push_str("</svg>");
    out
}

/// Render a self-contained HTML5 document embedding the given sections and
/// charts. Dark background, monospace font, no external assets.
pub fn render_html(title: &str, sections: &[ReportSection], charts: &[Chart]) -> String {
    let mut out = String::with_capacity(8192);
    out.push_str("<!DOCTYPE html>\n");
    out.push_str("<html lang=\"en\">\n<head>\n");
    out.push_str("<meta charset=\"utf-8\">\n");
    out.push_str(&format!(
        "<title>{}</title>\n",
        escape_text(title)
    ));
    out.push_str("<style>\n");
    out.push_str(
        "html,body{background:#0e0e0e;color:#dcdcdc;font-family:monospace;margin:0;padding:0;}\n",
    );
    out.push_str(".wrap{max-width:960px;margin:0 auto;padding:24px;}\n");
    out.push_str("h1{font-size:20px;margin:0 0 8px 0;color:#f0f0f0;}\n");
    out.push_str("h2{font-size:14px;margin:24px 0 8px 0;color:#f0f0f0;border-bottom:1px solid #2a2a2a;padding-bottom:4px;}\n");
    out.push_str(".section{margin-bottom:24px;}\n");
    out.push_str(".chart{margin:12px 0;}\n");
    out.push_str("pre{background:#161616;color:#dcdcdc;padding:8px;overflow:auto;font-size:12px;}\n");
    out.push_str("</style>\n");
    out.push_str("</head>\n<body>\n<div class=\"wrap\">\n");
    out.push_str(&format!("<h1>{}</h1>\n", escape_text(title)));

    for s in sections {
        out.push_str("<div class=\"section\">\n");
        out.push_str(&format!("<h2>{}</h2>\n", escape_text(&s.heading)));
        out.push_str("<pre>");
        out.push_str(&escape_text(&s.body));
        out.push_str("</pre>\n");
        out.push_str("</div>\n");
    }

    for c in charts {
        out.push_str("<div class=\"chart\">\n");
        out.push_str(&render_svg(c));
        out.push_str("\n</div>\n");
    }

    out.push_str("</div>\n</body>\n</html>\n");
    out
}

/// Build a set of charts that summarize an `Analysis`.
///
/// Always produces:
///   * An "absolute max" chart showing the sorted amax of every tensor.
///   * A "block role counts" bar-like chart.
///   * One chart per available spectrum in the analysis.
pub fn analysis_to_charts(analysis: &Analysis) -> Vec<Chart> {
    let mut charts = Vec::new();

    // 1) Per-tensor amax, sorted descending.
    let mut amax_values: Vec<f32> = analysis
        .blocks
        .iter()
        .flat_map(|b| b.tensors.iter().map(|t| t.stats.abs_max as f32))
        .filter(|v| v.is_finite() && *v > 0.0)
        .collect();
    amax_values.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    if !amax_values.is_empty() {
        charts.push(Chart {
            title: "per-tensor amax (sorted descending)".into(),
            width: 720,
            height: 240,
            series: vec![ChartSeries {
                label: "amax".into(),
                points: amax_values,
                color: "#7ec8e3".into(),
            }],
        });
    }

    // 2) Block-role distribution.
    let mut role_counts: Vec<(String, f32)> = Vec::new();
    let roles: [(&str, BlockRole); 5] = [
        ("embed", BlockRole::Embedding),
        ("output", BlockRole::OutputHead),
        ("norm", BlockRole::FinalNorm),
        ("block", BlockRole::Block),
        ("other", BlockRole::Other),
    ];
    for (name, role) in &roles {
        let count = analysis
            .blocks
            .iter()
            .filter(|b| b.role == *role)
            .count();
        if count > 0 {
            role_counts.push(((*name).to_string(), count as f32));
        }
    }
    if !role_counts.is_empty() {
        let max_y = role_counts
            .iter()
            .map(|(_, v)| *v)
            .fold(0.0f32, f32::max)
            .max(1.0);
        let series = vec![ChartSeries {
            label: "count".into(),
            points: role_counts.iter().map(|(_, v)| *v).collect(),
            color: "#a4d65e".into(),
        }];
        charts.push(Chart {
            title: format!(
                "block role distribution (max = {max_y:.0})"
            ),
            width: 720,
            height: 200,
            series,
        });
        // Replace x-axis labels by prepending a chart that uses category
        // labels via the series label list (we just keep the bar shape
        // here; the title is descriptive).
        let _ = role_counts; // labels used only for the title
    }

    // 3) Spectra (one chart per non-empty spectrum).
    for b in &analysis.blocks {
        for (name, spec) in &b.spectra {
            if spec.is_empty() {
                continue;
            }
            charts.push(Chart {
                title: format!("spectrum: {name}"),
                width: 720,
                height: 200,
                series: vec![ChartSeries {
                    label: format!("{name} σ"),
                    points: spec.clone(),
                    color: "#f0a868".into(),
                }],
            });
        }
    }

    charts
}

/// Build a self-contained HTML report of the given `Analysis`.
///
/// Embeds key numeric stats in the body, then appends the charts produced
/// by `analysis_to_charts`.
pub fn analysis_to_html(analysis: &Analysis) -> String {
    let mut body = String::new();
    body.push_str(&format!(
        "tensors:      {}\n",
        analysis.total_tensors
    ));
    body.push_str(&format!(
        "total bytes:  {:.2} MB\n",
        analysis.total_bytes as f64 / 1_048_576.0
    ));
    body.push_str(&format!(
        "sample/tensor:{}\n",
        analysis.sample_per_tensor
    ));
    body.push_str(&format!(
        "blocks:       {}\n",
        analysis.blocks.len()
    ));
    body.push_str(&format!(
        "recommend:    {} blocks -> {:.2} MB after\n",
        analysis.recommendation_count,
        analysis.estimated_bytes_after_prune as f64 / 1_048_576.0
    ));

    let mut section = ReportSection {
        heading: "summary".into(),
        body,
    };
    let _ = &mut section; // silence

    // Per-block role tally in its own section.
    let mut body2 = String::new();
    let roles: [(&str, BlockRole); 5] = [
        ("embed", BlockRole::Embedding),
        ("output", BlockRole::OutputHead),
        ("norm", BlockRole::FinalNorm),
        ("block", BlockRole::Block),
        ("other", BlockRole::Other),
    ];
    for (name, role) in &roles {
        let n = analysis
            .blocks
            .iter()
            .filter(|b| b.role == *role)
            .count();
        body2.push_str(&format!("{name:<8} {n}\n"));
    }
    let section2 = ReportSection {
        heading: "block roles".into(),
        body: body2,
    };

    let charts = analysis_to_charts(analysis);
    render_html("spynaltap analysis report", &[section, section2], &charts)
}

fn escape_text(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            c => out.push(c),
        }
    }
    out
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
        out.push('…');
        out
    }
}

#[cfg(test)]
mod tests {
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
}
