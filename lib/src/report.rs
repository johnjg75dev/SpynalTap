//! HTML report generation from analysis data.
//!
//! The library provides data structures (`Analysis`, `Chart`, `ReportSection`);
//! this module renders them to HTML using a template.

use crate::analysis::{analysis_to_charts, render_svg, Analysis};
use crate::error::{Error, Result};
use std::path::Path;

/// Built-in HTML template (embedded at compile time)
const DEFAULT_TEMPLATE: &str = include_str!("../templates/report.html");

/// Render an `Analysis` to HTML using the provided template.
///
/// If `template_path` is `None`, uses the built-in template.
/// Template placeholders are `{{PLACEHOLDER}}` format.
pub fn render_html_report(
    analysis: &Analysis,
    template_path: Option<&Path>,
) -> Result<String> {
    let template = if let Some(path) = template_path {
        std::fs::read_to_string(path).map_err(Error::Io)?
    } else {
        DEFAULT_TEMPLATE.to_string()
    };

    let mut html = template;

    // Basic metadata
    html = html.replace("{{TITLE}}", "SpynalTap Analysis Report");
    html = html.replace("{{DATE}}", &chrono_lite_date());

    // Model summary (placeholder values - actual model info not in Analysis)
    html = html.replace("{{MODEL_NAME}}", "Unknown");
    html = html.replace("{{ARCHITECTURE}}", "Unknown");
    html = html.replace("{{BLOCK_COUNT}}", &analysis.blocks.len().to_string());
    html = html.replace("{{TOTAL_SIZE}}", "N/A");

    // Recommendation
    html = html.replace(
        "{{RECOMMENDATION_COUNT}}",
        &analysis.recommendation_count.to_string(),
    );
    html = html.replace(
        "{{RECOMMENDATION}}",
        &format!("{:?}", analysis.recommendation),
    );
    html = html.replace("{{ESTIMATED_SIZE}}", "N/A");

    // Block rows
    let block_rows = analysis
        .blocks
        .iter()
        .map(|b| {
            format!(
                r#"<tr>
                    <td><code>{}</code></td>
                    <td>{}</td>
                    <td>{}</td>
                    <td>{:.2}</td>
                    <td>{:.3}</td>
                    <td><span class="tag {}">{}</span></td>
                </tr>"#,
                escape_html(&b.label),
                b.role.as_str(),
                b.tensor_count,
                b.total_bytes as f64 / 1_048_576.0,
                b.removable,
                if b.removable > 0.5 { "tag-prunable" } else { "tag-keep" },
                if b.removable > 0.5 { "Prunable" } else { "Keep" },
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    html = html.replace("{{BLOCK_ROWS}}", &block_rows);

    // Tensor stats rows (placeholder - actual tensor data not in Analysis)
    html = html.replace("{{TENSOR_ROWS}}", "<tr><td colspan=\"6\">Tensor data not available</td></tr>");
    html = html.replace("{{SAMPLE_PER_TENSOR}}", &analysis.sample_per_tensor.to_string());

    // Charts section - generate from analysis data
    let charts = analysis_to_charts(analysis);
    let charts_html = if charts.is_empty() {
        String::new()
    } else {
        let chart_svgs: Vec<String> = charts
            .iter()
            .map(|chart| {
                format!(
                    r#"<div class="chart-container"><h3>{}</h3>{}</div>"#,
                    escape_html(&chart.title),
                    render_svg(chart)
                )
            })
            .collect();
        format!(
            r#"<section class="card"><h2>Charts</h2>{}</section>"#,
            chart_svgs.join("\n")
        )
    };
    html = html.replace("{{CHARTS_SECTION}}", &charts_html);

    Ok(html)
}

fn escape_html(s: &str) -> String {
    s.replace('&', "&")
        .replace('<', "<")
        .replace('>', ">")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
}

fn chrono_lite_date() -> String {
    // Simple date without external dependency
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let days = now / 86400;
    let years = 1970 + (days / 365) as u32;
    let day_of_year = (days % 365) as u32;
    let month = ((day_of_year * 12) / 365) + 1;
    let day = ((day_of_year * 30) % 30) + 1;
    format!("{:04}-{:02}-{:02}", years, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_minimal() {
        let analysis = Analysis {
            sample_per_tensor: 1000,
            blocks: vec![],
            recommendation: vec![],
            recommendation_count: 0,
            estimated_bytes_after_prune: 0,
            total_tensors: 0,
            total_bytes: 0,
        };
        let html = render_html_report(&analysis, None).unwrap();
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("SpynalTap"));
    }
}