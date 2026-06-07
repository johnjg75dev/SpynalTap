pub mod analyzer;
pub mod report;
pub mod score;
pub mod spectrum;
pub mod stats;

pub use analyzer::Analyzer;
pub use report::{
    analysis_to_charts, analysis_to_html, render_html, render_svg, Chart, ChartSeries,
    ReportSection,
};
pub use score::{classify, BlockAnalysis, BlockRole, TensorAnalysis};
pub use spectrum::tensor_spectrum;
pub use stats::{Accum, Analysis, PerChannelStats, TensorStats};
