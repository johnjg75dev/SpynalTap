pub mod analyzer;
pub mod score;
pub mod stats;

pub use analyzer::Analyzer;
pub use score::{classify, BlockRole, BlockAnalysis, TensorAnalysis};
pub use stats::{Accum, TensorStats, Analysis};
