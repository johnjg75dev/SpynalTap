//! Shared types for the `merge` subsystem.

/// The high-level merge strategy to apply when combining two tensors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MergeStrategy {
    /// Elementwise mean: `0.5 * a + 0.5 * b`.
    Average,
    /// Spherical linear interpolation at the given `t` value in `[0, 1]`.
    Slerp(f32),
}

impl MergeStrategy {
    /// Convenience constructor for `MergeStrategy::Slerp(t)`.
    #[inline]
    pub fn slerp(t: f32) -> Self {
        Self::Slerp(t)
    }
}

/// In-memory layout of a 2-D weight matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    /// Row-major: row index varies slowest.
    RowMajor,
    /// Column-major: column index varies slowest.
    ColMajor,
}

impl Default for WeightFormat {
    fn default() -> Self {
        Self::RowMajor
    }
}

#[cfg(test)]
mod tests {
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
}
