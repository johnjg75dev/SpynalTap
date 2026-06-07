//! MoE (Mixture of Experts) expert merging.
//!
//! All merge operations take a `MoEWeights` value that already contains
//! the dequantized, row-major weight matrices for each expert and return
//! a single merged expert weight matrix of the same shape.

/// Per-expert weight data. Each `experts[i]` is a row-major matrix of
/// shape `expert_shape` (rows, cols) flattened to `rows * cols` floats.
#[derive(Debug, Clone)]
pub struct MoEWeights {
    pub experts: Vec<Vec<f32>>,
    pub expert_shape: (usize, usize),
}

impl MoEWeights {
    /// Build from a list of equal-shape, equal-length weight vectors.
    ///
    /// # Panics
    /// Panics if `experts` is empty or if the experts disagree on length.
    pub fn new(experts: Vec<Vec<f32>>, expert_shape: (usize, usize)) -> Self {
        assert!(!experts.is_empty(), "MoEWeights: at least one expert required");
        let (rows, cols) = expert_shape;
        assert_eq!(
            rows * cols,
            experts[0].len(),
            "MoEWeights: expert length {} does not match shape {rows}x{cols}",
            experts[0].len()
        );
        for (i, e) in experts.iter().enumerate().skip(1) {
            assert_eq!(
                e.len(),
                experts[0].len(),
                "MoEWeights: expert {i} has length {} (expected {})",
                e.len(),
                experts[0].len()
            );
        }
        Self { experts, expert_shape }
    }

    fn n_elements(&self) -> usize {
        self.expert_shape.0 * self.expert_shape.1
    }
}

/// Strategy for merging multiple experts into one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoEMergeStrategy {
    /// Elementwise mean across all experts.
    Average,
    /// Compute mean cosine similarity per expert; keep the top-k most
    /// similar experts and average those.
    Similarity { keep_top_k: usize },
}

/// Merge experts into a single weight matrix of the same shape.
///
/// # Panics
/// Panics if `Similarity { keep_top_k }` is `0` or exceeds the number of
/// experts, or if any expert in `moe` has the wrong length.
pub fn merge_experts(moe: &MoEWeights, strategy: MoEMergeStrategy) -> Vec<f32> {
    let n = moe.experts.len();
    let nelem = moe.n_elements();
    let mut out = vec![0.0f32; nelem];

    match strategy {
        MoEMergeStrategy::Average => {
            for expert in &moe.experts {
                for (o, x) in out.iter_mut().zip(expert.iter()) {
                    *o += x;
                }
            }
            let inv = 1.0 / n as f32;
            for o in out.iter_mut() {
                *o *= inv;
            }
        }
        MoEMergeStrategy::Similarity { keep_top_k } => {
            assert!(keep_top_k > 0, "keep_top_k must be > 0");
            assert!(
                keep_top_k <= n,
                "keep_top_k ({keep_top_k}) cannot exceed number of experts ({n})"
            );

            // Mean cosine similarity for each expert to all OTHERS.
            let mut mean_sim: Vec<f32> = Vec::with_capacity(n);
            for i in 0..n {
                let mut s = 0.0f64;
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    s += cosine_f32(&moe.experts[i], &moe.experts[j]);
                }
                let others = (n - 1) as f64;
                mean_sim.push(if others > 0.0 { (s / others) as f32 } else { 0.0 });
            }

            // Indices sorted by descending mean similarity.
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&x, &y| {
                mean_sim[y]
                    .partial_cmp(&mean_sim[x])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            for &idx in order.iter().take(keep_top_k) {
                for (o, x) in out.iter_mut().zip(moe.experts[idx].iter()) {
                    *o += x;
                }
            }
            let inv = 1.0 / keep_top_k as f32;
            for o in out.iter_mut() {
                *o *= inv;
            }
        }
    }

    out
}

fn cosine_f32(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..n {
        let x = a[i] as f64;
        let y = b[i] as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn experts() -> MoEWeights {
        // Three experts, each a 2x3 matrix.
        let e0: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let e1: Vec<f32> = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let e2: Vec<f32> = vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0];
        MoEWeights::new(vec![e0, e1, e2], (2, 3))
    }

    #[test]
    fn merge_average_basic() {
        let moe = experts();
        let m = merge_experts(&moe, MoEMergeStrategy::Average);
        // (1+2+10)/3, (2+3+10)/3, ... -> [13/3, 15/3, 17/3, 19/3, 21/3, 23/3]
        let want: Vec<f32> = vec![13.0, 15.0, 17.0, 19.0, 21.0, 23.0]
            .into_iter()
            .map(|x| x / 3.0)
            .collect();
        for (g, w) in m.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-5, "got {g} want {w}");
        }
        assert_eq!(m.len(), 6);
    }

    #[test]
    fn merge_similarity_keeps_top_k() {
        // e0 and e1 are similar (close numbers), e2 is a constant
        // constant vector that is far from both. The "mean similarity to
        // others" should rank e0 and e1 highest.
        let moe = experts();
        let m = merge_experts(&moe, MoEMergeStrategy::Similarity { keep_top_k: 2 });
        // Result should be the average of e0 and e1.
        let want: Vec<f32> = vec![
            (1.0 + 2.0) / 2.0,
            (2.0 + 3.0) / 2.0,
            (3.0 + 4.0) / 2.0,
            (4.0 + 5.0) / 2.0,
            (5.0 + 6.0) / 2.0,
            (6.0 + 7.0) / 2.0,
        ];
        for (g, w) in m.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-5, "got {g} want {w}");
        }
    }

    #[test]
    fn merge_similarity_top_k_one_picks_most_central() {
        let moe = experts();
        let m = merge_experts(&moe, MoEMergeStrategy::Similarity { keep_top_k: 1 });
        // The most-similar expert on average should be one of e0 or e1
        // (they are mutually closest); either way, the result is one of
        // those two weight matrices exactly.
        let is_e0 = m.iter().zip(moe.experts[0].iter()).all(|(a, b)| (a - b).abs() < 1e-6);
        let is_e1 = m.iter().zip(moe.experts[1].iter()).all(|(a, b)| (a - b).abs() < 1e-6);
        assert!(is_e0 || is_e1, "expected e0 or e1, got {:?}", m);
    }

    #[test]
    #[should_panic(expected = "keep_top_k must be > 0")]
    fn similarity_rejects_zero_top_k() {
        let moe = experts();
        let _ = merge_experts(&moe, MoEMergeStrategy::Similarity { keep_top_k: 0 });
    }
}
