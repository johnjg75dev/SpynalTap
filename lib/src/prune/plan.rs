//! Build a `PrunePlan` from a `Selection` and (optionally) the analyzer's
//! per-block scores. For `Auto(N)`, scores are required.

use crate::analysis::score::{classify, BlockAnalysis, BlockRole};
use crate::error::Result;
use crate::model::Model;
use crate::prune::selection::Selection;
use std::collections::{HashMap, HashSet};

#[derive(Debug)]
pub struct PrunePlan {
    /// (tensor name, keep?) for every tensor in the model.
    pub keep: Vec<(String, bool)>,
    /// old block index -> new (compacted) block index for kept blocks.
    pub remap: HashMap<i32, i32>,
    pub dropped_blocks: Vec<i32>,
    pub original_block_count: i32,
    pub new_block_count: i32,
}

pub fn build_plan<M: Model + ?Sized>(
    model: &M,
    sel: &Selection,
    scores: Option<&[BlockAnalysis]>,
) -> Result<PrunePlan> {
    // Discover all block indices in use.
    let mut all_blocks: HashSet<i32> = HashSet::new();
    let mut all_names: Vec<String> = Vec::with_capacity(model.tensors().len());
    for t in model.tensors() {
        all_names.push(t.name.clone());
        let (role, idx, _) = classify(&t.name);
        if role == BlockRole::Block {
            all_blocks.insert(idx);
        }
    }
    let mut sorted_blocks: Vec<i32> = all_blocks.into_iter().collect();
    sorted_blocks.sort();
    let original_block_count = sorted_blocks.len() as i32;

    let drop: HashSet<i32> = match sel {
        Selection::All => HashSet::new(),
        Selection::Keep(keeps) => sorted_blocks
            .iter()
            .filter(|i| !keeps.contains(i))
            .cloned()
            .collect(),
        Selection::Drop(d) => d.iter().cloned().collect(),
        Selection::Auto(n) => {
            let n = (*n).min(sorted_blocks.len());
            let scores = scores.ok_or_else(|| {
                crate::Error::InvalidSelection("auto".into(), "requires analysis scores".into())
            })?;
            let mut ranked: Vec<(i32, f64)> = sorted_blocks
                .iter()
                .filter_map(|i| {
                    scores
                        .iter()
                        .find(|b| b.index == *i)
                        .map(|b| (*i, b.removable))
                })
                .collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            ranked.iter().take(n).map(|(i, _)| *i).collect()
        }
        Selection::Pattern(re) => {
            let mut hits = HashSet::new();
            for t in model.tensors() {
                if re.is_match(&t.name) {
                    let (role, idx, _) = classify(&t.name);
                    if role == BlockRole::Block {
                        hits.insert(idx);
                    }
                }
            }
            hits
        }
    };

    let mut kept_sorted: Vec<i32> = sorted_blocks
        .iter()
        .filter(|i| !drop.contains(i))
        .cloned()
        .collect();
    kept_sorted.sort();
    let mut remap = HashMap::new();
    for (new_idx, old_idx) in kept_sorted.iter().enumerate() {
        remap.insert(*old_idx, new_idx as i32);
    }
    let new_block_count = kept_sorted.len() as i32;
    let mut dropped_blocks: Vec<i32> = drop.iter().cloned().collect();
    dropped_blocks.sort();

    let mut keep = Vec::with_capacity(all_names.len());
    for name in &all_names {
        let (role, idx, _) = classify(name);
        let k = match role {
            BlockRole::Block => !drop.contains(&idx),
            _ => true, // always keep embeddings, output, norms
        };
        keep.push((name.clone(), k));
    }

    Ok(PrunePlan {
        keep,
        remap,
        dropped_blocks,
        original_block_count,
        new_block_count,
    })
}
