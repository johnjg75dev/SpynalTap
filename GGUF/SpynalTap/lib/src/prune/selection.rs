//! Pruning selection grammar.
//!
//! ```text
//!   all                            — keep every block (no-op)
//!   keep:0-23                      — keep only blk.0..blk.23
//!   drop:5,6,7                     — drop blk.5, blk.6, blk.7
//!   drop:5-7                       — drop blk.5, blk.6, blk.7
//!   auto:N                         — auto-prune N blocks (highest removable first)
//!   pattern:REGEX                  — drop tensors matching regex
//! ```
//!
//! Pattern uses the `regex` crate (SIMD-accelerated DFA, faster than a
//! hand-rolled matcher). The previous hand-rolled `RegexLite` is gone.

use crate::error::{Error, Result};
use regex::Regex;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub enum Selection {
    All,
    Keep(Vec<i32>),
    Drop(Vec<i32>),
    Auto(usize),
    Pattern(Regex),
}

pub fn parse_selection(s: &str) -> Result<Selection> {
    if s == "all" { return Ok(Selection::All); }
    if let Some(rest) = s.strip_prefix("keep:") {
        return Ok(Selection::Keep(parse_index_list(rest)?));
    }
    if let Some(rest) = s.strip_prefix("drop:") {
        return Ok(Selection::Drop(parse_index_list(rest)?));
    }
    if let Some(rest) = s.strip_prefix("auto:") {
        let n: usize = rest.parse().map_err(|e| Error::InvalidSelection(s.into(), format!("bad auto:N ({e})")))?;
        return Ok(Selection::Auto(n));
    }
    if let Some(rest) = s.strip_prefix("pattern:") {
        let re = Regex::new(rest).map_err(|e| Error::InvalidSelection(s.into(), format!("bad regex: {e}")))?;
        return Ok(Selection::Pattern(re));
    }
    Err(Error::InvalidSelection(s.into(), "unknown selection".into()))
}

pub fn parse_index_list(s: &str) -> Result<Vec<i32>> {
    let mut out = HashSet::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() { continue; }
        if let Some((a, b)) = part.split_once('-') {
            let a: i32 = a.parse().map_err(|e| Error::InvalidSelection(s.into(), format!("bad range '{part}': {e}")))?;
            let b: i32 = b.parse().map_err(|e| Error::InvalidSelection(s.into(), format!("bad range '{part}': {e}")))?;
            let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
            for i in lo..=hi { out.insert(i); }
        } else {
            let i: i32 = part.parse().map_err(|e| Error::InvalidSelection(s.into(), format!("bad index '{part}': {e}")))?;
            out.insert(i);
        }
    }
    let mut v: Vec<i32> = out.into_iter().collect();
    v.sort();
    Ok(v)
}
