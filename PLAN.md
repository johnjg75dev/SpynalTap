# TensorKit Implementation Plan

> Generated from TODO.md code review. 19 items across 3 priority tiers.
> Codebase root: `C:/Users/John/Desktop/AI Gens/Rust/TensorKit`

---

## Phase 1 — P1 Bug Fixes (5 items, blocking CI / correctness)

### Task 1: Fix CLI Integration Test
**File:** `cli/tests/cli_quantize.rs`
**Problem:** Missing `"quant"` subcommand arg; wrong flag name `--quantize` → `--target`; no RAII cleanup.

**Changes:**
1. Add `struct Cleanup(PathBuf, PathBuf)` with `Drop` impl that removes both files.
2. Create `_cleanup` guard right after computing paths (line 36).
3. Fix `Command::new` block (lines 41–50):
   - Insert `.arg("quant")` as first arg after exe
   - Change `.arg("--quantize")` → `.arg("--target")`
4. Remove manual `let _ = std::fs::remove_file` calls (lines 72–73).
5. Verify assertion strings match actual JSON output format.

---

### Task 2: Fix `chrono_lite_date()` Wrong Dates
**File:** `lib/src/report.rs` lines 261–272
**Problem:** Ignores leap years; every month appears 30 days; off by ~14 days in 2026.

**Changes:**
1. Replace `chrono_lite_date()` body with Howard Hinnant's civil date algorithm (no deps).
2. Add unit test `date_format_is_valid` in `lib/tests/unit/report.rs` (or inline `#[cfg(test)]` module).

---

### Task 3: Implement `AutoQuant` Output Dtype for SVD
**Files:** `lib/src/svd/apply.rs`, `lib/src/svd/plan.rs` lines 152–163
**Problem:** `AutoQuant` treated identically to `F16`; never checks source tensor dtype.

**Changes:**
1. In `apply.rs`, inside the per-target GGUF loop where output type is resolved, add `AutoQuant` arm:
   - Check `ti.ggml_type` — if quantized → `Q8_0`, else → `F16`
2. In `plan.rs` line 152–163, fix `esz` computation for `AutoQuant`:
   - Check `t.ggml_type` to pick correct element size estimate
3. Add test in `lib/tests/unit/tests/svd.rs` verifying Q8_0 output for quantized source.

---

### Task 4: Implement Block Selection for `quant` CLI
**Files:** `cli/src/main.rs` lines 469–485, `lib/src/quantize/apply.rs`
**Problem:** `--blocks` accepted but discarded (`let _blocks = blocks_str`).

**Changes:**
1. In `main.rs::run_quant`, parse `blocks_str` via `parse_index_list` or handle `"all"` → `None`.
2. Add `blocks: Option<&[i32]>` parameter to `quantize_gguf` in `lib/src/quantize/apply.rs`.
3. Inside per-tensor loop, skip tensors whose block index isn't in the allowed set.
4. Update all callers (including `tests/unit/tests/quantize.rs`) to pass `None`.
5. Add test: quantize only block 0, verify block 1 retains original dtype.

---

### Task 5: Convert `debug_assert_eq!` to Release Assertions
**Files:** `lib/src/quantize/apply.rs` line 117, `lib/src/formats/gguf/writer.rs` line 61
**Problem:** Byte-size checks compiled out in release → silent malformed GGUF.

**Changes:**
1. In `quantize/apply.rs:117`: Replace `debug_assert_eq!` with `if` that returns `Err(Error::Quantize(...))`.
2. In `gguf/writer.rs:61`: Replace `debug_assert_eq!` with `if` that returns `io::Error` (or `assert_eq!` as interim).

---

## Phase 2 — P2 Missing Tests (10 items)

### Task 6: Tests for `merge/average.rs`
**File:** `lib/tests/unit/merge/average.rs` (exists, has 4 tests)
**Add:**
- `average_weighted_asymmetric` — weighted avg with different coefficients
- `average_empty_tensor` — zero-length inputs
- Verify existing tests still pass (they look correct)

---

### Task 7: Tests for `merge/depth.rs`
**File:** `lib/tests/unit/merge/depth.rs` (exists, has `FakeModel`)
**Add:**
- `insert_block_duplicate` — copy block 3 at position 5, verify re-indexing
- `insert_block_zero_fill` — zero-filled block at position 0
- `insert_plan_indices_valid` — plan index consistency
- `insert_result_tensor_count` — correct new tensor count

---

### Task 8: Tests for `merge/moe.rs`
**File:** `lib/tests/unit/merge/moe.rs` (exists, has 2 tests)
**Add:**
- `merge_two_identical_experts` — same expert → same output
- `merge_with_weights` — explicit weight vector
- `merge_strategy_average` — verify Average strategy
- `merge_empty_experts_panics` — edge case

---

### Task 9: Tests for Prune Metadata Handling
**File:** `lib/src/prune/apply.rs` (new test module needed)
**Create:** `lib/tests/unit/prune/` with:
- `shrink_array_correctness` — drop indices from array
- `shrink_array_empty` / `shrink_array_drop_all`
- `parse_block_key_basic` / `parse_block_key_no_block`
- `rename_metadata_block_key` / `rename_block_tensor_name`
- `integration_prune_metadata` — minimal GGUF prune round-trip

---

### Task 10: Tests for SVD Apply Paths
**File:** `lib/tests/unit/tests/svd.rs` (exists, 395 lines)
**Add:**
- `apply_to_gguf_creates_a_b_factors` — in-memory GGUF → SVD → verify factors
- `apply_to_gguf_metadata_written` — check metadata keys present
- `apply_to_gguf_preserves_non_target_tensors` — 3 tensors, 1 targeted
- `apply_to_gguf_energy_rank` — `RankSpec::Energy` deferred resolution

---

### Task 11: ONNX Round-Trip Test
**File:** `lib/tests/unit/formats/onnx.rs` (exists, 263 lines)
**Add:**
- `onnx_roundtrip_basic` — write 2 tensors, read back, verify
- `onnx_metadata_preserved` — metadata survives round-trip
- `onnx_graph_structure` — graph inputs/outputs preserved

---

### Task 12: Edge-Case Selection Parser Tests
**File:** `lib/tests/unit/tests/selection_parse.rs` (exists, 67 lines)
**Add:**
- `selection_empty_string` → error (not panic)
- `selection_keep_large_indices` → single element
- `selection_drop_duplicates` → deduped
- `parse_index_list_empty` → empty vec
- `parse_index_list_whitespace` → `[3, 5]`

---

### Task 13: Analyzer Sampling Path Tests
**File:** `lib/tests/unit/analysis/analyzer.rs` (exists, 148 lines)
**Add:**
- `analyzer_with_sampled_tensor` — large tensor, small sample
- `analyzer_small_tensor_not_sampled` — below threshold
- `analyzer_keeps_tensors_in_keep_list` — `keep()` builder respected

---

### Task 14: SVD Adjacent Pass Error Path Tests
**File:** `lib/tests/unit/svd/plan.rs` (exists, 440 lines)
**Add:**
- `adjacent_out_of_range_skipped`
- `adjacent_missing_tensor_skipped`
- `adjacent_non_2d_skipped`
- `adjacent_duplicate_not_added`
- `adjacent_basic_success`

---

### Task 15: Tests for `merge/strategy.rs`
**File:** `lib/tests/unit/merge/strategy.rs` (exists, 27 lines)
**Add:**
- `weight_format_from_str` — parse various dtype strings
- `merge_strategy_default` — verify Average is default
- `merge_strategy_slerp_parse` — `"slerp:0.5"` parsing

---

## Phase 3 — P3 Code Quality (4 items)

### Task 16: Improve `ModelFormat::from_path` Default
**File:** `lib/src/model.rs` lines 14–27
**Changes:**
- Add `Unknown(String)` variant to `ModelFormat`
- Match unknown extensions to `Unknown(ext)`
- Update CLI to produce clear error for unknown formats

---

### Task 17: Warn on Skipped I-quant/T-quant Tensors
**File:** `lib/src/analysis/analyzer.rs` lines 165–194
**Changes:**
- Change `dequant_for_dtype` to return richer type (Ok/Unsupported)
- Add `skipped_unsupported_tensors: usize` to `Analysis`
- Report in human-readable output

---

### Task 18: Clean Up Dead Code in `Accum`
**File:** `lib/src/analysis/stats.rs` lines 131–135, 247–251
**Changes:**
- Remove `seen_buckets` field and write at line 249
- Remove `near_zero` and `far_outlier` fields (only incremented in dead code)
- Remove `#[allow(dead_code)]` annotations

---

### Task 19: RAII Temp File Cleanup in Integration Tests
**File:** `cli/tests/cli_quantize.rs` (covered by Task 1)
**Changes:** Already handled in Task 1 — the `Cleanup` struct with `Drop` impl.

---

## Dependency Graph

```
Task 1 (CLI test)        — standalone
Task 2 (date calc)       — standalone
Task 3 (AutoQuant)       — standalone
Task 4 (block selection)  — standalone
Task 5 (debug asserts)   — standalone
Task 6–8 (merge tests)   — standalone, parallelizable
Task 9 (prune tests)     — standalone
Task 10 (SVD apply)      — depends on Task 3 (AutoQuant test)
Task 11 (ONNX)           — standalone
Task 12 (selection)      — standalone
Task 13 (analyzer)       — standalone
Task 14 (SVD adjacent)   — standalone
Task 15 (strategy)       — standalone
Task 16 (ModelFormat)    — standalone
Task 17 (warn skipped)   — standalone
Task 18 (dead code)      — standalone
```

## Execution Strategy

**Wave 1 (parallel):** Tasks 1, 2, 3, 4, 5 — all independent P1 bug fixes
**Wave 2 (parallel):** Tasks 6, 7, 8, 9, 11, 12, 13, 14, 15 — all independent P2 tests
**Wave 3 (after Wave 1):** Task 10 — SVD apply tests (needs AutoQuant from Task 3)
**Wave 4 (parallel):** Tasks 16, 17, 18 — P3 code quality

Each wave runs as parallel subagents. Final `cargo test` to verify everything passes.
