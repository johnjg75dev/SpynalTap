# TensorKit — TODO: Bug Fixes, Improvements & Test Coverage

> Generated from code review on 2026-06-12.
> 222/223 tests pass. 1 pre-existing CI failure.
> 5 bugs found, 2 potential issues, 1 regression, ~8 missing test areas.

---

## Priority 1 — Bug Fixes (Correctness / CI)

---

### 1. Fix Broken CLI Integration Test

**File:** `cli/tests/cli_quantize.rs`
**Lines:** 31–55
**Status:** 🔴 Regression — CI test failure

**Problem:**
The test runs the CLI binary with arguments in the wrong order. It passes the
model file path as the first positional argument before `--quantize`, but clap
interprets the positional `model` field of the `Quant` subcommand as the first
arg after the subcommand name. Because the test never passes the `"quant"`
subcommand name, clap sees the file path as a top-level unrecognized subcommand,
prints a usage error, and exits with a non-zero status.

The exact failing invocation looks like:

```
tensorkit.exe /path/to/file.gguf --quantize q4_0 --out /path/to/out.gguf --yes --json
```

Clap sees `/path/to/file.gguf` as an unrecognized subcommand because the
`Quant` subcommand was never dispatched. The fix is to insert `"quant"` as the
first argument before the model path.

**Changes:**

1. In `cli/tests/cli_quantize.rs`, change the `Command::new` block (lines 41–50)
   to:

   ```rust
   let out = Command::new(&exe)
       .arg("quant")              // ← NEW: dispatch the `quant` subcommand
       .arg(&in_path)
       .arg("--target")           // ← was "--quantize", now "--target"
       .arg("q4_0")
       .arg("--out")
       .arg(&out_path)
       .arg("--yes")
       .arg("--json")
       .output()
       .expect("run tensorkit");
   ```

2. Update the assertion on line 57 to match the new CLI output format (the
   `--json` output structure may have changed since the test was written).

3. Add a cleanup guard (`defer` pattern or explicit cleanup) so temp files are
   removed even on assertion failure (the current `let _ = std::fs::remove_file`
   at lines 72–73 won't run if an assertion above it panics).

4. Consider moving this test from `cli/tests/` (integration test) into a
   `#[cfg(test)]` unit test inside `main.rs` that calls `run_quant()` directly,
   avoiding the subprocess spawn and making the test more reliable.

---

### 2. Fix Incorrect Date Calculation in HTML Report

**File:** `lib/src/report.rs`
**Lines:** 261–272
**Status:** 🟡 Incorrect output (cosmetic)

**Problem:**
The `chrono_lite_date()` function computes approximate calendar dates from a
Unix timestamp without accounting for leap years. It uses:

```rust
let days = now / 86400;
let years = 1970 + (days / 365) as u32;
let day_of_year = (days % 365) as u32;
let month = ((day_of_year * 12) / 365) + 1;
let day = (day_of_year % 30) + 1;
```

This has multiple issues:
- Leap years are ignored entirely. By 2026, the output is off by ~14 days.
- `day_of_year % 30 + 1` maps day 31 → day 2, not day 31. Day 0 → day 1.
  Every month appears to have exactly 30 days.
- The `month` formula `(day_of_year * 12) / 365 + 1` gives incorrect results
  for months with different lengths (28/30/31 days).

**Changes:**

1. Replace `chrono_lite_date()` with a correct implementation. The simplest
   approach without adding a dependency:

   ```rust
   fn chrono_lite_date() -> String {
       use std::time::SystemTime;
       let secs = SystemTime::now()
           .duration_since(SystemTime::UNIX_EPOCH)
           .unwrap_or_default()
           .as_secs();

       // Days since epoch
       let days = secs / 86400;

       // Civil date from day count (Howard Hinnant's algorithm, public domain)
       let z = days as i64 + 719468;
       let era = if z >= 0 { z } else { z - 146096 } / 146097;
       let doe = (z - era * 146097) as u64;
       let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
       let y = yoe as i64 + era * 400;
       let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
       let mp = (5 * doy + 2) / 153;
       let d = doy - (153 * mp + 2) / 5 + 1;
       let m = if mp < 10 { mp + 3 } else { mp - 9 };
       let y = if m <= 2 { y + 1 } else { y };
       format!("{:04}-{:02}-{:02}", y, m, d)
   }
   ```

2. Add a unit test for `chrono_lite_date()`:

   ```rust
   #[cfg(test)]
   mod report_date_tests {
       #[test]
       fn date_format_is_valid() {
           let d = super::chrono_lite_date();
           assert_eq!(d.len(), 10);
           assert_eq!(&d[4..5], "-");
           assert_eq!(&d[7..8], "-");
       }
   }
   ```

---

### 3. Implement `AutoQuant` Output Dtype for SVD

**File:** `lib/src/svd/apply.rs`
**Lines:** (the match block on `cfg.dtype` inside the per-target loop)
**File:** `lib/src/svd/plan.rs`
**Lines:** 152–163
**Status:** 🔴 Feature stub — documented but unimplemented

**Problem:**
`OutputDtype::AutoQuant` is documented at `lib/src/svd/config.rs` line 436 as:

> "Auto-select a quantization format that matches the input tensor's on-disk
> precision. Currently picks Q8_0 (best accuracy/size ratio for factors) when
> the input is quantized, otherwise falls back to F16."

However, the actual `apply.rs` and `plan.rs` code treats `AutoQuant` the same as
`F16` (element size = 2 bytes). The intent to map quantized inputs to Q8_0 is
never executed.

**Changes:**

1. In `lib/src/svd/apply.rs`, inside the per-target loop where you resolve
   the output GGML type, add an `AutoQuant` arm that checks the source tensor's
   `ggml_type`:

   ```rust
   let (out_ggml, is_quantized) = match cfg.dtype {
       OutputDtype::F32 => (GgmlType::F32, false),
       OutputDtype::F16 => (GgmlType::F16, false),
       OutputDtype::Bf16 => (GgmlType::Bf16, false),
       OutputDtype::AutoQuant => {
           let src_is_quant = !matches!(
               ti.ggml_type,
               GgmlType::F32 | GgmlType::F16 | GgmlType::Bf16 | GgmlType::F64
           );
           if src_is_quant {
               (GgmlType::Q8_0, true)
           } else {
               (GgmlType::F16, false)
           }
       }
       OutputDtype::Ggml(ty) => (ty, true),
   };
   ```

2. Update `plan.rs` lines 152–163 to compute `new_bytes` accurately for
   `AutoQuant` by checking whether the source tensor is quantized. This
   requires passing the source `GgmlType` into the byte-estimate calculation:

   ```rust
   let esz = match cfg.dtype {
       OutputDtype::F32 => 4,
       OutputDtype::F16 | OutputDtype::Bf16 => 2,
       OutputDtype::AutoQuant => {
           // Estimate conservatively as F16 for non-quantized, Q8_0 for quantized
           match t.ggml_type {
               GgmlType::F32 | GgmlType::F16 | GgmlType::Bf16 | GgmlType::F64 => 2,
               _ => GgmlType::Q8_0.block_bytes().unwrap_or(34) as u64 / 32,
           }
       }
       OutputDtype::Ggml(ty) => ty.block_bytes().unwrap_or(34) as u64 / 32,
   };
   ```

3. Add a test in `tests/unit/tests/svd.rs` that creates a quantized source
   tensor, applies `AutoQuant`, and verifies the output factors are written
   as Q8_0.

---

### 4. Implement Block Selection for `quant` CLI Command

**File:** `cli/src/main.rs`
**Lines:** 469–485
**Status:** 🔴 Feature stub — CLI accepts argument but ignores it

**Problem:**
The `Quant` subcommand accepts a `--blocks` argument (line 160–161) with the
help text "Quantize only specified blocks/layers (comma-separated indices or
'all')". However, the implementation at line 478–479 simply discards it:

```rust
let _blocks = blocks_str;
```

All tensors in the model are quantized regardless of the `--blocks` value.

**Changes:**

1. Parse `blocks_str` into the existing `parse_selection` grammar or a
   new block-specific parser:

   ```rust
   use tensorkit::prune::selection::parse_index_list;

   let selected_blocks = if blocks_str == "all" {
       None // all blocks
   } else {
       Some(parse_index_list(blocks_str)?)
   };
   ```

2. Modify `quantize_gguf` in `lib/src/quantize/apply.rs` to accept an optional
   `&[i32]` block filter parameter:

   ```rust
   pub fn quantize_gguf(
       src: &Path,
       target: GgmlType,
       dst: &Path,
       blocks: Option<&[i32]>, // ← NEW
   ) -> Result<QuantizeReport> {
   ```

3. Inside the per-tensor loop (line 73–138), before dequantizing, check
   whether the tensor's block index (parsed from `blk.N.` in the name) is in
   the allowed set. If not, pass through verbatim:

   ```rust
   // Skip tensors not in the selected blocks
   if let Some(ref allowed) = blocks {
       if let Some(idx) = block_index_from_name(&ti.name) {
           if !allowed.contains(&idx) {
               writer.add_tensor(ti.name.clone(), ti.n_dims, ti.dims, ti.ggml_type, src_bytes);
               n_p += 1;
               total_in += ti.byte_size;
               total_out += ti.byte_size;
               continue;
           }
       }
   }
   ```

4. Update all callers of `quantize_gguf` (including the test in
   `tests/unit/tests/quantize.rs`) to pass `None` for the new parameter.

5. Add a test that quantizes only block 0 of a multi-block model and verifies
   block 1 retains its original dtype.

---

### 5. Make Debug Assertions into Release Assertions for Output Correctness

**Files:**
- `lib/src/quantize/apply.rs` line 117
- `lib/src/formats/gguf/writer.rs` line 61

**Status:** 🟡 Correctness risk in release builds

**Problem:**
`debug_assert_eq!` is used to verify that output byte sizes match expected values.
These assertions are **compiled out in release mode**, so a correctness bug in
the quantizer would silently produce a malformed GGUF file in production.

**Changes:**

1. In `lib/src/quantize/apply.rs` line 117, change:
   ```rust
   debug_assert_eq!(new_bz, byte_size_for(n_elems, target));
   ```
   to:
   ```rust
   if new_bz != byte_size_for(n_elems, target) {
       return Err(Error::Quantize(format!(
           "tensor '{}': quantized to {} bytes, expected {} for {} elements at {:?}",
           ti.name, new_bz, byte_size_for(n_elems, target), n_elems, target
       )));
   }
   ```

2. In `lib/src/formats/gguf/writer.rs` line 61, change:
   ```rust
   debug_assert_eq!(byte_size, expected, "tensor '{name}': byte count mismatch");
   ```
   to either a regular `assert_eq!` (will panic but not corrupt output) or
   return a `Result`:
   ```rust
   if byte_size != expected {
       return Err(std::io::Error::new(
           std::io::ErrorKind::InvalidData,
           format!("tensor '{name}': byte count mismatch ({byte_size} vs {expected})"),
       ));
   }
   ```
   (This requires `add_tensor` to return `Result<()>`, which is a larger
   refactor — consider deferring to a dedicated error-handling PR.)

---

## Priority 2 — Missing Tests

---

### 6. Add Unit Tests for `merge/average.rs`

**File:** `lib/src/merge/average.rs`
**Status:** 🔵 No tests at all

**Problem:**
`average_into` and `average_tensors` are public API functions with zero test
coverage. These are used for model merging operations.

**Changes:**

1. Create `lib/tests/unit/merge/average.rs` with tests:

   - **`average_two_equal_tensors`**: Two identical tensors should produce
     the same tensor back.
   - **`average_weighted_asymmetric`**: `[1.0, 2.0]` avg'd with `[3.0, 4.0]`
     at weights `[0.25, 0.75]` → `[2.5, 3.5]`.
   - **`average_length_mismatch_panics`**: Verify the function panics (or
     returns Err) when input lengths differ.
   - **`average_empty_tensor`**: Edge case with zero-length inputs.

2. Register the test module in `lib/src/merge/mod.rs` (it's already
   `#[path]`-linked but verify the module declaration covers `average.rs`).

---

### 7. Add Unit Tests for `merge/depth.rs`

**File:** `lib/src/merge/depth.rs`
**Status:** 🔵 No tests at all

**Problem:**
`insert_block` (block duplication, zero-fill insertion) has no tests.

**Changes:**

1. Create `lib/tests/unit/merge/depth.rs` with tests:

   - **`insert_block_duplicate`**: Insert a copy of block 3 at position 5.
     Verify tensor names are re-indexed correctly (block 5 becomes a copy
     of block 3, old blocks shift).
   - **`insert_block_zero_fill`**: Insert a zero-filled block at position 0.
     Verify original block 0 is now at index 1.
   - **`insert_plan_indices_valid`**: Verify the plan's index list is
     internally consistent.
   - **`insert_result_tensor_count`**: Verify the result reports the correct
     new tensor count.

---

### 8. Add Unit Tests for `merge/moe.rs`

**File:** `lib/src/merge/moe.rs`
**Status:** 🔵 No tests at all

**Problem:**
`merge_experts` is a complex operation with no test coverage.

**Changes:**

1. Create `lib/tests/unit/merge/moe.rs` with tests:

   - **`merge_two_identical_experts`**: Two copies of the same expert
     should produce the same expert (for average strategy).
   - **`merge_with_weights`**: Two experts with explicit weights.
   - **`merge_strategy_average`**: Verify MoEMergeStrategy::Average works.
   - **`merge_empty_experts_panics`**: Edge case.

---

### 9. Add Tests for Prune Metadata Handling

**File:** `lib/src/prune/apply.rs`
**Status:** 🔵 Critical path untested

**Problem:**
The GGUF prune path has sophisticated metadata handling (block re-indexing,
array shrinking, per-block key renaming) but no dedicated tests.

**Changes:**

1. Create `lib/tests/unit/prune/` module with tests:

   - **`shrink_array_correctness`**: Build an array of 10 elements, drop
     indices [2, 5, 8], verify 7 elements remain with correct values.
   - **`shrink_array_empty`**: Drop nothing from an array.
   - **`shrink_array_drop_all`**: Drop all indices.
   - **`parse_block_key_basic`**: `"llama.blk.3.attn_q.weight"` →
     `("llama.", 3, ".attn_q.weight")`.
   - **`parse_block_key_no_block`**: `"output.weight"` → `None`.
   - **`rename_metadata_block_key`**: Rename `blk.5` → `blk.3` in a key.
   - **`rename_block_tensor_name`**: Rename `blk.7.ffn_up.weight` with
     remap `{7→3}` → `blk.3.ffn_up.weight`.
   - **`rename_block_no_blk_prefix`**: `"token_embd.weight"` → unchanged.
   - **`integration_prune_metadata`**: Build a minimal GGUF, prune blocks
     1 and 3 from a 6-block model, verify `block_count` in output metadata
     is 4, and that tensor names are re-indexed.

2. Register the test module in `lib/src/prune/mod.rs`.

---

### 10. Add Tests for SVD Apply Paths

**File:** `lib/src/svd/apply.rs`
**Status:** 🔵 Core pipeline untested

**Problem:**
`apply_to_gguf`, `apply_to_safetensors`, and `apply_to_onnx` are the core
output paths but have no dedicated tests (the existing `svd.rs` tests only
cover `build_plan`, not `apply`).

**Changes:**

1. In `tests/unit/tests/svd.rs`, add:

   - **`apply_to_gguf_creates_a_b_factors`**: Build an in-memory GGUF with
     a 64×64 F32 tensor, apply a plan with rank=16, verify the output GGUF
     contains `*.svd_a` (shape [64, 16]) and `*.svd_b` (shape [16, 64]) and
     the original tensor is gone.
   - **`apply_to_gguf_metadata_written`**: Verify `tensorkit.svd.applied`,
     `tensorkit.svd.method`, `tensorkit.svd.targets` are present.
   - **`apply_to_gguf_preserves_non_target_tensors`**: A model with 3
     tensors, only 1 targeted — verify the other 2 are preserved verbatim.
   - **`apply_to_gguf_auto_quant`**: Test the `AutoQuant` path (once
     implemented, per item #3).
   - **`apply_to_gguf_energy_rank`**: Test `RankSpec::Energy` where `k` is
     resolved at apply time, not plan time.

---

### 11. Add ONNX Write/Read Round-Trip Test

**File:** `lib/src/formats/onnx.rs`
**Status:** 🔵 Writer untested

**Changes:**

1. Create `lib/tests/unit/formats/onnx.rs` with:

   - **`onnx_roundtrip_basic`**: Write a model with 2 tensors to bytes via
     `OnnxWriter`, read back via `OnnxFile::open`, verify tensor names,
     shapes, and byte data match.
   - **`onnx_metadata_preserved`**: Write metadata props, verify they survive
     the round-trip.
   - **`onnx_graph_structure`**: Verify graph inputs/outputs are preserved.

---

### 12. Add Edge-Case Tests for Selection Parser

**File:** `lib/src/prune/selection.rs`
**Status:** 🔵 Edge cases untested

**Changes:**

1. In `lib/tests/unit/tests/selection_parse.rs`, add:

   - **`selection_empty_string`**: `parse_selection("")` → `InvalidSelection`
     error (not panic).
   - **`selection_keep_large_indices`**: `keep:999999` → single element.
   - **`selection_drop_duplicates`**: `drop:5,5,5` → `[5]`.
   - **`parse_index_list_empty`**: `parse_index_list("")` → empty vec.
   - **`parse_index_list_whitespace`**: `parse_index_list(" 3 , 5 ")` →
     `[3, 5]`.

---

### 13. Add Tests for Analyzer with Sampled Tensors

**File:** `lib/src/analysis/analyzer.rs`
**Status:** 🔵 Sampling path untested

**Changes:**

1. In `lib/tests/unit/analysis/analyzer.rs`, add:

   - **`analyzer_with_sampled_tensor`**: Create a `FakeModel` with one very
     large tensor (element count > sample_per_tensor), run `Analyzer` with
     a small sample, verify `stats.n_sampled == true` and
     `stats.n == sample_per_tensor`.
   - **`analyzer_small_tensor_not_sampled`**: Element count < sample_per_tensor,
     verify `stats.n_sampled == false` and `stats.n == total_elements`.
   - **`analyzer_keeps_tensors_in_keep_list`**: Verify the `keep()` builder
     method is respected (analysis includes those tensors).

---

### 14. Add Tests for SVD Adjacent Pass Error Paths

**File:** `lib/src/svd/plan.rs` lines 184–277
**Status:** 🔵 Error paths untested

**Changes:**

1. In `tests/unit/svd/plan.rs`, add:

   - **`adjacent_out_of_range_skipped`**: Primary tensor at block 0 with
     offset `-1` → `adj_name` not found → appears in `skipped` list.
   - **`adjacent_missing_tensor_skipped`**: Adjacent name doesn't exist in
     model → appears in `skipped` list.
   - **`adjacent_non_2d_skipped`**: Adjacent tensor is 1D → skipped.
   - **`adjacent_duplicate_not_added`**: Two primary targets generate the
     same adjacent target → only one copy in output.
   - **`adjacent_basic_success`**: Primary at block 1 with offset +1 →
     adjacent at block 2 is added to targets.

---

### 15. Add Tests for `merge/strategy.rs`

**File:** `lib/src/merge/strategy.rs`
**Status:** 🔵 No tests

**Changes:**

1. Create `lib/tests/unit/merge/strategy_tests.rs` (or add to existing
   `tests.rs`):

   - **`weight_format_from_str`**: Parse `"f32"`, `"f16"`, `"q4_0"`, etc.
   - **`merge_strategy_default`**: Verify default is `Average`.
   - **`merge_strategy_slerp_parse`**: Verify `"slerp:0.5"` parsing.

---

## Priority 3 — Code Quality & Minor Issues

---

### 16. Improve `ModelFormat::from_path` Default

**File:** `lib/src/model.rs` lines 14–27
**Status:** 🟡 Potential user confusion

**Problem:**
Unknown file extensions silently default to `ModelFormat::Gguf`. A user who
runs `tensorkit analyze model.bin` will get a confusing "bad magic: 0x..."
error instead of a clear "unsupported file extension" message.

**Changes:**

1. Option A (minimal): Add a `Unknown` variant to `ModelFormat`:
   ```rust
   pub enum ModelFormat {
       Gguf,
       Safetensors,
       Onnx,
       Unknown(String),
   }
   ```
   Then match in the CLI to produce a clear error message.

2. Option B (stricter): Change `from_path` to return `Result`:
   ```rust
   pub fn from_path(path: &Path) -> Result<Self> {
       // ... existing match ...
       _ => Err(Error::UnsupportedType(format!(
           "unrecognized file extension for '{}'",
           path.display()
       ))),
   }
   ```

3. Update all callers of `ModelFormat::from_path` to handle the new error
   or unknown variant.

---

### 17. Add Warning When Analyzer Skips I-quant/T-quant Tensors

**File:** `lib/src/analysis/analyzer.rs` lines 170–193
**Status:** 🟡 Silent data loss

**Problem:**
The `dequant_for_dtype` function returns `None` for all I-quant and T-quant
types, causing those tensors to be silently dropped from the analysis. The
user sees fewer tensors in the report than exist in the model with no warning.

**Changes:**

1. Return a richer type from `dequant_for_dtype`:

   ```rust
   enum DequantResult {
       Ok(Vec<f32>),
       Unsupported(String),
   }
   ```

2. In `analyze_tensor`, when the result is `Unsupported(name)`, optionally
   store a `skipped_unsupported` count in the `Analysis` struct (or print
   a warning to stderr):

   ```rust
   // In the parallel collection pass:
   .filter_map(|r| match r {
       Ok(Some((t, s, v))) => Some(Ok((t, s, v))),
       Ok(None) => None,
       Err(e) => Some(Err(e)),
   })
   ```

3. Add a `skipped_unsupported_tensors: usize` field to `Analysis` and
   report it in the human-readable output.

---

### 18. Fix Dead Code in `Accum`

**File:** `lib/src/analysis/stats.rs` lines 131–134, 247–251
**Status:** 🟢 Cleanup

**Problem:**
`seen_buckets` is written to in `push()` (line 249) but never read in
`finalize()`. It's dead code. The `#[allow(dead_code)]` annotations are on
`near_zero` and `far_outlier` but not on `seen_buckets`.

**Changes:**

1. Remove the `seen_buckets` field and the write at line 249, or add
   `#[allow(dead_code)]` to it.

2. Verify `near_zero` and `far_outlier` are truly unused (they are — they're
   only incremented in a commented-out block). Consider removing them
   entirely or re-adding the logic that uses them.

---

### 19. Ensure Temp Files Are Cleaned Up in Integration Tests

**File:** `cli/tests/cli_quantize.rs` lines 72–73
**Status:** 🟢 Test hygiene

**Problem:**
The `let _ = std::fs::remove_file` cleanup calls at the end of the test won't
execute if any assertion above them panics, leaving temp files in the OS temp
directory.

**Changes:**

1. Use a RAII guard struct:

   ```rust
   struct Cleanup(PathBuf, PathBuf);
   impl Drop for Cleanup {
       fn drop(&mut self) {
           let _ = std::fs::remove_file(&self.0);
           let _ = std::fs::remove_file(&self.1);
       }
   }
   ```

2. Create the guard immediately after computing the paths, before any
   operations that could fail:

   ```rust
   let in_path = std::env::temp_dir().join(format!("tensorkit-cli-in-{}.gguf", std::process::id()));
   let out_path = std::env::temp_dir().join(format!("tensorkit-cli-out-{}.gguf", std::process::id()));
   let _cleanup = Cleanup(in_path.clone(), out_path.clone());
   ```

---

## Summary

| Priority | Item | Description |
|----------|------|-------------|
| **P1** | #1 | Fix CLI integration test (CI failure) |
| **P1** | #2 | Fix `chrono_lite_date()` wrong dates |
| **P1** | #3 | Implement `AutoQuant` SVD dtype |
| **P1** | #4 | Implement `--blocks` for quant CLI |
| **P1** | #5 | Convert debug_assert to real assertions |
| **P2** | #6 | Tests for `merge/average.rs` |
| **P2** | #7 | Tests for `merge/depth.rs` |
| **P2** | #8 | Tests for `merge/moe.rs` |
| **P2** | #9 | Tests for prune metadata handling |
| **P2** | #10 | Tests for SVD apply paths |
| **P2** | #11 | ONNX round-trip test |
| **P2** | #12 | Edge-case selection parser tests |
| **P2** | #13 | Analyzer sampling path tests |
| **P2** | #14 | SVD adjacent pass error path tests |
| **P2** | #15 | Tests for `merge/strategy.rs` |
| **P3** | #16 | Improve `ModelFormat::from_path` default |
| **P3** | #17 | Warn on skipped I-quant/T-quant tensors |
| **P3** | #18 | Clean up dead code in `Accum` |
| **P3** | #19 | RAII temp file cleanup in integration tests |

**Estimated effort:** ~2–3 days for P1+P2, ~1 day for P3.
