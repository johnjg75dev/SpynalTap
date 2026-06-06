# SpynalTap

Rust workspace for a fast AI model analyzer and transformer-block pruner. Reads
GGUF (v1–v3) and safetensors, scores per-block "removability", and can apply
prune / SVD-compress / quantize transformations. Theme is the Spinal Tap
mockumentary — spelling and binary names follow it (`spynaltap` / `spynaltape`).

Work directly inside `SpynalTap/` (the workspace root).

## Layout

- `lib/` — library crate `spynaltap` v0.1.3, edition 2024, `crate-type = ["cdylib", "rlib"]`
- `cli/` — binary crate `spynaltape` v0.1.3, edition 2024, thin `clap` wrapper over `lib`
- `lib/tests/` — integration tests (dequant, quantize, selection_parse, svd)
- `lib/benches/` — `criterion` benches (dequant, analyzer), `harness = false`
- `.cargo/config.toml` — x86_64 only: `target-cpu=native` for dev builds so SIMD wins are observable locally. The library itself uses `is_x86_feature_detected!` for portable runtime dispatch.

## Build / test / bench

Run all from `SpynalTap/`.

- `cargo build` / `cargo build --release` — release uses `lto = "fat"`, `codegen-units = 1`, `panic = "abort"`, `strip = "symbols"` (slow link).
- `cargo test` — runs the `lib` integration tests. Doctests are skipped (`doctest = false`) because the lib is `cdylib`-only.
- `cargo test -p spynaltape` — runs the CLI's `cli_quantize` test.
- `cargo bench -p spynaltap` — criterion benches; the `bench` profile inherits release with `debug = false`.
- `--features spynaltap/calibrate` — pulls in `candle-core` / `candle-nn` / `candle-transformers` / `tokenizers` / `hf-hub`. The integration is still a stub that returns a "not implemented" error (see `lib/src/calibrate/mod.rs` for the wiring plan).
- Single test: `cargo test -p spynaltap --test dequant` (and similarly for `quantize`, `selection_parse`, `svd`).

## Naming gotchas (do not "fix")

- Library package: `spynaltap` (no trailing `e`).
- CLI package and binary: `spynaltape` (extra `e`). Imports in `cli/src/main.rs` use `spynaltap::` to refer to the library — this is correct.

## Library architecture

- `lib/src/model.rs` — `Model` trait, `Tensor`, `TensorDtype`, `BlockRef`, `ModelFormat` (the dispatch key for GGUF vs safetensors).
- `lib/src/analysis/` — `Analyzer`, `BlockAnalysis`, scoring, stats. Per-tensor sampling: pass a positive count to `Analyzer::with_sample_per_tensor` (CLI default is 200_000).
- `lib/src/formats/gguf/` — reader, writer, types, `verify`, plus `dequant/{scalar,simd}`. SIMD path requires AVX2 + F16C and is dispatched via `is_x86_feature_detected!`. K-quants (`q4_k`, `q5_k`, `q6_k`) intentionally fall back to scalar — the SIMD module's `_ => return None`.
- `lib/src/formats/safetensors/` — reader, writer.
- `lib/src/prune/`, `lib/src/quantize/`, `lib/src/svd/` — all follow the same pattern: parse selection / build plan / apply to model. See re-exports at the bottom of `lib/src/lib.rs` for the public surface.
- `serde_json` is a temporary serializer; source comments say "until we write our own faster serializer".

## CLI surface (`spynaltape <model.gguf|st> ...`)

Default behavior: open, analyze, print human-readable recommendation, exit.

- `--list` dump every tensor; `--analyze` is implied by default; `--json` switches both `--list` and the report to JSON.
- `--prune <sel> --out <path>` writes a pruned file; `--verify` re-opens and structurally verifies it. Pruning prompts `y/N` (use `-y` to skip).
- `--svd <layer-sel> --out <path>` SVD-compresses. Layer grammar: `all | all-attn | all-ffn | all-mlp | 0,1,2 | 0-5,10 | regex:^blk\.0\.`. Tensor families (`--svd-tensors`, default `mlp`): `attn | ffn | mlp | all | attn_q,ffn_up | regex:...`. Rank spec (`--svd-rank`, default `0.5,min:4`): `64 | 0.5 | energy:0.99 | abs:64,min:8,max:256 | frac:0.5,min:4`. `--svd-dtype` is `f32 | f16 | bf16` (default `f16`).
- `--quantize <q4_0|q4_1|q5_0|q5_1|q8_0> --out <path>` — GGUF only. K-quants are not exposed yet.
- `--sample N` (default 200_000) sets per-tensor element sample size; `--sample 0` is rejected.

## Build profile notes

Root `Cargo.toml` defines:

- `[profile.release]`: `lto = "fat"`, `codegen-units = 1`, `panic = "abort"`, `strip = "symbols"`. Expect slow release links.
- `[profile.bench]`: `inherits = "release"`, `debug = false`. Crashes in benches will have no symbols — bump to a custom profile if you need them.

`.gitignore` excludes `target/`, `.cargo/`, `.idea/`, `.aiassistant/`, and `.aiignore` itself.

## Known pre-existing failures

3 K-quant roundtrip tests fail on some inputs (`q4_k_roundtrip_bipolar`,
`q5_k_roundtrip_bipolar`, `q6_k_roundtrip_bipolar`). These are edge cases in
the quantize/dequant path for K-quants; not caused by recent changes.