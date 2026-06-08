# SpynalTap — AI Model Analyzer, Pruner, SVD Compressor, Quantizer & Merger

**`spynaltape`** is a fast, format-agnostic CLI and library for analyzing, pruning, SVD-compressing, quantizing, merging, and MoE-merging transformer-based AI models. It supports **GGUF** (v1–v3), **safetensors**, and **ONNX** formats.

```text
spynaltape — these go to eleven.
```

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
  - [analyze](#analyze)
  - [prune](#prune)
  - [svd](#svd)
  - [quant](#quant)
  - [merge](#merge)
  - [moe](#moe)
  - [pipeline](#pipeline)
  - [interact](#interact)
  - [bench](#bench)
  - [test](#test)
- [Interactive Mode](#interactive-mode)
- [Pipeline Config](#pipeline-config)
- [Formats](#formats)
  - [GGUF](#gguf)
  - [Safetensors](#safetensors)
  - [ONNX](#onnx)
- [Quantization Types](#quantization-types)
- [Library API](#library-api)
- [Architecture](#architecture)
- [Comparison & Limitations](#comparison--limitations)
- [License](#license)

---

## Features

### Analysis (`analyze`)
- **Heuristic scoring** — scores each tensor on sparsity, outlier ratio, magnitude, and entropy; aggregates per-block scores weighted by byte size.
- **SVD spectra** — computes singular-value spectra for up to 12 tensors (8 prunable + 4 non-prunable) per model run using one-sided Jacobi SVD.
- **Full statistics** — per-tensor mean, variance, skewness, kurtosis, percentiles (p01, p50, p99, p999), sparsity, outlier ratio, entropy bits, L1/L2 norms, abs max, per-channel stats for 2-D tensors.
- **Multi-format** — works with GGUF, safetensors, and ONNX.
- **Interactive HTML reports** — self-contained HTML with Chart.js interactive charts (amax, role distribution, spectra), tensor stats tables, block analysis, dry-run impact summaries.
- **JSON output** — machine-readable analysis via `--json`.
- **Dry-run mode** — see expected prune savings and recommendations without writing any files.

### Pruning (`prune`)
- **Auto mode** — automatically selects the top 15% most-removable blocks (by heuristic score).
- **Keep/drop** — explicitly specify which blocks to keep or drop (`keep:0,1,2` or `drop:7,8`).
- **Pattern** — select blocks matching a regex (`pattern:.*ffn.*`).
- **Verification** — verify pruned output with `--verify`.
- **Multi-format** — supports GGUF and safetensors (ONNX TBD).

### SVD Compression (`svd`)
- **Per-layer rank** — set rank per layer by integer, fraction, or energy threshold (`energy:0.99`).
- **Per-tensor selection** — target `mlp`, `attn`, `all`, or comma-separated tensor names.
- **Adjacent tensors** — expand SVD to cover adjacent tensors with positive/negative offsets (`attn_v+1,ffn_up-2`).
- **Output dtype** — `f32`, `f16`, `bf16`, or `auto` (preserves input dtype).
- **Multi-format** — supports GGUF and safetensors (ONNX TBD).

### Quantization (`quant`)
- **23 quantization types** — from Q2_K to Q8_K, including IQ types (IQ1_S, IQ1_M, IQ2_S, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS) and TQ types (TQ1_0, TQ2_0).
- **GGUF only** — output format is always GGUF.

### Merging (`merge`)
- **Weighted average** — merge two or more models with per-model weights.
- **Slerp** — spherical linear interpolation between two models with configurable `t` parameter.
- **Same-file merge** — merge tensors within a single model file.
- **Deep merge** — insert/remove blocks with automatic renumbering.

### MoE Merging (`moe`)
- **Auto-detect** — automatically detects expert tensor groups by naming conventions.
- **Average** — average all experts in each group into one.
- **Similarity** — keep only the top-k most central experts per group.

### Interactive Wizard (`interact`)
- **Menu-driven** — build a pipeline step by step with guided prompts.
- **Add/remove/reorder** — manage pipeline steps interactively.
- **Save/load** — save pipeline configs as JSON, reload them later.
- **Approve before execute** — review the full pipeline before running.

### Pipeline Automation (`pipeline`)
- **JSON config** — define a sequence of operations in a JSON file.
- **Batch execution** — runs multiple operations sequentially, feeding the output of each step as the input to the next.
- **CI/CD friendly** — ideal for automated model optimization workflows.

---

## Installation

### From source

```bash
git clone https://github.com/anomalyco/spynaltap.git
cd spynaltap
cargo build --release
./target/release/spynaltape --help
```

### Prerequisites

- **Rust** 1.75+ (edition 2024)
- Standard build tools (`gcc`, `make`, etc. on Linux; MSVC Build Tools on Windows)

---

## Quick Start

### 1. Analyze a model

```bash
# Basic analysis
spynaltape analyze model.gguf

# With HTML report + dry-run
spynaltape analyze model.gguf --report report.html --dry-run

# JSON output
spynaltape analyze model.safetensors --json | jq .
```

### 2. Prune blocks

```bash
# Auto-prune 5 blocks
spynaltape prune model.gguf --selection auto:5 --out pruned.gguf -y

# Keep specific blocks
spynaltape prune model.gguf --selection keep:0,1,2,3,4 --out pruned.gguf --verify
```

### 3. SVD compress

```bash
# Compress MLP tensors at 50% rank
spynaltape svd model.gguf --layers all --tensors mlp --rank 0.5 --out compressed.gguf

# Energy-based rank with adjacent tensors
spynaltape svd model.gguf --layers all-attn --rank energy:0.99 --adjacent attn_v+1 --out compressed.gguf
```

### 4. Quantize

```bash
# Quantize to Q4_K
spynaltape quant model.gguf --target q4_k --out quantized.gguf -y

# Quantize to IQ2_S (extreme compression)
spynaltape quant model.gguf --target iq2_s --out quantized.gguf -y
```

### 5. Merge models

```bash
# Average two models
spynaltape merge model_a.gguf --model-b model_b.gguf --out merged.gguf

# Slerp with t=0.3
spynaltape merge model_a.gguf --model-b model_b.gguf --mode slerp:0.3 --out merged.gguf
```

### 6. Interactive pipeline

```bash
# Launch the interactive wizard
spynaltape interact
```

### 7. Pipeline from JSON config

```bash
spynaltape pipeline my-pipeline.json
```

### 8. ONNX operations

```bash
# Analyze
spynaltape analyze model.onnx --json

# Prune blocks (write output as ONNX)
spynaltape prune model.onnx --selection auto:3 --out pruned.onnx -y

# SVD compress (write output as ONNX)
spynaltape svd model.onnx --layers all --tensors mlp --rank 0.5 --out compressed.onnx
```

---

## CLI Reference

### `analyze`

```
spynaltape analyze [OPTIONS] <MODEL>

Arguments:
  <MODEL>  Path to the model file

Options:
  --sample <SAMPLE>    Samples per tensor [default: 200000]
  --json               Output analysis as JSON
  --report <PATH>      Write interactive HTML report to PATH
  --template <PATH>    Custom HTML template (uses built-in if omitted)
  --dry-run            Show expected impact without writing files
  -h, --help           Print help
```

**Dry-run mode** (`--dry-run`): runs the full analysis and prints:
- Model size and block count
- Recommended prune blocks with savings estimates
- Candidate block scores
- "No files were written" confirmation

### `prune`

```
spynaltape prune [OPTIONS] <MODEL> --selection <SELECTION> --out <OUT>

Options:
  --selection <SEL>  Selection grammar: auto:<N>, keep:<list>, drop:<list>, pattern:<regex>
  --out <PATH>       Output path
  --verify           Verify output after writing
  -y, --yes          Skip confirmation prompt
  -h, --help         Print help
```

**Selection grammar:**
- `auto:5` — automatically select 5 most-removable blocks
- `keep:0,1,2,3,4` — keep only blocks 0–4
- `drop:7,8` — drop blocks 7 and 8
- `pattern:.*ffn.*` — select blocks matching regex
- `all` — select all blocks (for list/dry-run purposes)

### `svd`

```
spynaltape svd [OPTIONS] <MODEL> --layers <LAYERS> --out <OUT>

Options:
  --layers <LAYERS>    Layer selection: all, all-attn, all-ffn, 0-5,10, regex:^blk\.0\.
  --tensors <TENSORS>  Tensor selection: mlp, attn, all, ffn_gate,ffn_up [default: mlp]
  --rank <RANK>        Rank spec: int, fraction, energy:<frac>, frac:<frac>,min:<min> [default: 0.5,min:4]
  --dtype <DTYPE>      Output dtype: f32, f16, bf16, auto [default: f16]
  --min-dim <N>        Minimum dimension for SVD [default: 16]
  --adjacent <SPEC>    Adjacent tensors: attn_v+1,ffn_up-2
  --out <PATH>         Output path
  -y, --yes            Skip confirmation
  -h, --help           Print help
```

**Rank specifications:**
- `64` — fixed rank 64
- `0.5` — fraction (50% of original rank)
- `energy:0.99` — keep top singular values that capture 99% of energy
- `frac:0.5,min:4` — fraction with minimum rank 4
- `frac:0.5,min:4,max:128` — fraction with clamp

**Adjacent tensor syntax:**
- `attn_v+1` — include the tensor 1 position after `attn_v`
- `ffn_up-2` — include the tensor 2 positions before `ffn_up`
- `attn_v+1,ffn_gate-1,ffn_up-1` — multiple adjacent specs

### `quant`

```
spynaltape quant [OPTIONS] <MODEL> --target <TYPE> --out <OUT>

Options:
  --target <TYPE>  Quantization type (see list below)
  --blocks <BLCKS> Blocks to quantize: all or comma-separated [default: all]
  --out <PATH>     Output path
  -y, --yes        Skip confirmation
  -h, --help       Print help
```

### `merge`

```
spynaltape merge [OPTIONS] <MODEL_A> --out <OUT>

Options:
  --model-b <PATH>   Second model (omit for same-file merge)
  --mode <MODE>      Merge mode: average or slerp:<t> [default: average]
  --weights <W>      Comma-separated weights: 0.5,0.5
  --config <PATH>    JSON config file (overrides --mode, --weights)
  --out <PATH>       Output path
  -y, --yes          Skip confirmation
  -h, --help         Print help
```

### `moe`

```
spynaltape moe <MODEL> --out <OUT>

Options:
  --strategy <S>    average or similarity:<k> [default: average]
  --expert-pattern  Expert tensor name regex override
  --out <PATH>      Output path
  -y, --yes         Skip confirmation
  -h, --help        Print help
```

### `pipeline`

```
spynaltape pipeline <CONFIG>
```

Runs a sequence of operations defined in a JSON pipeline config file. The output of each step becomes the input of the next.

**Example pipeline JSON:**

```json
{
  "model": "input.gguf",
  "steps": [
    { "action": "Analyze", "sample": 200000, "report": "analysis.html" },
    { "action": "Prune", "selection": "auto:5", "out": "pruned.gguf", "verify": true },
    { "action": "Quant", "quant_type": "q4_k", "out": "final.gguf" }
  ]
}
```

### `interact`

```
spynaltape interact
```

Launches the interactive pipeline builder with a menu-driven interface. Supports all six operations with guided parameter input, step management (add/remove/reorder), save/load of config files, and approval-before-execution.

### `bench`

```
spynaltape bench [OPTIONS] [OP]

Arguments:
  [OP]  Operation to benchmark: analyze, svd, quant [default: analyze]

Options:
  --model <PATH>   Model path
  --iterations <N> Iterations [default: 3]
```

### `test`

```
spynaltape test [CATEGORY]

Arguments:
  [CATEGORY]  Test category: all, dequant, svd, quant [default: all]
```

---

## Interactive Mode

The `interact` subcommand provides a menu-driven wizard for building complex model processing pipelines without needing to write JSON config files.

**Flow:**

```
1. Set model path ──→ 2. Add steps ──→ 3. Review ──→ 4. Approve ──→ 5. Execute
                        ↓
                   Reorder / Remove / Save / Load
```

**Step types with guided prompts:**

| Step | Parameters prompted |
|------|-------------------|
| **Analyze** | Samples, JSON output, HTML report path, template path |
| **Quantize** | Quantization type (23 options), output path, verify |
| **Prune** | Selection string, output path, verify |
| **SVD** | Output path, tensor selection, rank, dry-run, verify |
| **Merge** | Number of models, model paths, merge mode (average/slerp), slerp t, verify |
| **MoE** | Strategy (average/similarity), output path, dry-run |

**Features:**
- Save/load pipeline configs as JSON files
- Undo via step removal
- Full pipeline preview before execution
- Unsaved-changes warning on exit

---

## Pipeline Config

Pipeline configs are JSON files that define a sequence of operations. Each step feeds its output as the input to the next step (the `model` field sets the initial input).

```json
{
  "model": "base_model.gguf",
  "steps": [
    { "action": "Analyze", "sample": 100000, "json": true, "report": "report.html" },
    { "action": "Prune", "selection": "auto:3", "out": "pruned.gguf", "verify": true },
    { "action": "Svd", "out": "svd.gguf", "selection": "mlp", "rank": "0.5,min:4" },
    { "action": "Quant", "quant_type": "q4_k", "out": "final.gguf" }
  ]
}
```

---

## Formats

### GGUF

**GGUF** (GGML Universal Format) is a binary format designed for efficient storage and loading of quantized transformer models. It is the primary format used by `llama.cpp`.

- **Versions:** v1, v2, v3 (auto-detected)
- **Quantized types:** Full support for all 20+ GGML quantized types
- **Write support:** Full read/write for all tensor types
- **Dequantization:** SIMD-accelerated (Q4_0, Q8_0) and scalar fallback
- **Verification:** Built-in verify subcommand for pruned output

### Safetensors

**Safetensors** is a safe, fast file format for storing tensors, created by Hugging Face.

- **Reader:** Full support for all standard dtypes (F32, F16, BF16, F64, I8, I16, I32, I64, U8, U16, U32, U64, BOOL)
- **Writer:** Full support for creating safetensors files
- **Metadata:** Reads `general.name`, `general.architecture` and arbitrary metadata
- **Limitations:** No native quantized tensor storage (output is always standard dtypes)

### ONNX

**ONNX** (Open Neural Network Exchange) is an open format for representing machine learning models.

- **Reader:** Parses `ModelProto` protobuf using `prost` (lightweight, no external proto compiler needed)
- **Tensor access:** Reads tensor metadata (names, shapes, dtypes) and raw data from `initializer` lists
- **Graph metadata:** Extracts graph inputs, outputs, producer info, and custom metadata properties
- **Block detection:** Recognizes common ONNX block naming conventions (`layer.N.*`, `transformer.h.N.*`, `encoder.layer.N.*`, etc.)
- **Dtype mapping:** ONNX data types → internal TensorDtype (FLOAT→F32, FLOAT16→F16, BFLOAT16→BF16, INT8→I8, etc.)
- **Write support:** Full ONNX writer via `OnnxWriter` — serialize `ModelProto` with tensors, metadata, graph inputs/outputs; used by prune and SVD operations
- **Dtype mapping:** Converts `TensorDtype` (F32, F16, BF16, F64, INT8–64, UINT8–64, BOOL) to ONNX data types; quantized GGML types (Q4_0, IQ2_S, etc.) are not supported and will error
- **Limitations:** Quantize operation not yet supported for ONNX input (use GGUF output); ONNX write does not preserve graph nodes, only initializer tensors and metadata

---

## Quantization Types

SpynalTap supports 23 quantization types, mapped to the canonical GGML enum IDs used by `llama.cpp`:

| Type | Enum Name | ID | Block Size | Description |
|------|-----------|----|------------|-------------|
| Q2_K | `Q2K` | 10 | 256 | 2-bit K-quant |
| Q3_K | `Q3K` | 11 | 256 | 3-bit K-quant |
| Q4_0 | `Q4_0` | 2 | 32 | 4-bit round-to-nearest |
| Q4_1 | `Q4_1` | 3 | 32 | 4-bit with min/max |
| Q4_K | `Q4K` | 12 | 256 | 4-bit K-quant |
| Q5_0 | `Q5_0` | 6 | 32 | 5-bit round-to-nearest |
| Q5_1 | `Q5_1` | 7 | 32 | 5-bit with min/max |
| Q5_K | `Q5K` | 13 | 256 | 5-bit K-quant |
| Q6_K | `Q6K` | 14 | 256 | 6-bit K-quant |
| Q8_0 | `Q8_0` | 8 | 32 | 8-bit round-to-nearest |
| Q8_1 | `Q8_1` | 9 | 32 | 8-bit (unused) |
| Q8_K | `Q8K` | 15 | 256 | 8-bit K-quant |
| IQ1_S | `Iq1S` | 19 | 32 | 1.5-bit ternary |
| IQ1_M | `Iq1M` | 29 | 56 | 1.5-bit with 12-bit FP16 d |
| IQ2_XXS | `Iq2Xxs` | 17 | 32 | 2.0625-bit |
| IQ2_XS | `Iq2Xs` | 18 | 32 | 2.3125-bit |
| IQ2_S | `Iq2S` | 22 | 82 | 2.5625-bit |
| IQ3_XXS | `Iq3Xxs` | 20 | 32 | 3.0625-bit |
| IQ3_S | `Iq3S` | 21 | 32 | 3.5-bit |
| IQ4_NL | `Iq4Nl` | 24 | 32 | 4-bit null |
| IQ4_XS | `Iq4Xs` | 23 | 32 | 4-bit |
| TQ1_0 | `Tq1_0` | 27 | 32 | 1-bit ternary |
| TQ2_0 | `Tq2_0` | 28 | 32 | 2-bit ternary |

---

## Library API

The `spynaltap` library crate provides the core functionality. Use it in your own Rust projects:

```toml
[dependencies]
spynaltap = { git = "https://github.com/anomalyco/spynaltap.git" }
```

### Basic usage

```rust
use spynaltap::{Analyzer, formats::gguf::GgufFile, formats::onnx::OnnxFile};

// Analyze a GGUF model
let model = GgufFile::open("model.gguf")?;
let analysis = Analyzer::with_sample_per_tensor(200_000).analyze(&model)?;
println!("recommended: {:?}", analysis.recommendation);

// Analyze an ONNX model
let onnx = OnnxFile::open("model.onnx")?;
let analysis = Analyzer::new().analyze(&onnx)?;
println!("blocks: {} tensors: {}", analysis.blocks.len(), analysis.total_tensors);
```

### Key types

| Type | Module | Description |
|------|--------|-------------|
| `GgufFile` | `formats::gguf` | GGUF model reader |
| `GgufWriter` | `formats::gguf::writer` | GGUF model writer |
| `SafetensorsFile` | `formats::safetensors` | Safetensors model reader |
| `SafetensorsWriter` | `formats::safetensors::writer` | Safetensors model writer |
| `OnnxFile` | `formats::onnx` | ONNX model reader |
| `OnnxWriter` | `formats::onnx` | ONNX model writer |
| `Model` | `model` | Format-agnostic trait |
| `Tensor` | `model` | Tensor metadata |
| `TensorDtype` | `model` | Data type enum (35+ variants) |
| `Analysis` | `analysis` | Full analysis results |
| `PrunePlan` | `prune` | Pruning plan |
| `SvdConfig` | `svd` | SVD configuration |
| `QuantizeReport` | `quantize::apply` | Quantization results |

### Feature flags

- `calibrate` — Enables calibration-based block scoring using `candle` (requires CUDA or Metal). Runs a forward pass through the model to rank blocks by activation delta.

---

## Architecture

### Data flow

```
Input File (GGUF / Safetensors / ONNX)
        │
        ▼
   Model trait ───► Analyzer ──► Analysis
        │                          │
        │                     ┌────┴────┐
        │                     │         │
        ▼                     ▼         ▼
   Prune / SVD / Quant    JSON      HTML Report
        │                 output    (Chart.js +
        ▼                             SVG)
   Output File
```

### Module structure

```
spynaltap/
├── lib/                          # Library crate
│   ├── src/
│   │   ├── analysis/             # Analyzer, scoring, stats, spectra
│   │   │   ├── analyzer.rs       # Main analyzer (heuristic scoring)
│   │   │   ├── report.rs         # SVG chart rendering
│   │   │   ├── score.rs          # Per-block/per-tensor scoring
│   │   │   ├── spectrum.rs       # SVD-based spectrum computation
│   │   │   └── stats.rs          # Streaming statistics (Welford/Pebay)
│   │   ├── formats/
│   │   │   ├── gguf/             # GGUF reader, writer, types, verify, dequant
│   │   │   ├── onnx.rs           # ONNX reader (prost-based protobuf)
│   │   │   └── safetensors/      # Safetensors reader & writer
│   │   ├── merge/                # Model merging (average, slerp, MoE, depth, tying)
│   │   ├── prune/                # Pruning plans & application
│   │   ├── quantize/             # Quantization algorithms (23 types + SIMD)
│   │   ├── svd/                  # SVD compression (Jacobi + randomized)
│   │   ├── error.rs              # Error types
│   │   ├── model.rs              # Model trait + common types
│   │   ├── report.rs             # HTML report generation (template-based)
│   │   └── lib.rs                # Crate root
│   └── templates/
│       └── report.html           # Built-in HTML report template
├── cli/                          # CLI binary crate
│   ├── src/
│   │   ├── main.rs               # CLI entry point + all run_* functions
│   │   └── pipeline.rs           # Pipeline config + interactive wizard
│   └── tests/
│       └── cli_quantize.rs       # CLI integration tests
└── Cargo.toml                    # Workspace root
```

### Key design decisions

- **Streaming stats** — All per-tensor statistics are computed in a single pass using Welford/Pebay recurrence for numeric stability and O(1) memory.
- **Brute-force quantization** — IQ quantizers use brute-force nearest-neighbor search over the codebook grid. This is slower but simpler and more accurate than the reference implementation's recursive approach.
- **Format-agnostic core** — The `Model` trait allows the analyzer, pruner, and SVD compressor to work with any model format. Adding a new format requires only implementing the trait.
- **SIMD dequantization** — Q4_0 and Q8_0 dequantization use SIMD (SSE/AVX2 on x86, NEON on ARM) for performance. Other types use scalar fallback.
- **No external ML runtime** — Analysis is purely statistical/heuristic; no forward pass needed (unless the `calibrate` feature is enabled).

### Comparison & Limitations

| Feature | SpynalTap | llama.cpp | ONNX Runtime |
|---------|-----------|-----------|--------------|
| Heuristic block scoring | ✅ | ❌ | ❌ |
| SVD compression | ✅ | ❌ | ❌ |
| Model merging (slerp) | ✅ | ❌ | ❌ |
| MoE merging | ✅ | ❌ | ❌ |
| GGUF quantize (23 types) | ✅ | ✅ | ❌ |
| Interactive pipeline | ✅ | ❌ | ❌ |
| Full model inference | ❌ | ✅ | ✅ |
| Calibration-based scoring | ✅ (optional) | ❌ | ❌ |
| ONNX format | ✅ (read only) | ❌ | ✅ |
| Safetensors format | ✅ | ❌ | ✅ |

### Known limitations

- **Quantization output** is always GGUF format (even for safetensors/ONNX input).
- **ONNX** supports analysis only (prune, SVD, quant operations not yet implemented).
- **Safetensors** cannot store quantized tensor types natively.
- **Merge** currently supports at most 2 models for cross-file operations (same-file merge is a no-op pass-through).
- **Bench** and **test** subcommands are stubs for future implementation.
- SIMD dequantization is only available for Q4_0 and Q8_0 on x86-64 with SSE/AVX2.

---

## License

Licensed under either of [MIT License](LICENSE-MIT) or [Apache License 2.0](LICENSE-APACHE) at your option.
