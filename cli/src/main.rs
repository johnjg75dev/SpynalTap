//! `spynaltape` — analyze, prune, SVD-compress, merge, and quantize AI model files.
//!
//! Subcommands:
//! - `analyze` — run heuristic analysis, print recommendation
//! - `prune` — remove specified blocks/layers
//! - `svd` — SVD-compress specified tensors
//! - `quant` — quantize model to specified GGML type
//! - `merge` — merge two models (average, slerp, moe)
//! - `bench` — benchmark operations
//! - `test` — run self-tests

use clap::{Parser, Subcommand};
use spynaltap::formats::gguf::GgufFile;
use spynaltap::formats::gguf::types::GgmlType;
use spynaltap::formats::safetensors::SafetensorsFile;
use spynaltap::model::ModelFormat;
use spynaltap::prune::{
    apply_to_gguf, apply_to_safetensors, build_plan, parse_selection,
};
use spynaltap::quantize::apply::quantize_gguf as quantize_gguf_apply;
use spynaltap::svd::{
    AdjacentSelection, LayerSelection, OutputDtype, RankSpecWithClamps, SvdConfig, TensorSelection,
    apply_to_gguf as svd_apply_gguf, apply_to_safetensors as svd_apply_st,
    build_plan as build_svd_plan,
};
use spynaltap::{Analyzer, Error};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[derive(Parser, Debug)]
#[command(
    name = "spynaltape",
    version,
    about = "spynaltape — these go to eleven.",
    long_about = "Analyze, prune, SVD-compress, merge, and quantize AI model files.\n\n\
                  Run `spynaltape <command> --help` for command-specific options."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Analyze a model and print recommendations
    Analyze {
        /// Path to the model file
        model: PathBuf,

        /// Number of elements to sample per tensor
        #[arg(long, default_value_t = 200_000)]
        sample: usize,

        /// Output as JSON
        #[arg(long)]
        json: bool,

        /// Write HTML report to this path
        #[arg(long, value_name = "PATH")]
        report: Option<PathBuf>,

        /// Path to HTML template file (optional, uses built-in if not provided)
        #[arg(long, value_name = "PATH")]
        template: Option<PathBuf>,
    },

    /// Prune specified blocks/layers from a model
    Prune {
        /// Path to the model file
        model: PathBuf,

        /// Selection grammar (e.g., "auto:5", "keep:0,1,2", "drop:7,8", "pattern:.*ffn.*")
        #[arg(long)]
        selection: String,

        /// Output path
        #[arg(long, short = 'o')]
        out: PathBuf,

        /// Verify output after writing
        #[arg(long)]
        verify: bool,

        /// Skip confirmation prompt
        #[arg(long, short = 'y')]
        yes: bool,
    },

    /// SVD-compress specified tensors
    Svd {
        /// Path to the model file
        model: PathBuf,

        /// Layer selection (e.g., "all", "all-attn", "0-5,10", "regex:^blk\\.0\\.")
        #[arg(long)]
        layers: String,

        /// Tensor selection within layers (e.g., "mlp", "attn", "all", "ffn_gate,ffn_up")
        #[arg(long, default_value = "mlp")]
        tensors: String,

        /// Rank specification (e.g., "64", "0.5", "energy:0.99", "frac:0.5,min:4")
        #[arg(long, default_value = "0.5,min:4")]
        rank: String,

        /// Output dtype (f32, f16, bf16, auto)
        #[arg(long, default_value = "f16")]
        dtype: String,

        /// Minimum dimension to qualify for SVD
        #[arg(long, default_value_t = 16)]
        min_dim: usize,

        /// Adjacent tensor selection (e.g., "attn_v+1,ffn_up-2")
        #[arg(long)]
        adjacent: Option<String>,

        /// Output path
        #[arg(long, short = 'o')]
        out: PathBuf,

        /// Skip confirmation prompt
        #[arg(long, short = 'y')]
        yes: bool,
    },

    /// Quantize model to a GGML type
    Quant {
        /// Path to the model file (GGUF only for now)
        model: PathBuf,

        /// Target quantization type
        #[arg(long)]
        target: String,

        /// Output path
        #[arg(long, short = 'o')]
        out: PathBuf,

        /// Quantize only specified blocks/layers (comma-separated indices or "all")
        #[arg(long, default_value = "all")]
        blocks: String,

        /// Skip confirmation prompt
        #[arg(long, short = 'y')]
        yes: bool,
    },

    /// Merge two models
    Merge {
        /// Path to model A
        model_a: PathBuf,

        /// Path to model B
        #[arg(long)]
        model_b: PathBuf,

        /// Merge mode: "average", "slerp:<t>", "moe:<strategy>:<k>"
        #[arg(long)]
        mode: String,

        /// Output path
        #[arg(long, short = 'o')]
        out: PathBuf,

        /// Skip confirmation prompt
        #[arg(long, short = 'y')]
        yes: bool,
    },

    /// Benchmark operations
    Bench {
        /// Operation to benchmark (analyze, svd, quant)
        #[arg(default_value = "analyze")]
        op: String,

        /// Model path
        #[arg(long)]
        model: Option<PathBuf>,

        /// Iterations
        #[arg(long, default_value_t = 3)]
        iterations: usize,
    },

    /// Run self-tests
    Test {
        /// Test category (all, dequant, svd, quant)
        #[arg(default_value = "all")]
        category: String,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
    }
}

fn run(cli: Cli) -> Result<(), Error> {
    match cli.command {
        Commands::Analyze { model, sample, json, report, template } => {
            run_analyze(&model, sample, json, report.as_ref().map(|v| &**v), template.as_ref().map(|v| &**v))
        }
        Commands::Prune { model, selection, out, verify, yes } => {
            run_prune(&model, &selection, &out, verify, yes)
        }
        Commands::Svd { model, layers, tensors, rank, dtype, min_dim, adjacent, out, yes } => {
            run_svd(&model, &layers, &tensors, &rank, &dtype, min_dim, adjacent.as_deref(), &out, yes)
        }
        Commands::Quant { model, target, out, blocks, yes } => {
            run_quant(&model, &target, &out, &blocks, yes)
        }
        Commands::Merge { model_a, model_b, mode, out, yes } => {
            run_merge(&model_a, &model_b, &mode, &out, yes)
        }
        Commands::Bench { op, model, iterations } => {
            run_bench(&op, model.as_ref().map(|v| &**v), iterations)
        }
        Commands::Test { category } => {
            run_test(&category)
        }
    }
}

fn run_analyze(model: &Path, sample: usize, json: bool, report_path: Option<&Path>, template_path: Option<&Path>) -> Result<(), Error> {
    let format = ModelFormat::from_path(model);
    eprintln!("[open] {} (format: {})", model.display(), format.as_str());

    match format {
        ModelFormat::Gguf => {
            let gg = GgufFile::open(model)?;
            let analysis = Analyzer::with_sample_per_tensor(sample).analyze(&gg)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&analysis)?);
            } else {
                print_human_report(&gg, &analysis);
            }
            if let Some(path) = report_path {
                write_html_report(&analysis, path, template_path)?;
                eprintln!("[report] wrote {}", path.display());
            }
        }
        ModelFormat::Safetensors => {
            let st = SafetensorsFile::open(model)?;
            let analysis = Analyzer::with_sample_per_tensor(sample).analyze(&st)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&analysis)?);
            } else {
                print_human_report(&st, &analysis);
            }
            if let Some(path) = report_path {
                write_html_report(&analysis, path, template_path)?;
                eprintln!("[report] wrote {}", path.display());
            }
        }
    }
    Ok(())
}

fn run_prune(model: &Path, selection_str: &str, out: &Path, verify: bool, yes: bool) -> Result<(), Error> {
    let format = ModelFormat::from_path(model);
    eprintln!("[open] {} (format: {})", model.display(), format.as_str());

    let selection = parse_selection(selection_str)?;
    
    match format {
        ModelFormat::Gguf => {
            let gg = GgufFile::open(model)?;
            let analysis = Analyzer::new().analyze(&gg)?;
            let plan = build_plan(&gg, &selection, Some(&analysis.blocks))?;
            confirm_or_exit(&format!("prune {:?}", plan.dropped_blocks), out, yes)?;
            let report = apply_to_gguf(&gg, &plan, out)?;
            print_prune_report(&report);
            if verify {
                verify_gguf(out, &plan)?;
            }
        }
        ModelFormat::Safetensors => {
            let st = SafetensorsFile::open(model)?;
            let analysis = Analyzer::new().analyze(&st)?;
            let plan = build_plan(&st, &selection, Some(&analysis.blocks))?;
            confirm_or_exit(&format!("prune {:?}", plan.dropped_blocks), out, yes)?;
            let report = apply_to_safetensors(&st, &plan, out)?;
            print_prune_report(&report);
        }
    }
    Ok(())
}

fn run_svd(
    model: &Path, layers_str: &str, tensors_str: &str, rank_str: &str, dtype_str: &str,
    min_dim: usize, adjacent_str: Option<&str>, out: &Path, yes: bool
) -> Result<(), Error> {
    let format = ModelFormat::from_path(model);
    eprintln!("[open] {} (format: {})", model.display(), format.as_str());

    let layers = LayerSelection::parse(layers_str)
        .map_err(|e| Error::Gguf(format!("--layers: {e}")))?;
    let tensors = TensorSelection::parse(tensors_str)
        .map_err(|e| Error::Gguf(format!("--tensors: {e}")))?;
    let rank = RankSpecWithClamps::parse(rank_str)
        .map_err(|e| Error::Gguf(format!("--rank: {e}")))?;
    let dtype = OutputDtype::parse(dtype_str)
        .map_err(|e| Error::Gguf(format!("--dtype: {e}")))?;
    
    let adjacent = if let Some(s) = adjacent_str {
        Some(AdjacentSelection::parse(s)
            .map_err(|e| Error::Gguf(format!("--adjacent: {e}")))?)
    } else {
        None
    };

    let cfg = SvdConfig {
        layers, tensors, rank, dtype, min_dim,
        randomized: false, randomized_oversample: 8, randomized_power_iters: 2, randomized_min_elems: 262_144,
        suffix_a: ".svd_a".into(), suffix_b: ".svd_b".into(),
        per_layer: Default::default(), per_tensor: Vec::new(), adjacent: adjacent.flatten(),
    };

    match format {
        ModelFormat::Gguf => {
            let gg = GgufFile::open(model)?;
            let plan = build_svd_plan(&gg, &cfg)?;
            print_svd_plan_summary(&plan);
            confirm_or_exit(&format!("SVD compress {} targets", plan.targets.len()), out, yes)?;
            let report = svd_apply_gguf(&gg, &plan, out)?;
            print_svd_report(&report);
        }
        ModelFormat::Safetensors => {
            let st = SafetensorsFile::open(model)?;
            let plan = build_svd_plan(&st, &cfg)?;
            print_svd_plan_summary(&plan);
            confirm_or_exit(&format!("SVD compress {} targets", plan.targets.len()), out, yes)?;
            let report = svd_apply_st(&st, &plan, out)?;
            print_svd_report(&report);
        }
    }
    Ok(())
}

fn run_quant(model: &Path, target_str: &str, out: &Path, blocks_str: &str, yes: bool) -> Result<(), Error> {
    let format = ModelFormat::from_path(model);
    if format != ModelFormat::Gguf {
        return Err(Error::Gguf("quantize currently only supports GGUF".into()));
    }
    
    let target = parse_quant_type(target_str)?;
    eprintln!("[quant] {} -> {} as {}", model.display(), out.display(), target.as_str());
    
    // TODO: implement block selection
    let _blocks = blocks_str;
    
    confirm_or_exit(&format!("quantize to {}", target.as_str()), out, yes)?;
    let report = quantize_gguf_apply(model, target, out)?;
    print_quant_report(&report);
    Ok(())
}

fn run_merge(model_a: &Path, model_b: &Path, mode: &str, out: &Path, yes: bool) -> Result<(), Error> {
    // TODO: implement merge
    eprintln!("[merge] {} + {} -> {} (mode: {})", model_a.display(), model_b.display(), out.display(), mode);
    confirm_or_exit(&format!("merge ({})", mode), out, yes)?;
    Err(Error::Gguf("merge not yet implemented".into()))
}

fn run_bench(op: &str, _model: Option<&Path>, _iterations: usize) -> Result<(), Error> {
    eprintln!("[bench] op={} iterations={}", op, _iterations);
    // TODO: implement benchmarking
    Ok(())
}

fn run_test(_category: &str) -> Result<(), Error> {
    eprintln!("[test] category={}", _category);
    // TODO: implement self-tests
    Ok(())
}

// ---- Helpers -------------------------------------------------------------

fn confirm_or_exit(action: &str, out: &Path, yes: bool) -> Result<(), Error> {
    eprintln!("[confirm] about to {} -> {}", action, out.display());
    if yes {
        eprintln!("           --yes set; proceeding.");
        return Ok(());
    }
    eprint!("           continue? [y/N] ");
    std::io::stderr().flush().ok();
    let mut s = String::new();
    std::io::stdin().read_line(&mut s).map_err(Error::Io)?;
    let s = s.trim().to_ascii_lowercase();
    if s == "y" || s == "yes" {
        Ok(())
    } else {
        eprintln!("           cancelled.");
        std::process::exit(0);
    }
}

fn print_human_report<M: spynaltap::Model + ?Sized>(m: &M, a: &spynaltap::Analysis) {
    let total_bytes: u64 = m.tensors().iter().map(|t| t.byte_size).sum();
    println!("\n=== model summary ===");
    if let Some(n) = m.name() { println!("  model.name:     {n}"); }
    if let Some(a_) = m.architecture() { println!("  architecture:   {a_}"); }
    if let Some(bc) = m.block_count() { println!("  block_count:    {bc}"); }
    println!("  tensors:        {}", m.tensors().len());
    println!("  total size:     {:.2} MB", total_bytes as f64 / 1_048_576.0);
    println!("  sample/tensor:  {}", a.sample_per_tensor);

    println!("\n=== block summary ===");
    println!("{:<10}  {:<7}  {:>7}  {:>10}  {:>9}", "label", "role", "tensors", "bytes", "MB");
    let mut sortable: Vec<_> = a.blocks.iter().collect();
    sortable.sort_by_key(|b| (b.index, b.label.clone()));
    for b in &sortable {
        println!("{:<10}  {:<7}  {:>7}  {:>10}  {:>9.2}", b.label, b.role.as_str(), b.tensor_count, b.total_bytes, b.total_bytes as f64 / 1_048_576.0);
    }

    println!("\n=== recommended ===");
    println!("  auto-prune {} blocks: {:?}", a.recommendation_count, a.recommendation);
}

fn print_prune_report(r: &spynaltap::PruneReport) {
    println!("\n=== prune report ===");
    println!("  bytes:   {} -> {} ({:.2} MB saved)", r.bytes_in, r.bytes_out, (r.bytes_in as f64 - r.bytes_out as f64) / 1_048_576.0);
    println!("  tensors: kept={} dropped={}", r.tensors_kept, r.tensors_dropped);
    println!("  blocks:  {} -> {}", r.original_block_count, r.new_block_count);
    println!("  output:  {}", r.output_path);
}

fn print_svd_plan_summary(plan: &spynaltap::SvdPlan) {
    eprintln!("\n[SVD plan]");
    eprintln!("  blocks:           {}", plan.original_block_count);
    eprintln!("  targets:          {}", plan.targets.len());
    eprintln!("  skipped:          {}", plan.skipped.len());
    eprintln!("  orig:             {:.2} MB", plan.orig_bytes() as f64 / 1_048_576.0);
    eprintln!("  est. new:         {:.2} MB", plan.new_bytes() as f64 / 1_048_576.0);
    eprintln!("  est. compression: {:.1}%", plan.compression_ratio() * 100.0);
}

fn print_svd_report(r: &spynaltap::SvdReport) {
    println!("\n=== SVD report ===");
    println!("  output:       {}", r.output_path);
    println!("  file size:    {} -> {} bytes ({:.2} MB -> {:.2} MB)", r.bytes_in, r.bytes_out, r.bytes_in as f64 / 1_048_576.0, r.bytes_out as f64 / 1_048_576.0);
    println!("  compression:  {:.1}%", r.compression_ratio * 100.0);
    println!("  targets:      {}", r.applied.len());
    println!("  skipped:      {}", r.skipped.len());
}

fn print_quant_report(r: &spynaltap::quantize::apply::QuantizeReport) {
    println!("\n=== quantize report ===");
    println!("  output:       {}", r.output_path);
    println!("  target:       {}", r.target);
    println!("  tensors:      {} total, {} quantized", r.tensors_total, r.tensors_quantized);
    println!("  bytes:        {} -> {} ({:.2} MB -> {:.2} MB)", r.bytes_in, r.bytes_out, r.bytes_in as f64 / 1_048_576.0, r.bytes_out as f64 / 1_048_576.0);
    println!("  compression:  {:.1}%", r.compression_ratio * 100.0);
}

fn verify_gguf(path: &Path, plan: &spynaltap::PrunePlan) -> Result<(), Error> {
    use spynaltap::formats::gguf::verify;
    let kept: Vec<String> = plan.keep.iter().filter(|(_, k)| *k).map(|(n, _)| n.clone()).collect();
    let r = verify::verify(path, &kept)?;
    println!("\n=== verify ===");
    if r.ok { println!("  PASS"); } else { println!("  FAIL"); }
    println!("  tensors: {}", r.kept_tensors);
    println!("  bytes:   {} ({:.2} MB)", r.total_bytes, r.total_bytes as f64 / 1_048_576.0);
    for e in &r.errors { println!("  ERROR: {e}"); }
    for w in &r.warnings { println!("  warn:  {w}"); }
    Ok(())
}

fn parse_quant_type(s: &str) -> Result<GgmlType, Error> {
    match s.to_ascii_lowercase().as_str() {
        "q2_k" => Ok(GgmlType::Q2K),
        "q3_k" => Ok(GgmlType::Q3K),
        "q4_0" => Ok(GgmlType::Q4_0),
        "q4_1" => Ok(GgmlType::Q4_1),
        "q4_k" => Ok(GgmlType::Q4K),
        "q5_0" => Ok(GgmlType::Q5_0),
        "q5_1" => Ok(GgmlType::Q5_1),
        "q5_k" => Ok(GgmlType::Q5K),
        "q6_k" => Ok(GgmlType::Q6K),
        "q8_0" => Ok(GgmlType::Q8_0),
        "q8_1" => Ok(GgmlType::Q8_1),
        "q8_k" => Ok(GgmlType::Q8K),
        other => Err(Error::Gguf(format!(
            "unsupported quant type {:?} (supported: q2_k, q3_k, q4_0, q4_1, q4_k, q5_0, q5_1, q5_k, q6_k, q8_0, q8_1, q8_k)", other
        ))),
    }
}

fn write_html_report(analysis: &spynaltap::Analysis, path: &Path, _template_path: Option<&Path>) -> Result<(), Error> {
    // TODO: load template, fill in data from analysis, write HTML
    // For now, write a minimal placeholder
    let html = format!(
        r#"<!DOCTYPE html><html><head><title>SpynalTap Report</title></head><body><h1>Analysis Report</h1><p>Blocks: {}</p><p>Recommendation: {:?}</p></body></html>"#,
        analysis.blocks.len(),
        analysis.recommendation
    );
    std::fs::write(path, html).map_err(Error::Io)?;
    Ok(())
}