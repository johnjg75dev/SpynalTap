//! `spynaltape` — analyze and prune an AI model file (GGUF or safetensors).
//!
//! Default: open, analyze, print a recommendation, exit.
//! With `--prune` and `--out`: analyze, prompt to confirm, then prune + write.
//! With `--svd` and `--out`: SVD-compress the requested layers/tensors, then write.

use clap::Parser;
use spynaltap::analysis::score::BlockRole;
use spynaltap::formats::gguf::GgufFile;
use spynaltap::formats::safetensors::SafetensorsFile;
use spynaltap::model::ModelFormat;
use spynaltap::prune::{apply_to_gguf, apply_to_safetensors, build_plan, parse_selection, Selection};
use spynaltap::svd::{
    apply_to_gguf as svd_apply_gguf, apply_to_safetensors as svd_apply_st,
    build_plan as build_svd_plan, LayerSelection, OutputDtype, RankSpec, RankSpecWithClamps,
    SvdConfig, TensorSelection,
};
use spynaltap::{Analyzer, Error};
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser, Debug)]
#[command(name = "spynaltape", version, about = "Analyze, prune, and SVD-compress AI model files", long_about = None)]
struct Cli {
    /// Path to the model file (GGUF or safetensors).
    model: PathBuf,

    /// List every tensor in the model.
    #[arg(long)]
    list: bool,

    /// Number of elements to sample per tensor (default 200_000).
    #[arg(long, default_value_t = 200_000)]
    sample: usize,

    /// Run a heuristic analysis and print a recommendation (default behavior).
    #[arg(long)]
    analyze: bool,

    /// Output the report as JSON.
    #[arg(long)]
    json: bool,

    /// Prune per SELECTION (see grammar in the docs).
    #[arg(long)]
    prune: Option<String>,

    /// Output file path (required for --prune / --svd).
    #[arg(long)]
    out: Option<PathBuf>,

    /// Re-open the pruned file and verify its structural integrity.
    #[arg(long)]
    verify: bool,

    /// Skip the y/N confirmation prompt.
    #[arg(long, short = 'y')]
    yes: bool,

    // ---- SVD compression options -----------------------------------------

    /// SVD-compress the model. Value is the layer-selection grammar:
    ///   all | all-attn | all-ffn | all-mlp | 0,1,2 | 0-5,10 | regex:^blk\.0\.
    /// Requires --out.
    #[arg(long)]
    svd: Option<String>,

    /// Which tensor families to compress within each selected layer:
    ///   attn | ffn | mlp | all | attn_q,ffn_up | regex:...
    /// (default: mlp)
    #[arg(long, default_value = "mlp")]
    svd_tensors: String,

    /// Rank spec for the low-rank factorization:
    ///   64 | 0.5 | energy:0.99 | abs:64,min:8,max:256 | frac:0.5,min:4
    /// (default: 0.5,min:4)
    #[arg(long, default_value = "0.5,min:4")]
    svd_rank: String,

    /// Output dtype for the packed (A, B) factors: f32 | f16 | bf16
    /// (default: f16)
    #[arg(long, default_value = "f16")]
    svd_dtype: String,

    /// Minimum min(m, n) of a tensor to be eligible (default: 16).
    #[arg(long, default_value_t = 16)]
    svd_min_dim: usize,

    /// Use randomized SVD for matrices with >= this many elements (0 = never).
    /// (default: 262144 = 256K)
    #[arg(long, default_value_t = 262_144)]
    svd_randomized_min: usize,

    /// Randomized SVD oversampling (extra test columns).
    #[arg(long, default_value_t = 8)]
    svd_oversample: usize,

    /// Randomized SVD power iterations.
    #[arg(long, default_value_t = 2)]
    svd_power_iters: usize,

    /// Suffix appended to the original name to form the "A" (m x k) factor.
    #[arg(long, default_value = ".svd_a")]
    svd_suffix_a: String,

    /// Suffix appended to the original name to form the "B" (k x n) factor.
    #[arg(long, default_value = ".svd_b")]
    svd_suffix_b: String,
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
    let format = ModelFormat::from_path(&cli.model);
    eprintln!("[open] {} (format: {})", cli.model.display(), format.as_str());

    match format {
        ModelFormat::Gguf => run_gguf(cli),
        ModelFormat::Safetensors => run_safetensors(cli),
    }
}

fn run_gguf(cli: Cli) -> Result<(), Error> {
    let gg = GgufFile::open(&cli.model)?;

    if cli.list {
        print_tensor_list(&gg, cli.json);
        return Ok(());
    }

    if cli.sample == 0 {
        return Err(Error::Gguf("--sample must be > 0".into()));
    }

    let analysis = Analyzer::with_sample_per_tensor(cli.sample).analyze(&gg)?;
    print_human_report(&gg, &analysis, cli.json);

    if let Some(sel_str) = &cli.prune {
        let selection = parse_selection(sel_str)?;
        let out_path = cli.out.as_ref()
            .ok_or_else(|| Error::Gguf("--prune requires --out".into()))?
            .clone();
        confirm_or_exit_prune(&out_path, &selection, &analysis.recommendation, cli.yes)?;
        let plan = build_plan(&gg, &selection, Some(&analysis.blocks))?;
        print_plan_summary(&plan);
        let report = apply_to_gguf(&gg, &plan, &out_path)?;
        println!("\n=== prune report ===");
        if cli.json {
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
        } else {
            print_prune_report(&report);
        }
        if cli.verify {
            use spynaltap::formats::gguf::verify;
            let kept: Vec<String> = plan.keep.iter().filter(|(_, k)| *k).map(|(n, _)| n.clone()).collect();
            let r = verify::verify(&out_path, &kept)?;
            println!("\n=== verify ===");
            if r.ok { println!("  PASS"); } else { println!("  FAIL"); }
            println!("  tensors: {}", r.kept_tensors);
            println!("  bytes:   {} ({:.2} MB)", r.total_bytes, r.total_bytes as f64 / 1_048_576.0);
            for e in &r.errors { println!("  ERROR: {e}"); }
            for w in &r.warnings { println!("  warn:  {w}"); }
        }
    }

    if let Some(layer_sel) = &cli.svd {
        let out_path = cli.out.as_ref()
            .ok_or_else(|| Error::Gguf("--svd requires --out".into()))?
            .clone();
        let cfg = build_svd_config(&cli, layer_sel)?;
        print_svd_config_summary(&cfg);
        let plan = build_svd_plan(&gg, &cfg)?;
        print_svd_plan_summary(&plan);
        confirm_or_exit_svd(&out_path, &plan, cli.yes)?;
        let report = svd_apply_gguf(&gg, &plan, &out_path)?;
        println!("\n=== SVD report ===");
        if cli.json {
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
        } else {
            print_svd_report(&report);
        }
    }
    Ok(())
}

fn run_safetensors(cli: Cli) -> Result<(), Error> {
    let st = SafetensorsFile::open(&cli.model)?;

    if cli.list {
        print_tensor_list(&st, cli.json);
        return Ok(());
    }

    if cli.sample == 0 {
        return Err(Error::Safetensors("--sample must be > 0".into()));
    }

    let analysis = Analyzer::with_sample_per_tensor(cli.sample).analyze(&st)?;
    print_human_report(&st, &analysis, cli.json);

    if let Some(sel_str) = &cli.prune {
        let selection = parse_selection(sel_str)?;
        let out_path = cli.out.as_ref()
            .ok_or_else(|| Error::Safetensors("--prune requires --out".into()))?
            .clone();
        confirm_or_exit_prune(&out_path, &selection, &analysis.recommendation, cli.yes)?;
        let plan = build_plan(&st, &selection, Some(&analysis.blocks))?;
        print_plan_summary(&plan);
        let report = apply_to_safetensors(&st, &plan, &out_path)?;
        println!("\n=== prune report ===");
        if cli.json {
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
        } else {
            print_prune_report(&report);
        }
    }

    if let Some(layer_sel) = &cli.svd {
        let out_path = cli.out.as_ref()
            .ok_or_else(|| Error::Safetensors("--svd requires --out".into()))?
            .clone();
        let cfg = build_svd_config(&cli, layer_sel)?;
        print_svd_config_summary(&cfg);
        let plan = build_svd_plan(&st, &cfg)?;
        print_svd_plan_summary(&plan);
        confirm_or_exit_svd(&out_path, &plan, cli.yes)?;
        let report = svd_apply_st(&st, &plan, &out_path)?;
        println!("\n=== SVD report ===");
        if cli.json {
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
        } else {
            print_svd_report(&report);
        }
    }
    Ok(())
}

// ---- SVD helpers --------------------------------------------------------

fn build_svd_config(cli: &Cli, layer_sel: &str) -> Result<SvdConfig, Error> {
    let layers = LayerSelection::parse(layer_sel)
        .map_err(|e| Error::Gguf(format!("--svd: {e}")))?;
    let tensors = TensorSelection::parse(&cli.svd_tensors)
        .map_err(|e| Error::Gguf(format!("--svd-tensors: {e}")))?;
    let rank = RankSpecWithClamps::parse(&cli.svd_rank)
        .map_err(|e| Error::Gguf(format!("--svd-rank: {e}")))?;
    let dtype = OutputDtype::parse(&cli.svd_dtype)
        .map_err(|e| Error::Gguf(format!("--svd-dtype: {e}")))?;
    Ok(SvdConfig {
        layers,
        tensors,
        rank,
        dtype,
        min_dim: cli.svd_min_dim,
        randomized: cli.svd_randomized_min > 0,
        randomized_oversample: cli.svd_oversample,
        randomized_power_iters: cli.svd_power_iters,
        randomized_min_elems: cli.svd_randomized_min,
        suffix_a: cli.svd_suffix_a.clone(),
        suffix_b: cli.svd_suffix_b.clone(),
        per_layer: std::collections::BTreeMap::new(),
        per_tensor: Vec::new(),
    })
}

fn print_svd_config_summary(cfg: &SvdConfig) {
    eprintln!("\n[SVD config]");
    eprintln!("  layers:       {:?}", cfg.layers);
    eprintln!("  tensors:      {:?}", cfg.tensors);
    eprintln!("  rank spec:    {:?}", cfg.rank.spec);
    eprintln!("  rank clamps:  min={} max={:?}", cfg.rank.clamps.min, cfg.rank.clamps.max);
    eprintln!("  output dtype: {}", cfg.dtype.as_str());
    eprintln!("  min_dim:      {}", cfg.min_dim);
    eprintln!("  randomized:   {} (oversample={}, power_iters={}, min_elems={})",
              cfg.randomized, cfg.randomized_oversample,
              cfg.randomized_power_iters, cfg.randomized_min_elems);
    eprintln!("  suffixes:     a={:?} b={:?}", cfg.suffix_a, cfg.suffix_b);
}

fn print_svd_plan_summary(plan: &spynaltap::SvdPlan) {
    eprintln!("\n[SVD plan]");
    eprintln!("  blocks:           {}", plan.original_block_count);
    eprintln!("  targets:          {}", plan.targets.len());
    eprintln!("  skipped:          {}", plan.skipped.len());
    eprintln!("  orig target bytes:{:.2} MB", plan.orig_bytes() as f64 / 1_048_576.0);
    eprintln!("  est. new bytes:   {:.2} MB", plan.new_bytes() as f64 / 1_048_576.0);
    eprintln!("  est. compression: {:.1}%", plan.compression_ratio() * 100.0);
    for t in plan.targets.iter().take(20) {
        eprintln!("    {:<48}  m={:<6}  n={:<6}  k={:<6}  ({} -> {} bytes)",
                 t.name, t.m, t.n, t.k, t.orig_bytes, t.new_bytes);
    }
    if plan.targets.len() > 20 {
        eprintln!("    ... and {} more", plan.targets.len() - 20);
    }
    for s in plan.skipped.iter().take(10) {
        eprintln!("    [skip] {:<48}  {}", s.name, s.reason);
    }
}

fn print_svd_report(r: &spynaltap::SvdReport) {
    println!("  output:                {}", r.output_path);
    println!("  file size:             {} -> {} bytes ({:.2} MB -> {:.2} MB)",
             r.bytes_in, r.bytes_out,
             r.bytes_in as f64 / 1_048_576.0,
             r.bytes_out as f64 / 1_048_576.0);
    println!("  targets applied:       {}", r.applied.len());
    println!("  target bytes:          {} -> {} bytes (saved {:.2} MB)",
             r.orig_tensor_bytes, r.new_tensor_bytes,
             (r.orig_tensor_bytes as f64 - r.new_tensor_bytes as f64) / 1_048_576.0);
    println!("  compression ratio:     {:.1}%", r.compression_ratio * 100.0);
    println!("  skipped:               {}", r.skipped.len());
    println!();
    println!("  {:<48}  {:>4}  {:>4}  {:>4}  {:>10}  {:>10}  {:>10}  {:>9}",
             "tensor", "m", "n", "k", "orig B", "new B", "saved B", "err");
    for a in r.applied.iter().take(40) {
        println!("  {:<48}  {:>4}  {:>4}  {:>4}  {:>10}  {:>10}  {:>10}  {:>9.4}",
                 truncate(&a.name, 48), a.m, a.n, a.k,
                 a.orig_bytes, a.new_bytes,
                 a.orig_bytes.saturating_sub(a.new_bytes),
                 a.approx_error);
    }
    if r.applied.len() > 40 {
        println!("  ... and {} more", r.applied.len() - 40);
    }
}

fn confirm_or_exit_svd(out: &std::path::Path, plan: &spynaltap::SvdPlan, yes: bool) -> Result<(), Error> {
    eprintln!("\n[confirm] about to SVD-compress {} targets -> {}", plan.targets.len(), out.display());
    eprintln!("           est. compression: {:.1}%", plan.compression_ratio() * 100.0);
    if yes {
        eprintln!("           --yes set; proceeding.");
        return Ok(());
    }
    eprint!("           continue? [y/N] ");
    std::io::stderr().flush().ok();
    let mut s = String::new();
    std::io::stdin().read_line(&mut s).map_err(Error::Io)?;
    let s = s.trim().to_ascii_lowercase();
    if s == "y" || s == "yes" { Ok(()) } else {
        eprintln!("           cancelled.");
        std::process::exit(0);
    }
}

fn confirm_or_exit_prune(out: &std::path::Path, sel: &Selection, recommendation: &[i32], yes: bool) -> Result<(), Error> {
    let sel_desc = match sel {
        Selection::All => "all (no-op)".to_string(),
        Selection::Keep(ks) => format!("keep {:?}", ks),
        Selection::Drop(ds) => format!("drop {:?}", ds),
        Selection::Auto(n) => format!("auto (drop {n} blocks)"),
        Selection::Pattern(p) => format!("pattern /{p}/", p = p.as_str()),
    };
    eprintln!("\n[confirm] about to prune with {sel_desc} -> {}", out.display());
    if !recommendation.is_empty() {
        eprintln!("           recommendation: drop {:?}", recommendation);
    }
    if yes {
        eprintln!("           --yes set; proceeding.");
        return Ok(());
    }
    eprint!("           continue? [y/N] ");
    std::io::stderr().flush().ok();
    let mut s = String::new();
    std::io::stdin().read_line(&mut s).map_err(Error::Io)?;
    let s = s.trim().to_ascii_lowercase();
    if s == "y" || s == "yes" { Ok(()) } else {
        eprintln!("           cancelled.");
        std::process::exit(0);
    }
}

fn print_tensor_list<M: spynaltap::Model + ?Sized>(m: &M, json: bool) {
    if json {
        let list: Vec<_> = m.tensors().iter().map(|t| serde_json::json!({
            "name": t.name,
            "dtype": t.dtype.as_str(),
            "shape": t.shape,
            "bytes": t.byte_size,
        })).collect();
        println!("{}", serde_json::to_string_pretty(&list).unwrap());
    } else {
        println!("{:<60}  {:<8}  {:>14}  {:>10}",
                 "name", "dtype", "elements", "MB");
        println!("{}", "-".repeat(110));
        let mut total_bytes = 0u64;
        for t in m.tensors() {
            let elems: u64 = t.shape.iter().product();
            println!("{:<60}  {:<8}  {:>14}  {:>10.2}",
                     truncate(&t.name, 60), t.dtype.as_str(), elems,
                     t.byte_size as f64 / 1_048_576.0);
            total_bytes += t.byte_size;
        }
        println!("\n{} tensors, {:.2} MB total", m.tensors().len(),
                 total_bytes as f64 / 1_048_576.0);
    }
}

fn print_human_report<M: spynaltap::Model + ?Sized>(m: &M, a: &spynaltap::Analysis, json: bool) {
    if json {
        let v = serde_json::to_string_pretty(a).unwrap();
        println!("{v}");
        return;
    }
    let total_bytes: u64 = m.tensors().iter().map(|t| t.byte_size).sum();
    println!("\n=== model summary ===");
    if let Some(n) = m.name() { println!("  model.name:     {n}"); }
    if let Some(a_) = m.architecture() { println!("  architecture:   {a_}"); }
    if let Some(bc) = m.block_count() { println!("  block_count:    {bc}"); }
    println!("  tensors:        {}", m.tensors().len());
    println!("  total size:     {:.2} MB", total_bytes as f64 / 1_048_576.0);
    println!("  sample/tensor:  {}", a.sample_per_tensor);

    println!("\n=== block summary ===");
    let mut sortable: Vec<_> = a.blocks.iter().collect();
    sortable.sort_by_key(|b| (b.index, b.label.clone()));
    println!("{:<10}  {:<7}  {:>7}  {:>10}  {:>9}  {:>9}",
             "label", "role", "tensors", "bytes", "MB", "removable");
    for b in &sortable {
        println!("{:<10}  {:<7}  {:>7}  {:>10}  {:>9.2}  {:>9.3}",
                 b.label, b.role.as_str(), b.tensor_count, b.total_bytes,
                 b.total_bytes as f64 / 1_048_576.0, b.removable);
    }

    println!("\n=== top blocks by removable score (higher = safer to prune) ===");
    let mut ranked: Vec<_> = a.blocks.iter()
        .filter(|b| b.role == BlockRole::Block)
        .collect();
    ranked.sort_by(|x, y| y.removable.partial_cmp(&x.removable).unwrap_or(std::cmp::Ordering::Equal));
    for b in ranked.iter().take(10) {
        println!("  {:<8}  removable={:.3}  size={:.2} MB",
                 b.label, b.removable, b.total_bytes as f64 / 1_048_576.0);
    }

    println!("\n=== recommended ===");
    println!("  auto-prune {} blocks:", a.recommendation_count);
    println!("  {:?}", a.recommendation);
    println!("  estimated bytes after: {:.2} MB", a.estimated_bytes_after_prune as f64 / 1_048_576.0);
    if let Some(_first) = a.recommendation.first() {
        let n = a.recommendation_count;
        println!("  command:  spynaltape {} --prune auto:{n} --out pruned.gguf", m.name().unwrap_or("model"));
    }
}

fn print_plan_summary(plan: &spynaltap::PrunePlan) {
    eprintln!("\n[prune plan]");
    eprintln!("  blocks original: {}", plan.original_block_count);
    eprintln!("  blocks dropped:  {:?}", plan.dropped_blocks);
    eprintln!("  blocks new:      {}", plan.new_block_count);
    let kept: usize = plan.keep.iter().filter(|(_, k)| *k).count();
    let dropped: usize = plan.keep.iter().filter(|(_, k)| !*k).count();
    eprintln!("  tensors kept:    {kept}");
    eprintln!("  tensors dropped: {dropped}");
}

fn print_prune_report(r: &spynaltap::PruneReport) {
    println!("  bytes:   {} -> {} ({:.2} MB saved)",
             r.bytes_in, r.bytes_out,
             (r.bytes_in as f64 - r.bytes_out as f64) / 1_048_576.0);
    println!("  tensors: kept={} dropped={}", r.tensors_kept, r.tensors_dropped);
    println!("  blocks:  {} -> {}", r.original_block_count, r.new_block_count);
    println!("  output:  {}", r.output_path);
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max { s.to_string() }
    else { format!("{}…", &s[..max.saturating_sub(1)]) }
}

// silence "unused" on RankSpec
#[allow(dead_code)]
fn _rsv_keepalive(_: RankSpec) {}
