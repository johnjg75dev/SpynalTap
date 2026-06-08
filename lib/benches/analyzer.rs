//! Analyzer throughput bench.

use criterion::{criterion_group, criterion_main, Criterion};
use tensorkit::formats::gguf::dequant as gguf_dequant;
use tensorkit::formats::gguf::types::GgmlType;
use tensorkit::Analyzer;

fn bench_analyzer_push(c: &mut Criterion) {
    let bytes = vec![0u8; 1 << 20];
    let values = gguf_dequant::dequantize(GgmlType::Q8_0, &bytes, None).unwrap();
    c.bench_function("accum_push_q8_0_1MiB", |b| {
        b.iter(|| {
            let mut acc = tensorkit::analysis::Accum::new();
            for &v in &values {
                acc.push(v);
            }
            acc
        })
    });
}

criterion_group!(benches, bench_analyzer_push);
criterion_main!(benches);
