//! Dequantization throughput bench.
//!
//! Run with: `cargo bench --bench dequant`
//! Add `RUSTFLAGS=-Ctarget-cpu=native` for peak numbers on your machine.

use criterion::{criterion_group, criterion_main, Criterion};
use spynaltap::formats::gguf::dequant as gguf_dequant;
use spynaltap::formats::gguf::types::GgmlType;

fn bench_dequant_f16(c: &mut Criterion) {
    let bytes: Vec<u8> = (0..(1 << 20))
        .map(|i| ((i * 7 + 3) & 0xFFFF) as u16 as u8)
        .cycle()
        .take(1 << 21)
        .collect();
    c.bench_function("dequant_f16_2MiB", |b| {
        b.iter(|| gguf_dequant::dequantize(GgmlType::F16, &bytes, None))
    });
}

fn bench_dequant_q8_0(c: &mut Criterion) {
    // Q8_0: 34 bytes per 32-element block. 1 MiB of q8_0 = 32768 blocks = 1M elements.
    let bytes = vec![0u8; 1 << 20];
    c.bench_function("dequant_q8_0_1MiB", |b| {
        b.iter(|| gguf_dequant::dequantize(GgmlType::Q8_0, &bytes, None))
    });
}

fn bench_dequant_q4_0(c: &mut Criterion) {
    // Q4_0: 18 bytes per 32 elements. 1 MiB = ~58250 blocks = ~1.86M elements.
    let bytes = vec![0u8; 1 << 20];
    c.bench_function("dequant_q4_0_1MiB", |b| {
        b.iter(|| gguf_dequant::dequantize(GgmlType::Q4_0, &bytes, None))
    });
}

criterion_group!(
    benches,
    bench_dequant_f16,
    bench_dequant_q8_0,
    bench_dequant_q4_0
);
criterion_main!(benches);
