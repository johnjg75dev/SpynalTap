use super::*;

fn nearly(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn stats_skewness_known_value() {
    // {1,2,3,4,5} is perfectly symmetric around its mean (3.0),
    // so the third central moment is 0 and the standardized skewness
    // must be 0 to within numerical noise.
    let mut acc = Accum::new();
    for &v in &[1.0f32, 2.0, 3.0, 4.0, 5.0] {
        acc.push(v);
    }
    let st = acc.finalize(1e-3, false);
    assert!(st.skewness.abs() < 0.01, "skewness = {}", st.skewness);
    // For a symmetric 5-point dataset, the excess kurtosis is -1.2,
    // which is well within sanity.
    assert!(st.kurtosis.is_finite());
}

#[test]
fn stats_kurtosis_normal() {
    // 1024 pseudo-random samples drawn from a near-Gaussian
    // distribution (sum of four independent sinusoids by CLT). The
    // excess kurtosis of a near-Gaussian distribution is small.
    let mut acc = Accum::new();
    for i in 0..1024u32 {
        let f = i as f64;
        // Four sinusoids with incommensurate frequencies.
        let a = (f * 0.013).sin();
        let b = (f * 0.021).cos();
        let c = (f * 0.037).sin();
        let d = (f * 0.051).cos();
        // Sum + scale; the sum of four sinusoids converges to a
        // Gaussian shape under CLT.
        let v = (a + b + c + d) * 0.5;
        acc.push(v as f32);
    }
    let st = acc.finalize(1e-3, false);
    assert!(st.kurtosis.is_finite(), "kurtosis is not finite");
    assert!(
        st.kurtosis.abs() < 0.5,
        "kurtosis = {} (expected |kurtosis| < 0.5)",
        st.kurtosis
    );
}

#[test]
fn stats_per_channel() {
    // 3 rows of 4 values each; known row means.
    let rows: [[f32; 4]; 3] = [
        [1.0, 2.0, 3.0, 4.0],   // mean = 2.5
        [10.0, 20.0, 30.0, 40.0], // mean = 25.0
        [-1.0, -2.0, -3.0, -4.0], // mean = -2.5
    ];
    let mut acc = Accum::new_2d(3, 4);
    for r in &rows {
        for &v in r {
            acc.push(v);
        }
    }
    let st = acc.finalize(1e-3, false);
    let pc = st.per_channel.expect("per_channel should be Some for 2D");
    assert_eq!(pc.n_channels, 3);
    assert_eq!(pc.channel_means.len(), 3);
    assert_eq!(pc.channel_stds.len(), 3);
    let means = pc.channel_means;
    assert!(nearly(means[0] as f64, 2.5, 1e-4));
    assert!(nearly(means[1] as f64, 25.0, 1e-4));
    assert!(nearly(means[2] as f64, -2.5, 1e-4));
    // Stds should be positive (non-constant rows).
    assert!(pc.channel_stds[0] > 0.0);
    assert!(pc.channel_stds[1] > 0.0);
    assert!(pc.channel_stds[2] > 0.0);
}

#[test]
fn stats_per_channel_1d_is_none() {
    // 1-D accumulators should not produce per_channel stats.
    let mut acc = Accum::new();
    for &v in &[1.0f32, 2.0, 3.0, 4.0] {
        acc.push(v);
    }
    let st = acc.finalize(1e-3, false);
    assert!(st.per_channel.is_none());
}

#[test]
fn stats_reservoir_skewness() {
    // 1000 random-ish values, sampled mode (reservoir path). Skewness
    // must be finite (not NaN/Inf) even with sampling.
    let mut acc = Accum::new();
    for i in 0..1000u32 {
        let v = (((i.wrapping_mul(2654435761)) as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
        acc.push(v);
    }
    let st = acc.finalize(1e-3, true);
    assert!(st.skewness.is_finite(), "sampled skewness = {}", st.skewness);
    assert!(st.kurtosis.is_finite(), "sampled kurtosis = {}", st.kurtosis);
}

#[test]
fn stats_serialize_roundtrip() {
    let mut acc = Accum::new_2d(2, 3);
    for &v in &[0.5f32, 1.5, -0.5, 2.0, 0.25, -1.0] {
        acc.push(v);
    }
    let st = acc.finalize(1e-3, false);
    let json = serde_json::to_string(&st).expect("serialize");
    let back: TensorStats = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back.n, st.n);
    assert_eq!(back.n_sampled, st.n_sampled);
    assert!((back.mean - st.mean).abs() < 1e-12);
    assert!((back.std - st.std).abs() < 1e-12);
    assert!((back.skewness - st.skewness).abs() < 1e-12);
    assert!((back.kurtosis - st.kurtosis).abs() < 1e-12);
    let pc_back = back.per_channel.expect("per_channel present");
    let pc_orig = st.per_channel.expect("per_channel present");
    assert_eq!(pc_back.n_channels, pc_orig.n_channels);
    assert_eq!(pc_back.channel_means.len(), pc_orig.channel_means.len());
    for (a, b) in pc_back.channel_means.iter().zip(&pc_orig.channel_means) {
        assert!((a - b).abs() < 1e-6);
    }
}

#[test]
fn stats_existing_fields_unchanged() {
    // Sanity: the existing behavior is preserved when a small f32
    // slice is pushed and finalized. min/max/mean should match.
    let mut acc = Accum::new();
    let vals = [-0.5f32, 0.0, 0.5, 1.0, 1.5];
    for &v in &vals {
        acc.push(v);
    }
    let st = acc.finalize(1e-3, false);
    assert_eq!(st.n, vals.len() as u64);
    assert!((st.mean - 0.5).abs() < 1e-6);
    assert!((st.min - (-0.5)).abs() < 1e-6);
    assert!((st.max - 1.5).abs() < 1e-6);
}
