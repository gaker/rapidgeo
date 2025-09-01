use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rapidgeo_polyline::{decode, encode, LngLat};

#[cfg(feature = "batch")]
use rapidgeo_polyline::batch::encode_batch;

fn generate_test_coordinates(count: usize) -> Vec<LngLat> {
    (0..count)
        .map(|i| {
            let lng = -180.0 + (i as f64 * 360.0 / count as f64);
            let lat = -90.0 + ((i * 137) % 180) as f64; // Use golden ratio for distribution
            LngLat::new_deg(lng, lat)
        })
        .collect()
}

fn generate_realistic_route(count: usize) -> Vec<LngLat> {
    // Simulate GPS track with small incremental changes
    let mut coords = Vec::with_capacity(count);
    let mut lng = -122.4194; // Start in San Francisco
    let mut lat = 37.7749;

    coords.push(LngLat::new_deg(lng, lat));

    for _ in 1..count {
        // Small random walk to simulate real GPS track
        lng += (fastrand::f64() - 0.5) * 0.001; // ~100m steps
        lat += (fastrand::f64() - 0.5) * 0.001;
        coords.push(LngLat::new_deg(lng, lat));
    }
    coords
}

fn bench_encode_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode_by_size");

    for size in [10, 100, 1_000, 10_000, 100_000].iter() {
        let coords = generate_test_coordinates(*size);
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            b.iter(|| encode(black_box(&coords), black_box(5)));
        });
    }
    group.finish();
}

fn bench_decode_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode_by_size");

    for size in [10, 100, 1_000, 10_000, 100_000].iter() {
        let coords = generate_test_coordinates(*size);
        let encoded = encode(&coords, 5).unwrap();

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &encoded, |b, encoded| {
            b.iter(|| decode(black_box(encoded), black_box(5)));
        });
    }
    group.finish();
}

fn bench_precision_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("precision_impact");
    let coords = generate_test_coordinates(1000);

    for precision in [5, 6, 7, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("encode", precision),
            precision,
            |b, &precision| {
                b.iter(|| encode(black_box(&coords), black_box(precision)));
            },
        );
    }
    group.finish();
}

fn bench_coordinate_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("coordinate_patterns");

    // Test different coordinate patterns that affect delta encoding
    let uniform_coords = generate_test_coordinates(1000);
    let realistic_route = generate_realistic_route(1000);
    let extreme_coords: Vec<LngLat> = (0..1000)
        .map(|i| {
            if i % 2 == 0 {
                LngLat::new_deg(-180.0, -90.0)
            } else {
                LngLat::new_deg(180.0, 90.0)
            }
        })
        .collect();

    group.bench_function("uniform_distribution", |b| {
        b.iter(|| encode(black_box(&uniform_coords), black_box(5)))
    });

    group.bench_function("realistic_gps_track", |b| {
        b.iter(|| encode(black_box(&realistic_route), black_box(5)))
    });

    group.bench_function("extreme_deltas", |b| {
        b.iter(|| encode(black_box(&extreme_coords), black_box(5)))
    });

    group.finish();
}

fn bench_roundtrip_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("roundtrip");

    for size in [100, 1_000, 10_000].iter() {
        let coords = generate_test_coordinates(*size);
        group.throughput(Throughput::Elements(*size as u64 * 2)); // encode + decode

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            b.iter(|| {
                let encoded = encode(black_box(&coords), black_box(5)).unwrap();
                let _decoded = decode(black_box(&encoded), black_box(5)).unwrap();
            });
        });
    }
    group.finish();
}

#[cfg(feature = "batch")]
fn bench_batch_vs_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_vs_sequential");

    // Generate multiple small polylines
    let small_polylines: Vec<Vec<LngLat>> =
        (0..200).map(|_| generate_realistic_route(50)).collect();

    // Generate fewer large polylines
    let large_polylines: Vec<Vec<LngLat>> =
        (0..10).map(|_| generate_realistic_route(5000)).collect();

    group.bench_function("small_polylines_sequential", |b| {
        b.iter(|| {
            let _encoded: Result<Vec<_>, _> = small_polylines
                .iter()
                .map(|coords| encode(coords, 5))
                .collect();
        })
    });

    group.bench_function("small_polylines_batch", |b| {
        b.iter(|| {
            let _encoded = encode_batch(black_box(&small_polylines), black_box(5));
        })
    });

    group.bench_function("large_polylines_sequential", |b| {
        b.iter(|| {
            let _encoded: Result<Vec<_>, _> = large_polylines
                .iter()
                .map(|coords| encode(coords, 5))
                .collect();
        })
    });

    group.bench_function("large_polylines_batch", |b| {
        b.iter(|| {
            let _encoded = encode_batch(black_box(&large_polylines), black_box(5));
        })
    });

    group.finish();
}

fn bench_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_patterns");

    // Test string capacity estimation accuracy
    let coords = generate_test_coordinates(1000);
    let encoded = encode(&coords, 5).unwrap();

    group.bench_function("string_growth_current", |b| {
        b.iter(|| {
            // Current implementation with inadequate capacity
            let mut result = String::with_capacity(coords.len() * 4);
            for i in 0..coords.len() {
                result.push_str(&format!("test{}", i)); // Simulate growing string
            }
            black_box(result);
        })
    });

    group.bench_function("string_growth_optimal", |b| {
        b.iter(|| {
            // Better capacity estimation
            let mut result = String::with_capacity(coords.len() * 8);
            for i in 0..coords.len() {
                result.push_str(&format!("test{}", i));
            }
            black_box(result);
        })
    });

    group.bench_function("decode_with_vec_collection", |b| {
        b.iter(|| {
            // Current approach: collect bytes into Vec
            let bytes: Vec<u8> = encoded.bytes().collect();
            black_box(bytes);
        })
    });

    group.bench_function("decode_with_direct_iteration", |b| {
        b.iter(|| {
            // Better approach: direct byte iteration
            for byte in encoded.bytes() {
                black_box(byte);
            }
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_encode_sizes,
    bench_decode_sizes,
    bench_precision_impact,
    bench_coordinate_patterns,
    bench_roundtrip_performance,
    bench_memory_patterns
);

#[cfg(feature = "batch")]
criterion_group!(batch_benches, bench_batch_vs_sequential);

#[cfg(not(feature = "batch"))]
criterion_main!(benches);

#[cfg(feature = "batch")]
criterion_main!(benches, batch_benches);
