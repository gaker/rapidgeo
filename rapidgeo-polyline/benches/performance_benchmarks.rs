use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rapidgeo_distance::LngLat;
use rapidgeo_polyline::{decode, encode, encode_simplified, simplify_polyline};
use rapidgeo_simplify::SimplifyMethod;

#[cfg(feature = "batch")]
use rapidgeo_polyline::batch::{decode_batch, encode_batch, encode_simplified_batch};

fn generate_coordinates(count: usize) -> Vec<LngLat> {
    (0..count)
        .map(|i| {
            LngLat::new_deg(
                -180.0 + (i as f64 * 0.001) % 360.0, // lng
                -90.0 + (i as f64 * 0.0007) % 180.0, // lat
            )
        })
        .collect()
}

fn generate_detailed_route(count: usize) -> Vec<LngLat> {
    // Generate a realistic route with small incremental changes
    let mut coords = Vec::with_capacity(count);
    let mut lng = -122.0;
    let mut lat = 37.0;

    for i in 0..count {
        coords.push(LngLat::new_deg(lng, lat));
        lng += if i % 3 == 0 { 0.001 } else { 0.0001 };
        lat += if i % 5 == 0 { 0.001 } else { 0.0001 };
    }
    coords
}

fn bench_encode_by_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode_by_size");

    for size in [10, 100, 1000, 5000].iter() {
        let coords = generate_coordinates(*size);
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            b.iter(|| encode(black_box(&coords), black_box(5)).unwrap());
        });
    }
    group.finish();
}

fn bench_decode_by_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode_by_size");

    for size in [10, 100, 1000, 5000].iter() {
        let coords = generate_coordinates(*size);
        let encoded = encode(&coords, 5).unwrap();

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            b.iter(|| decode(black_box(&encoded), black_box(5)).unwrap());
        });
    }
    group.finish();
}

fn bench_encode_decode_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("roundtrip_by_size");

    for size in [100, 1000, 5000].iter() {
        let coords = generate_coordinates(*size);

        group.throughput(Throughput::Elements(*size as u64 * 2)); // encode + decode
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &_size| {
            b.iter(|| {
                let encoded = encode(black_box(&coords), black_box(5)).unwrap();
                decode(black_box(&encoded), black_box(5)).unwrap()
            });
        });
    }
    group.finish();
}

fn bench_simplification(c: &mut Criterion) {
    let mut group = c.benchmark_group("simplification");

    // Test different route complexities
    for size in [100, 500, 2000].iter() {
        let coords = generate_detailed_route(*size);
        let polyline = encode(&coords, 5).unwrap();

        // Bench coordinate simplification
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("simplify_coords", size),
            size,
            |b, &_size| {
                b.iter(|| {
                    encode_simplified(
                        black_box(&coords),
                        black_box(100.0),
                        black_box(SimplifyMethod::GreatCircleMeters),
                        black_box(5),
                    )
                    .unwrap()
                });
            },
        );

        // Bench polyline simplification
        group.bench_with_input(
            BenchmarkId::new("simplify_polyline", size),
            size,
            |b, &_size| {
                b.iter(|| {
                    simplify_polyline(
                        black_box(&polyline),
                        black_box(100.0),
                        black_box(SimplifyMethod::GreatCircleMeters),
                        black_box(5),
                    )
                    .unwrap()
                });
            },
        );
    }
    group.finish();
}

#[cfg(feature = "batch")]
fn bench_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_operations");

    // Create batches of different sizes
    for batch_size in [10, 50, 200].iter() {
        let coords_per_polyline = 100;
        let batch: Vec<Vec<LngLat>> = (0..*batch_size)
            .map(|_| generate_coordinates(coords_per_polyline))
            .collect();

        let encoded_batch: Vec<String> = batch
            .iter()
            .map(|coords| encode(coords, 5).unwrap())
            .collect();

        group.throughput(Throughput::Elements(
            (*batch_size * coords_per_polyline) as u64,
        ));

        // Bench batch encoding
        group.bench_with_input(
            BenchmarkId::new("encode_batch", batch_size),
            batch_size,
            |b, &_size| {
                b.iter(|| encode_batch(black_box(&batch), black_box(5)).unwrap());
            },
        );

        // Bench batch decoding
        group.bench_with_input(
            BenchmarkId::new("decode_batch", batch_size),
            batch_size,
            |b, &_size| {
                b.iter(|| decode_batch(black_box(&encoded_batch), black_box(5)).unwrap());
            },
        );

        // Bench batch simplification
        group.bench_with_input(
            BenchmarkId::new("simplify_batch", batch_size),
            batch_size,
            |b, &_size| {
                b.iter(|| {
                    encode_simplified_batch(
                        black_box(&batch),
                        black_box(100.0),
                        black_box(SimplifyMethod::GreatCircleMeters),
                        black_box(5),
                    )
                    .unwrap()
                });
            },
        );
    }
    group.finish();
}

fn bench_precision_levels(c: &mut Criterion) {
    let mut group = c.benchmark_group("precision_levels");

    let coords = generate_coordinates(1000);

    for precision in [5, 6, 8, 10, 11].iter() {
        group.bench_with_input(BenchmarkId::new("encode", precision), precision, |b, &p| {
            b.iter(|| encode(black_box(&coords), black_box(p)).unwrap());
        });

        let encoded = encode(&coords, *precision).unwrap();
        group.bench_with_input(BenchmarkId::new("decode", precision), precision, |b, &p| {
            b.iter(|| decode(black_box(&encoded), black_box(p)).unwrap());
        });
    }
    group.finish();
}

fn bench_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_patterns");

    // Test different coordinate patterns that stress memory allocation
    let sparse_coords = (0..1000)
        .step_by(10)
        .map(|i| LngLat::new_deg(-180.0 + i as f64 * 0.36, -90.0 + i as f64 * 0.18))
        .collect::<Vec<_>>();

    let dense_coords = generate_coordinates(1000);

    // Large deltas (will produce longer encoded strings)
    let large_delta_coords = (0..1000)
        .map(|i| {
            LngLat::new_deg(
                (i as f64 * 0.1) % 360.0 - 180.0,
                (i as f64 * 0.05) % 180.0 - 90.0,
            )
        })
        .collect::<Vec<_>>();

    group.bench_function("sparse_coords", |b| {
        b.iter(|| encode(black_box(&sparse_coords), black_box(5)).unwrap());
    });

    group.bench_function("dense_coords", |b| {
        b.iter(|| encode(black_box(&dense_coords), black_box(5)).unwrap());
    });

    group.bench_function("large_delta_coords", |b| {
        b.iter(|| encode(black_box(&large_delta_coords), black_box(5)).unwrap());
    });

    group.finish();
}

#[cfg(feature = "batch")]
criterion_group!(
    benches,
    bench_encode_by_size,
    bench_decode_by_size,
    bench_encode_decode_roundtrip,
    bench_simplification,
    bench_batch_operations,
    bench_precision_levels,
    bench_memory_patterns
);

#[cfg(not(feature = "batch"))]
criterion_group!(
    benches,
    bench_encode_by_size,
    bench_decode_by_size,
    bench_encode_decode_roundtrip,
    bench_simplification,
    bench_precision_levels,
    bench_memory_patterns
);

criterion_main!(benches);
