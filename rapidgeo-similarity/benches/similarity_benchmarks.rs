use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rapidgeo_similarity::{
    frechet::{discrete_frechet_distance, discrete_frechet_distance_with_threshold},
    hausdorff::{hausdorff_distance, hausdorff_distance_with_threshold},
    LngLat,
};

#[cfg(feature = "batch")]
use rapidgeo_similarity::batch;

// ---------- helpers ----------
fn generate_realistic_path(n: usize) -> Vec<LngLat> {
    let start_lng = -122.4194; // San Francisco
    let start_lat = 37.7749;

    (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            // Create a realistic winding path across the US
            let lng = start_lng + t * 48.0; // SF to NYC longitude span
            let lat = start_lat + (t * std::f64::consts::PI * 4.0).sin() * 5.0; // Sine wave variation
            LngLat::new_deg(lng, lat)
        })
        .collect()
}

fn generate_polyline(num_points: usize, spread: f64) -> Vec<LngLat> {
    (0..num_points)
        .map(|i| {
            let t = i as f64 / (num_points - 1) as f64;
            LngLat::new_deg(-122.0 + t * spread, 37.0 + t * spread)
        })
        .collect()
}

fn generate_curved_polyline(num_points: usize) -> Vec<LngLat> {
    (0..num_points)
        .map(|i| {
            let t = i as f64 / (num_points - 1) as f64;
            let angle = t * std::f64::consts::PI;
            LngLat::new_deg(-122.0 + angle.cos(), 37.0 + angle.sin())
        })
        .collect()
}

// VM-friendly Criterion config
fn crit_cfg() -> Criterion {
    Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(3))
        .measurement_time(std::time::Duration::from_secs(15))
        .sample_size(150)
        .noise_threshold(0.05)
}

// ---------- benches ----------
fn bench_single_similarity_operations(c: &mut Criterion) {
    let path1 = vec![
        LngLat::new_deg(-122.4194, 37.7749), // San Francisco
        LngLat::new_deg(-74.0060, 40.7128),  // New York
    ];
    let path2 = vec![
        LngLat::new_deg(-122.4, 37.8), // Close to SF
        LngLat::new_deg(-74.1, 40.6),  // Close to NYC
    ];

    let mut group = c.benchmark_group("single_ops");
    group.throughput(Throughput::Elements(1));

    group.bench_function("frechet_two_points", |b| {
        b.iter(|| {
            black_box(discrete_frechet_distance(
                black_box(&path1),
                black_box(&path2),
            ))
        })
    });

    group.bench_function("hausdorff_two_points", |b| {
        b.iter(|| black_box(hausdorff_distance(black_box(&path1), black_box(&path2))))
    });

    // Test with threshold for early termination
    let threshold = 50000.0; // 50km threshold
    group.bench_function("frechet_with_threshold", |b| {
        b.iter(|| {
            black_box(discrete_frechet_distance_with_threshold(
                black_box(&path1),
                black_box(&path2),
                black_box(threshold),
            ))
        })
    });

    group.bench_function("hausdorff_with_threshold", |b| {
        b.iter(|| {
            black_box(hausdorff_distance_with_threshold(
                black_box(&path1),
                black_box(&path2),
                black_box(threshold),
            ))
        })
    });

    group.finish();
}

fn bench_scaling_complexity(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling_complexity");

    for &size in &[10usize, 25, 50, 100, 200] {
        let polyline_a = generate_realistic_path(size);
        let polyline_b = generate_realistic_path(size);
        let complexity = (size * size) as u64; // O(n²) for Fréchet

        group.throughput(Throughput::Elements(complexity));
        group.bench_with_input(BenchmarkId::new("frechet_scaling", size), &size, |b, _| {
            b.iter(|| {
                black_box(discrete_frechet_distance(
                    black_box(&polyline_a),
                    black_box(&polyline_b),
                ))
            })
        });

        // Hausdorff is also O(n²) in distance calculations
        group.throughput(Throughput::Elements(complexity));
        group.bench_with_input(
            BenchmarkId::new("hausdorff_scaling", size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(hausdorff_distance(
                        black_box(&polyline_a),
                        black_box(&polyline_b),
                    ))
                })
            },
        );
    }

    group.finish();
}

fn bench_asymmetric_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("asymmetric_sizes");

    let test_cases = [(10, 50), (25, 100), (50, 200)];

    for (size_a, size_b) in test_cases {
        let polyline_a = generate_realistic_path(size_a);
        let polyline_b = generate_realistic_path(size_b);
        let complexity = (size_a * size_b) as u64;

        group.throughput(Throughput::Elements(complexity));
        group.bench_with_input(
            BenchmarkId::new("frechet_asymmetric", format!("{}x{}", size_a, size_b)),
            &(size_a, size_b),
            |b, _| {
                b.iter(|| {
                    black_box(discrete_frechet_distance(
                        black_box(&polyline_a),
                        black_box(&polyline_b),
                    ))
                })
            },
        );

        group.throughput(Throughput::Elements(complexity));
        group.bench_with_input(
            BenchmarkId::new("hausdorff_asymmetric", format!("{}x{}", size_a, size_b)),
            &(size_a, size_b),
            |b, _| {
                b.iter(|| {
                    black_box(hausdorff_distance(
                        black_box(&polyline_a),
                        black_box(&polyline_b),
                    ))
                })
            },
        );
    }

    group.finish();
}

fn bench_edge_cases(c: &mut Criterion) {
    let mut group = c.benchmark_group("edge_cases");
    group.throughput(Throughput::Elements(1));

    let single_point = vec![LngLat::new_deg(-122.0, 37.0)];
    let small_line = vec![LngLat::new_deg(-122.0, 37.0), LngLat::new_deg(-122.1, 37.1)];
    let large_line = generate_realistic_path(100);

    // Single point comparisons
    group.bench_function("frechet_single_vs_single", |b| {
        b.iter(|| {
            black_box(discrete_frechet_distance(
                black_box(&single_point),
                black_box(&single_point),
            ))
        })
    });

    group.bench_function("hausdorff_single_vs_single", |b| {
        b.iter(|| {
            black_box(hausdorff_distance(
                black_box(&single_point),
                black_box(&single_point),
            ))
        })
    });

    // Single vs large comparisons
    group.throughput(Throughput::Elements(100));
    group.bench_function("frechet_single_vs_large", |b| {
        b.iter(|| {
            black_box(discrete_frechet_distance(
                black_box(&single_point),
                black_box(&large_line),
            ))
        })
    });

    group.bench_function("hausdorff_single_vs_large", |b| {
        b.iter(|| {
            black_box(hausdorff_distance(
                black_box(&single_point),
                black_box(&large_line),
            ))
        })
    });

    // Small vs large
    group.throughput(Throughput::Elements(200));
    group.bench_function("frechet_small_vs_large", |b| {
        b.iter(|| {
            black_box(discrete_frechet_distance(
                black_box(&small_line),
                black_box(&large_line),
            ))
        })
    });

    group.bench_function("hausdorff_small_vs_large", |b| {
        b.iter(|| {
            black_box(hausdorff_distance(
                black_box(&small_line),
                black_box(&large_line),
            ))
        })
    });

    group.finish();
}

fn bench_curve_shapes(c: &mut Criterion) {
    let mut group = c.benchmark_group("curve_shapes");
    group.throughput(Throughput::Elements(10000)); // 100x100 complexity

    let straight_line = generate_polyline(100, 1.0);
    let curved_line = generate_curved_polyline(100);
    let realistic_path = generate_realistic_path(100);

    group.bench_function("frechet_straight_vs_straight", |b| {
        b.iter(|| {
            black_box(discrete_frechet_distance(
                black_box(&straight_line),
                black_box(&straight_line),
            ))
        })
    });

    group.bench_function("frechet_straight_vs_curved", |b| {
        b.iter(|| {
            black_box(discrete_frechet_distance(
                black_box(&straight_line),
                black_box(&curved_line),
            ))
        })
    });

    group.bench_function("frechet_realistic_paths", |b| {
        b.iter(|| {
            black_box(discrete_frechet_distance(
                black_box(&realistic_path),
                black_box(&curved_line),
            ))
        })
    });

    group.bench_function("hausdorff_straight_vs_curved", |b| {
        b.iter(|| {
            black_box(hausdorff_distance(
                black_box(&straight_line),
                black_box(&curved_line),
            ))
        })
    });

    group.bench_function("hausdorff_realistic_paths", |b| {
        b.iter(|| {
            black_box(hausdorff_distance(
                black_box(&realistic_path),
                black_box(&curved_line),
            ))
        })
    });

    group.finish();
}

fn bench_threshold_early_termination(c: &mut Criterion) {
    let mut group = c.benchmark_group("threshold_early_termination");

    let path_a = generate_realistic_path(100);
    let path_b = generate_realistic_path(100);

    // Different threshold levels to test early termination behavior
    let thresholds = [
        ("very_low", 1000.0),     // Should terminate early
        ("low", 10000.0),         // Might terminate early
        ("high", 100000.0),       // Unlikely to terminate early
        ("very_high", 1000000.0), // Never terminates early
    ];

    for (name, threshold) in thresholds {
        group.bench_with_input(
            BenchmarkId::new("frechet_with_threshold", name),
            &threshold,
            |b, &thresh| {
                b.iter(|| {
                    black_box(discrete_frechet_distance_with_threshold(
                        black_box(&path_a),
                        black_box(&path_b),
                        black_box(thresh),
                    ))
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("hausdorff_with_threshold", name),
            &threshold,
            |b, &thresh| {
                b.iter(|| {
                    black_box(hausdorff_distance_with_threshold(
                        black_box(&path_a),
                        black_box(&path_b),
                        black_box(thresh),
                    ))
                })
            },
        );
    }

    group.finish();
}

#[cfg(feature = "batch")]
fn bench_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_operations");

    for &batch_size in &[10usize, 50, 100] {
        let polylines_a: Vec<Vec<LngLat>> = (0..batch_size)
            .map(|_| generate_realistic_path(50))
            .collect();
        let polylines_b: Vec<Vec<LngLat>> = (0..batch_size)
            .map(|_| generate_realistic_path(50))
            .collect();

        let operations = batch_size as u64;
        group.throughput(Throughput::Elements(operations));

        let polyline_pairs: Vec<(Vec<LngLat>, Vec<LngLat>)> = polylines_a
            .iter()
            .zip(polylines_b.iter())
            .map(|(a, b)| (a.clone(), b.clone()))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("batch_frechet", batch_size),
            &batch_size,
            |b, _| b.iter(|| black_box(batch::batch_frechet_distance(black_box(&polyline_pairs)))),
        );

        // Test with threshold
        let threshold = 50000.0;
        group.bench_with_input(
            BenchmarkId::new("batch_frechet_threshold", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    black_box(batch::batch_frechet_distance_threshold(
                        black_box(&polyline_pairs),
                        black_box(threshold),
                    ))
                })
            },
        );
    }

    group.finish();
}

#[cfg(not(feature = "batch"))]
fn bench_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_operations_manual");

    for &batch_size in &[10usize, 50] {
        let polylines_a: Vec<Vec<LngLat>> = (0..batch_size)
            .map(|_| generate_realistic_path(50))
            .collect();
        let polylines_b: Vec<Vec<LngLat>> = (0..batch_size)
            .map(|_| generate_realistic_path(50))
            .collect();

        let operations = batch_size as u64;
        group.throughput(Throughput::Elements(operations));

        group.bench_with_input(
            BenchmarkId::new("manual_batch_frechet", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    let results: Vec<_> = polylines_a
                        .iter()
                        .zip(polylines_b.iter())
                        .map(|(a, b)| discrete_frechet_distance(a, b).unwrap())
                        .collect();
                    black_box(results);
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    name = benches;
    config = crit_cfg();
    targets =
        bench_single_similarity_operations,
        bench_scaling_complexity,
        bench_asymmetric_sizes,
        bench_edge_cases,
        bench_curve_shapes,
        bench_threshold_early_termination,
        bench_batch_operations
);
criterion_main!(benches);
