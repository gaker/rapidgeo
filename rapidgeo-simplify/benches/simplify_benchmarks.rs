use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rapidgeo_distance::LngLat;
use rapidgeo_simplify::{simplify_dp_into, simplify_dp_mask, SimplifyMethod};

#[cfg(feature = "batch")]
use rapidgeo_simplify::batch;

fn generate_dense_gps_trace(num_points: usize) -> Vec<LngLat> {
    let mut points = Vec::with_capacity(num_points);

    // Create a realistic GPS trace from SF to LA with noise
    let start_lng = -122.4194;
    let start_lat = 37.7749;
    let end_lng = -118.2437;
    let end_lat = 34.0522;

    for i in 0..num_points {
        let t = i as f64 / (num_points - 1) as f64;

        // Linear interpolation with random noise
        let base_lng = start_lng + t * (end_lng - start_lng);
        let base_lat = start_lat + t * (end_lat - start_lat);

        // Add GPS-like noise (±0.0001 degrees ≈ ±11m)
        let noise_lng = (i as f64 * 13.7).sin() * 0.0001;
        let noise_lat = (i as f64 * 17.3).cos() * 0.0001;

        points.push(LngLat::new_deg(base_lng + noise_lng, base_lat + noise_lat));
    }

    points
}

fn generate_great_circle_trace(num_points: usize) -> Vec<LngLat> {
    let mut points = Vec::with_capacity(num_points);

    // Great circle from London to Sydney (crosses antimeridian)
    let start_lng = -0.1276;
    let start_lat = 51.5074;
    let end_lng = 151.2093;
    let end_lat = -33.8688;

    for i in 0..num_points {
        let t = i as f64 / (num_points - 1) as f64;

        // Simple linear interpolation (not true great circle, but good for testing)
        let mut lng = start_lng + t * (end_lng - start_lng);
        let lat = start_lat + t * (end_lat - start_lat);

        // Handle antimeridian crossing
        if lng > 180.0 {
            lng -= 360.0;
        }

        points.push(LngLat::new_deg(lng, lat));
    }

    points
}

fn generate_sawtooth_adversary(num_points: usize) -> Vec<LngLat> {
    let mut points = Vec::with_capacity(num_points);

    // Create worst-case sawtooth pattern that forces O(n²) behavior
    let base_lng = -122.0;
    let base_lat = 37.0;
    let amplitude = 0.1; // Large enough to exceed most tolerances

    for i in 0..num_points {
        let t = i as f64 / (num_points - 1) as f64;
        let lng = base_lng + t;

        // Sawtooth pattern
        let lat = if i % 2 == 0 {
            base_lat + amplitude
        } else {
            base_lat - amplitude
        };

        points.push(LngLat::new_deg(lng, lat));
    }

    points
}

// VM-friendly Criterion config
fn crit_cfg() -> Criterion {
    Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(3))
        .measurement_time(std::time::Duration::from_secs(15))
        .sample_size(150)
        .noise_threshold(0.05)
}

fn bench_single_simplify_operations(c: &mut Criterion) {
    let points = generate_dense_gps_trace(1000);
    let tolerance = 25.0;

    let mut group = c.benchmark_group("single_ops");
    group.throughput(Throughput::Elements(points.len() as u64));

    // Test each method
    for (method_name, method) in [
        ("great_circle", SimplifyMethod::GreatCircleMeters),
        ("planar_enu", SimplifyMethod::PlanarMeters),
        ("euclidean", SimplifyMethod::EuclidRaw),
    ] {
        group.bench_function(format!("dp_mask_{}", method_name), |b| {
            let mut mask = Vec::new();
            b.iter(|| {
                simplify_dp_mask(
                    black_box(&points),
                    black_box(tolerance),
                    black_box(method),
                    black_box(&mut mask),
                );
            })
        });

        group.bench_function(format!("dp_into_{}", method_name), |b| {
            let mut output = Vec::new();
            b.iter(|| {
                simplify_dp_into(
                    black_box(&points),
                    black_box(tolerance),
                    black_box(method),
                    black_box(&mut output),
                );
            })
        });
    }

    group.finish();
}

fn bench_simplify_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling_tests");

    for &size in &[100usize, 1_000, 5_000, 25_000, 100_000] {
        let points = generate_dense_gps_trace(size);
        let tolerance = 25.0;
        let method = SimplifyMethod::GreatCircleMeters;

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("dp_mask_scaling", size),
            &points,
            |b, pts| {
                let mut mask = Vec::new();
                b.iter(|| {
                    simplify_dp_mask(
                        black_box(pts),
                        black_box(tolerance),
                        black_box(method),
                        black_box(&mut mask),
                    );
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("dp_into_scaling", size),
            &points,
            |b, pts| {
                let mut output = Vec::new();
                b.iter(|| {
                    simplify_dp_into(
                        black_box(pts),
                        black_box(tolerance),
                        black_box(method),
                        black_box(&mut output),
                    );
                })
            },
        );
    }

    group.finish();
}

fn bench_different_geometries(c: &mut Criterion) {
    let size = 5000;
    let tolerance = 25.0;
    let method = SimplifyMethod::GreatCircleMeters;

    let mut group = c.benchmark_group("geometry_types");
    group.throughput(Throughput::Elements(size as u64));

    // Dense GPS trace (realistic case)
    let gps_trace = generate_dense_gps_trace(size);
    group.bench_function("dense_gps_trace", |b| {
        let mut mask = Vec::new();
        b.iter(|| {
            simplify_dp_mask(
                black_box(&gps_trace),
                black_box(tolerance),
                black_box(method),
                black_box(&mut mask),
            );
        });
    });

    // Great circle trace (long distance)
    let gc_trace = generate_great_circle_trace(size);
    group.bench_function("great_circle_trace", |b| {
        let mut mask = Vec::new();
        b.iter(|| {
            simplify_dp_mask(
                black_box(&gc_trace),
                black_box(tolerance),
                black_box(method),
                black_box(&mut mask),
            );
        });
    });

    // Sawtooth adversary (worst case)
    let sawtooth = generate_sawtooth_adversary(size);
    group.bench_function("sawtooth_adversary", |b| {
        let mut mask = Vec::new();
        b.iter(|| {
            simplify_dp_mask(
                black_box(&sawtooth),
                black_box(tolerance),
                black_box(method),
                black_box(&mut mask),
            );
        });
    });

    group.finish();
}

fn bench_tolerance_scaling(c: &mut Criterion) {
    let points = generate_dense_gps_trace(10000);
    let method = SimplifyMethod::GreatCircleMeters;

    let mut group = c.benchmark_group("tolerance_scaling");
    group.throughput(Throughput::Elements(points.len() as u64));

    // Different tolerances to see how simplification ratio affects performance
    for tolerance in [1.0, 5.0, 25.0, 100.0, 500.0] {
        group.bench_with_input(
            BenchmarkId::new("tolerance", format!("{}m", tolerance as u32)),
            &tolerance,
            |b, &tol| {
                let mut mask = Vec::new();
                b.iter(|| {
                    simplify_dp_mask(
                        black_box(&points),
                        black_box(tol),
                        black_box(method),
                        black_box(&mut mask),
                    );
                });
            },
        );
    }

    group.finish();
}

fn bench_memory_allocation_vs_reuse(c: &mut Criterion) {
    let points = generate_dense_gps_trace(5000);
    let tolerance = 25.0;
    let method = SimplifyMethod::GreatCircleMeters;

    let mut group = c.benchmark_group("allocation_vs_reuse");
    group.throughput(Throughput::Elements(points.len() as u64));

    // Pre-allocate buffers for reuse tests
    let mut mask_buffer = Vec::with_capacity(points.len());
    let mut output_buffer = Vec::with_capacity(points.len());

    // Mask variants
    group.bench_function("dp_mask_reuse_buffer", |b| {
        b.iter(|| {
            simplify_dp_mask(
                black_box(&points),
                black_box(tolerance),
                black_box(method),
                black_box(&mut mask_buffer),
            );
            black_box(&mask_buffer);
        });
    });

    group.bench_function("dp_mask_allocating", |b| {
        b.iter(|| {
            let mut mask = Vec::new();
            simplify_dp_mask(
                black_box(&points),
                black_box(tolerance),
                black_box(method),
                black_box(&mut mask),
            );
            black_box(&mask);
        });
    });

    // Into variants
    group.bench_function("dp_into_reuse_buffer", |b| {
        b.iter(|| {
            simplify_dp_into(
                black_box(&points),
                black_box(tolerance),
                black_box(method),
                black_box(&mut output_buffer),
            );
            black_box(&output_buffer);
        });
    });

    group.bench_function("dp_into_allocating", |b| {
        b.iter(|| {
            let mut output = Vec::new();
            simplify_dp_into(
                black_box(&points),
                black_box(tolerance),
                black_box(method),
                black_box(&mut output),
            );
            black_box(&output);
        });
    });

    group.finish();
}

#[cfg(feature = "batch")]
fn bench_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_operations");

    for &polyline_count in &[10usize, 100, 1_000] {
        for &points_per_polyline in &[100usize, 1_000] {
            let polylines: Vec<Vec<LngLat>> = (0..polyline_count)
                .map(|i| {
                    generate_dense_gps_trace(points_per_polyline)
                        .into_iter()
                        .map(|mut pt| {
                            // Offset each polyline slightly
                            pt.lng_deg += i as f64 * 0.01;
                            pt
                        })
                        .collect()
                })
                .collect();

            let total_points = (polyline_count * points_per_polyline) as u64;
            group.throughput(Throughput::Elements(total_points));

            let bench_name = format!("{}x{}", polyline_count, points_per_polyline);

            group.bench_with_input(
                BenchmarkId::new("batch_simplify_parallel", &bench_name),
                &polylines,
                |b, polys| {
                    b.iter(|| {
                        let results = batch::simplify_batch_par(
                            black_box(polys),
                            black_box(25.0),
                            black_box(SimplifyMethod::GreatCircleMeters),
                        );
                        black_box(results);
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new("batch_simplify_serial", &bench_name),
                &polylines,
                |b, polys| {
                    b.iter(|| {
                        let results = batch::simplify_batch(
                            black_box(polys),
                            black_box(25.0),
                            black_box(SimplifyMethod::GreatCircleMeters),
                        );
                        black_box(results);
                    });
                },
            );
        }
    }

    group.finish();
}

#[cfg(not(feature = "batch"))]
fn bench_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_operations_manual");

    let polylines: Vec<Vec<LngLat>> = (0..100)
        .map(|i| {
            generate_dense_gps_trace(1000)
                .into_iter()
                .map(|mut pt| {
                    pt.lng_deg += i as f64 * 0.01;
                    pt
                })
                .collect()
        })
        .collect();

    let total_points = (100 * 1000) as u64;
    group.throughput(Throughput::Elements(total_points));

    group.bench_function("batch_simplify_manual", |b| {
        b.iter(|| {
            let results: Vec<Vec<LngLat>> = polylines
                .iter()
                .map(|polyline| {
                    let mut output = Vec::new();
                    simplify_dp_into(
                        polyline,
                        25.0,
                        SimplifyMethod::GreatCircleMeters,
                        &mut output,
                    );
                    output
                })
                .collect();
            black_box(results);
        });
    });

    group.finish();
}

criterion_group!(
    name = benches;
    config = crit_cfg();
    targets =
        bench_single_simplify_operations,
        bench_simplify_scaling,
        bench_different_geometries,
        bench_tolerance_scaling,
        bench_memory_allocation_vs_reuse,
        bench_batch_operations
);
criterion_main!(benches);
