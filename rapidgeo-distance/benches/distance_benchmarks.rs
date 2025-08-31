use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rapidgeo_distance::{euclid, geodesic, LngLat};

#[cfg(feature = "batch")]
use rapidgeo_distance::batch;

// ---------- helpers ----------
fn generate_test_points(n: usize) -> Vec<LngLat> {
    (0..n)
        .map(|i| {
            let lng = -180.0 + (i as f64 / n as f64) * 360.0;
            let lat = -90.0 + ((i * 7) % n) as f64 / n as f64 * 180.0;
            LngLat::new_deg(lng, lat)
        })
        .collect()
}

fn generate_realistic_path(n: usize) -> Vec<LngLat> {
    let start_lng = -122.4194;
    let start_lat = 37.7749;

    (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            let lng = start_lng + t * 48.0;
            let lat = start_lat + (t * std::f64::consts::PI * 4.0).sin() * 5.0;
            LngLat::new_deg(lng, lat)
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
fn bench_single_distance_functions(c: &mut Criterion) {
    let p1 = LngLat::new_deg(-122.4194, 37.7749);
    let p2 = LngLat::new_deg(-74.0060, 40.7128);

    let mut group = c.benchmark_group("single_ops");
    group.throughput(Throughput::Elements(1));

    group.bench_function("haversine_single", |b| {
        b.iter(|| black_box(geodesic::haversine(black_box(p1), black_box(p2))))
    });

    group.bench_function("vincenty_single", |b| {
        b.iter(|| black_box(geodesic::vincenty_distance_m(black_box(p1), black_box(p2))))
    });

    group.bench_function("euclidean_single", |b| {
        b.iter(|| black_box(euclid::distance_euclid(black_box(p1), black_box(p2))))
    });

    // Point-to-segment
    let segment = (p1, p2);
    let point = LngLat::new_deg(-98.5, 39.5);

    group.bench_function("point_to_segment_euclidean", |b| {
        b.iter(|| {
            black_box(euclid::point_to_segment(
                black_box(point),
                black_box(segment),
            ))
        })
    });

    group.bench_function("point_to_segment_enu", |b| {
        b.iter(|| {
            black_box(geodesic::point_to_segment_enu_m(
                black_box(point),
                black_box(segment),
            ))
        })
    });

    group.bench_function("point_to_segment_gc", |b| {
        b.iter(|| {
            black_box(geodesic::great_circle_point_to_seg(
                black_box(point),
                black_box(segment),
            ))
        })
    });

    group.finish();
}

#[cfg(feature = "batch")]
fn bench_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_operations");

    for &size in &[10usize, 100, 1_000, 10_000, 100_000] {
        let points = generate_realistic_path(size);
        let pairs = (size - 1) as u64;

        // serial pairwise
        group.throughput(Throughput::Elements(pairs));
        group.bench_with_input(
            BenchmarkId::new("pairwise_haversine_serial", size),
            &points,
            |b, pts| {
                b.iter(|| {
                    // allocate is intentional here — measuring allocating variant
                    let v: Vec<f64> = batch::pairwise_haversine(black_box(pts)).collect();
                    black_box(v);
                })
            },
        );

        // parallel pairwise
        group.throughput(Throughput::Elements(pairs));
        group.bench_with_input(
            BenchmarkId::new("pairwise_haversine_parallel", size),
            &points,
            |b, pts| {
                b.iter(|| {
                    let v = batch::pairwise_haversine_par(black_box(pts));
                    black_box(v);
                })
            },
        );

        // path length serial
        group.throughput(Throughput::Elements(pairs));
        group.bench_with_input(
            BenchmarkId::new("path_length_haversine", size),
            &points,
            |b, pts| b.iter(|| black_box(batch::path_length_haversine(black_box(pts)))),
        );

        // path length parallel
        group.throughput(Throughput::Elements(pairs));
        group.bench_with_input(
            BenchmarkId::new("path_length_haversine_parallel", size),
            &points,
            |b, pts| b.iter(|| black_box(batch::path_length_haversine_par(black_box(pts)))),
        );

        #[cfg(feature = "vincenty")]
        {
            group.throughput(Throughput::Elements(pairs));
            group.bench_with_input(
                BenchmarkId::new("path_length_vincenty", size),
                &points,
                |b, pts| b.iter(|| black_box(batch::path_length_vincenty_m(black_box(pts)))),
            );
        }
    }

    group.finish();
}

#[cfg(not(feature = "batch"))]
fn bench_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_operations_manual");

    for &size in &[10usize, 100, 1_000, 10_000, 100_000] {
        let points = generate_realistic_path(size);
        let pairs = (size - 1) as u64;

        group.throughput(Throughput::Elements(pairs));
        group.bench_with_input(
            BenchmarkId::new("pairwise_haversine_manual", size),
            &points,
            |b, pts| {
                b.iter(|| {
                    let v: Vec<f64> = pts
                        .windows(2)
                        .map(|pair| geodesic::haversine(pair[0], pair[1]))
                        .collect();
                    black_box(v);
                })
            },
        );

        group.throughput(Throughput::Elements(pairs));
        group.bench_with_input(
            BenchmarkId::new("path_length_haversine_manual", size),
            &points,
            |b, pts| {
                b.iter(|| {
                    let s: f64 = pts
                        .windows(2)
                        .map(|pair| geodesic::haversine(pair[0], pair[1]))
                        .sum();
                    black_box(s);
                })
            },
        );
    }

    group.finish();
}

#[cfg(feature = "batch")]
fn bench_distances_to_point(c: &mut Criterion) {
    let mut group = c.benchmark_group("distances_to_point");
    let target = LngLat::new_deg(-98.5, 39.5);

    for &size in &[100usize, 1_000, 10_000, 100_000] {
        let points = generate_test_points(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("distances_to_point_parallel", size),
            &points,
            |b, pts| {
                b.iter(|| {
                    let v = batch::distances_to_point_par(black_box(pts), black_box(target));
                    black_box(v);
                })
            },
        );
    }

    group.finish();
}

#[cfg(not(feature = "batch"))]
fn bench_distances_to_point(c: &mut Criterion) {
    let mut group = c.benchmark_group("distances_to_point_manual");
    let target = LngLat::new_deg(-98.5, 39.5);

    for &size in &[100usize, 1_000, 10_000] {
        let points = generate_test_points(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("distances_to_point_manual_serial", size),
            &points,
            |b, pts| {
                b.iter(|| {
                    let v: Vec<f64> = pts
                        .iter()
                        .map(|&p| geodesic::haversine(p, black_box(target)))
                        .collect();
                    black_box(v);
                })
            },
        );
    }

    group.finish();
}

#[cfg(feature = "batch")]
fn bench_memory_allocation_vs_into(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocation_vs_into");
    let points = generate_realistic_path(1_000);
    let target = LngLat::new_deg(-98.5, 39.5);

    // Pre-allocate output buffer once; reuse across iters
    let mut out = vec![0.0; points.len().max(1)];

    // Pairwise (999 distances)
    let pair_count = (points.len() - 1) as u64;

    group.throughput(Throughput::Elements(pair_count));
    group.bench_function("pairwise_allocating", |b| {
        b.iter(|| {
            let v: Vec<f64> = batch::pairwise_haversine(black_box(&points)).collect();
            black_box(v);
        })
    });

    group.throughput(Throughput::Elements(pair_count));
    group.bench_function("pairwise_into", |b| {
        b.iter(|| {
            batch::pairwise_haversine_into(
                black_box(&points),
                black_box(&mut out[..points.len() - 1]),
            );
            black_box(&out[..points.len() - 1]);
        })
    });

    group.throughput(Throughput::Elements(pair_count));
    group.bench_function("pairwise_parallel_allocating", |b| {
        b.iter(|| {
            let v = batch::pairwise_haversine_par(black_box(&points));
            black_box(v);
        })
    });

    group.throughput(Throughput::Elements(pair_count));
    group.bench_function("pairwise_parallel_into", |b| {
        b.iter(|| {
            batch::pairwise_haversine_par_into(
                black_box(&points),
                black_box(&mut out[..points.len() - 1]),
            );
            black_box(&out[..points.len() - 1]);
        })
    });

    // Distances-to-point for first 500 points
    let n = 500usize;
    let pts500 = &points[..n];

    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("distances_to_point_allocating", |b| {
        b.iter(|| {
            let v = batch::distances_to_point_par(black_box(pts500), black_box(target));
            black_box(v);
        })
    });

    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("distances_to_point_into", |b| {
        b.iter(|| {
            batch::distances_to_point_into(
                black_box(pts500),
                black_box(target),
                black_box(&mut out[..n]),
            );
            black_box(&out[..n]);
        })
    });

    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("distances_to_point_parallel_into", |b| {
        b.iter(|| {
            batch::distances_to_point_par_into(
                black_box(pts500),
                black_box(target),
                black_box(&mut out[..n]),
            );
            black_box(&out[..n]);
        })
    });

    group.finish();
}

#[cfg(not(feature = "batch"))]
fn bench_memory_allocation_vs_into(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocation_manual");
    let points = generate_realistic_path(1_000);
    let target = LngLat::new_deg(-98.5, 39.5);

    let pair_count = (points.len() - 1) as u64;

    group.throughput(Throughput::Elements(pair_count));
    group.bench_function("pairwise_manual", |b| {
        b.iter(|| {
            let v: Vec<f64> = points
                .windows(2)
                .map(|pair| geodesic::haversine(pair[0], pair[1]))
                .collect();
            black_box(v);
        })
    });

    let n = 500usize;
    let pts500 = &points[..n];

    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("distances_to_point_manual", |b| {
        b.iter(|| {
            let v: Vec<f64> = pts500
                .iter()
                .map(|&p| geodesic::haversine(p, target))
                .collect();
            black_box(v);
        })
    });

    group.finish();
}

fn bench_numerical_stability(c: &mut Criterion) {
    let mut group = c.benchmark_group("numerical_stability");
    group.throughput(Throughput::Elements(1));

    let base = LngLat::new_deg(0.0, 0.0);

    let small = [
        ("1mm", 0.001),
        ("1cm", 0.01),
        ("10cm", 0.1),
        ("1m", 1.0),
        ("10m", 10.0),
    ];
    for (name, meters) in small {
        // 1 deg lat ≈ 111_320 m near equator
        let dlat = meters / 111_320.0;
        let p = LngLat::new_deg(0.0, dlat);
        group.bench_with_input(
            BenchmarkId::new("haversine_small_distance", name),
            &p,
            |b, p| b.iter(|| black_box(geodesic::haversine(black_box(base), black_box(*p)))),
        );
    }

    let large = [
        ("across_usa", LngLat::new_deg(-74.0060, 40.7128)),
        ("across_pacific", LngLat::new_deg(139.6917, 35.6895)),
        ("antipodal", LngLat::new_deg(179.9, 0.1)),
    ];
    for (name, p) in large {
        group.bench_with_input(
            BenchmarkId::new("haversine_large_distance", name),
            &p,
            |b, p| b.iter(|| black_box(geodesic::haversine(black_box(base), black_box(*p)))),
        );
        group.bench_with_input(
            BenchmarkId::new("vincenty_large_distance", name),
            &p,
            |b, p| {
                b.iter(|| {
                    black_box(geodesic::vincenty_distance_m(
                        black_box(base),
                        black_box(*p),
                    ))
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
        bench_single_distance_functions,
        bench_batch_operations,
        bench_distances_to_point,
        bench_memory_allocation_vs_into,
        bench_numerical_stability
);
criterion_main!(benches);
