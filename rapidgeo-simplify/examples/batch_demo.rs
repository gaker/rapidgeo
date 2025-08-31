use rapidgeo_distance::LngLat;
use rapidgeo_simplify::{simplify_dp_into, SimplifyMethod};

#[cfg(feature = "batch")]
use rapidgeo_simplify::batch::{simplify_batch_par, simplify_dp_into_par};

fn create_large_polyline(size: usize) -> Vec<LngLat> {
    let mut points = Vec::with_capacity(size);

    // Create a realistic GPS trace with noise
    let start_lng = -122.4194;
    let start_lat = 37.7749;
    let end_lng = -118.2437;
    let end_lat = 34.0522;

    for i in 0..size {
        let t = i as f64 / (size - 1) as f64;

        let base_lng = start_lng + t * (end_lng - start_lng);
        let base_lat = start_lat + t * (end_lat - start_lat);

        // Add GPS-like noise
        let noise_lng = (i as f64 * 13.7).sin() * 0.0001;
        let noise_lat = (i as f64 * 17.3).cos() * 0.0001;

        points.push(LngLat::new_deg(base_lng + noise_lng, base_lat + noise_lat));
    }

    points
}

fn main() {
    println!("rapidgeo-simplify batch processing demo");

    // Test single large polyline
    let large_polyline = create_large_polyline(10000);
    println!(
        "Created large polyline with {} points",
        large_polyline.len()
    );

    let tolerance = 25.0; // 25 meters
    let method = SimplifyMethod::GreatCircleMeters;

    // Serial processing
    let start = std::time::Instant::now();
    let mut serial_result = Vec::new();
    let serial_count = simplify_dp_into(&large_polyline, tolerance, method, &mut serial_result);
    let serial_time = start.elapsed();

    println!(
        "Serial: {} -> {} points in {:?}",
        large_polyline.len(),
        serial_count,
        serial_time
    );

    #[cfg(feature = "batch")]
    {
        // Parallel processing
        let start = std::time::Instant::now();
        let mut par_result = Vec::new();
        let par_count = simplify_dp_into_par(&large_polyline, tolerance, method, &mut par_result);
        let par_time = start.elapsed();

        println!(
            "Parallel: {} -> {} points in {:?}",
            large_polyline.len(),
            par_count,
            par_time
        );

        // Verify results are identical
        assert_eq!(serial_count, par_count);
        assert_eq!(serial_result, par_result);
        println!("✅ Serial and parallel results match");

        // Test batch processing of multiple polylines
        let polylines = vec![
            create_large_polyline(2000),
            create_large_polyline(3000),
            create_large_polyline(1500),
            create_large_polyline(2500),
        ];

        let total_points: usize = polylines.iter().map(|p| p.len()).sum();
        println!(
            "\nBatch processing {} polylines with {} total points",
            polylines.len(),
            total_points
        );

        // Serial batch
        let start = std::time::Instant::now();
        let serial_batch = rapidgeo_simplify::batch::simplify_batch(&polylines, tolerance, method);
        let serial_batch_time = start.elapsed();

        let serial_total_out: usize = serial_batch.iter().map(|p| p.len()).sum();
        println!(
            "Serial batch: {} -> {} points in {:?}",
            total_points, serial_total_out, serial_batch_time
        );

        // Parallel batch
        let start = std::time::Instant::now();
        let par_batch = simplify_batch_par(&polylines, tolerance, method);
        let par_batch_time = start.elapsed();

        let par_total_out: usize = par_batch.iter().map(|p| p.len()).sum();
        println!(
            "Parallel batch: {} -> {} points in {:?}",
            total_points, par_total_out, par_batch_time
        );

        // Verify results match
        assert_eq!(serial_batch, par_batch);
        println!("✅ Serial and parallel batch results match");

        if par_batch_time < serial_batch_time {
            let speedup = serial_batch_time.as_secs_f64() / par_batch_time.as_secs_f64();
            println!("🚀 Parallel batch is {:.2}x faster!", speedup);
        }
    }

    #[cfg(not(feature = "batch"))]
    {
        println!("To see parallel processing demo, run with: cargo run --example batch_demo --features batch");
    }
}
