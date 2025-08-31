use rapidgeo_distance::{geodesic::haversine, LngLat};
use std::time::Instant;

fn main() {
    println!("testing performance with simulated parallel vs serial computation");

    // set rayon threads to all cores if not already set (for potential future use)
    if std::env::var("RAYON_NUM_THREADS").is_err() {
        let num_cpus = std::thread::available_parallelism().unwrap().get();
        std::env::set_var("RAYON_NUM_THREADS", num_cpus.to_string());
        println!("set RAYON_NUM_THREADS={}", num_cpus);
    }

    // create a bunch of points
    let mut points = Vec::new();
    for i in 0..5000 {
        // Reduced for demo
        let lng = -180.0 + (i as f64 / 5000.0) * 360.0;
        let lat = -85.0 + (i as f64 / 5000.0) * 170.0;
        points.push(LngLat::new_deg(lng, lat));
    }

    println!("created {} points", points.len());

    // serial path length calculation
    let start = Instant::now();
    let mut serial_total = 0.0;
    for i in 1..points.len() {
        serial_total += haversine(points[i - 1], points[i]);
    }
    let serial_time = start.elapsed();

    println!(
        "serial path: {:.0} km in {:?}",
        serial_total / 1000.0,
        serial_time
    );

    // simulate "parallel" computation (really just chunked serial)
    let start = Instant::now();
    let chunk_size = points.len() / 4; // simulate 4 threads
    let mut chunk_totals = Vec::new();

    for chunk_start in (0..points.len()).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(points.len());
        let mut chunk_total = 0.0;

        for i in (chunk_start + 1)..chunk_end {
            chunk_total += haversine(points[i - 1], points[i]);
        }
        chunk_totals.push(chunk_total);
    }

    let simulated_parallel_total: f64 = chunk_totals.iter().sum();
    let simulated_parallel_time = start.elapsed();

    println!(
        "simulated parallel: {:.0} km in {:?}",
        simulated_parallel_total / 1000.0,
        simulated_parallel_time
    );

    // check results match
    let diff = (serial_total - simulated_parallel_total).abs();
    println!("difference: {:.3} m", diff);
}
