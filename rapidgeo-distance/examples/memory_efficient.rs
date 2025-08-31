use rapidgeo_distance::{geodesic::haversine, LngLat};

fn main() {
    println!("testing memory-efficient distance calculations");

    let points = [
        LngLat::new_deg(-122.4194, 37.7749), // sf
        LngLat::new_deg(-87.6298, 41.8781),  // chicago
        LngLat::new_deg(-74.0060, 40.7128),  // nyc
        LngLat::new_deg(-0.1276, 51.5074),   // london
    ];

    // allocating version (creates new vec)
    let mut distances_alloc = Vec::new();
    for i in 1..points.len() {
        distances_alloc.push(haversine(points[i - 1], points[i]));
    }
    println!("allocated distances: {} segments", distances_alloc.len());
    for (i, d) in distances_alloc.iter().enumerate() {
        println!("  {}: {:.0} km", i, d / 1000.0);
    }

    // memory-efficient version (reuses buffer)
    let mut buffer = vec![0.0; points.len() - 1]; // pre-allocate
    for i in 1..points.len() {
        buffer[i - 1] = haversine(points[i - 1], points[i]);
    }

    println!("reused buffer distances:");
    for (i, d) in buffer.iter().enumerate() {
        println!("  {}: {:.0} km", i, d / 1000.0);
    }

    // verify they're the same
    let all_match = distances_alloc
        .iter()
        .zip(buffer.iter())
        .all(|(a, b)| (a - b).abs() < 1e-6);

    println!("results match: {}", all_match);

    // distances to a point
    let origin = LngLat::new_deg(0.0, 0.0);
    let test_points = [
        LngLat::new_deg(1.0, 0.0),
        LngLat::new_deg(0.0, 1.0),
        LngLat::new_deg(1.0, 1.0),
    ];

    let mut point_distances = vec![0.0; test_points.len()];
    for (i, pt) in test_points.iter().enumerate() {
        point_distances[i] = haversine(*pt, origin);
    }

    println!("distances to origin:");
    for (i, d) in point_distances.iter().enumerate() {
        println!("  point {}: {:.0} km", i, d / 1000.0);
    }
}
