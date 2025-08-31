use rapidgeo_distance::{geodesic::haversine, LngLat};

fn main() {
    println!("testing batch operations using basic geodesic functions");

    // simple 3-city route
    let route = [
        LngLat::new_deg(-122.4194, 37.7749), // sf
        LngLat::new_deg(-87.6298, 41.8781),  // chicago
        LngLat::new_deg(-74.0060, 40.7128),  // nyc
    ];

    // get total path length
    let mut total = 0.0;
    for i in 1..route.len() {
        total += haversine(route[i - 1], route[i]);
    }
    println!("total route: {:.0} km", total / 1000.0);

    // get individual segments
    println!("segments:");
    for i in 1..route.len() {
        let dist = haversine(route[i - 1], route[i]);
        println!("  {}: {:.0} km", i - 1, dist / 1000.0);
    }

    // test with more points
    let mut longer_route = vec![];
    for i in 0..10 {
        let lng = -122.0 + (i as f64 * 5.0); // west to east
        let lat = 37.0 + (i as f64 * 0.5); // south to north
        longer_route.push(LngLat::new_deg(lng, lat));
    }

    let mut longer_total = 0.0;
    for i in 1..longer_route.len() {
        longer_total += haversine(longer_route[i - 1], longer_route[i]);
    }
    println!(
        "longer route ({} points): {:.0} km",
        longer_route.len(),
        longer_total / 1000.0
    );

    // distance from many points to target
    let target = LngLat::new_deg(0.0, 0.0); // origin
    let points = [
        LngLat::new_deg(1.0, 0.0),
        LngLat::new_deg(0.0, 1.0),
        LngLat::new_deg(1.0, 1.0),
        LngLat::new_deg(2.0, 2.0),
    ];

    println!("distances to origin:");
    for (i, pt) in points.iter().enumerate() {
        let dist = haversine(*pt, target);
        println!("  point {}: {:.0} km", i, dist / 1000.0);
    }
}
