use rapidgeo_distance::LngLat;
use rapidgeo_simplify::{simplify_dp_into, simplify_dp_mask, SimplifyMethod};

fn main() {
    // Create a simple polyline - a path from SF to LA with some detours
    let points = vec![
        LngLat::new_deg(-122.4194, 37.7749), // San Francisco
        LngLat::new_deg(-122.0, 37.5),       // Small detour east
        LngLat::new_deg(-121.5, 37.0),       // Another point
        LngLat::new_deg(-120.0, 36.0),       // Central Valley
        LngLat::new_deg(-119.0, 35.0),       // More south
        LngLat::new_deg(-118.2437, 34.0522), // Los Angeles
    ];

    println!("Original polyline has {} points", points.len());

    // Method 1: Using mask to see which points are kept
    let mut mask = Vec::new();
    simplify_dp_mask(
        &points,
        50000.0,
        SimplifyMethod::GreatCircleMeters,
        &mut mask,
    );

    println!("With 50km tolerance, keep mask: {:?}", mask);
    let kept_count = mask.iter().filter(|&&x| x).count();
    println!("Keeping {} out of {} points", kept_count, points.len());

    // Method 2: Getting simplified points directly
    let mut simplified = Vec::new();
    let count = simplify_dp_into(
        &points,
        50000.0,
        SimplifyMethod::GreatCircleMeters,
        &mut simplified,
    );

    println!("Simplified polyline has {} points:", count);
    for (i, point) in simplified.iter().enumerate() {
        println!(
            "  Point {}: ({:.4}, {:.4})",
            i + 1,
            point.lng_deg,
            point.lat_deg
        );
    }

    // Try with different methods
    println!("\nComparing different distance methods:");

    for (method_name, method) in [
        ("Great Circle", SimplifyMethod::GreatCircleMeters),
        ("ENU Planar", SimplifyMethod::PlanarMeters),
        ("Raw Euclidean", SimplifyMethod::EuclidRaw),
    ] {
        let mut method_simplified = Vec::new();
        let method_count = simplify_dp_into(&points, 50000.0, method, &mut method_simplified);
        println!("  {}: {} points kept", method_name, method_count);
    }
}
