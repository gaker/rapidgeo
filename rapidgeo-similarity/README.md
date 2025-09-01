# rapidgeo-similarity

[![Crates.io](https://img.shields.io/crates/v/rapidgeo-similarity)](https://crates.io/crates/rapidgeo-similarity)
[![Documentation](https://docs.rs/rapidgeo-similarity/badge.svg)](https://docs.rs/rapidgeo-similarity)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/gaker/rapidgeo)

Curve similarity measures for geographic polylines. Measures how similar two paths are using geometric algorithms.

## Installation

```toml
[dependencies]
rapidgeo-similarity = "0.1"

# Enable batch processing with Rayon
rapidgeo-similarity = { version = "0.1", features = ["batch"] }
```

## Quick Start

```rust
use rapidgeo_similarity::{LngLat, frechet::discrete_frechet_distance, hausdorff::hausdorff_distance};

let route_a = vec![
    LngLat::new_deg(-122.0, 37.0),  // San Francisco
    LngLat::new_deg(-122.1, 37.1),
    LngLat::new_deg(-122.2, 37.2),
];

let route_b = vec![
    LngLat::new_deg(-122.0, 37.05), // Similar but slightly different
    LngLat::new_deg(-122.1, 37.15),
    LngLat::new_deg(-122.2, 37.25),
];

// Fréchet distance: considers point order (good for trajectories)
let frechet = discrete_frechet_distance(&route_a, &route_b)?;
println!("Fréchet distance: {:.0}m", frechet);

// Hausdorff distance: ignores point order (good for shape matching)
let hausdorff = hausdorff_distance(&route_a, &route_b)?;
println!("Hausdorff distance: {:.0}m", hausdorff);
```

## What These Algorithms Actually Do

### Fréchet Distance

Think of two people walking their dogs along different paths. The Fréchet distance is the shortest leash you'd need so both dogs can reach their owners at any point during the walk, walking at whatever speed they want but never going backwards.

**When to use:**
- Comparing GPS trajectories where order matters
- Route similarity for navigation
- Animal movement pattern analysis
- Time-series path comparison

```rust
use rapidgeo_similarity::{LngLat, frechet::DiscreteFrechet, SimilarityMeasure};

let measure = DiscreteFrechet;

// These routes go in the same direction
let morning_route = vec![
    LngLat::new_deg(-122.419, 37.775), // Home
    LngLat::new_deg(-122.418, 37.773), // Turn right
    LngLat::new_deg(-122.416, 37.774), // Coffee shop
];

let evening_route = vec![
    LngLat::new_deg(-122.419, 37.776), // Home (slightly different GPS)
    LngLat::new_deg(-122.417, 37.772), // Different turn
    LngLat::new_deg(-122.416, 37.775), // Same coffee shop
];

let similarity = measure.distance(&morning_route, &evening_route)?;
println!("Route similarity: {:.0}m", similarity);
```

**Important:** Fréchet distance is directional. If you reverse one path, you get a different (usually larger) distance.

### Hausdorff Distance

The Hausdorff distance finds the point in one set that's farthest from the other set, then measures that distance. It's like asking "what's the worst mismatch between these two shapes?"

**When to use:**
- Comparing building outlines or property boundaries
- Shape matching where direction doesn't matter
- Finding how well one path covers another
- Quality checking simplified paths

```rust
use rapidgeo_similarity::{LngLat, hausdorff::Hausdorff, SimilarityMeasure};

let measure = Hausdorff;

// Original detailed path
let detailed_path = vec![
    LngLat::new_deg(-122.0, 37.0),
    LngLat::new_deg(-122.05, 37.05),
    LngLat::new_deg(-122.1, 37.1),
    LngLat::new_deg(-122.15, 37.15),
    LngLat::new_deg(-122.2, 37.2),
];

// Simplified version
let simplified_path = vec![
    LngLat::new_deg(-122.0, 37.0),
    LngLat::new_deg(-122.1, 37.1),
    LngLat::new_deg(-122.2, 37.2),
];

let max_error = measure.distance(&detailed_path, &simplified_path)?;
println!("Max error from simplification: {:.0}m", max_error);
```

## Key Differences

| Aspect | Fréchet Distance | Hausdorff Distance |
|--------|------------------|-------------------|
| **Point order** | Matters | Doesn't matter |
| **Use case** | Trajectories, routes | Shapes, boundaries |
| **Direction** | Considers path direction | Order-independent |
| **What it measures** | Alignment similarity | Maximum mismatch |
| **Good for** | "Did they take the same route?" | "How different are these shapes?" |

## Error Handling

Both algorithms return `Result<f64, SimilarityError>`:

```rust
use rapidgeo_similarity::{SimilarityError, frechet::discrete_frechet_distance};

match discrete_frechet_distance(&path_a, &path_b) {
    Ok(distance) => println!("Distance: {:.2}m", distance),
    Err(SimilarityError::EmptyInput) => eprintln!("One or both paths are empty"),
    Err(SimilarityError::InvalidInput(msg)) => eprintln!("Invalid input: {}", msg),
}
```

## Batch Processing

Process multiple path pairs in parallel with the `batch` feature:

```rust
use rapidgeo_similarity::batch::{batch_frechet_distance, pairwise_frechet_matrix};

// Compare many path pairs at once
let path_pairs = vec![
    (route_1.clone(), route_2.clone()),
    (route_1.clone(), route_3.clone()),
    (route_2.clone(), route_3.clone()),
];

let distances = batch_frechet_distance(&path_pairs)?;
for (i, distance) in distances.iter().enumerate() {
    println!("Pair {}: {:.0}m", i + 1, distance);
}

// Create a full distance matrix
let all_paths = vec![route_1, route_2, route_3];
let matrix = pairwise_frechet_matrix(&all_paths)?;

// matrix[i][j] is the distance between path i and path j
```

## Advanced Usage

### Early Termination with Thresholds

Stop calculating if distance exceeds a threshold:

```rust
use rapidgeo_similarity::frechet::discrete_frechet_distance_with_threshold;

let threshold = 500.0; // 500 meters
match discrete_frechet_distance_with_threshold(&path_a, &path_b, threshold)? {
    d if d > threshold => println!("Paths are too different (>{:.0}m)", threshold),
    distance => println!("Paths are similar: {:.0}m", distance),
}
```

### Batch Threshold Filtering

Find which path pairs are similar enough:

```rust
use rapidgeo_similarity::batch::batch_frechet_distance_threshold;

let similar_pairs = batch_frechet_distance_threshold(&path_pairs, 200.0)?;
for (i, is_similar) in similar_pairs.iter().enumerate() {
    if *is_similar {
        println!("Pair {} is within 200m", i + 1);
    }
}
```

## Coordinate System

Uses longitude-first coordinates (lng, lat) and returns distances in meters via Haversine calculation. Works anywhere on Earth but assumes WGS84.

```rust
// Longitude first, latitude second
let point = LngLat::new_deg(-122.4194, 37.7749); // San Francisco
```

## Real-World Examples

### Finding Similar Delivery Routes

```rust
use rapidgeo_similarity::{frechet::discrete_frechet_distance, LngLat};

fn find_similar_routes(target_route: &[LngLat], candidate_routes: &[Vec<LngLat>]) -> Vec<(usize, f64)> {
    let mut similar = Vec::new();
    
    for (idx, route) in candidate_routes.iter().enumerate() {
        if let Ok(distance) = discrete_frechet_distance(target_route, route) {
            if distance < 1000.0 { // Within 1km
                similar.push((idx, distance));
            }
        }
    }
    
    similar.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    similar
}
```

### Validating GPS Track Quality

```rust
use rapidgeo_similarity::hausdorff::hausdorff_distance;

fn assess_gps_quality(raw_track: &[LngLat], smoothed_track: &[LngLat]) -> f64 {
    hausdorff_distance(raw_track, smoothed_track).unwrap_or(f64::INFINITY)
}

let raw_gps = load_raw_gps_data();
let smoothed = apply_kalman_filter(&raw_gps);

let max_error = assess_gps_quality(&raw_gps, &smoothed);
if max_error > 50.0 {
    println!("Warning: GPS smoothing introduced {:.0}m maximum error", max_error);
}
```

## Performance Notes

- Both algorithms use Haversine distance calculation (accurate for most use cases)
- Memory usage: O(n×m) for Fréchet, O(1) for Hausdorff
- CPU usage: O(n×m) for Fréchet, O(n×m) for Hausdorff
- Threshold versions can terminate early for better performance

## Limitations

- Assumes points represent a path in order (for Fréchet)
- Uses spherical Earth model (Haversine), not suitable for millimeter precision
- No support for 3D coordinates or time-aware distance
- Does not handle coordinate system transformations

## Related Crates

Part of the RapidGeo family:
- `rapidgeo-distance` - Point-to-point distance calculations
- `rapidgeo-polyline` - Google Polyline encoding/decoding  
- `rapidgeo-simplify` - Path simplification algorithms

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.