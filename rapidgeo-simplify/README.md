# rapidgeo-simplify

[![Crates.io](https://img.shields.io/crates/v/rapidgeo-simplify)](https://crates.io/crates/rapidgeo-simplify)
[![Documentation](https://docs.rs/rapidgeo-simplify/badge.svg)](https://docs.rs/rapidgeo-simplify)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/gaker/rapidgeo)

Polyline simplification using the [Douglas-Peucker algorithm](https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm) with pluggable distance backends.

## Installation

```toml
[dependencies]
rapidgeo-simplify = "0.1"

# For parallel processing support
rapidgeo-simplify = { version = "0.1", features = ["batch"] }
```

## Quick Start

```rust
use rapidgeo_distance::LngLat;
use rapidgeo_simplify::{simplify_dp_into, SimplifyMethod};

let points = vec![
    LngLat::new_deg(-122.4194, 37.7749), // San Francisco
    LngLat::new_deg(-122.0, 37.5),       // Detour
    LngLat::new_deg(-121.5, 37.0),       // Another point
    LngLat::new_deg(-118.2437, 34.0522), // Los Angeles
];

let mut simplified = Vec::new();
let count = simplify_dp_into(
    &points,
    50000.0, // 50km tolerance
    SimplifyMethod::GreatCircleMeters,
    &mut simplified,
);

println!("Simplified from {} to {} points", points.len(), count);
```

## Why This Exists

The [Douglas-Peucker algorithm](https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm) simplifies polylines by removing points that don't significantly change the shape. Points are kept only if they deviate more than the specified tolerance from the simplified line. This implementation provides multiple distance calculation backends:

- **GreatCircleMeters**: Spherical distance calculations for global accuracy
- **PlanarMeters**: ENU projection for regional work with better performance
- **EuclidRaw**: Raw coordinate differences for non-geographic data

## Examples

### Basic Simplification

```rust
use rapidgeo_distance::LngLat;
use rapidgeo_simplify::{simplify_dp_mask, SimplifyMethod};

let points = vec![
    LngLat::new_deg(-122.0, 37.0),
    LngLat::new_deg(-121.5, 37.5),
    LngLat::new_deg(-121.0, 37.0),
];

let mut mask = Vec::new();
simplify_dp_mask(
    &points,
    1000.0, // 1km tolerance
    SimplifyMethod::GreatCircleMeters,
    &mut mask,
);

// mask[i] is true if points[i] should be kept
for (i, &keep) in mask.iter().enumerate() {
    if keep {
        println!("Keep point {}: {:?}", i, points[i]);
    }
}
```

### Batch Processing

```rust
use rapidgeo_simplify::batch::simplify_batch;

let polylines = vec![
    vec![/* first polyline points */],
    vec![/* second polyline points */],
];

let simplified = simplify_batch(
    &polylines,
    100.0,
    SimplifyMethod::GreatCircleMeters,
);
```

### Parallel Processing (requires `batch` feature)

```rust
use rapidgeo_simplify::batch::simplify_batch_par;

let simplified = simplify_batch_par(
    &polylines,
    100.0,
    SimplifyMethod::GreatCircleMeters,
);
```

## Distance Methods

### GreatCircleMeters
Uses spherical distance calculations. Best for:
- Global datasets
- High accuracy requirements
- Cross-country or intercontinental paths

### PlanarMeters
Projects to East-North-Up coordinates around the polyline's midpoint. Best for:
- Regional datasets (city/state level)
- Better performance than great circle
- Reasonable accuracy for smaller areas

### EuclidRaw
Direct coordinate subtraction. Best for:
- Non-geographic coordinate systems
- Screen coordinates
- Already-projected data

## API

### Core Functions

- `simplify_dp_into(points, tolerance, method, output)` - Simplify into output vector
- `simplify_dp_mask(points, tolerance, method, mask)` - Generate keep/drop mask

### Batch Functions (feature = "batch")

- `simplify_batch(polylines, tolerance, method)` - Process multiple polylines
- `simplify_batch_par(polylines, tolerance, method)` - Parallel batch processing
- `simplify_dp_into_par(points, tolerance, method, output)` - Parallel single polyline

## Algorithm

Implements the [Douglas-Peucker algorithm](https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm):

1. Draw line between first and last points
2. Find point with maximum perpendicular distance to this line
3. If distance > tolerance, keep the point and recurse on both segments
4. Otherwise, drop all intermediate points

The algorithm guarantees that:
- First and last points are always preserved
- No point deviates more than tolerance from the simplified line
- Shape characteristics are maintained

## Performance

- Single-threaded: ~1M points/second for typical GPS traces
- Memory: O(n) for input, O(log n) stack depth
- Parallel processing available for large datasets with `batch` feature

## Limitations

- Requires at least 2 points (endpoints always preserved)
- GreatCircleMeters doesn't handle antimeridian crossing optimally
- PlanarMeters accuracy decreases for very large regions

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/gaker/rapidgeo/blob/main/LICENSE-APACHE))
- MIT license ([LICENSE-MIT](https://github.com/gaker/rapidgeo/blob/main/LICENSE-MIT))

at your option.