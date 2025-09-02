# rapidgeo-distance

[![Crates.io](https://img.shields.io/crates/v/rapidgeo-distance.svg)](https://crates.io/crates/rapidgeo-distance)
[![Documentation](https://docs.rs/rapidgeo-distance/badge.svg)](https://docs.rs/rapidgeo-distance)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![CI](https://github.com/gaker/rapidgeo/workflows/CI/badge.svg)](https://github.com/gaker/rapidgeo/actions)
[![Coverage](https://img.shields.io/codecov/c/github/gaker/rapidgeo)](https://codecov.io/gh/gaker/rapidgeo)

Geographic and planar distance calculations.

All coordinates use **longitude, latitude** ordering (lng, lat).

## Installation

```toml
[dependencies]
rapidgeo-distance = "0.1"

# Or with optional features
rapidgeo-distance = { version = "0.1", features = ["batch", "vincenty"] }
```

## Quick Start

```rust
use rapidgeo_distance::{LngLat, geodesic, euclid};

let sf = LngLat::new_deg(-122.4194, 37.7749);   // San Francisco
let nyc = LngLat::new_deg(-74.0060, 40.7128);   // New York City

// Haversine: ±0.5% accuracy for distances <1000km
let distance = geodesic::haversine(sf, nyc);
println!("Distance: {:.1} km", distance / 1000.0);

// Vincenty: Sub-meter accuracy globally (requires "vincenty" feature)
let precise = geodesic::vincenty_distance_m(sf, nyc)?;
println!("Precise: {:.3} km", precise / 1000.0);

// Euclidean: Fast but inaccurate for large distances
let euclidean = euclid::distance_euclid(sf, nyc);
println!("Euclidean: {:.6}°", euclidean);
```

## What This Crate Does

This crate calculates distances between geographic coordinates using three approaches:

1. **Geodesic algorithms** - Account for Earth's shape (haversine, Vincenty)
2. **Euclidean distance** - Treats coordinates as flat plane points  
3. **Batch operations** - Process many points efficiently with optional parallelization

All geodesic calculations use the WGS84 ellipsoid.

## API Overview

### Core Types

```rust
// All functions work with LngLat coordinates
let point = LngLat::new_deg(longitude, latitude);
let (lng_rad, lat_rad) = point.to_radians();
```

### Coordinate Format Detection

The crate automatically detects and converts coordinate data from various formats (tuples, flat arrays, GeoJSON) to the internal `LngLat` representation:

```rust
use rapidgeo_distance::formats::{coords_to_lnglat_vec, CoordinateInput};

// Automatically detects lng,lat vs lat,lng ordering
let coords = vec![
    (37.7749, -122.4194),  // Latitude,longitude format (detected)
    (40.7128, -74.0060),   // Automatically corrected to lng,lat
];

let input = CoordinateInput::Tuples(coords);
let lnglat_coords = coords_to_lnglat_vec(&input);
```

See [Coordinate Format Documentation](docs/coordinate-formats.md) for detailed examples of supported formats.

### Geodesic Distances (Earth-Aware)

```rust
use rapidgeo_distance::geodesic::{haversine, vincenty_distance_m, VincentyError};

// Haversine: Fast, good accuracy for distances <1000km
let distance_m = haversine(point1, point2);

// Vincenty: Slower, very accurate, may fail for antipodal points
match vincenty_distance_m(point1, point2) {
    Ok(distance) => println!("{:.3} m", distance),
    Err(VincentyError::DidNotConverge) => {
        // Use haversine as fallback
        let fallback = haversine(point1, point2);
    }
    Err(VincentyError::Domain) => {
        // Invalid coordinates (NaN/infinite)
    }
}
```

### Euclidean Distances (Flat Plane)

```rust
use rapidgeo_distance::euclid::{distance_euclid, distance_squared, point_to_segment};

// Basic distance in degrees (not meters)
let dist_deg = distance_euclid(point1, point2);

// Squared distance (avoids sqrt for performance)
let dist_sq = distance_squared(point1, point2);

// Point to line segment distance
let segment = (point1, point2);
let distance = point_to_segment(test_point, segment);
```

### Point-to-Segment Distances

```rust
use rapidgeo_distance::geodesic::{point_to_segment_enu_m, great_circle_point_to_seg};

let segment = (start_point, end_point);

// ENU projection (good for small areas)
let distance = point_to_segment_enu_m(point, segment);

// Great circle method (accurate but slower)
let distance = great_circle_point_to_seg(point, segment);
```

### Batch Operations

```rust
use rapidgeo_distance::batch::{
    pairwise_haversine, path_length_haversine,
    pairwise_haversine_into, distances_to_point_into
};

let path = vec![point1, point2, point3];

// Process consecutive pairs
let distances: Vec<f64> = pairwise_haversine(&path).collect();

// Total path length
let total = path_length_haversine(&path);

// Write to pre-allocated buffer (no allocation)
let mut buffer = vec![0.0; path.len() - 1];
pairwise_haversine_into(&path, &mut buffer);
```

### Parallel Processing (requires "batch" feature)

```rust
#[cfg(feature = "batch")]
use rapidgeo_distance::batch::{
    pairwise_haversine_par, path_length_haversine_par,
    distances_to_point_par
};

let large_dataset = load_many_points();

// Parallel processing (beneficial for >1000 points)
let distances = pairwise_haversine_par(&large_dataset);
let total = path_length_haversine_par(&large_dataset);
```

## Algorithm Selection

| Algorithm | Speed | Accuracy | Best For |
|-----------|-------|----------|----------|
| [Haversine](https://en.wikipedia.org/wiki/Haversine_formula) | 46ns | ±0.5% | Distances <1000km |
| [Vincenty](https://en.wikipedia.org/wiki/Vincenty%27s_formulae) | 271ns | ±1mm | High precision, any distance |
| Euclidean | 1ns | Poor at scale | Small areas, relative comparisons |

### Accuracy Details

**Haversine**: Uses spherical approximation with ellipsoidal correction for the [WGS84 ellipsoid](https://en.wikipedia.org/wiki/World_Geodetic_System). Good tradeoff for accuracy vs speed.

**Vincenty**: Implements [Vincenty's inverse formula](https://en.wikipedia.org/wiki/Vincenty%27s_formulae) on WGS84 ellipsoid. May fail to converge for nearly antipodal points.

**Euclidean**: Simple [Pythagorean theorem](https://en.wikipedia.org/wiki/Pythagorean_theorem) in degree space. Ignores Earth curvature. Error increases with distance and latitude.

## Features

- **Default**: Haversine and Euclidean functions
- **`vincenty`**: Enables high-precision Vincenty calculations
- **`batch`**: Enables Rayon-based parallel processing

## Performance Notes

**Serial vs Parallel**: Parallel functions are faster for large datasets (>1000 points) but have overhead. Use serial for small datasets.

**Memory Allocation**: Functions ending in `_into` write to pre-allocated buffers, avoiding allocation overhead.

**Benchmarks**: On Intel i9-10900F:
- Euclidean: ~1ns per calculation  
- Haversine: ~46ns per calculation
- Vincenty: ~271ns per calculation
- Pre-allocated buffers: ~60% faster than allocating

## Coordinate System

All coordinates use **longitude, latitude** ordering:
- Longitude: -180.0 to +180.0° (West to East)
- Latitude: -90.0 to +90.0° (South to North)

```rust
let coord = LngLat::new_deg(lng, lat);  // Note: lng first
```

## Limitations

- Vincenty may fail for nearly antipodal points
- Euclidean accuracy degrades significantly with distance and latitude
- Parallel functions require the `batch` feature
- All geodesic calculations assume WGS84 ellipsoid
- Point-to-segment functions assume segments shorter than hemisphere

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or https://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or https://opensource.org/licenses/MIT)

at your option.