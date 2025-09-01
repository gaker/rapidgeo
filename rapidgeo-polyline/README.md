# rapidgeo-polyline

Fast Google Polyline Algorithm encoding/decoding for geographic coordinates.

[![Crates.io](https://img.shields.io/crates/v/rapidgeo-polyline)](https://crates.io/crates/rapidgeo-polyline)
[![Documentation](https://docs.rs/rapidgeo-polyline/badge.svg)](https://docs.rs/rapidgeo-polyline)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/gaker/rapidgeo#license)

## Installation

```toml
[dependencies]
rapidgeo-polyline = "0.1"
```

For batch processing with parallel support:

```toml
[dependencies]
rapidgeo-polyline = { version = "0.1", features = ["batch"] }
```

## Quick Start

```rust
use rapidgeo_polyline::{encode, decode};
use rapidgeo_distance::LngLat;

let coords = vec![
    LngLat::new_deg(-120.2, 38.5),
    LngLat::new_deg(-120.95, 40.7),
];

let encoded = encode(&coords, 5).unwrap();
let decoded = decode(&encoded, 5).unwrap();
```

## What This Does

Implements [Google's Polyline Algorithm](https://developers.google.com/maps/documentation/utilities/polylinealgorithm) for encoding sequences of geographic coordinates into compact ASCII strings. Commonly used in mapping applications, route optimization, and GPS data storage.

**Coordinate Order**: All functions use **longitude, latitude** ordering (x, y).

## Examples

### Basic Encoding/Decoding

```rust
use rapidgeo_polyline::{encode, decode};
use rapidgeo_distance::LngLat;

// Google's test vector
let coords = vec![
    LngLat::new_deg(-120.2, 38.5),
    LngLat::new_deg(-120.95, 40.7),
    LngLat::new_deg(-126.453, 43.252),
];

let polyline = encode(&coords, 5).unwrap();
assert_eq!(polyline, "_p~iF~ps|U_ulLnnqC_mqNvxq`@");

let decoded = decode(&polyline, 5).unwrap();
assert_eq!(decoded.len(), 3);
```

### Route Simplification

Uses the [Douglas-Peucker algorithm](https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm) to reduce coordinate density while preserving route shape:

```rust
use rapidgeo_polyline::{encode_simplified, simplify_polyline};
use rapidgeo_simplify::SimplifyMethod;
use rapidgeo_distance::LngLat;

let detailed_route = vec![
    LngLat::new_deg(-122.0, 37.0),
    LngLat::new_deg(-122.01, 37.01),  // Close intermediate point
    LngLat::new_deg(-122.02, 37.02),  // Close intermediate point
    LngLat::new_deg(-122.1, 37.1),
];

// Encode with 1km simplification tolerance
let simplified = encode_simplified(
    &detailed_route,
    1000.0, // meters
    SimplifyMethod::GreatCircleMeters,
    5
).unwrap();
```

### Batch Processing

Process multiple polylines in parallel (requires `batch` feature):

```rust
#[cfg(feature = "batch")]
use rapidgeo_polyline::batch::{encode_batch, decode_batch};

#[cfg(feature = "batch")]
{
    let routes = vec![
        vec![LngLat::new_deg(-120.2, 38.5), LngLat::new_deg(-120.95, 40.7)],
        vec![LngLat::new_deg(-126.453, 43.252), LngLat::new_deg(-122.4194, 37.7749)],
    ];
    
    let encoded = encode_batch(&routes, 5).unwrap();
    let decoded = decode_batch(&encoded, 5).unwrap();
}
```

## API

### Core Functions

- `encode(coords, precision)` - Encode coordinates to polyline string
- `decode(polyline, precision)` - Decode polyline string to coordinates  
- `encode_simplified(coords, tolerance, method, precision)` - Encode with simplification
- `simplify_polyline(polyline, tolerance, method, precision)` - Simplify existing polyline

### Batch Functions (feature = "batch")

- `encode_batch()` - Parallel encoding of multiple coordinate sequences
- `decode_batch()` - Parallel decoding of multiple polylines
- `encode_simplified_batch()` - Parallel encoding with simplification

### Precision

- **5** (default): ~1 meter accuracy, standard for most mapping applications
- **6**: ~10 centimeter accuracy, used for high-precision applications

Valid range: 1-11

## Performance

- Encoding: ~2-4 million coordinates/second
- Decoding: ~3-5 million coordinates/second
- Memory: O(n) where n = number of coordinates
- Batch processing automatically uses parallel processing for >100 polylines

## Limitations

- Coordinates must be within valid WGS84 bounds (±180° longitude, ±90° latitude)
- No support for 3D coordinates (elevation)
- Precision limited to 11 decimal places
- Does not handle coordinate reference system transformations

## Algorithm References

- [Google Polyline Algorithm Specification](https://developers.google.com/maps/documentation/utilities/polylinealgorithm)
- [Douglas-Peucker Line Simplification](https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm)

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.