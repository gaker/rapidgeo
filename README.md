# rapidgeo

[![CI](https://github.com/gaker/rapidgeo/workflows/CI/badge.svg)](https://github.com/gaker/rapidgeo/actions)
[![Coverage](https://codecov.io/gh/gaker/rapidgeo/branch/main/graph/badge.svg)](https://codecov.io/gh/gaker/rapidgeo)
[![PyPI](https://img.shields.io/pypi/v/rapidgeo.svg)](https://pypi.org/project/rapidgeo/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-APACHE)

Geographic distance calculations, polyline encoding/decoding, and coordinate simplification libraries for Rust and Python.

## Workspace Structure

This repository contains four crates:

- **rapidgeo-distance** - Geographic and planar distance calculations
- **rapidgeo-polyline** - Google Polyline Algorithm encoding/decoding
- **rapidgeo-simplify** - Douglas-Peucker polyline simplification
- **rapidgeo-py** - Python bindings for all three libraries

## Installation

### Rust Crates

```toml
[dependencies]
rapidgeo-distance = "0.1"
rapidgeo-polyline = "0.1"
rapidgeo-simplify = "0.1"
rapidgeo-similarity = "0.1"
```

### Python Package

```bash
pip install rapidgeo
pip install rapidgeo[numpy]  # With NumPy array support
```

## Usage

### Distance Calculations

```rust
use rapidgeo_distance::{LngLat, haversine, vincenty};

let point1 = LngLat::degrees(-122.4194, 37.7749);  // San Francisco
let point2 = LngLat::degrees(-74.0060, 40.7128);   // New York City

let distance_m = haversine(&point1, &point2);
let precise_distance_m = vincenty(&point1, &point2);
```

```python
from rapidgeo import LngLat
from rapidgeo.distance.geo import haversine

sf = LngLat(-122.4194, 37.7749)
nyc = LngLat(-74.0060, 40.7128)
distance_meters = haversine(sf, nyc)
```

### Polyline Encoding/Decoding

```rust
use rapidgeo_polyline::{encode_coordinates, decode_polyline};

let coords = vec![
    (-122.4194, 37.7749),
    (-122.4094, 37.7849),
];
let polyline = encode_coordinates(&coords, 5);
let decoded = decode_polyline(&polyline, 5);
```

### Coordinate Simplification

```rust
use rapidgeo_simplify::{simplify_coords_dp, DistanceType};

let coords = vec![
    (-122.4194, 37.7749),
    (-122.4144, 37.7799),
    (-122.4094, 37.7849),
];
let simplified = simplify_coords_dp(&coords, 0.001, DistanceType::Haversine);
```

## Important Notes

- All coordinates use longitude, latitude ordering (lng, lat)
- Python package requires Python 3.8+
- Optional features available: `batch` (parallel processing), `numpy` (Python array support)

## Documentation

### Rust Crates
- [rapidgeo-distance](https://docs.rs/rapidgeo-distance) - Distance calculations
- [rapidgeo-polyline](https://docs.rs/rapidgeo-polyline) - Polyline encoding/decoding  
- [rapidgeo-simplify](https://docs.rs/rapidgeo-simplify) - Coordinate simplification

### Python Package
- [Python API Documentation](https://rapidgeo.readthedocs.io/) - Complete Python documentation on Read the Docs

## License

Licensed under either [Apache License 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.