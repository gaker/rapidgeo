# rapidgeo

[![PyPI](https://img.shields.io/pypi/v/rapidgeo.svg)](https://pypi.org/project/rapidgeo/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)

Fast geographic algorithms for Python, including distance calculations, polyline encoding/decoding, line simplification, and curve similarity measures.

## Installation

```bash
pip install rapidgeo          # Base package
pip install rapidgeo[numpy]   # With NumPy support
```

## Quick Start

```python
from rapidgeo.distance import LngLat
from rapidgeo.distance.geo import haversine, vincenty_distance

# Create coordinates (longitude, latitude)
sf = LngLat.new_deg(-122.4194, 37.7749)   # San Francisco
nyc = LngLat.new_deg(-74.0060, 40.7128)   # New York City

# Calculate distance using Haversine formula
distance = haversine(sf, nyc)
print(f"Distance: {distance / 1000:.1f} km")  # ~4,130 km
```

## Features

**Distance Calculations:**
- Haversine: Fast, 0.5% accuracy for distances <1000km
- Vincenty: High precision, 1mm accuracy globally  
- Euclidean: Planar distance for projected coordinates
- Batch operations for processing large datasets

**Polyline Encoding/Decoding:**
- Google Polyline Algorithm implementation
- Compress GPS tracks for efficient storage/transmission
- Full precision support

**Line Simplification:**
- Douglas-Peucker algorithm
- Reduce GPS track complexity while preserving shape
- Configurable tolerance thresholds

**Curve Similarity:**
- Fréchet distance: Compare trajectories (considers point order)
- Hausdorff distance: Compare shapes (order-independent)  
- Early termination for threshold-based queries
- Ideal for GPS track analysis and map matching

## Coordinate System

All coordinates use **longitude, latitude** ordering (lng, lat):

```python
# Correct
point = LngLat.new_deg(-122.4194, 37.7749)  # lng first, lat second

# Common mistake
# point = LngLat.new_deg(37.7749, -122.4194)  # lat, lng - WRONG
```

## Usage Examples

**Compare GPS Tracks:**
```python
from rapidgeo.similarity.frechet import discrete_frechet

# Two similar walking routes
route_a = [LngLat.new_deg(-122.419, 37.775), LngLat.new_deg(-122.418, 37.776)]
route_b = [LngLat.new_deg(-122.419, 37.775), LngLat.new_deg(-122.417, 37.777)]

similarity = discrete_frechet(route_a, route_b)
print(f"Routes differ by {similarity:.1f} meters")
```

**Simplify GPS Tracks:**
```python
from rapidgeo.simplify import douglas_peucker

# Reduce GPS track complexity
detailed_track = [LngLat.new_deg(-122.4, 37.7), LngLat.new_deg(-122.39, 37.71), ...]
simplified = douglas_peucker(detailed_track, tolerance=10.0)  # 10m tolerance
print(f"Reduced from {len(detailed_track)} to {len(simplified)} points")
```

**Encode/Decode Polylines:**
```python  
from rapidgeo.polyline import encode, decode

# Compress GPS data for storage/transmission
points = [LngLat.new_deg(-122.4, 37.7), LngLat.new_deg(-122.3, 37.8)]
encoded = encode(points, precision=5)
decoded = decode(encoded, precision=5)
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.