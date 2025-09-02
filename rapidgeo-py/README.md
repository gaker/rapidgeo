# rapidgeo

[![PyPI](https://img.shields.io/pypi/v/rapidgeo.svg)](https://pypi.org/project/rapidgeo/)
[![Documentation](https://img.shields.io/badge/docs-rapidgeo-blue.svg)](https://rapidgeo.readthedocs.io/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)

Geographic calculations for Python: distance calculations, polyline encoding/decoding, line simplification, and curve similarity measures.

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

## What it does

**Distance Calculations:**
- Haversine: Spherical Earth approximation, good for most uses
- Vincenty: Ellipsoidal Earth model for higher precision  
- Euclidean: Flat plane calculations
- Batch operations for multiple points

**Polyline Encoding/Decoding:**
- Google Polyline Algorithm implementation
- Compress coordinate sequences to text strings
- Configurable precision levels

**Line Simplification:**
- Douglas-Peucker algorithm
- Reduce point count while preserving shape
- Configurable distance tolerance

**Curve Similarity:**
- Fréchet distance: Compare trajectories (point order matters)
- Hausdorff distance: Compare shapes (point order doesn't matter)  
- Useful for GPS track analysis and route comparison

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