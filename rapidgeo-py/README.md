# rapidgeo

[![PyPI](https://img.shields.io/pypi/v/rapidgeo.svg)](https://pypi.org/project/rapidgeo/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)

Fast geographic and planar distance calculations for Python.

## Installation

```bash
pip install rapidgeo          # Base package
pip install rapidgeo[numpy]   # With NumPy support
```

## Quick Start

```python
from rapidgeo.distance import LngLat
from rapidgeo.distance.geo import haversine, vincenty_distance
from rapidgeo.distance.euclid import euclid
from rapidgeo.distance.batch import pairwise_haversine

# Create coordinates (longitude, latitude)
sf = LngLat.new_deg(-122.4194, 37.7749)   # San Francisco
nyc = LngLat.new_deg(-74.0060, 40.7128)   # New York City

# Haversine: 0.5% accuracy for distances <1000km
distance = haversine(sf, nyc)
print(f"Distance: {distance / 1000:.1f} km")

# Vincenty: 1mm accuracy globally
precise = vincenty_distance(sf, nyc)
print(f"Precise: {precise / 1000:.3f} km")

# Euclidean: Fast but inaccurate for large distances
euclidean = euclid(sf, nyc)
print(f"Euclidean: {euclidean:.6f}�")

# Batch processing
path = [sf, nyc, LngLat.new_deg(-87.6298, 41.8781)]  # Add Chicago
distances = list(pairwise_haversine(path))
```

## Coordinate System

All coordinates use **longitude, latitude** ordering (lng, lat):

```python
# Correct
point = LngLat.new_deg(-122.4194, 37.7749)  # lng first, lat second

# Common mistake
# point = LngLat.new_deg(37.7749, -122.4194)  # lat, lng - WRONG
```

## Algorithm Selection

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.