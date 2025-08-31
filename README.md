# RapidGo

[![CI](https://github.com/gaker/rapidgo/workflows/CI/badge.svg)](https://github.com/gaker/rapidgo/actions?query=workflow%3ACI)
[![PyPI](https://img.shields.io/pypi/v/rapidgeo.svg)](https://pypi.org/project/rapidgeo/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)

Fast geographic and planar distance calculations for Rust and Python.

## Components

### 🦀 map-distance (Rust)

Core Rust library for geographic distance calculations.

- **Haversine**: ±0.5% accuracy for distances <1000km
- **Vincenty**: ±1mm accuracy globally  
- **Euclidean**: Fast planar distance calculations
- **Batch operations** with optional parallelization

[**📖 Documentation**](https://docs.rs/map-distance)

### 🐍 rapidgeo (Python)

Python bindings with NumPy integration.

```bash
pip install rapidgeo          # Base package
pip install rapidgeo[numpy]   # With NumPy support
```

```python
from rapidgeo.distance import LngLat
from rapidgeo.distance.geo import haversine

sf = LngLat.new_deg(-122.4194, 37.7749)
nyc = LngLat.new_deg(-74.0060, 40.7128)
distance = haversine(sf, nyc)
```

[**📖 Python Docs**](https://github.com/gaker/rapidgo/tree/main/rapidgeo-py)

## Performance

| Algorithm | Speed | Accuracy | Best For |
|-----------|-------|----------|----------|
| Haversine | 46ns | ±0.5% | Distances <1000km |
| Vincenty | 271ns | ±1mm | High precision |
| Euclidean | 1ns | Poor at scale | Relative comparisons |

## Coordinate System

All coordinates use **longitude, latitude** ordering (lng, lat).

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.