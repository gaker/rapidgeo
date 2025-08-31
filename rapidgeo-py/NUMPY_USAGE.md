# NumPy Integration for RapidGeo

## Installation Options

### Basic Installation
```bash
pip install rapidgeo
```
Includes optimized distance calculations with GIL release and signal handling.

### With NumPy Support  
```bash
pip install rapidgeo[numpy]
# or during development:
maturin develop --features "batch,numpy"
```
Adds ultra-high-performance NumPy array functions.

## Usage Examples

### Standard Python Interface
```python
from rapidgeo.distance import LngLat
from rapidgeo.distance.batch import pairwise_haversine, path_length_haversine

# Works with regular Python lists
points = [LngLat(0.0, 0.0), LngLat(1.0, 0.0), LngLat(1.0, 1.0)]
distances = pairwise_haversine(points)  # ~10K points/sec, GIL-released
total = path_length_haversine(points)
```

### NumPy High-Performance Interface
```python
# Only available when installed with [numpy] extra
import numpy as np
from rapidgeo.distance.numpy import (
    pairwise_haversine_numpy,
    distances_to_point_numpy, 
    path_length_haversine_numpy
)

# Ultra-fast with NumPy arrays
lng = np.array([0.0, 1.0, 2.0], dtype=np.float64)
lat = np.array([0.0, 0.0, 0.0], dtype=np.float64)

distances = pairwise_haversine_numpy(lng, lat)      # 14M+ points/sec!
distances_to_target = distances_to_point_numpy(lng, lat, 0.5, 0.5)
total_length = path_length_haversine_numpy(lng, lat)
```

## Performance Comparison

| Method | Speed | Memory | Use Case |
|--------|-------|--------|----------|
| Standard Python | ~10K points/sec | Efficient | General use |
| NumPy Arrays | **14M+ points/sec** | Zero-copy | High-performance |

## Features

✅ **Memory Optimized**: Single allocation instead of 3x  
✅ **GIL Released**: True parallelism with `py.allow_threads()`  
✅ **Signal Handling**: Interruptible with Control-C  
✅ **Zero-Copy NumPy**: Direct array operations  
✅ **Optional Dependency**: NumPy only when needed

## Checking NumPy Availability

```python
import rapidgeo.distance as dist

if hasattr(dist, 'numpy'):
    print("NumPy functions available!")
    # Use high-performance NumPy functions
else:
    print("Using standard Python interface")
    # Use regular batch functions
```