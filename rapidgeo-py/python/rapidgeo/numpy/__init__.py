"""
NumPy-optimized batch operations for geographic calculations
"""

try:
    from .._rapidgeo import numpy as _numpy_module

    # Re-export numpy functions
    pairwise_haversine = _numpy_module.pairwise_haversine
    distances_to_point = _numpy_module.distances_to_point
    path_length_haversine = _numpy_module.path_length_haversine

    __all__ = [
        "pairwise_haversine",
        "distances_to_point",
        "path_length_haversine",
    ]
except (ImportError, AttributeError):
    # NumPy not available
    __all__ = []
