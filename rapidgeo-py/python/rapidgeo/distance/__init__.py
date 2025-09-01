"""
Distance calculation submodule
"""

from .._rapidgeo import distance as _distance_module

# Re-export everything from the compiled distance module
LngLat = _distance_module.LngLat

# Import submodules
from . import geo, euclid, batch

# Conditionally import numpy if available
_all_modules = ["LngLat", "geo", "euclid", "batch"]
try:
    numpy = _distance_module.numpy
    _all_modules.append("numpy")
except AttributeError:
    # numpy feature not enabled during compilation
    pass

__all__ = _all_modules
