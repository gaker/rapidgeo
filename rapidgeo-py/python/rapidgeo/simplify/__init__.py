"""
Polyline simplification submodule
"""

from .._rapidgeo import simplify as _simplify_module

# Re-export the main function
douglas_peucker = _simplify_module.douglas_peucker

# Import submodules
from . import batch

__all__ = ["douglas_peucker", "batch"]
