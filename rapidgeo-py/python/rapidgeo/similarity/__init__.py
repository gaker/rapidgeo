"""
Curve similarity measures for geographic polylines.

This module provides algorithms for measuring the similarity between two polygonal curves:

- Fréchet distance: Considers point ordering along curves, ideal for trajectories
- Hausdorff distance: Maximum distance between point sets, order-independent

All functions work with sequences of LngLat coordinates and return distances in meters.
"""

from .. import _rapidgeo

# Re-export submodules
from . import frechet, hausdorff

__all__ = ["frechet", "hausdorff"]
