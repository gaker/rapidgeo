"""
rapidgeo: Fast geographic and planar distance calculations
"""

from ._rapidgeo import LngLat, __version__
from . import distance, simplify, polyline, similarity, formats

__all__ = ["LngLat", "distance", "simplify", "polyline", "similarity", "formats", "__version__"]
