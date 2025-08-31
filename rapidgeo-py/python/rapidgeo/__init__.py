"""
rapidgeo: Fast geographic and planar distance calculations
"""

from ._rapidgeo import LngLat, __version__
from . import distance, simplify

__all__ = ["LngLat", "distance", "simplify", "__version__"]