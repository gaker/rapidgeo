"""
rapidgeo: Fast geographic and planar distance calculations
"""

from ._rapidgeo import LngLat, __version__
from . import distance

__all__ = ["LngLat", "distance", "__version__"]