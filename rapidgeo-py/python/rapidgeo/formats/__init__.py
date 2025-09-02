"""
Coordinate format detection and conversion utilities
"""

from .._rapidgeo import formats

# Export the main function
coords_to_lnglat = formats.coords_to_lnglat

__all__ = ["coords_to_lnglat"]