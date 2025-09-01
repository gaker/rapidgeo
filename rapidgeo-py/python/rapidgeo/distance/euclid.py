"""
Euclidean distance functions
"""

from .._rapidgeo import distance

# Re-export functions from the compiled module
euclid = distance.euclid.euclid
squared = distance.euclid.squared
point_to_segment = distance.euclid.point_to_segment
point_to_segment_squared = distance.euclid.point_to_segment_squared

__all__ = ["euclid", "squared", "point_to_segment", "point_to_segment_squared"]
