"""
Batch distance and bearing functions
"""

from .._rapidgeo import distance

# Re-export functions from the compiled module
pairwise_haversine = distance.batch.pairwise_haversine
path_length_haversine = distance.batch.path_length_haversine
path_length_haversine_batch = distance.batch.path_length_haversine_batch
pairwise_bearings = distance.batch.pairwise_bearings

__all__ = ["pairwise_haversine", "path_length_haversine", "path_length_haversine_batch", "pairwise_bearings"]
