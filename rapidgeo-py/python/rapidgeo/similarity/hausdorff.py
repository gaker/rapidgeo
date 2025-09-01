"""
Hausdorff distance implementation.

The Hausdorff distance measures the maximum distance between any point in one set
and the closest point in another set. Order-independent similarity measure.
"""

from .. import _rapidgeo

hausdorff = _rapidgeo.similarity.hausdorff.hausdorff
hausdorff_with_threshold = _rapidgeo.similarity.hausdorff.hausdorff_with_threshold

__all__ = [
    "hausdorff",
    "hausdorff_with_threshold",
]
