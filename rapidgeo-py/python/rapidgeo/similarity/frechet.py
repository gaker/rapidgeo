"""
Discrete Fréchet distance implementation.

The Fréchet distance measures similarity between polygonal curves while considering
the ordering of points along each curve. Often described as the "dog walking" distance.
"""

from .. import _rapidgeo

discrete_frechet = _rapidgeo.similarity.frechet.discrete_frechet
discrete_frechet_with_threshold = _rapidgeo.similarity.frechet.discrete_frechet_with_threshold

__all__ = [
    "discrete_frechet",
    "discrete_frechet_with_threshold",
]
