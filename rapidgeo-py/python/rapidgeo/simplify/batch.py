"""
Batch simplification operations
"""

from .._rapidgeo import simplify as _simplify_module

# Re-export batch functions
simplify_multiple = _simplify_module.batch.simplify_multiple

__all__ = ["simplify_multiple"]
