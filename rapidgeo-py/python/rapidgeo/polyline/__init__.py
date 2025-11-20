"""
Google Polyline Algorithm encoding/decoding with simplification support
"""

from .._rapidgeo import polyline as _polyline

encode = _polyline.py_encode
decode = _polyline.py_decode
encode_simplified = _polyline.py_encode_simplified
simplify_polyline = _polyline.py_simplify_polyline
encode_batch = _polyline.py_encode_batch
decode_batch = _polyline.py_decode_batch
encode_simplified_batch = _polyline.py_encode_simplified_batch
encode_column = _polyline.py_encode_column

__all__ = [
    "encode",
    "decode",
    "encode_simplified",
    "simplify_polyline",
    "encode_batch",
    "decode_batch",
    "encode_simplified_batch",
    "encode_column",
]
