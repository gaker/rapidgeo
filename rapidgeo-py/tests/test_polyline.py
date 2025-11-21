#!/usr/bin/env python3

import pytest

from rapidgeo import LngLat
from rapidgeo.polyline import (
    encode,
    decode,
    encode_simplified,
    simplify_polyline,
    encode_batch,
    decode_batch,
    encode_simplified_batch,
)


@pytest.fixture
def sample_coords():
    return [
        LngLat(-120.2, 38.5),
        LngLat(-120.95, 40.7),
        LngLat(-126.453, 43.252),
    ]


@pytest.fixture
def sample_coords_tuples():
    return [
        (-120.2, 38.5),
        (-120.95, 40.7),
        (-126.453, 43.252),
    ]


@pytest.fixture
def sample_coords_lists():
    return [
        [-120.2, 38.5],
        [-120.95, 40.7],
        [-126.453, 43.252],
    ]


def test_basic_encode_decode(sample_coords):
    encoded = encode(sample_coords, 5)
    assert isinstance(encoded, str)
    assert len(encoded) > 0

    decoded = decode(encoded, 5)
    assert len(decoded) == len(sample_coords)

    for i, coord in enumerate(decoded):
        assert abs(coord.lng - sample_coords[i].lng) < 0.00001
        assert abs(coord.lat - sample_coords[i].lat) < 0.00001


def test_encode_simplified(sample_coords):
    encoded = encode_simplified(
        sample_coords, tolerance_m=1000.0, method="great_circle", precision=5
    )
    assert isinstance(encoded, str)
    assert len(encoded) > 0


def test_simplify_polyline(sample_coords):
    encoded = encode(sample_coords, 5)
    simplified = simplify_polyline(
        encoded, tolerance_m=1000.0, method="great_circle", precision=5
    )
    assert isinstance(simplified, str)


def test_encode_batch_lnglat_objects(sample_coords):
    coord_batches = [sample_coords, sample_coords[:2]]
    encoded = encode_batch(coord_batches, 5)

    assert len(encoded) == 2
    assert all(isinstance(p, str) for p in encoded)


def test_encode_batch_tuples(sample_coords_tuples):
    coord_batches = [sample_coords_tuples, sample_coords_tuples[:2]]
    encoded = encode_batch(coord_batches, 5)

    assert len(encoded) == 2
    assert all(isinstance(p, str) for p in encoded)


def test_encode_batch_lists(sample_coords_lists):
    coord_batches = [sample_coords_lists, sample_coords_lists[:2]]
    encoded = encode_batch(coord_batches, 5)

    assert len(encoded) == 2
    assert all(isinstance(p, str) for p in encoded)


def test_encode_batch_different_batches_same_result(
    sample_coords, sample_coords_tuples, sample_coords_lists
):
    # Each batch format should produce the same encoding
    encoded_lnglat = encode_batch([sample_coords], 5)
    encoded_tuples = encode_batch([sample_coords_tuples], 5)
    encoded_lists = encode_batch([sample_coords_lists], 5)

    assert len(encoded_lnglat) == 1
    assert len(encoded_tuples) == 1
    assert len(encoded_lists) == 1
    assert encoded_lnglat[0] == encoded_tuples[0] == encoded_lists[0]


def test_encode_simplified_batch_lnglat_objects(sample_coords):
    coord_batches = [sample_coords, sample_coords[:2]]
    encoded = encode_simplified_batch(
        coord_batches, tolerance_m=5.0, method="great_circle", precision=5
    )

    assert len(encoded) == 2
    assert all(isinstance(p, str) for p in encoded)


def test_encode_simplified_batch_tuples(sample_coords_tuples):
    coord_batches = [sample_coords_tuples, sample_coords_tuples[:2]]
    encoded = encode_simplified_batch(
        coord_batches, tolerance_m=5.0, method="great_circle", precision=5
    )

    assert len(encoded) == 2
    assert all(isinstance(p, str) for p in encoded)


def test_encode_simplified_batch_lists(sample_coords_lists):
    coord_batches = [sample_coords_lists, sample_coords_lists[:2]]
    encoded = encode_simplified_batch(
        coord_batches, tolerance_m=5.0, method="great_circle", precision=5
    )

    assert len(encoded) == 2
    assert all(isinstance(p, str) for p in encoded)


def test_encode_simplified_batch_tuple_coordinates():
    # Individual coordinates can be tuples, but the batch container must be a list
    coords_with_tuples = [
        (-120.2, 38.5),
        (-120.95, 40.7),
        (-126.453, 43.252),
    ]
    coord_batches = [coords_with_tuples, coords_with_tuples[:2]]
    encoded = encode_simplified_batch(
        coord_batches, tolerance_m=5.0, method="great_circle", precision=5
    )

    assert len(encoded) == 2
    assert all(isinstance(p, str) for p in encoded)


@pytest.mark.skipif(
    not pytest.importorskip("numpy", reason="numpy not installed"),
    reason="numpy not available",
)
def test_encode_simplified_batch_numpy_array():
    import numpy as np

    coords_np = np.array(
        [
            [-120.2, 38.5],
            [-120.95, 40.7],
            [-126.453, 43.252],
        ]
    )

    coord_batches = [coords_np, coords_np[:2]]
    encoded = encode_simplified_batch(
        coord_batches, tolerance_m=5.0, method="great_circle", precision=5
    )

    assert len(encoded) == 2
    assert all(isinstance(p, str) for p in encoded)


@pytest.mark.skipif(
    not pytest.importorskip("numpy", reason="numpy not installed"),
    reason="numpy not available",
)
def test_encode_batch_numpy_array():
    import numpy as np

    coords_np = np.array(
        [
            [-120.2, 38.5],
            [-120.95, 40.7],
            [-126.453, 43.252],
        ]
    )

    coord_batches = [coords_np, coords_np[:2]]
    encoded = encode_batch(coord_batches, 5)

    assert len(encoded) == 2
    assert all(isinstance(p, str) for p in encoded)


def test_decode_batch(sample_coords):
    coord_batches = [sample_coords, sample_coords[:2]]
    encoded = encode_batch(coord_batches, 5)
    decoded = decode_batch(encoded, 5)

    assert len(decoded) == 2
    assert len(decoded[0]) == 3
    assert len(decoded[1]) == 2


def test_encode_simplified_batch_methods(sample_coords):
    coord_batches = [sample_coords]

    for method in ["great_circle", "planar", "euclidean"]:
        encoded = encode_simplified_batch(
            coord_batches, tolerance_m=5.0, method=method, precision=5
        )
        assert len(encoded) == 1
        assert isinstance(encoded[0], str)


def test_large_batch(sample_coords):
    coord_batches = [sample_coords for _ in range(100)]
    encoded = encode_simplified_batch(coord_batches, tolerance_m=5.0, precision=5)

    assert len(encoded) == 100
    assert all(isinstance(p, str) for p in encoded)
