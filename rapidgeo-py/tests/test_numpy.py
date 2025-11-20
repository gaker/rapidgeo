#!/usr/bin/env python3

import pytest

numpy = pytest.importorskip("numpy")

from rapidgeo.numpy import pairwise_haversine, distances_to_point, path_length_haversine


def test_numpy_pairwise_haversine():
    lng = numpy.array([-122.4, -122.3, -122.2])
    lat = numpy.array([37.7, 37.8, 37.9])

    distances = pairwise_haversine(lng, lat)

    assert isinstance(distances, numpy.ndarray)
    assert len(distances) == 2
    assert all(d > 0 for d in distances)


def test_numpy_distances_to_point():
    lng = numpy.array([-122.4, -122.3, -122.2])
    lat = numpy.array([37.7, 37.8, 37.9])

    target_lng = -122.0
    target_lat = 37.5

    distances = distances_to_point(lng, lat, target_lng, target_lat)

    assert isinstance(distances, numpy.ndarray)
    assert len(distances) == 3
    assert all(d > 0 for d in distances)


def test_numpy_path_length_haversine():
    lng = numpy.array([-122.4, -122.3, -122.2])
    lat = numpy.array([37.7, 37.8, 37.9])

    total_distance = path_length_haversine(lng, lat)

    assert isinstance(total_distance, float)
    assert total_distance > 0


def test_numpy_empty_arrays():
    lng = numpy.array([])
    lat = numpy.array([])

    distances = pairwise_haversine(lng, lat)
    assert len(distances) == 0

    total = path_length_haversine(lng, lat)
    assert total == 0.0


def test_numpy_single_point():
    lng = numpy.array([-122.4])
    lat = numpy.array([37.7])

    distances = pairwise_haversine(lng, lat)
    assert len(distances) == 0

    total = path_length_haversine(lng, lat)
    assert total == 0.0


def test_numpy_mismatched_lengths():
    lng = numpy.array([-122.4, -122.3])
    lat = numpy.array([37.7])

    with pytest.raises(ValueError, match="must have same length"):
        pairwise_haversine(lng, lat)

    with pytest.raises(ValueError, match="must have same length"):
        path_length_haversine(lng, lat)


def test_numpy_large_batch():
    n = 1000
    lng = numpy.linspace(-122.0, -121.0, n)
    lat = numpy.linspace(37.0, 38.0, n)

    distances = pairwise_haversine(lng, lat)
    assert len(distances) == n - 1
    assert all(d > 0 for d in distances)

    total = path_length_haversine(lng, lat)
    assert total > 0
    assert abs(total - sum(distances)) < 0.01


def test_numpy_distances_to_point_batch():
    n = 500
    lng = numpy.linspace(-122.0, -121.0, n)
    lat = numpy.linspace(37.0, 38.0, n)

    target_lng = -121.5
    target_lat = 37.5

    distances = distances_to_point(lng, lat, target_lng, target_lat)

    assert len(distances) == n
    assert all(d >= 0 for d in distances)
    min_idx = numpy.argmin(distances)
    assert min_idx > 0 and min_idx < n - 1


def test_numpy_real_world_gps_track():
    # GPS track moving roughly northeast through San Francisco
    # Points are 0.001 degrees apart in both directions
    lng = numpy.array([
        -122.4194, -122.4184, -122.4174, -122.4164,
        -122.4154, -122.4144, -122.4134, -122.4124
    ])
    lat = numpy.array([
        37.7749, 37.7759, 37.7769, 37.7779,
        37.7789, 37.7799, 37.7809, 37.7819
    ])

    distances = pairwise_haversine(lng, lat)
    total = path_length_haversine(lng, lat)

    assert len(distances) == 7
    # Each segment is roughly 141 meters (0.001 deg ~ 111m lat + 85m lng at this latitude)
    for d in distances:
        assert 130 < d < 155

    assert abs(total - sum(distances)) < 0.01
