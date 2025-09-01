#!/usr/bin/env python3

import rapidgeo
from rapidgeo.distance import LngLat
from rapidgeo.distance.geo import haversine, vincenty_distance
from rapidgeo.distance.euclid import (
    euclid,
    squared,
    point_to_segment,
    point_to_segment_squared,
)
from rapidgeo.distance.batch import pairwise_haversine, path_length_haversine
from rapidgeo.simplify import douglas_peucker
from rapidgeo.simplify.batch import simplify_multiple


def test_basic_functionality():
    print(f"rapidgeo version: {rapidgeo.__version__}")

    sf = LngLat(-122.4194, 37.7749)
    nyc = LngLat(-74.0060, 40.7128)

    print(f"San Francisco: {sf}")
    print(f"New York City: {nyc}")

    # Test haversine distance
    distance_haversine = haversine(sf, nyc)
    print(f"Haversine distance SF->NYC: {distance_haversine:.2f} meters")

    # Test vincenty distance
    try:
        distance_vincenty = vincenty_distance(sf, nyc)
        print(f"Vincenty distance SF->NYC: {distance_vincenty:.2f} meters")
    except ValueError as e:
        print(f"Vincenty failed: {e}")

    # Test euclidean distance
    distance_euclidean = euclid(sf, nyc)
    print(f"Euclidean distance SF->NYC: {distance_euclidean:.6f} degrees")

    # Test squared distance
    distance_squared = squared(sf, nyc)
    print(f"Squared distance SF->NYC: {distance_squared:.6f} degrees²")

    # Test point to segment
    p1 = LngLat(-122.0, 37.0)
    p2 = LngLat(-121.0, 37.0)
    point = LngLat(-121.5, 37.01)

    segment_distance = point_to_segment(point, p1, p2)
    print(f"Point to segment distance: {segment_distance:.6f} degrees")

    segment_distance_sq = point_to_segment_squared(point, p1, p2)
    print(f"Point to segment squared distance: {segment_distance_sq:.6f} degrees²")

    # Test batch operations
    points = [LngLat(0.0, 0.0), LngLat(1.0, 0.0), LngLat(1.0, 1.0), LngLat(0.0, 1.0)]

    # Test pairwise distances
    pairwise_distances = pairwise_haversine(points)
    print(f"Pairwise distances: {[f'{d:.2f}' for d in pairwise_distances]}")

    # Test path length
    total_length = path_length_haversine(points)
    print(f"Total path length: {total_length:.2f} meters")


def test_simplify_functionality():
    print("\n=== Testing Simplify Functionality ===")

    # Create a polyline with points that should be simplified
    points = [
        LngLat(-122.0, 37.0),
        LngLat(-121.99, 37.001),  # Close to first point
        LngLat(-121.98, 37.002),  # Close intermediate point
        LngLat(-121.5, 37.5),  # Significant deviation
        LngLat(-121.0, 37.0),
    ]

    print(f"Original polyline has {len(points)} points")

    # Test douglas_peucker with different methods
    for method in ["great_circle", "planar", "euclidean"]:
        simplified = douglas_peucker(points, tolerance_m=1000.0, method=method)
        print(f"Simplified with {method}: {len(simplified)} points")

        # Test mask return
        mask = douglas_peucker(
            points, tolerance_m=1000.0, method=method, return_mask=True
        )
        print(f"Mask for {method}: {mask}")
        assert len(mask) == len(points), "Mask length should match input length"

    # Test batch simplification
    polylines = [
        [LngLat(-122.0, 37.0), LngLat(-121.9, 37.1), LngLat(-121.0, 37.0)],
        [LngLat(-74.0, 40.0), LngLat(-73.9, 40.1), LngLat(-73.0, 40.0)],
    ]

    simplified_batch = simplify_multiple(polylines, tolerance_m=5000.0)
    print(f"Batch simplified: {len(simplified_batch)} polylines")

    batch_masks = simplify_multiple(polylines, tolerance_m=5000.0, return_masks=True)
    print(f"Batch masks: {batch_masks}")


def test_distance_smoke_test():
    print("\n=== Distance Smoke Test ===")

    # Quick smoke test to ensure all distance functions work
    try:
        sf = LngLat(-122.4194, 37.7749)
        nyc = LngLat(-74.0060, 40.7128)

        # Test geodesic distances
        haversine_dist = haversine(sf, nyc)
        vincenty_dist = vincenty_distance(sf, nyc)
        assert haversine_dist > 4_000_000  # ~4130 km
        assert vincenty_dist > 4_000_000
        print("Geodesic distances work")

        # Test Euclidean distances
        euclid_dist = euclid(sf, nyc)
        squared_dist = squared(sf, nyc)
        assert euclid_dist > 0
        assert squared_dist > 0
        print("Euclidean distances work")

        # Test point to segment
        seg_dist = point_to_segment(LngLat(-98.0, 39.0), sf, nyc)
        seg_dist_sq = point_to_segment_squared(LngLat(-98.0, 39.0), sf, nyc)
        assert seg_dist > 0
        assert seg_dist_sq > 0
        print("Point to segment works")

        # Test batch operations
        points = [sf, LngLat(-98.0, 39.0), nyc]
        pairwise_dists = pairwise_haversine(points)
        path_length = path_length_haversine(points)
        assert len(pairwise_dists) == 2
        assert path_length > 0
        print("Batch operations work")

        print("All distance functionality working!")

    except Exception as e:
        print(f"X Distance smoke test failed: {e}")
        raise


def test_simplify_smoke_test():
    print("\n=== Simplify Smoke Test ===")

    # Quick smoke test to ensure imports and basic functionality work
    try:
        from rapidgeo.simplify import douglas_peucker
        from rapidgeo.simplify.batch import simplify_multiple

        print("Imports successful")

        # Basic functionality test
        points = [LngLat(-122.0, 37.0), LngLat(-121.0, 37.0)]
        simplified = douglas_peucker(points, tolerance_m=100.0)
        assert len(simplified) == 2
        print("Basic simplification works")

        # Batch test
        polylines = [[LngLat(-122.0, 37.0), LngLat(-121.0, 37.0)]]
        batch_result = simplify_multiple(polylines, tolerance_m=100.0)
        assert len(batch_result) == 1
        print("Batch simplification works")

    except Exception as e:
        print(f"Simplify smoke test failed: {e}")
        raise


if __name__ == "__main__":
    test_basic_functionality()
    test_simplify_functionality()
    test_distance_smoke_test()
    test_simplify_smoke_test()
