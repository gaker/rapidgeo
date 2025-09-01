#!/usr/bin/env python3

import pytest
import math
from rapidgeo import LngLat
from rapidgeo.distance.geo import haversine, vincenty_distance
from rapidgeo.distance.euclid import (
    euclid,
    squared,
    point_to_segment,
    point_to_segment_squared,
)
from rapidgeo.distance.batch import pairwise_haversine, path_length_haversine


class TestLngLat:
    """Test LngLat coordinate class"""

    def test_lnglat_creation(self):
        """Test LngLat creation and properties"""
        pt = LngLat(-122.4194, 37.7749)
        assert pt.lng == -122.4194
        assert pt.lat == 37.7749

    def test_lnglat_repr(self):
        """Test string representation"""
        pt = LngLat(-122.0, 37.0)
        assert repr(pt) == "LngLat(-122, 37)"

    def test_lnglat_edge_coordinates(self):
        """Test edge case coordinates"""
        # Antimeridian
        pt1 = LngLat(180.0, 0.0)
        assert pt1.lng == 180.0

        pt2 = LngLat(-180.0, 0.0)
        assert pt2.lng == -180.0

        # Poles
        pt3 = LngLat(0.0, 90.0)
        assert pt3.lat == 90.0

        pt4 = LngLat(0.0, -90.0)
        assert pt4.lat == -90.0


class TestGeodesicDistances:
    """Test geodesic distance calculations"""

    def test_haversine_basic(self):
        """Test basic haversine distance calculation"""
        sf = LngLat(-122.4194, 37.7749)
        nyc = LngLat(-74.0060, 40.7128)

        distance = haversine(sf, nyc)

        # SF to NYC is approximately 4130 km
        assert 4_000_000 < distance < 4_200_000
        assert isinstance(distance, float)

    def test_haversine_zero_distance(self):
        """Test haversine with identical points"""
        pt = LngLat(-122.0, 37.0)

        distance = haversine(pt, pt)
        assert distance == 0.0

    def test_haversine_small_distance(self):
        """Test haversine with very small distances"""
        pt1 = LngLat(0.0, 0.0)
        pt2 = LngLat(0.001, 0.001)  # ~157 meters

        distance = haversine(pt1, pt2)
        assert 100 < distance < 200

    def test_haversine_antipodal(self):
        """Test haversine with antipodal points"""
        pt1 = LngLat(0.0, 0.0)
        pt2 = LngLat(180.0, 0.0)

        distance = haversine(pt1, pt2)
        # Half circumference of Earth (~20,015 km)
        assert 20_000_000 < distance < 20_100_000

    def test_vincenty_basic(self):
        """Test basic Vincenty distance calculation"""
        sf = LngLat(-122.4194, 37.7749)
        nyc = LngLat(-74.0060, 40.7128)

        distance = vincenty_distance(sf, nyc)

        # Vincenty should be very close to haversine for this distance
        haversine_dist = haversine(sf, nyc)
        assert abs(distance - haversine_dist) < 20_000  # Within 20km

    def test_vincenty_zero_distance(self):
        """Test Vincenty with identical points"""
        pt = LngLat(-122.0, 37.0)

        distance = vincenty_distance(pt, pt)
        assert distance == 0.0

    def test_vincenty_precision(self):
        """Test Vincenty higher precision vs haversine"""
        # Points where Vincenty should be more accurate
        pt1 = LngLat(0.0, 0.0)
        pt2 = LngLat(0.0001, 0.0001)  # Very close points

        vincenty_dist = vincenty_distance(pt1, pt2)
        haversine_dist = haversine(pt1, pt2)

        # Both should be small and close
        assert vincenty_dist > 0
        assert haversine_dist > 0
        assert abs(vincenty_dist - haversine_dist) < 1.0  # Within 1 meter


class TestEuclideanDistances:
    """Test Euclidean distance calculations"""

    def test_euclid_basic(self):
        """Test basic Euclidean distance"""
        pt1 = LngLat(0.0, 0.0)
        pt2 = LngLat(1.0, 1.0)

        distance = euclid(pt1, pt2)
        expected = math.sqrt(2.0)  # sqrt(1^2 + 1^2)
        assert abs(distance - expected) < 1e-10

    def test_euclid_zero_distance(self):
        """Test Euclidean with identical points"""
        pt = LngLat(-122.0, 37.0)

        distance = euclid(pt, pt)
        assert distance == 0.0

    def test_squared_basic(self):
        """Test squared Euclidean distance"""
        pt1 = LngLat(0.0, 0.0)
        pt2 = LngLat(3.0, 4.0)

        distance_sq = squared(pt1, pt2)
        expected = 3.0 * 3.0 + 4.0 * 4.0  # 25.0
        assert abs(distance_sq - expected) < 1e-10

    def test_squared_vs_euclid_consistency(self):
        """Test squared distance consistency with Euclidean"""
        pt1 = LngLat(-122.0, 37.0)
        pt2 = LngLat(-121.0, 38.0)

        euclid_dist = euclid(pt1, pt2)
        squared_dist = squared(pt1, pt2)

        assert abs(squared_dist - euclid_dist**2) < 1e-10

    def test_point_to_segment_basic(self):
        """Test basic point to segment distance"""
        # Segment from (0,0) to (2,0), point at (1,1)
        seg_start = LngLat(0.0, 0.0)
        seg_end = LngLat(2.0, 0.0)
        point = LngLat(1.0, 1.0)

        distance = point_to_segment(point, seg_start, seg_end)
        # Distance should be 1.0 (perpendicular to horizontal segment)
        assert abs(distance - 1.0) < 1e-10

    def test_point_to_segment_endpoint(self):
        """Test point to segment when closest point is an endpoint"""
        # Segment from (0,0) to (1,0), point at (-1,0)
        seg_start = LngLat(0.0, 0.0)
        seg_end = LngLat(1.0, 0.0)
        point = LngLat(-1.0, 0.0)

        distance = point_to_segment(point, seg_start, seg_end)
        # Distance should be 1.0 (to start point)
        assert abs(distance - 1.0) < 1e-10

    def test_point_to_segment_on_segment(self):
        """Test point to segment when point is on segment"""
        seg_start = LngLat(0.0, 0.0)
        seg_end = LngLat(2.0, 0.0)
        point = LngLat(1.0, 0.0)  # On the segment

        distance = point_to_segment(point, seg_start, seg_end)
        assert distance == 0.0

    def test_point_to_segment_squared_consistency(self):
        """Test squared point to segment consistency"""
        seg_start = LngLat(0.0, 0.0)
        seg_end = LngLat(2.0, 0.0)
        point = LngLat(1.0, 1.0)

        distance = point_to_segment(point, seg_start, seg_end)
        distance_sq = point_to_segment_squared(point, seg_start, seg_end)

        assert abs(distance_sq - distance**2) < 1e-10


class TestBatchOperations:
    """Test batch distance operations"""

    def test_pairwise_haversine_basic(self):
        """Test basic pairwise haversine calculation"""
        points = [LngLat(0.0, 0.0), LngLat(1.0, 0.0), LngLat(1.0, 1.0)]

        distances = pairwise_haversine(points)

        assert len(distances) == 2  # n-1 distances
        assert all(d > 0 for d in distances)
        assert isinstance(distances, list)

    def test_pairwise_haversine_single_point(self):
        """Test pairwise with single point"""
        points = [LngLat(0.0, 0.0)]

        distances = pairwise_haversine(points)
        assert len(distances) == 0

    def test_pairwise_haversine_two_points(self):
        """Test pairwise with two points"""
        points = [LngLat(0.0, 0.0), LngLat(1.0, 0.0)]

        distances = pairwise_haversine(points)
        assert len(distances) == 1

        # Should match individual haversine calculation
        expected = haversine(points[0], points[1])
        assert abs(distances[0] - expected) < 1e-10

    def test_path_length_haversine_basic(self):
        """Test path length calculation"""
        points = [
            LngLat(0.0, 0.0),
            LngLat(1.0, 0.0),
            LngLat(1.0, 1.0),
            LngLat(0.0, 1.0),
        ]

        total_length = path_length_haversine(points)

        # Should be sum of pairwise distances
        pairwise = pairwise_haversine(points)
        expected = sum(pairwise)

        assert abs(total_length - expected) < 1e-10

    def test_path_length_haversine_single_point(self):
        """Test path length with single point"""
        points = [LngLat(0.0, 0.0)]

        total_length = path_length_haversine(points)
        assert total_length == 0.0

    def test_path_length_haversine_empty(self):
        """Test path length with empty list"""
        points = []

        total_length = path_length_haversine(points)
        assert total_length == 0.0

    def test_batch_consistency(self):
        """Test batch operations consistency with individual calls"""
        points = [LngLat(-122.0, 37.0), LngLat(-121.0, 37.5), LngLat(-120.0, 38.0)]

        # Pairwise batch vs individual
        batch_distances = pairwise_haversine(points)
        individual_distances = [
            haversine(points[0], points[1]),
            haversine(points[1], points[2]),
        ]

        assert len(batch_distances) == len(individual_distances)
        for batch_dist, individual_dist in zip(batch_distances, individual_distances):
            assert abs(batch_dist - individual_dist) < 1e-10

    def test_large_batch_performance(self):
        """Test batch operations with larger dataset"""
        # Create a path with many points
        points = []
        for i in range(100):
            points.append(LngLat(-122.0 + i * 0.01, 37.0 + i * 0.01))

        distances = pairwise_haversine(points)
        total_length = path_length_haversine(points)

        assert len(distances) == 99
        assert total_length > 0
        assert total_length == sum(distances)


class TestDistanceEdgeCases:
    """Test edge cases and error conditions"""

    def test_vincenty_convergence_failure(self):
        """Test Vincenty with potentially problematic points"""
        # Test with antipodal points which can be problematic for Vincenty
        pt1 = LngLat(0.0, 0.0)
        pt2 = LngLat(180.0, 0.0)

        try:
            distance = vincenty_distance(pt1, pt2)
            # Vincenty might return 0.0 or a valid distance for antipodal points
            assert distance >= 0
        except ValueError:
            # Vincenty can fail to converge for antipodal points
            pass

    def test_extreme_coordinates(self):
        """Test with extreme but valid coordinates"""
        # Near poles
        pt1 = LngLat(0.0, 89.9)
        pt2 = LngLat(180.0, 89.9)

        distance = haversine(pt1, pt2)
        assert distance > 0

    def test_coordinate_precision(self):
        """Test with high precision coordinates"""
        pt1 = LngLat(-122.123456789, 37.987654321)
        pt2 = LngLat(-122.123456790, 37.987654322)

        distance = haversine(pt1, pt2)
        assert distance > 0
        assert distance < 1.0  # Very small distance

    def test_zero_length_segment(self):
        """Test point to segment with zero-length segment"""
        seg_start = LngLat(1.0, 1.0)
        seg_end = LngLat(1.0, 1.0)  # Same point
        point = LngLat(0.0, 0.0)

        distance = point_to_segment(point, seg_start, seg_end)
        expected = euclid(point, seg_start)
        assert abs(distance - expected) < 1e-10

    def test_batch_operations_empty_input(self):
        """Test batch operations with empty input"""
        empty_points = []

        # These should handle empty input gracefully
        distances = pairwise_haversine(empty_points)
        assert len(distances) == 0

        total_length = path_length_haversine(empty_points)
        assert total_length == 0.0


class TestDistanceMethodComparison:
    """Compare different distance calculation methods"""

    def test_haversine_vs_vincenty_accuracy(self):
        """Compare haversine vs Vincenty for various distances"""
        test_cases = [
            # Short distances - should be very close
            (LngLat(0.0, 0.0), LngLat(0.01, 0.01)),
            # Medium distances - should be close
            (LngLat(-122.0, 37.0), LngLat(-74.0, 40.0)),
            # Different hemispheres
            (LngLat(-122.0, 37.0), LngLat(139.69, 35.68)),
        ]

        for pt1, pt2 in test_cases:
            try:
                haversine_dist = haversine(pt1, pt2)
                vincenty_dist = vincenty_distance(pt1, pt2)

                # They should be reasonably close
                relative_diff = abs(haversine_dist - vincenty_dist) / max(
                    haversine_dist, vincenty_dist
                )
                assert relative_diff < 0.01  # Less than 1% difference

            except ValueError:
                # Vincenty might fail for some edge cases
                pass

    def test_euclidean_vs_geodesic_small_distances(self):
        """Compare Euclidean vs geodesic for small distances"""
        # For very small distances, Euclidean should approximate geodesic
        pt1 = LngLat(0.0, 0.0)
        pt2 = LngLat(0.001, 0.001)  # ~157 meters

        euclidean_dist = euclid(pt1, pt2) * 111_320  # Convert degrees to meters (rough)
        geodesic_dist = haversine(pt1, pt2)

        # Should be reasonably close for small distances
        relative_diff = abs(euclidean_dist - geodesic_dist) / geodesic_dist
        assert relative_diff < 0.1  # Within 10%


if __name__ == "__main__":
    pytest.main([__file__])
