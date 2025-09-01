#!/usr/bin/env python3

import pytest
from rapidgeo import LngLat
from rapidgeo.simplify import douglas_peucker
from rapidgeo.simplify.batch import simplify_multiple


class TestSimplifyBasic:
    """Basic functionality tests for simplify module"""

    def test_douglas_peucker_basic(self):
        """Test basic Douglas-Peucker simplification"""
        points = [LngLat(-122.0, 37.0), LngLat(-121.5, 37.5), LngLat(-121.0, 37.0)]

        simplified = douglas_peucker(points, tolerance_m=1000.0)

        assert len(simplified) >= 2  # At least endpoints
        assert isinstance(simplified[0], LngLat)
        assert simplified[0].lng == -122.0
        assert simplified[0].lat == 37.0
        assert simplified[-1].lng == -121.0
        assert simplified[-1].lat == 37.0

    def test_douglas_peucker_mask_return(self):
        """Test mask return functionality"""
        points = [LngLat(-122.0, 37.0), LngLat(-121.5, 37.5), LngLat(-121.0, 37.0)]

        mask = douglas_peucker(points, tolerance_m=1000.0, return_mask=True)

        assert len(mask) == len(points)
        assert isinstance(mask, list)
        assert all(isinstance(b, bool) for b in mask)
        assert mask[0] is True  # First endpoint always kept
        assert mask[-1] is True  # Last endpoint always kept

    def test_douglas_peucker_methods(self):
        """Test all three methods work"""
        points = [LngLat(-122.0, 37.0), LngLat(-121.5, 37.5), LngLat(-121.0, 37.0)]

        for method in ["great_circle", "planar", "euclidean"]:
            simplified = douglas_peucker(points, tolerance_m=1000.0, method=method)
            assert len(simplified) >= 2
            assert simplified[0].lng == -122.0
            assert simplified[-1].lng == -121.0

    def test_douglas_peucker_invalid_method(self):
        """Test invalid method raises error"""
        points = [LngLat(-122.0, 37.0), LngLat(-121.0, 37.0)]

        with pytest.raises(ValueError, match="Invalid method"):
            douglas_peucker(points, tolerance_m=100.0, method="invalid")

    def test_single_point(self):
        """Test single point handling"""
        points = [LngLat(-122.0, 37.0)]

        simplified = douglas_peucker(points, tolerance_m=100.0)
        assert len(simplified) == 1
        assert simplified[0].lng == -122.0
        assert simplified[0].lat == 37.0

    def test_two_points(self):
        """Test two points handling"""
        points = [LngLat(-122.0, 37.0), LngLat(-121.0, 37.0)]

        simplified = douglas_peucker(points, tolerance_m=100.0)
        assert len(simplified) == 2
        assert simplified[0].lng == -122.0
        assert simplified[1].lng == -121.0

    def test_zero_tolerance(self):
        """Test zero tolerance preserves all points"""
        points = [LngLat(-122.0, 37.0), LngLat(-121.5, 37.5), LngLat(-121.0, 37.0)]

        simplified = douglas_peucker(points, tolerance_m=0.0)
        assert len(simplified) == len(points)

    def test_high_tolerance(self):
        """Test very high tolerance only keeps endpoints"""
        points = [LngLat(-122.0, 37.0), LngLat(-121.5, 37.5), LngLat(-121.0, 37.0)]

        simplified = douglas_peucker(points, tolerance_m=1000000.0)
        assert len(simplified) == 2  # Only endpoints
        assert simplified[0].lng == -122.0
        assert simplified[1].lng == -121.0


class TestSimplifyBatch:
    """Test batch simplification functionality"""

    def test_batch_simplify_basic(self):
        """Test basic batch simplification"""
        polylines = [
            [LngLat(-122.0, 37.0), LngLat(-121.5, 37.5), LngLat(-121.0, 37.0)],
            [LngLat(-74.0, 40.0), LngLat(-73.5, 40.5), LngLat(-73.0, 40.0)],
        ]

        simplified_batch = simplify_multiple(polylines, tolerance_m=1000.0)

        assert len(simplified_batch) == 2
        for simplified in simplified_batch:
            assert len(simplified) >= 2
            assert isinstance(simplified[0], LngLat)

    def test_batch_simplify_masks(self):
        """Test batch mask return"""
        polylines = [
            [LngLat(-122.0, 37.0), LngLat(-121.5, 37.5), LngLat(-121.0, 37.0)],
            [LngLat(-74.0, 40.0), LngLat(-73.5, 40.5), LngLat(-73.0, 40.0)],
        ]

        masks = simplify_multiple(polylines, tolerance_m=1000.0, return_masks=True)

        assert len(masks) == 2
        for i, mask in enumerate(masks):
            assert len(mask) == len(polylines[i])
            assert mask[0] is True  # First endpoint
            assert mask[-1] is True  # Last endpoint

    def test_batch_empty_polyline(self):
        """Test batch with empty polyline"""
        polylines = [
            [],  # Empty
            [LngLat(-122.0, 37.0), LngLat(-121.0, 37.0)],  # Normal
        ]

        simplified_batch = simplify_multiple(polylines, tolerance_m=1000.0)

        assert len(simplified_batch) == 2
        assert len(simplified_batch[0]) == 0  # Empty stays empty
        assert len(simplified_batch[1]) == 2  # Two points stay two

    def test_batch_methods(self):
        """Test batch with different methods"""
        polylines = [[LngLat(-122.0, 37.0), LngLat(-121.5, 37.5), LngLat(-121.0, 37.0)]]

        for method in ["great_circle", "planar", "euclidean"]:
            simplified_batch = simplify_multiple(
                polylines, tolerance_m=1000.0, method=method
            )
            assert len(simplified_batch) == 1
            assert len(simplified_batch[0]) >= 2


class TestSimplifyEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_list(self):
        """Test empty point list"""
        points = []

        simplified = douglas_peucker(points, tolerance_m=100.0)
        assert len(simplified) == 0

    def test_identical_points(self):
        """Test with identical points"""
        points = [LngLat(-122.0, 37.0), LngLat(-122.0, 37.0), LngLat(-122.0, 37.0)]

        simplified = douglas_peucker(points, tolerance_m=100.0)
        assert len(simplified) == 3  # All points kept when identical

    def test_negative_tolerance(self):
        """Test behavior with negative tolerance (should work like zero)"""
        points = [LngLat(-122.0, 37.0), LngLat(-121.5, 37.5), LngLat(-121.0, 37.0)]

        simplified = douglas_peucker(points, tolerance_m=-10.0)
        # Negative tolerance should behave like zero tolerance
        assert len(simplified) == len(points)

    def test_antimeridian_crossing(self):
        """Test points crossing antimeridian"""
        points = [
            LngLat(179.0, 0.0),
            LngLat(179.5, 0.1),
            LngLat(-179.5, 0.2),
            LngLat(-179.0, 0.0),
        ]

        simplified = douglas_peucker(points, tolerance_m=1000.0, method="great_circle")
        assert len(simplified) >= 2
        assert simplified[0].lng == 179.0
        assert simplified[-1].lng == -179.0


if __name__ == "__main__":
    pytest.main([__file__])
