import pytest
from rapidgeo import LngLat
from rapidgeo.similarity.frechet import (
    discrete_frechet,
    discrete_frechet_with_threshold,
)
from rapidgeo.similarity.hausdorff import hausdorff, hausdorff_with_threshold


class TestFrechetDistance:
    def test_identical_single_points(self):
        a = [LngLat(0.0, 0.0)]
        b = [LngLat(0.0, 0.0)]

        result = discrete_frechet(a, b)
        assert result < 1e-10

    def test_different_single_points(self):
        a = [LngLat(0.0, 0.0)]
        b = [LngLat(1.0, 1.0)]

        result = discrete_frechet(a, b)
        assert result > 0

    def test_identical_curves(self):
        curve = [
            LngLat(0.0, 0.0),
            LngLat(1.0, 1.0),
            LngLat(2.0, 0.0),
        ]

        result = discrete_frechet(curve, curve)
        assert result < 1e-10

    def test_simple_curves(self):
        a = [LngLat(0.0, 0.0), LngLat(1.0, 0.0)]
        b = [LngLat(0.0, 1.0), LngLat(1.0, 1.0)]

        result = discrete_frechet(a, b)
        assert result > 0

    def test_empty_input_raises_error(self):
        a = []
        b = [LngLat(0.0, 0.0)]

        with pytest.raises(ValueError):
            discrete_frechet(a, b)

    def test_with_threshold(self):
        a = [LngLat(0.0, 0.0), LngLat(1.0, 0.0)]
        b = [LngLat(0.0, 1.0), LngLat(1.0, 1.0)]

        full_distance = discrete_frechet(a, b)
        threshold = full_distance / 2.0

        result = discrete_frechet_with_threshold(a, b, threshold)
        assert result > threshold


class TestHausdorffDistance:
    def test_identical_single_points(self):
        a = [LngLat(0.0, 0.0)]
        b = [LngLat(0.0, 0.0)]

        result = hausdorff(a, b)
        assert result < 1e-10

    def test_different_single_points(self):
        a = [LngLat(0.0, 0.0)]
        b = [LngLat(1.0, 1.0)]

        result = hausdorff(a, b)
        assert result > 0

    def test_identical_curves(self):
        curve = [
            LngLat(0.0, 0.0),
            LngLat(1.0, 1.0),
            LngLat(2.0, 0.0),
        ]

        result = hausdorff(curve, curve)
        assert result < 1e-10

    def test_simple_hausdorff(self):
        a = [LngLat(0.0, 0.0), LngLat(1.0, 0.0)]
        b = [LngLat(0.0, 1.0), LngLat(1.0, 1.0)]

        result = hausdorff(a, b)
        assert result > 0

    def test_empty_input_raises_error(self):
        a = []
        b = [LngLat(0.0, 0.0)]

        with pytest.raises(ValueError):
            hausdorff(a, b)

    def test_with_threshold(self):
        a = [LngLat(0.0, 0.0), LngLat(1.0, 0.0)]
        b = [LngLat(0.0, 1.0), LngLat(1.0, 1.0)]

        full_distance = hausdorff(a, b)
        threshold = full_distance / 2.0

        result = hausdorff_with_threshold(a, b, threshold)
        assert result > threshold


class TestSecurityAndLimits:
    def test_memory_limit_enforcement_frechet(self):
        # Test that large polylines are rejected to prevent memory exhaustion
        large_polyline = [LngLat(i * 0.001, i * 0.001) for i in range(5000)]

        # Should reject computation that would exceed memory limits
        with pytest.raises(Exception):  # MemoryError or ValueError
            discrete_frechet(large_polyline, large_polyline)

    def test_size_limit_enforcement_frechet(self):
        # Test that oversized polylines are rejected
        oversized_polyline = [LngLat(i * 0.001, i * 0.001) for i in range(15000)]
        small_polyline = [LngLat(0.0, 0.0)]

        with pytest.raises(ValueError, match="size limited"):
            discrete_frechet(oversized_polyline, small_polyline)

    def test_size_limit_enforcement_hausdorff(self):
        # Test that oversized polylines are rejected for Hausdorff
        oversized_polyline = [LngLat(i * 0.001, i * 0.001) for i in range(60000)]
        small_polyline = [LngLat(0.0, 0.0)]

        with pytest.raises(ValueError, match="size limited"):
            hausdorff(oversized_polyline, small_polyline)

    def test_reasonable_size_polylines_work(self):
        # Test that reasonably sized polylines work fine
        medium_polyline_a = [LngLat(i * 0.01, 0.0) for i in range(100)]
        medium_polyline_b = [LngLat(0.0, i * 0.01) for i in range(100)]

        # Should work without issues
        result_frechet = discrete_frechet(medium_polyline_a, medium_polyline_b)
        result_hausdorff = hausdorff(medium_polyline_a, medium_polyline_b)

        assert result_frechet > 0
        assert result_hausdorff > 0
