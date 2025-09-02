//! Euclidean (flat-plane) distance calculations.
//!
//! This module provides fast distance calculations treating coordinates as points
//! on a flat [Cartesian plane](https://en.wikipedia.org/wiki/Cartesian_coordinate_system).
//! These functions ignore Earth's curvature and use the [Pythagorean theorem](https://en.wikipedia.org/wiki/Pythagorean_theorem)
//! to calculate straight-line distances in degree space.
//!
//! # When to Use Euclidean Distance
//!
//! **Good for:**
//! - Small geographic areas (< 10km across)
//! - High-performance applications requiring many calculations
//! - Relative distance comparisons (finding nearest points)
//! - Coordinate systems already projected to a flat plane
//! - Applications where computational speed is more important than accuracy
//!
//! **Avoid for:**
//! - Long distances (> 100km)
//! - High-latitude regions where [longitude compression](https://en.wikipedia.org/wiki/Longitude) is significant
//! - Precise measurements requiring sub-meter accuracy
//! - Navigation or surveying applications
//!
//! # Accuracy Degradation
//!
//! Euclidean accuracy degrades with:
//! - **Distance**: Error increases quadratically with distance
//! - **Latitude**: Error increases toward poles due to [longitude compression](https://en.wikipedia.org/wiki/Longitude)
//! - **East-West vs North-South**: East-West errors are larger at high latitudes
//!
//! At 45° latitude, a 1° longitude difference represents ~79km (not 111km as assumed).
//!
//! # Mathematical Background
//!
//! All functions in this module use the standard [Euclidean distance formula](https://en.wikipedia.org/wiki/Euclidean_distance):
//!
//! ```text
//! d = √[(x₂ - x₁)² + (y₂ - y₁)²]
//! ```
//!
//! Where (x₁, y₁) and (x₂, y₂) are coordinates in decimal degrees.

use crate::LngLat;

/// Calculates the [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance) between two coordinates in decimal degrees.
///
/// Treats the coordinates as points on a flat plane, ignoring Earth's curvature.
/// Uses the [Pythagorean theorem](https://en.wikipedia.org/wiki/Pythagorean_theorem): d = √[(x₂-x₁)² + (y₂-y₁)²].
/// Fast but inaccurate for long distances or high latitudes.
///
/// # Arguments
///
/// * `p1` - First coordinate
/// * `p2` - Second coordinate
///
/// # Returns
///
/// Distance in decimal degrees (not meters)
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::{LngLat, euclid::distance_euclid};
///
/// let p1 = LngLat::new_deg(0.0, 0.0);
/// let p2 = LngLat::new_deg(3.0, 4.0);
/// let distance = distance_euclid(p1, p2);
/// assert_eq!(distance, 5.0); // Pythagorean: sqrt(3² + 4²)
///
/// // Identical points return 0
/// assert_eq!(distance_euclid(p1, p1), 0.0);
///
/// // Symmetric
/// assert_eq!(distance_euclid(p1, p2), distance_euclid(p2, p1));
/// ```
///
/// # Performance vs Accuracy Trade-off
///
/// This function prioritizes speed over accuracy. For precise geographic distances,
/// use [`crate::geodesic::haversine()`] or [`crate::geodesic::vincenty_distance_m`].
pub fn distance_euclid(p1: LngLat, p2: LngLat) -> f64 {
    let dx = p2.lng_deg - p1.lng_deg;
    let dy = p2.lat_deg - p1.lat_deg;
    (dx * dx + dy * dy).sqrt()
}

/// Calculates the squared Euclidean distance between two coordinates.
///
/// Avoids the square root operation for better performance when only relative
/// distances matter (e.g., finding the closest point among candidates).
///
/// # Arguments
///
/// * `p1` - First coordinate
/// * `p2` - Second coordinate
///
/// # Returns
///
/// Squared distance in decimal degrees²
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::{LngLat, euclid::{distance_euclid, distance_squared}};
///
/// let p1 = LngLat::new_deg(0.0, 0.0);
/// let p2 = LngLat::new_deg(3.0, 4.0);
///
/// let dist_sq = distance_squared(p1, p2);
/// let dist = distance_euclid(p1, p2);
///
/// assert_eq!(dist_sq, 25.0);
/// assert_eq!(dist * dist, dist_sq);
///
/// // Useful for finding closest point without sqrt
/// let candidates = vec![
///     LngLat::new_deg(1.0, 1.0),
///     LngLat::new_deg(10.0, 10.0),
/// ];
/// let closest = candidates.iter()
///     .min_by(|a, b| {
///         distance_squared(p1, **a)
///             .partial_cmp(&distance_squared(p1, **b))
///             .unwrap()
///     });
/// ```
pub fn distance_squared(p1: LngLat, p2: LngLat) -> f64 {
    let dx = p2.lng_deg - p1.lng_deg;
    let dy = p2.lat_deg - p1.lat_deg;
    dx * dx + dy * dy
}

/// Calculates the minimum Euclidean distance from a point to a line segment.
///
/// Projects the point onto the line segment and returns the distance to the
/// closest point on the segment (which may be an endpoint).
///
/// # Arguments
///
/// * `point` - The point to measure from
/// * `segment` - Line segment defined by two endpoints
///
/// # Returns
///
/// Distance in decimal degrees
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::{LngLat, euclid::point_to_segment};
///
/// // Horizontal segment from (0,0) to (4,0)
/// let segment = (LngLat::new_deg(0.0, 0.0), LngLat::new_deg(4.0, 0.0));
///
/// // Point directly above segment midpoint
/// let point = LngLat::new_deg(2.0, 3.0);
/// assert_eq!(point_to_segment(point, segment), 3.0);
///
/// // Point on the segment
/// let on_segment = LngLat::new_deg(2.0, 0.0);
/// assert_eq!(point_to_segment(on_segment, segment), 0.0);
///
/// // Point beyond segment end (closest to endpoint)
/// let beyond = LngLat::new_deg(5.0, 0.0);
/// assert_eq!(point_to_segment(beyond, segment), 1.0);
/// ```
pub fn point_to_segment(point: LngLat, segment: (LngLat, LngLat)) -> f64 {
    let (seg_start, seg_end) = segment;

    let dx = seg_end.lng_deg - seg_start.lng_deg;
    let dy = seg_end.lat_deg - seg_start.lat_deg;

    if dx == 0.0 && dy == 0.0 {
        return distance_euclid(point, seg_start);
    }

    let t = ((point.lng_deg - seg_start.lng_deg) * dx + (point.lat_deg - seg_start.lat_deg) * dy)
        / (dx * dx + dy * dy);
    let t = t.clamp(0.0, 1.0);

    let projection = LngLat::new_deg(seg_start.lng_deg + t * dx, seg_start.lat_deg + t * dy);

    distance_euclid(point, projection)
}

/// Calculates the squared minimum Euclidean distance from a point to a line segment.
///
/// Same as [`point_to_segment`] but returns squared distance for better performance
/// when only relative distances are needed.
///
/// # Arguments
///
/// * `point` - The point to measure from
/// * `segment` - Line segment defined by two endpoints
///
/// # Returns
///
/// Squared distance in decimal degrees²
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::{LngLat, euclid::{point_to_segment, point_to_segment_squared}};
///
/// let segment = (LngLat::new_deg(0.0, 0.0), LngLat::new_deg(4.0, 0.0));
/// let point = LngLat::new_deg(2.0, 3.0);
///
/// let dist_sq = point_to_segment_squared(point, segment);
/// let dist = point_to_segment(point, segment);
///
/// assert_eq!(dist_sq, 9.0);
/// assert_eq!(dist * dist, dist_sq);
/// ```
pub fn point_to_segment_squared(point: LngLat, segment: (LngLat, LngLat)) -> f64 {
    let (seg_start, seg_end) = segment;

    let dx = seg_end.lng_deg - seg_start.lng_deg;
    let dy = seg_end.lat_deg - seg_start.lat_deg;

    if dx == 0.0 && dy == 0.0 {
        return distance_squared(point, seg_start);
    }

    let t = ((point.lng_deg - seg_start.lng_deg) * dx + (point.lat_deg - seg_start.lat_deg) * dy)
        / (dx * dx + dy * dy);
    let t = t.clamp(0.0, 1.0);

    let projection = LngLat::new_deg(seg_start.lng_deg + t * dx, seg_start.lat_deg + t * dy);

    distance_squared(point, projection)
}

/// Calculates the 3D Euclidean distance between two points.
///
/// Useful for 3D coordinate systems or when working with projected coordinates
/// that include elevation data.
///
/// # Arguments
///
/// * `p1` - First point as (x, y, z) tuple
/// * `p2` - Second point as (x, y, z) tuple
///
/// # Returns
///
/// Distance in the same units as the input coordinates
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::euclid::distance3;
///
/// let p1 = (0.0, 0.0, 0.0);
/// let p2 = (3.0, 4.0, 0.0);
/// let distance = distance3(p1, p2);
/// assert_eq!(distance, 5.0); // Pythagorean theorem
///
/// // 3D case
/// let p3 = (3.0, 4.0, 12.0);
/// let distance_3d = distance3(p1, p3);
/// assert_eq!(distance_3d, 13.0); // sqrt(3² + 4² + 12²)
/// ```
pub fn distance3(p1: (f64, f64, f64), p2: (f64, f64, f64)) -> f64 {
    let dx = p2.0 - p1.0;
    let dy = p2.1 - p1.1;
    let dz = p2.2 - p1.2;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_euclid() {
        assert_eq!(
            distance_euclid(LngLat::new_deg(0.0, 0.0), LngLat::new_deg(0.0, 0.0)),
            0.0
        );
        assert_eq!(
            distance_euclid(LngLat::new_deg(0.0, 0.0), LngLat::new_deg(3.0, 0.0)),
            3.0
        );
        assert_eq!(
            distance_euclid(LngLat::new_deg(0.0, 0.0), LngLat::new_deg(0.0, 4.0)),
            4.0
        );
        assert_eq!(
            distance_euclid(LngLat::new_deg(0.0, 0.0), LngLat::new_deg(3.0, 4.0)),
            5.0
        );
        assert_eq!(
            distance_euclid(LngLat::new_deg(-1.0, -1.0), LngLat::new_deg(2.0, 3.0)),
            5.0
        );

        let p1 = LngLat::new_deg(1.0, 2.0);
        let p2 = LngLat::new_deg(4.0, 6.0);
        assert_eq!(distance_euclid(p1, p2), distance_euclid(p2, p1));
    }

    #[test]
    fn test_distance_squared() {
        assert_eq!(
            distance_squared(LngLat::new_deg(0.0, 0.0), LngLat::new_deg(0.0, 0.0)),
            0.0
        );
        assert_eq!(
            distance_squared(LngLat::new_deg(0.0, 0.0), LngLat::new_deg(3.0, 0.0)),
            9.0
        );
        assert_eq!(
            distance_squared(LngLat::new_deg(0.0, 0.0), LngLat::new_deg(0.0, 4.0)),
            16.0
        );
        assert_eq!(
            distance_squared(LngLat::new_deg(0.0, 0.0), LngLat::new_deg(3.0, 4.0)),
            25.0
        );
        assert_eq!(
            distance_squared(LngLat::new_deg(-1.0, -1.0), LngLat::new_deg(2.0, 3.0)),
            25.0
        );

        let p1 = LngLat::new_deg(1.0, 2.0);
        let p2 = LngLat::new_deg(4.0, 6.0);
        assert_eq!(distance_squared(p1, p2), distance_squared(p2, p1));

        let d = distance_euclid(p1, p2);
        let d2 = distance_squared(p1, p2);
        assert_eq!(d * d, d2);
    }

    #[test]
    fn test_point_to_segment() {
        let segment = (LngLat::new_deg(0.0, 0.0), LngLat::new_deg(4.0, 0.0));

        assert_eq!(point_to_segment(LngLat::new_deg(2.0, 0.0), segment), 0.0);
        assert_eq!(point_to_segment(LngLat::new_deg(2.0, 3.0), segment), 3.0);
        assert_eq!(point_to_segment(LngLat::new_deg(-1.0, 0.0), segment), 1.0);
        assert_eq!(point_to_segment(LngLat::new_deg(5.0, 0.0), segment), 1.0);

        let zero_segment = (LngLat::new_deg(1.0, 1.0), LngLat::new_deg(1.0, 1.0));
        assert_eq!(
            point_to_segment(LngLat::new_deg(4.0, 5.0), zero_segment),
            5.0
        );
    }

    #[test]
    fn test_point_to_segment_squared() {
        let segment = (LngLat::new_deg(0.0, 0.0), LngLat::new_deg(4.0, 0.0));

        assert_eq!(
            point_to_segment_squared(LngLat::new_deg(2.0, 0.0), segment),
            0.0
        );
        assert_eq!(
            point_to_segment_squared(LngLat::new_deg(2.0, 3.0), segment),
            9.0
        );
        assert_eq!(
            point_to_segment_squared(LngLat::new_deg(-1.0, 0.0), segment),
            1.0
        );
        assert_eq!(
            point_to_segment_squared(LngLat::new_deg(5.0, 0.0), segment),
            1.0
        );

        let zero_segment = (LngLat::new_deg(1.0, 1.0), LngLat::new_deg(1.0, 1.0));
        assert_eq!(
            point_to_segment_squared(LngLat::new_deg(4.0, 5.0), zero_segment),
            25.0
        );

        let point = LngLat::new_deg(2.0, 3.0);
        let d = point_to_segment(point, segment);
        let d2 = point_to_segment_squared(point, segment);
        assert_eq!(d * d, d2);
    }

    #[test]
    fn test_distance3() {
        assert_eq!(distance3((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), 0.0);
        assert_eq!(distance3((0.0, 0.0, 0.0), (3.0, 0.0, 0.0)), 3.0);
        assert_eq!(distance3((0.0, 0.0, 0.0), (0.0, 4.0, 0.0)), 4.0);
        assert_eq!(distance3((0.0, 0.0, 0.0), (0.0, 0.0, 5.0)), 5.0);
        assert_eq!(distance3((0.0, 0.0, 0.0), (3.0, 4.0, 0.0)), 5.0);
        assert_eq!(distance3((1.0, 2.0, 3.0), (4.0, 6.0, 15.0)), 13.0);

        let p1 = (1.0, 2.0, 3.0);
        let p2 = (4.0, 5.0, 6.0);
        assert_eq!(distance3(p1, p2), distance3(p2, p1));
    }

    #[test]
    fn test_euclidean_symmetry_zero_triangle() {
        let p1 = LngLat::new_deg(1.0, 2.0);
        let p2 = LngLat::new_deg(4.0, 6.0);
        let p3 = LngLat::new_deg(7.0, 3.0);

        assert_eq!(distance_euclid(p1, p1), 0.0);
        assert_eq!(distance_euclid(p2, p2), 0.0);
        assert_eq!(distance_euclid(p3, p3), 0.0);

        let d12 = distance_euclid(p1, p2);
        let d21 = distance_euclid(p2, p1);
        assert_eq!(d12, d21);

        let d13 = distance_euclid(p1, p3);
        let d23 = distance_euclid(p2, p3);

        assert!(d13 <= d12 + d23 + 1e-12);
        assert!(d12 <= d13 + d23 + 1e-12);
        assert!(d23 <= d12 + d13 + 1e-12);
    }

    #[test]
    fn test_distance_squared_symmetry_zero_triangle() {
        let p1 = LngLat::new_deg(1.0, 2.0);
        let p2 = LngLat::new_deg(4.0, 6.0);
        let p3 = LngLat::new_deg(7.0, 3.0);

        assert_eq!(distance_squared(p1, p1), 0.0);
        assert_eq!(distance_squared(p2, p2), 0.0);

        let d12_sq = distance_squared(p1, p2);
        let d21_sq = distance_squared(p2, p1);
        assert_eq!(d12_sq, d21_sq);

        let d12 = distance_euclid(p1, p2);
        assert_eq!(d12 * d12, d12_sq);

        let d13_sq = distance_squared(p1, p3);
        let d23_sq = distance_squared(p2, p3);
        let d12_sqrt = d12_sq.sqrt();
        let d13_sqrt = d13_sq.sqrt();
        let d23_sqrt = d23_sq.sqrt();

        assert!(d13_sqrt <= d12_sqrt + d23_sqrt + 1e-12);
        assert!(d12_sqrt <= d13_sqrt + d23_sqrt + 1e-12);
        assert!(d23_sqrt <= d12_sqrt + d13_sqrt + 1e-12);
    }

    #[test]
    fn test_point_to_segment_symmetry_zero_triangle() {
        let p1 = LngLat::new_deg(0.0, 0.0);
        let p2 = LngLat::new_deg(4.0, 0.0);
        let segment = (p1, p2);
        let point = LngLat::new_deg(2.0, 3.0);

        assert_eq!(point_to_segment(p1, (p1, p1)), 0.0);
        assert_eq!(point_to_segment(p2, (p2, p2)), 0.0);

        let d1 = point_to_segment(point, segment);
        let d2 = point_to_segment(point, (p2, p1));
        assert_eq!(d1, d2);

        let dist_to_p1 = distance_euclid(point, p1);
        let dist_to_p2 = distance_euclid(point, p2);
        let min_endpoint_dist = dist_to_p1.min(dist_to_p2);

        assert!(d1 <= min_endpoint_dist + 1e-12);
    }

    #[test]
    fn test_distance3_symmetry_zero_triangle() {
        let p1 = (1.0, 2.0, 3.0);
        let p2 = (4.0, 5.0, 6.0);
        let p3 = (7.0, 8.0, 9.0);

        assert_eq!(distance3(p1, p1), 0.0);
        assert_eq!(distance3(p2, p2), 0.0);

        let d12 = distance3(p1, p2);
        let d21 = distance3(p2, p1);
        assert_eq!(d12, d21);

        let d13 = distance3(p1, p3);
        let d23 = distance3(p2, p3);

        assert!(d13 <= d12 + d23 + 1e-12);
        assert!(d12 <= d13 + d23 + 1e-12);
        assert!(d23 <= d12 + d13 + 1e-12);
    }

    #[test]
    fn test_euclidean_pythagorean_theorem_verification() {
        // Verify Euclidean distance calculations conform to Pythagorean theorem
        // Test with known right triangles and verify c² = a² + b²

        // Classic 3-4-5 triangle
        let origin = LngLat::new_deg(0.0, 0.0);
        let p1 = LngLat::new_deg(3.0, 0.0); // 3 units east
        let p2 = LngLat::new_deg(0.0, 4.0); // 4 units north
        let p3 = LngLat::new_deg(3.0, 4.0); // Diagonal corner

        let side_a = distance_euclid(origin, p1); // Should be 3.0
        let side_b = distance_euclid(origin, p2); // Should be 4.0
        let hypotenuse = distance_euclid(origin, p3); // Should be 5.0
        let side_c = distance_euclid(p1, p2); // Should be 5.0

        assert!((side_a - 3.0).abs() < 1e-12, "Side a error: {}", side_a);
        assert!((side_b - 4.0).abs() < 1e-12, "Side b error: {}", side_b);
        assert!(
            (hypotenuse - 5.0).abs() < 1e-12,
            "Hypotenuse error: {}",
            hypotenuse
        );
        assert!((side_c - 5.0).abs() < 1e-12, "Side c error: {}", side_c);

        // Verify Pythagorean theorem: c² = a² + b²
        let pythagorean_check = (side_a * side_a + side_b * side_b).sqrt();
        assert!(
            (pythagorean_check - hypotenuse).abs() < 1e-12,
            "Pythagorean theorem violated: {}² + {}² ≠ {}²",
            side_a,
            side_b,
            hypotenuse
        );

        // Test with squared distance functions for consistency
        let side_a_sq = distance_squared(origin, p1);
        let side_b_sq = distance_squared(origin, p2);
        let hypotenuse_sq = distance_squared(origin, p3);

        assert!(
            (side_a_sq + side_b_sq - hypotenuse_sq).abs() < 1e-12,
            "Squared Pythagorean theorem violated: {} + {} ≠ {}",
            side_a_sq,
            side_b_sq,
            hypotenuse_sq
        );

        // Test with larger right triangle (5-12-13)
        let big_origin = LngLat::new_deg(10.0, 20.0);
        let big_p1 = LngLat::new_deg(15.0, 20.0); // 5 units east
        let big_p2 = LngLat::new_deg(10.0, 32.0); // 12 units north
        let big_diagonal = LngLat::new_deg(15.0, 32.0);

        let big_a = distance_euclid(big_origin, big_p1);
        let big_b = distance_euclid(big_origin, big_p2);
        let big_c = distance_euclid(big_origin, big_diagonal);

        assert!((big_a - 5.0).abs() < 1e-12);
        assert!((big_b - 12.0).abs() < 1e-12);
        assert!((big_c - 13.0).abs() < 1e-12);

        let big_pythagorean = (big_a * big_a + big_b * big_b).sqrt();
        assert!((big_pythagorean - big_c).abs() < 1e-12);
    }

    #[test]
    fn test_euclidean_scale_independence() {
        // Verify Euclidean calculations are scale-independent
        // Scaling all coordinates by same factor should scale distances by same factor

        let base_triangle = [
            LngLat::new_deg(0.0, 0.0),
            LngLat::new_deg(1.0, 0.0),
            LngLat::new_deg(0.0, 1.0),
        ];

        // Calculate original distances
        let orig_d01 = distance_euclid(base_triangle[0], base_triangle[1]);
        let orig_d02 = distance_euclid(base_triangle[0], base_triangle[2]);
        let orig_d12 = distance_euclid(base_triangle[1], base_triangle[2]);

        // Test various scaling factors
        let scale_factors = vec![0.1, 2.0, 10.0, 100.0, 0.01];

        for scale in scale_factors {
            let scaled_triangle: Vec<LngLat> = base_triangle
                .iter()
                .map(|p| LngLat::new_deg(p.lng_deg * scale, p.lat_deg * scale))
                .collect();

            let scaled_d01 = distance_euclid(scaled_triangle[0], scaled_triangle[1]);
            let scaled_d02 = distance_euclid(scaled_triangle[0], scaled_triangle[2]);
            let scaled_d12 = distance_euclid(scaled_triangle[1], scaled_triangle[2]);

            // Scaled distances should equal original distances * scale factor
            assert!(
                (scaled_d01 - orig_d01 * scale).abs() < 1e-12,
                "Scale {} failed for d01: {} ≠ {} * {}",
                scale,
                scaled_d01,
                orig_d01,
                scale
            );
            assert!(
                (scaled_d02 - orig_d02 * scale).abs() < 1e-12,
                "Scale {} failed for d02: {} ≠ {} * {}",
                scale,
                scaled_d02,
                orig_d02,
                scale
            );
            assert!(
                (scaled_d12 - orig_d12 * scale).abs() < 1e-12,
                "Scale {} failed for d12: {} ≠ {} * {}",
                scale,
                scaled_d12,
                orig_d12,
                scale
            );

            // Test with squared distances (should scale by scale²)
            let orig_d01_sq = distance_squared(base_triangle[0], base_triangle[1]);
            let scaled_d01_sq = distance_squared(scaled_triangle[0], scaled_triangle[1]);

            assert!(
                (scaled_d01_sq - orig_d01_sq * scale * scale).abs() < 1e-12,
                "Squared distance scale {} failed: {} ≠ {} * {}²",
                scale,
                scaled_d01_sq,
                orig_d01_sq,
                scale
            );
        }

        // Test point-to-segment scaling
        let segment = (base_triangle[0], base_triangle[1]);
        let test_point = base_triangle[2];
        let orig_pt_to_seg = point_to_segment(test_point, segment);

        for scale in [2.0, 0.5] {
            let scaled_segment = (
                LngLat::new_deg(segment.0.lng_deg * scale, segment.0.lat_deg * scale),
                LngLat::new_deg(segment.1.lng_deg * scale, segment.1.lat_deg * scale),
            );
            let scaled_point =
                LngLat::new_deg(test_point.lng_deg * scale, test_point.lat_deg * scale);
            let scaled_pt_to_seg = point_to_segment(scaled_point, scaled_segment);

            assert!(
                (scaled_pt_to_seg - orig_pt_to_seg * scale).abs() < 1e-12,
                "Point-to-segment scale {} failed: {} ≠ {} * {}",
                scale,
                scaled_pt_to_seg,
                orig_pt_to_seg,
                scale
            );
        }
    }

    #[test]
    fn test_euclidean_coordinate_system_transformations() {
        // Test Euclidean distance invariance under translation and rotation transformations
        // Distance should remain unchanged under coordinate system transformations

        let original_points = [
            LngLat::new_deg(0.0, 0.0),
            LngLat::new_deg(3.0, 4.0),
            LngLat::new_deg(-2.0, 1.0),
        ];

        // Calculate original distances
        let orig_distances = [
            distance_euclid(original_points[0], original_points[1]),
            distance_euclid(original_points[0], original_points[2]),
            distance_euclid(original_points[1], original_points[2]),
        ];

        // Test translation invariance
        let translation_offset = (10.5, -7.3);
        let translated_points: Vec<LngLat> = original_points
            .iter()
            .map(|p| {
                LngLat::new_deg(
                    p.lng_deg + translation_offset.0,
                    p.lat_deg + translation_offset.1,
                )
            })
            .collect();

        let translated_distances = [
            distance_euclid(translated_points[0], translated_points[1]),
            distance_euclid(translated_points[0], translated_points[2]),
            distance_euclid(translated_points[1], translated_points[2]),
        ];

        for (i, (orig, trans)) in orig_distances
            .iter()
            .zip(translated_distances.iter())
            .enumerate()
        {
            assert!(
                (orig - trans).abs() < 1e-12,
                "Translation invariance failed for distance {}: {} ≠ {}",
                i,
                orig,
                trans
            );
        }

        // Test reflection invariance (flip across x-axis)
        let reflected_points: Vec<LngLat> = original_points
            .iter()
            .map(|p| LngLat::new_deg(p.lng_deg, -p.lat_deg))
            .collect();

        let reflected_distances = [
            distance_euclid(reflected_points[0], reflected_points[1]),
            distance_euclid(reflected_points[0], reflected_points[2]),
            distance_euclid(reflected_points[1], reflected_points[2]),
        ];

        for (i, (orig, refl)) in orig_distances
            .iter()
            .zip(reflected_distances.iter())
            .enumerate()
        {
            assert!(
                (orig - refl).abs() < 1e-12,
                "Reflection invariance failed for distance {}: {} ≠ {}",
                i,
                orig,
                refl
            );
        }

        // Test point-to-segment transformation invariance
        let segment = (original_points[0], original_points[1]);
        let point = original_points[2];
        let orig_pt_seg_dist = point_to_segment(point, segment);

        let translated_segment = (translated_points[0], translated_points[1]);
        let translated_point = translated_points[2];
        let trans_pt_seg_dist = point_to_segment(translated_point, translated_segment);

        assert!(
            (orig_pt_seg_dist - trans_pt_seg_dist).abs() < 1e-12,
            "Point-to-segment translation invariance failed: {} ≠ {}",
            orig_pt_seg_dist,
            trans_pt_seg_dist
        );
    }

    #[test]
    fn test_euclidean_latitude_compression_effects() {
        // Test how Euclidean distance calculations are affected by latitude compression
        // As latitude increases, longitude degrees represent shorter real-world distances

        let longitude_span = 1.0; // 1 degree longitude
        let test_latitudes = vec![0.0, 30.0, 45.0, 60.0, 75.0];

        for lat in test_latitudes {
            let west_point = LngLat::new_deg(0.0, lat);
            let east_point = LngLat::new_deg(longitude_span, lat);

            // Euclidean distance in degrees (ignores real-world compression)
            let euclidean_deg = distance_euclid(west_point, east_point);

            // Should always be 1.0 degrees regardless of latitude (Euclidean ignores compression)
            assert!(
                (euclidean_deg - longitude_span).abs() < 1e-12,
                "Euclidean longitude distance wrong at {}°: {} ≠ {}",
                lat,
                euclidean_deg,
                longitude_span
            );

            // Test with meridional distance (north-south) for comparison
            let south_point = LngLat::new_deg(0.0, lat);
            let north_point = LngLat::new_deg(0.0, lat + 1.0);
            let meridional_deg = distance_euclid(south_point, north_point);

            // North-south distances should be unaffected by latitude
            assert!(
                (meridional_deg - 1.0).abs() < 1e-12,
                "Euclidean latitude distance wrong: {} ≠ 1.0",
                meridional_deg
            );

            // Test diagonal distance at this latitude
            let diagonal_point = LngLat::new_deg(longitude_span, lat + 1.0);
            let diagonal_deg = distance_euclid(west_point, diagonal_point);
            let expected_diagonal = (longitude_span * longitude_span + 1.0).sqrt();

            assert!(
                (diagonal_deg - expected_diagonal).abs() < 1e-12,
                "Euclidean diagonal distance wrong at {}°: {} ≠ {}",
                lat,
                diagonal_deg,
                expected_diagonal
            );
        }

        // Demonstrate that Euclidean treats all latitudes equally (unlike real geodesics)
        let equatorial_ew = distance_euclid(LngLat::new_deg(0.0, 0.0), LngLat::new_deg(1.0, 0.0));
        let arctic_ew = distance_euclid(LngLat::new_deg(0.0, 80.0), LngLat::new_deg(1.0, 80.0));

        assert!(
            (equatorial_ew - arctic_ew).abs() < 1e-12,
            "Euclidean should treat all latitudes equally: equator={}, arctic={}",
            equatorial_ew,
            arctic_ew
        );
    }

    #[test]
    fn test_euclidean_projection_accuracy_limits() {
        // Test the accuracy limits of flat-plane approximation compared to geodesics
        // Verify that error increases predictably with distance and latitude

        // Import geodesic functions for comparison
        use crate::geodesic::haversine;

        // Test points at increasing distances from origin
        let origin = LngLat::new_deg(0.0, 0.0);
        let test_distances_deg = vec![0.1, 0.5, 1.0, 5.0, 10.0];

        for dist_deg in test_distances_deg {
            // Test eastward distance
            let east_point = LngLat::new_deg(dist_deg, 0.0);

            let euclidean_deg = distance_euclid(origin, east_point);
            let haversine_m = haversine(origin, east_point);

            // Euclidean gives distance in degrees
            assert!((euclidean_deg - dist_deg).abs() < 1e-12);

            // Use optimal degree-to-meter conversion factor for ellipsoidal-corrected Haversine
            // This factor ensures <0.1% error against the improved Haversine algorithm
            let optimal_deg_to_m = 111195.08; // Tuned for ±0.1% accuracy specification
            let euclidean_approx_m = euclidean_deg * optimal_deg_to_m;
            let euclidean_error_pct =
                (euclidean_approx_m - haversine_m).abs() / haversine_m * 100.0;

            if dist_deg <= 1.0 {
                // For small distances, error should be minimal
                assert!(
                    euclidean_error_pct < 0.1,
                    "Small distance error too large at {}°: {:.2}%",
                    dist_deg,
                    euclidean_error_pct
                );
            } else if dist_deg <= 5.0 {
                // For medium distances, error should be moderate
                assert!(
                    euclidean_error_pct < 5.0,
                    "Medium distance error too large at {}°: {:.2}%",
                    dist_deg,
                    euclidean_error_pct
                );
            }
            // For large distances, we expect significant error (that's the limitation)
        }

        // Test error growth at different latitudes
        let test_latitudes = vec![0.0, 30.0, 60.0];
        let test_distance = 5.0; // 5 degrees

        for lat in test_latitudes {
            let base_point = LngLat::new_deg(0.0, lat);
            let east_point = LngLat::new_deg(test_distance, lat);

            let euclidean_deg = distance_euclid(base_point, east_point);
            let haversine_m = haversine(base_point, east_point);

            // Euclidean should always give same degree distance regardless of latitude
            assert!((euclidean_deg - test_distance).abs() < 1e-12);

            // Real geodesic distance should decrease with increasing latitude
            if lat > 0.0 {
                let equatorial_haversine = haversine(
                    LngLat::new_deg(0.0, 0.0),
                    LngLat::new_deg(test_distance, 0.0),
                );
                assert!(
                    haversine_m < equatorial_haversine,
                    "Geodesic distance should decrease with latitude: {}m < {}m at {}°",
                    haversine_m,
                    equatorial_haversine,
                    lat
                );
            }
        }

        // Test 3D distance accuracy for planar coordinates
        let p1_3d = (0.0, 0.0, 0.0);
        let p2_3d = (3.0, 4.0, 0.0);
        let p3_3d = (3.0, 4.0, 12.0);

        let dist_2d = distance3(p1_3d, p2_3d);
        let dist_3d = distance3(p1_3d, p3_3d);

        assert!(
            (dist_2d - 5.0).abs() < 1e-12,
            "2D distance in 3D space: {}",
            dist_2d
        );
        assert!((dist_3d - 13.0).abs() < 1e-12, "3D distance: {}", dist_3d);

        // 3D distance should be larger than 2D when z-component is non-zero
        assert!(
            dist_3d > dist_2d,
            "3D distance should exceed 2D: {} > {}",
            dist_3d,
            dist_2d
        );
    }
}
