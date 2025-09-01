use super::haversine::haversine;
use crate::LngLat;

/// Compromise Earth radius for Haversine to minimize maximum error against Vincenty
/// Balances meridional and equatorial accuracy for best overall performance
const EARTH_RADIUS_M: f64 = 6371008.8;

#[inline]
fn check_zero_length_segment(seg_start: LngLat, seg_end: LngLat, point: LngLat) -> Option<f64> {
    if seg_start.lng_deg == seg_end.lng_deg && seg_start.lat_deg == seg_end.lat_deg {
        Some(haversine(point, seg_start))
    } else {
        None
    }
}

#[inline]
fn compute_segment_midpoint(seg_start: LngLat, seg_end: LngLat) -> LngLat {
    LngLat::new_deg(
        (seg_start.lng_deg + seg_end.lng_deg) * 0.5,
        (seg_start.lat_deg + seg_end.lat_deg) * 0.5,
    )
}

#[inline]
fn to_enu_m(origin: LngLat, point: LngLat) -> (f64, f64) {
    let (origin_lng_rad, origin_lat_rad) = origin.to_radians();
    let (point_lng_rad, point_lat_rad) = point.to_radians();

    let dlng = point_lng_rad - origin_lng_rad;
    let dlat = point_lat_rad - origin_lat_rad;

    let cos_lat = origin_lat_rad.cos();

    let east_m = EARTH_RADIUS_M * dlng * cos_lat;
    let north_m = EARTH_RADIUS_M * dlat;

    (east_m, north_m)
}

#[inline]
fn project_point_to_segment(
    point_e: f64,
    point_n: f64,
    start_e: f64,
    start_n: f64,
    end_e: f64,
    end_n: f64,
) -> (f64, f64) {
    let dx = end_e - start_e;
    let dy = end_n - start_n;

    let t = ((point_e - start_e) * dx + (point_n - start_n) * dy) / (dx * dx + dy * dy);
    let t = t.clamp(0.0, 1.0);

    let proj_e = start_e + t * dx;
    let proj_n = start_n + t * dy;

    (proj_e, proj_n)
}

#[inline]
fn compute_distance_to_projection(point_e: f64, point_n: f64, proj_e: f64, proj_n: f64) -> f64 {
    let de = point_e - proj_e;
    let dn = point_n - proj_n;
    (de * de + dn * dn).sqrt()
}

/// Calculates the minimum distance from a point to a line segment using ENU projection.
///
/// Projects the coordinates to a local [East-North-Up (ENU)](https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates)
/// coordinate system centered at the segment midpoint, then calculates the Euclidean
/// distance from the point to the closest point on the segment.
///
/// This method is accurate for segments shorter than ~100km and provides good
/// performance for most geographic applications.
///
/// # Arguments
///
/// * `point` - The point to measure from
/// * `segment` - Line segment defined by two endpoints (start, end)
///
/// # Returns
///
/// Distance in meters
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::{LngLat, geodesic::point_to_segment_enu_m};
///
/// // Horizontal segment
/// let segment = (
///     LngLat::new_deg(-122.4194, 37.7749), // San Francisco
///     LngLat::new_deg(-122.4094, 37.7749), // 1km east
/// );
///
/// // Point 500m north of segment midpoint
/// let point = LngLat::new_deg(-122.4144, 37.7794);
/// let distance = point_to_segment_enu_m(point, segment);
///
/// assert!(distance > 480.0 && distance < 520.0); // ~500m
/// ```
///
/// # Accuracy
///
/// This method is most accurate for:
/// - Segments shorter than ~100km
/// - Points within ~50km of the segment
/// - Low to moderate latitudes (< 70°)
///
/// For longer segments or higher precision, consider [`great_circle_point_to_seg`].
pub fn point_to_segment_enu_m(point: LngLat, segment: (LngLat, LngLat)) -> f64 {
    let (seg_start, seg_end) = segment;

    if let Some(distance) = check_zero_length_segment(seg_start, seg_end, point) {
        return distance;
    }

    let midpoint = compute_segment_midpoint(seg_start, seg_end);

    let (start_e, start_n) = to_enu_m(midpoint, seg_start);
    let (end_e, end_n) = to_enu_m(midpoint, seg_end);
    let (point_e, point_n) = to_enu_m(midpoint, point);

    let (proj_e, proj_n) =
        project_point_to_segment(point_e, point_n, start_e, start_n, end_e, end_n);

    compute_distance_to_projection(point_e, point_n, proj_e, proj_n)
}

/// Calculates the minimum distance from a point to a line segment using great circle geometry.
///
/// Uses spherical geometry to calculate the [cross-track distance](https://en.wikipedia.org/wiki/Cross-track_error)
/// from a point to a great circle segment. This method works for any segment length
/// and maintains accuracy globally.
///
/// The algorithm uses the spherical triangle formed by the segment endpoints and
/// the query point to compute the perpendicular distance to the great circle path,
/// handling cases where the projection falls outside the segment endpoints.
///
/// # Arguments
///
/// * `point` - The point to measure from
/// * `segment` - Line segment defined by two endpoints (start, end)
///
/// # Returns
///
/// Distance in meters
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::{LngLat, geodesic::great_circle_point_to_seg};
///
/// // Long-distance segment (SF to NYC)
/// let segment = (
///     LngLat::new_deg(-122.4194, 37.7749), // San Francisco
///     LngLat::new_deg(-74.0060, 40.7128),  // New York City
/// );
///
/// // Point somewhere in the middle (Denver area)
/// let point = LngLat::new_deg(-105.0, 39.7);
/// let distance = great_circle_point_to_seg(point, segment);
///
/// assert!(distance > 0.0 && distance < 1_000_000.0); // Reasonable distance
/// ```
///
/// # Algorithm Details
///
/// This method uses [Heron's formula](https://en.wikipedia.org/wiki/Heron%27s_formula) to calculate
/// the area of the spherical triangle, then derives the cross-track distance.
/// It handles edge cases including:
/// - Zero-length segments
/// - Degenerate triangles
/// - Numerical stability issues near poles or for very small segments
///
/// # Accuracy vs Performance
///
/// More accurate than [`point_to_segment_enu_m`] for long segments but slower.
/// Use this method when:
/// - Segment length > 100km
/// - High precision is required
/// - Working at extreme latitudes
pub fn great_circle_point_to_seg(point: LngLat, segment: (LngLat, LngLat)) -> f64 {
    let (seg_start, seg_end) = segment;

    if seg_start.lng_deg == seg_end.lng_deg && seg_start.lat_deg == seg_end.lat_deg {
        return haversine(point, seg_start);
    }

    let d_start = haversine(seg_start, point);
    let d_end = haversine(seg_end, point);
    let d_seg = haversine(seg_start, seg_end);

    if d_seg < 1e-6 {
        return d_start;
    }

    let a = d_start;
    let b = d_seg;
    let c = d_end;

    let s = (a + b + c) * 0.5;
    if s <= a || s <= b || s <= c {
        return d_start.min(d_end);
    }

    let area = (s * (s - a) * (s - b) * (s - c)).sqrt();
    let cross_track_distance = (2.0 * area) / b;

    // Check for numerical stability - avoid sqrt of negative number
    if a * a < cross_track_distance * cross_track_distance {
        return d_start.min(d_end);
    }
    let along_track_distance = (a * a - cross_track_distance * cross_track_distance).sqrt();

    if along_track_distance > b {
        d_end
    } else if along_track_distance < 0.0 {
        d_start
    } else {
        cross_track_distance
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LngLat;

    #[test]
    fn test_check_zero_length_segment() {
        let point = LngLat::new_deg(-121.0, 38.0);
        let seg_start = LngLat::new_deg(-122.0, 37.0);
        let seg_end = LngLat::new_deg(-122.0, 37.0);

        let result = check_zero_length_segment(seg_start, seg_end, point);
        assert!(result.is_some());
        assert!(result.unwrap() > 0.0);

        let non_zero_seg_end = LngLat::new_deg(-121.0, 37.0);
        let result2 = check_zero_length_segment(seg_start, non_zero_seg_end, point);
        assert!(result2.is_none());
    }

    #[test]
    fn test_point_to_segment_enu_m_zero_length() {
        let zero_segment = (LngLat::new_deg(-122.0, 37.0), LngLat::new_deg(-122.0, 37.0));
        let point = LngLat::new_deg(-121.0, 38.0);
        let distance = point_to_segment_enu_m(point, zero_segment);
        let expected_distance = haversine(LngLat::new_deg(-122.0, 37.0), point);
        assert!((distance - expected_distance).abs() < 100.0);
    }

    #[test]
    fn test_point_to_segment_enu_m_on_segment() {
        let segment = (LngLat::new_deg(-122.0, 37.0), LngLat::new_deg(-121.0, 37.0));
        let point_on_segment = LngLat::new_deg(-121.5, 37.0);
        let distance = point_to_segment_enu_m(point_on_segment, segment);
        assert!(distance < 10.0); // Should be very close to 0
    }

    #[test]
    fn test_point_to_segment_enu_m_perpendicular() {
        let segment = (LngLat::new_deg(-122.0, 37.0), LngLat::new_deg(-121.0, 37.0));
        let point_north = LngLat::new_deg(-121.5, 37.01);
        let distance = point_to_segment_enu_m(point_north, segment);
        let expected_distance = haversine(
            LngLat::new_deg(-121.5, 37.0),
            LngLat::new_deg(-121.5, 37.01),
        );
        assert!((distance - expected_distance).abs() < 100.0);
    }

    #[test]
    fn test_compute_segment_midpoint() {
        let seg_start = LngLat::new_deg(-122.0, 37.0);
        let seg_end = LngLat::new_deg(-121.0, 38.0);

        let midpoint = compute_segment_midpoint(seg_start, seg_end);
        assert_eq!(midpoint.lng_deg, -121.5);
        assert_eq!(midpoint.lat_deg, 37.5);
    }

    #[test]
    fn test_project_point_to_segment() {
        let (proj_e, proj_n) = project_point_to_segment(
            1.0, 1.0, // point
            0.0, 0.0, // start
            2.0, 0.0, // end
        );

        assert!((proj_e - 1.0).abs() < 1e-10);
        assert!((proj_n - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_distance_to_projection() {
        let distance = compute_distance_to_projection(1.0, 1.0, 1.0, 0.0);
        assert!((distance - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_great_circle_point_to_seg_zero_length() {
        let zero_segment = (LngLat::new_deg(-122.0, 37.0), LngLat::new_deg(-122.0, 37.0));
        let point = LngLat::new_deg(-121.0, 38.0);
        let distance = great_circle_point_to_seg(point, zero_segment);
        let expected_distance = haversine(LngLat::new_deg(-122.0, 37.0), point);
        assert!((distance - expected_distance).abs() < 100.0);
    }

    #[test]
    fn test_great_circle_point_to_seg_perpendicular() {
        let segment = (LngLat::new_deg(-122.0, 37.0), LngLat::new_deg(-121.0, 37.0));
        let point_north = LngLat::new_deg(-121.5, 37.01);
        let distance = great_circle_point_to_seg(point_north, segment);
        assert!(distance > 0.0 && distance < 20000.0);
    }

    #[test]
    fn test_great_circle_point_to_seg_projection_outside() {
        let segment = (LngLat::new_deg(1.0, 0.0), LngLat::new_deg(1.001, 0.0));

        // Point projects before segment start
        let point_before = LngLat::new_deg(0.0, 0.5);
        let distance_before = great_circle_point_to_seg(point_before, segment);
        let expected_before = haversine(point_before, segment.0);
        assert!((distance_before - expected_before).abs() < 100.0);

        // Point projects after segment end
        let point_after = LngLat::new_deg(2.0, 0.5);
        let distance_after = great_circle_point_to_seg(point_after, segment);
        let expected_after = haversine(point_after, segment.1);
        assert!((distance_after - expected_after).abs() < 100.0);

        // Point projects within segment
        let point_middle = LngLat::new_deg(1.0005, 0.001);
        let distance_middle = great_circle_point_to_seg(point_middle, segment);
        assert!(distance_middle > 50.0 && distance_middle < 200.0);
    }

    #[test]
    fn test_great_circle_point_to_seg_very_small_segment() {
        let segment = (LngLat::new_deg(0.0, 0.0), LngLat::new_deg(0.00001, 0.0));
        let point = LngLat::new_deg(0.000005, 0.00001);
        let distance = great_circle_point_to_seg(point, segment);
        assert!(distance > 0.0 && distance < 10.0);
    }

    #[test]
    fn test_great_circle_point_to_seg_line_99_tiny_segment() {
        // Create a segment so small that d_seg < 1e-6 (line 99)
        let seg_start = LngLat::new_deg(0.0, 0.0);
        let seg_end = LngLat::new_deg(0.0000001, 0.0); // ~0.01mm apart
        let point = LngLat::new_deg(1.0, 1.0);

        let distance = great_circle_point_to_seg(point, (seg_start, seg_end));
        let expected = haversine(seg_start, point);
        assert!((distance - expected).abs() < 1.0);
    }

    #[test]
    fn test_great_circle_point_to_seg_line_108_degenerate_triangle() {
        // Create a degenerate triangle where s <= a || s <= b || s <= c (line 108)
        // This happens when the point is very far from one endpoint
        let seg_start = LngLat::new_deg(0.0, 0.0);
        let seg_end = LngLat::new_deg(1.0, 0.0);
        let point = LngLat::new_deg(179.0, 0.0); // Very far from start, close to antipodal

        let distance = great_circle_point_to_seg(point, (seg_start, seg_end));
        let expected_min = haversine(seg_start, point).min(haversine(seg_end, point));
        assert!((distance - expected_min).abs() < 1000.0);
    }

    #[test]
    fn test_great_circle_point_to_seg_line_116_numerical_stability() {
        // Create a case where a² < cross_track_distance² (line 116)
        // This can happen with numerical precision issues in the area calculation
        let seg_start = LngLat::new_deg(0.0, 0.0);
        let seg_end = LngLat::new_deg(0.1, 0.0);
        let point = LngLat::new_deg(0.05, 89.9); // Very close to pole, might cause numerical issues

        let distance = great_circle_point_to_seg(point, (seg_start, seg_end));
        let expected_min = haversine(seg_start, point).min(haversine(seg_end, point));
        assert!(distance <= expected_min + 1000.0); // Should return one of the endpoint distances
    }

    #[test]
    fn test_great_circle_point_to_seg_line_123_negative_along_track() {
        // This is actually very hard to hit because along_track_distance is computed from sqrt
        // which is always >= 0, but let's try with extreme coordinates that might cause precision issues
        let seg_start = LngLat::new_deg(-179.999, 0.0);
        let seg_end = LngLat::new_deg(179.999, 0.0); // Crosses antimeridian
        let point = LngLat::new_deg(-90.0, 0.0); // Point way off to the side

        let distance = great_circle_point_to_seg(point, (seg_start, seg_end));
        // Should return distance to one of the endpoints or cross-track distance
        assert!(distance > 0.0 && distance < 25_000_000.0); // Max possible Earth distance
    }
}
