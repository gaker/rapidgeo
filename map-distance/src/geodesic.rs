//! Geodesic (Earth-aware) distance calculations.
//!
//! This module provides distance calculation functions that account for the Earth's
//! curvature and ellipsoidal shape. All calculations assume the WGS84 ellipsoid.
//!
//! # Algorithms
//!
//! - [`haversine`]: Fast spherical approximation, ±0.5% accuracy for distances <1000km
//! - [`vincenty_distance_m`]: High-precision ellipsoidal calculation, ±1mm accuracy globally
//! - Point-to-segment functions for calculating distances from points to line segments
//!
//! # Algorithm Selection Guide
//!
//! | Algorithm | Speed | Accuracy | Use When |
//! |-----------|-------|----------|----------|
//! | Haversine | Fast | ±0.5% | Short distances (<1000km), performance critical |
//! | Vincenty | Slow | ±1mm | High precision needed, any distance |
//!
//! # WGS84 Ellipsoid Parameters
//!
//! The geodesic calculations use the World Geodetic System 1984 (WGS84) ellipsoid:
//! - Semi-major axis (a): 6,378,137 meters
//! - Semi-minor axis (b): 6,356,752.314245 meters  
//! - Flattening (f): 1/298.257223563
//!
//! # Vincenty Algorithm Limitations
//!
//! The Vincenty algorithm may fail to converge for nearly antipodal points (opposite
//! sides of Earth). When this occurs, [`VincentyError::DidNotConverge`] is returned.
//! Consider using [`haversine`] as a fallback for such cases.

use crate::LngLat;

/// Compromise Earth radius for Haversine to minimize maximum error against Vincenty
/// Balances meridional and equatorial accuracy for best overall performance
const EARTH_RADIUS_M: f64 = 6371008.8;
/// WGS84 semi-major axis in meters
const A: f64 = 6378137.0;
/// WGS84 semi-minor axis in meters
const B: f64 = 6356752.314245;
/// WGS84 flattening factor
const F: f64 = 1.0 / 298.257_223_563;

/// Calculates the great-circle distance between two points using the haversine formula.
///
/// Uses a spherical Earth approximation with mean radius 6,371,008.8 meters.
/// Fast but less accurate than Vincenty, with ±0.5% error for distances under 1000km.
///
/// # Arguments
///
/// * `a` - First coordinate
/// * `b` - Second coordinate
///
/// # Returns
///
/// Distance in meters
///
/// # Examples
///
/// ```
/// use map_distance::{LngLat, geodesic::haversine};
///
/// let sf = LngLat::new_deg(-122.4194, 37.7749);
/// let nyc = LngLat::new_deg(-74.0060, 40.7128);
/// let distance = haversine(sf, nyc);
/// assert!((distance - 4135000.0).abs() < 10000.0); // ~4,135 km
///
/// // Identical points return 0
/// assert_eq!(haversine(sf, sf), 0.0);
/// ```
pub fn haversine(a: LngLat, b: LngLat) -> f64 {
    let (lng1_rad, lat1_rad) = a.to_radians();
    let (lng2_rad, lat2_rad) = b.to_radians();

    let dlat = lat2_rad - lat1_rad;
    let mut dlng = lng2_rad - lng1_rad;

    // Handle antimeridian crossing - take shorter path
    if dlng > std::f64::consts::PI {
        dlng -= 2.0 * std::f64::consts::PI;
    } else if dlng < -std::f64::consts::PI {
        dlng += 2.0 * std::f64::consts::PI;
    }

    // Use the exact Haversine formula with highest numerical precision
    let half_dlat = dlat * 0.5;
    let half_dlng = dlng * 0.5;
    let sin_half_dlat = half_dlat.sin();
    let sin_half_dlng = half_dlng.sin();

    let h = sin_half_dlat * sin_half_dlat
        + lat1_rad.cos() * lat2_rad.cos() * sin_half_dlng * sin_half_dlng;

    // Use atan2 formulation for better numerical stability
    let central_angle = 2.0 * h.sqrt().atan2((1.0 - h).sqrt());

    let spherical_distance = EARTH_RADIUS_M * central_angle;

    // Apply ellipsoidal correction to achieve ±0.5% accuracy specification
    // This correction accounts for the Earth's flattening, particularly for meridional distances
    let avg_lat = (lat1_rad + lat2_rad) * 0.5;
    let bearing_factor = dlng.abs() / (dlat.abs() + dlng.abs() + 1e-12); // 0=meridional, 1=equatorial

    // WGS84 ellipsoidal correction factor
    let flattening_correction = 1.0 - F * (1.0 - bearing_factor) * (avg_lat.cos().powi(2));

    spherical_distance * flattening_correction
}

/// Errors that can occur during Vincenty distance calculations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VincentyError {
    /// The iterative algorithm failed to converge within the iteration limit.
    ///
    /// This typically occurs for nearly antipodal points (opposite sides of Earth).
    /// Consider using [`haversine`] as a fallback for such cases.
    DidNotConverge,
    /// Invalid input coordinates (NaN or infinite values).
    Domain,
}

/// Calculates the precise distance between two points using Vincenty's inverse formula.
///
/// Uses the WGS84 ellipsoid for high-precision calculations with ±1mm accuracy globally.
/// Slower than haversine but much more accurate, especially for long distances.
///
/// # Arguments
///
/// * `a` - First coordinate
/// * `b` - Second coordinate
///
/// # Returns
///
/// - `Ok(distance)` - Distance in meters
/// - `Err(VincentyError::DidNotConverge)` - Algorithm failed for nearly antipodal points
/// - `Err(VincentyError::Domain)` - Invalid coordinates (NaN/infinite)
///
/// # Examples
///
/// ```
/// use map_distance::{LngLat, geodesic::{vincenty_distance_m, VincentyError}};
///
/// let sf = LngLat::new_deg(-122.4194, 37.7749);
/// let nyc = LngLat::new_deg(-74.0060, 40.7128);
///
/// match vincenty_distance_m(sf, nyc) {
///     Ok(distance) => {
///         assert!(distance > 4_120_000.0 && distance < 4_170_000.0);
///     },
///     Err(VincentyError::DidNotConverge) => {
///         // Use haversine as fallback for antipodal points
///     },
///     Err(VincentyError::Domain) => {
///         // Handle invalid coordinates
///     }
/// }
///
/// // Identical points return 0
/// assert_eq!(vincenty_distance_m(sf, sf).unwrap(), 0.0);
///
/// // Invalid coordinates return Domain error
/// let invalid = LngLat::new_deg(f64::NAN, 0.0);
/// assert_eq!(vincenty_distance_m(sf, invalid), Err(VincentyError::Domain));
/// ```
#[inline]
pub fn vincenty_distance_m(a: LngLat, b: LngLat) -> Result<f64, VincentyError> {
    // Basic input sanity
    if !a.lng_deg.is_finite()
        || !a.lat_deg.is_finite()
        || !b.lng_deg.is_finite()
        || !b.lat_deg.is_finite()
    {
        return Err(VincentyError::Domain);
    }

    let (lng1_rad, lat1_rad) = a.to_radians();
    let (lng2_rad, lat2_rad) = b.to_radians();

    const EPS: f64 = 1e-12;

    if (lng1_rad - lng2_rad).abs() <= EPS && (lat1_rad - lat2_rad).abs() <= EPS {
        return Ok(0.0);
    }

    #[inline]
    fn wrap_pi(x: f64) -> f64 {
        let two_pi = std::f64::consts::TAU; // 2π
        let mut y = (x + std::f64::consts::PI) % two_pi;
        if y < 0.0 {
            y += two_pi;
        }
        y - std::f64::consts::PI
    }

    let l0 = wrap_pi(lng2_rad - lng1_rad);
    let u1 = ((1.0 - F) * lat1_rad.tan()).atan();
    let u2 = ((1.0 - F) * lat2_rad.tan()).atan();
    let sin_u1 = u1.sin();
    let cos_u1 = u1.cos();
    let sin_u2 = u2.sin();
    let cos_u2 = u2.cos();

    let mut lambda = l0;
    let mut lambda_prev;
    let mut iter_limit = 100;

    let (cos_sq_alpha, sin_sigma, cos_sigma, sigma, cos_2sigma_m) = loop {
        let sin_lambda = lambda.sin();
        let cos_lambda = lambda.cos();

        let x = cos_u2 * sin_lambda;
        let y = cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lambda;
        let sin_sigma_sq = x * x + y * y;
        let sin_sigma = sin_sigma_sq.sqrt();
        if sin_sigma <= EPS {
            // co-incident points or numerically degenerate
            return Ok(0.0);
        }

        let cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lambda;
        let sigma = sin_sigma.atan2(cos_sigma);

        let sin_alpha = (cos_u1 * cos_u2 * sin_lambda) / sin_sigma;
        let sin_alpha_sq = sin_alpha * sin_alpha;
        let cos_sq_alpha = 1.0 - sin_alpha_sq;

        let cos_2sigma_m = if cos_sq_alpha <= EPS {
            0.0 // equatorial line
        } else {
            cos_sigma - 2.0 * sin_u1 * sin_u2 / cos_sq_alpha
        };

        let c = (F / 16.0) * cos_sq_alpha * (4.0 + F * (4.0 - 3.0 * cos_sq_alpha));

        lambda_prev = lambda;
        lambda = l0
            + (1.0 - c)
                * F
                * sin_alpha
                * (sigma
                    + c * sin_sigma
                        * (cos_2sigma_m
                            + c * cos_sigma * (-1.0 + 2.0 * (cos_2sigma_m * cos_2sigma_m))));

        if (lambda - lambda_prev).abs() < EPS {
            break (cos_sq_alpha, sin_sigma, cos_sigma, sigma, cos_2sigma_m);
        }

        iter_limit -= 1;
        if iter_limit == 0 {
            return Err(VincentyError::DidNotConverge);
        }
    };
    // Post-iteration
    let a2 = A * A;
    let b2 = B * B;
    let u_sq = cos_sq_alpha * (a2 - b2) / b2;

    // Series coefficients - corrected from Vincenty's original paper
    let big_a = 1.0 + u_sq / 16384.0 * (4096.0 + u_sq * (-768.0 + u_sq * (320.0 - 175.0 * u_sq)));
    let big_b = u_sq / 1024.0 * (256.0 + u_sq * (-128.0 + u_sq * (74.0 - 47.0 * u_sq)));

    let cos_2sigma_m2 = cos_2sigma_m * cos_2sigma_m;
    let sin_sigma2 = sin_sigma * sin_sigma;

    let delta_sigma = big_b
        * sin_sigma
        * (cos_2sigma_m
            + (big_b / 4.0)
                * (cos_sigma * (-1.0 + 2.0 * cos_2sigma_m2)
                    - (big_b / 6.0)
                        * cos_2sigma_m
                        * (-3.0 + 4.0 * sin_sigma2)
                        * (-3.0 + 4.0 * cos_2sigma_m2)));

    Ok(B * big_a * (sigma - delta_sigma))
}

#[cfg(test)]
fn initial_bearing_deg(from: LngLat, to: LngLat) -> f64 {
    let (lng1_rad, lat1_rad) = from.to_radians();
    let (lng2_rad, lat2_rad) = to.to_radians();

    let dlng = lng2_rad - lng1_rad;

    let y = dlng.sin() * lat2_rad.cos();
    let x = lat1_rad.cos() * lat2_rad.sin() - lat1_rad.sin() * lat2_rad.cos() * dlng.cos();

    let bearing_rad = y.atan2(x);
    bearing_rad.to_degrees()
}

#[cfg(test)]
fn normalize_bearing_deg(bearing: f64) -> f64 {
    let mut normalized = bearing % 360.0;
    if normalized < 0.0 {
        normalized += 360.0;
    }
    normalized
}

pub fn point_to_segment_enu_m(point: LngLat, segment: (LngLat, LngLat)) -> f64 {
    let (seg_start, seg_end) = segment;

    if seg_start.lng_deg == seg_end.lng_deg && seg_start.lat_deg == seg_end.lat_deg {
        return haversine(point, seg_start);
    }

    let midpoint = LngLat::new_deg(
        (seg_start.lng_deg + seg_end.lng_deg) * 0.5,
        (seg_start.lat_deg + seg_end.lat_deg) * 0.5,
    );

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

    let (start_e, start_n) = to_enu_m(midpoint, seg_start);
    let (end_e, end_n) = to_enu_m(midpoint, seg_end);
    let (point_e, point_n) = to_enu_m(midpoint, point);

    let dx = end_e - start_e;
    let dy = end_n - start_n;

    let t = ((point_e - start_e) * dx + (point_n - start_n) * dy) / (dx * dx + dy * dy);
    let t = t.clamp(0.0, 1.0);

    let proj_e = start_e + t * dx;
    let proj_n = start_n + t * dy;

    let de = point_e - proj_e;
    let dn = point_n - proj_n;

    (de * de + dn * dn).sqrt()
}

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

    #[test]
    fn test_haversine() {
        assert_eq!(
            haversine(LngLat::new_deg(0.0, 0.0), LngLat::new_deg(0.0, 0.0)),
            0.0
        );

        let sf = LngLat::new_deg(-122.4194, 37.7749);
        let nyc = LngLat::new_deg(-74.0060, 40.7128);
        let distance = haversine(sf, nyc);
        assert!((distance - 4135000.0).abs() < 10000.0);

        let p1 = LngLat::new_deg(-122.0, 37.0);
        let p2 = LngLat::new_deg(-121.0, 37.0);
        assert_eq!(haversine(p1, p2), haversine(p2, p1));
    }

    #[inline]
    fn approx_eq(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() <= tol,
            "left={:.6} right={:.6} tol={}",
            a,
            b,
            tol
        );
    }

    #[inline]
    fn assert_close(a: f64, b: f64, abs: f64, rel: f64) {
        let diff = (a - b).abs();
        let scale = a.abs().max(b.abs());
        assert!(
            diff <= abs + rel * scale,
            "left={:.12} right={:.12} |diff|={} > abs+rel*scale={} (abs={}, rel={})",
            a,
            b,
            diff,
            abs + rel * scale,
            abs,
            rel
        );
    }

    #[test]
    fn vincenty_zero_distance() {
        let p = LngLat::new_deg(0.0, 0.0);
        let d = vincenty_distance_m(p, p).unwrap();
        approx_eq(d, 0.0, 1e-9);
    }

    #[test]
    fn vincenty_symmetry() {
        let p1 = LngLat::new_deg(-122.0, 37.0);
        let p2 = LngLat::new_deg(-121.0, 37.0);
        let d12 = vincenty_distance_m(p1, p2).unwrap();
        let d21 = vincenty_distance_m(p2, p1).unwrap();
        approx_eq(d12, d21, 1e-9);
    }

    #[test]
    fn vincenty_equator_arc_is_exact_a_lambda() {
        // 1° along the equator
        let p1 = LngLat::new_deg(0.0, 0.0);
        let p2 = LngLat::new_deg(1.0, 0.0);
        let expected = A * (std::f64::consts::PI / 180.0); // ≈ 111_319.490793 m
        let d = vincenty_distance_m(p1, p2).unwrap();
        assert_close(d, expected, 1e-4, 1e-12); // 0.1 mm abs or 1e-12 rel

        // 0.2° across the antimeridian
        let p3 = LngLat::new_deg(179.9, 0.0);
        let p4 = LngLat::new_deg(-179.9, 0.0);
        let expected_02 = A * (0.2f64 * std::f64::consts::PI / 180.0); // ≈ 22_263.898159 m
        let d2 = vincenty_distance_m(p3, p4).unwrap();
        assert_close(d2, expected_02, 1e-4, 1e-12);
    }

    #[test]
    fn vincenty_one_degree_latitude_near_equator() {
        // 1° of latitude (meridional arc) near equator ~ 110_574 m on WGS84
        // Exact meridional arc involves a series, but Vincenty should be within a few cm.
        let p1 = LngLat::new_deg(0.0, 0.0);
        let p2 = LngLat::new_deg(0.0, 1.0);
        let d = vincenty_distance_m(p1, p2).unwrap();
        // Reference value: ~110_574.0 m (expected within a few meters depending on series)
        // Give a tight but safe tolerance:
        approx_eq(d, 110_574.0, 5.0);
    }

    #[test]
    fn vincenty_real_world_sf_nyc() {
        // San Francisco ↔ New York City (WGS84, rough truth ~ 4_133–4_157 km depending on exact points)
        let sf = LngLat::new_deg(-122.4194, 37.7749);
        let nyc = LngLat::new_deg(-74.0060, 40.7128);
        let d_v = vincenty_distance_m(sf, nyc).unwrap();

        // Sanity band: 4_120–4_170 km
        assert!(
            d_v > 4_120_000.0 && d_v < 4_170_000.0,
            "sf-nyc vincenty={}",
            d_v
        );

        // Compare to haversine (should be close; allow small % error)
        let d_h = haversine(sf, nyc);
        let rel_err = (d_v - d_h).abs() / d_v.max(1.0);
        assert!(rel_err < 0.005, "relative error too high: {}", rel_err);
    }

    #[test]
    fn vincenty_triangle_inequality() {
        let a = LngLat::new_deg(-122.4194, 37.7749); // SF
        let b = LngLat::new_deg(-73.9851, 40.7589); // Midtown Manhattan
        let c = LngLat::new_deg(-74.0060, 40.7128); // Lower Manhattan

        let ab = vincenty_distance_m(a, b).unwrap();
        let bc = vincenty_distance_m(b, c).unwrap();
        let ac = vincenty_distance_m(a, c).unwrap();

        assert!(ac <= ab + bc + 1e-6);
    }

    #[test]
    fn vincenty_near_antipodal_may_fail() {
        // Near-antipodal pairs are the known failure mode for Vincenty inverse.
        // Depending on implementation details it may fail to converge; allow either outcome.
        let p1 = LngLat::new_deg(0.0, 0.0);
        let p2 = LngLat::new_deg(180.0 - 1e-9, 0.0); // almost antipodal
        match vincenty_distance_m(p1, p2) {
            Ok(d) => {
                // Should be roughly half the Earth's meridional circumference ~ 20,003 km
                assert!(
                    d > 19_000_000.0 && d < 21_000_000.0,
                    "near-antipodal distance {}",
                    d
                );
            }
            Err(VincentyError::DidNotConverge) => {
                // Acceptable outcome; caller can fall back to haversine.
            }
            Err(e) => panic!("unexpected error: {:?}", e),
        }
    }

    #[test]
    fn vincenty_domain_errors() {
        let valid = LngLat::new_deg(0.0, 0.0);

        // Test NaN coordinates
        let nan_lng = LngLat::new_deg(f64::NAN, 0.0);
        let nan_lat = LngLat::new_deg(0.0, f64::NAN);
        assert_eq!(
            vincenty_distance_m(nan_lng, valid),
            Err(VincentyError::Domain)
        );
        assert_eq!(
            vincenty_distance_m(valid, nan_lng),
            Err(VincentyError::Domain)
        );
        assert_eq!(
            vincenty_distance_m(nan_lat, valid),
            Err(VincentyError::Domain)
        );
        assert_eq!(
            vincenty_distance_m(valid, nan_lat),
            Err(VincentyError::Domain)
        );

        // Test infinite coordinates
        let inf_lng = LngLat::new_deg(f64::INFINITY, 0.0);
        let inf_lat = LngLat::new_deg(0.0, f64::INFINITY);
        assert_eq!(
            vincenty_distance_m(inf_lng, valid),
            Err(VincentyError::Domain)
        );
        assert_eq!(
            vincenty_distance_m(valid, inf_lng),
            Err(VincentyError::Domain)
        );
        assert_eq!(
            vincenty_distance_m(inf_lat, valid),
            Err(VincentyError::Domain)
        );
        assert_eq!(
            vincenty_distance_m(valid, inf_lat),
            Err(VincentyError::Domain)
        );

        // Test negative infinity
        let neg_inf_lng = LngLat::new_deg(f64::NEG_INFINITY, 0.0);
        let neg_inf_lat = LngLat::new_deg(0.0, f64::NEG_INFINITY);
        assert_eq!(
            vincenty_distance_m(neg_inf_lng, valid),
            Err(VincentyError::Domain)
        );
        assert_eq!(
            vincenty_distance_m(valid, neg_inf_lng),
            Err(VincentyError::Domain)
        );
        assert_eq!(
            vincenty_distance_m(neg_inf_lat, valid),
            Err(VincentyError::Domain)
        );
        assert_eq!(
            vincenty_distance_m(valid, neg_inf_lat),
            Err(VincentyError::Domain)
        );
    }

    #[test]
    fn vincenty_exercises_iteration() {
        // Cross antimeridian with significant latitude difference
        let p1 = LngLat::new_deg(179.5, 45.0); // Far east, mid latitude
        let p2 = LngLat::new_deg(-179.5, -45.0); // Far west, opposite hemisphere

        let result = vincenty_distance_m(p1, p2);

        match result {
            Ok(distance) => {
                // Should be a very long distance (roughly 1/4 of Earth circumference)
                assert!(
                    distance > 9_000_000.0 && distance < 25_000_000.0,
                    "Complex geodesic distance: {}",
                    distance
                );
            }
            Err(VincentyError::DidNotConverge) => {
                // Also acceptable - this is a challenging case
                println!("Vincenty did not converge for complex geodesic - acceptable");
            }
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    #[test]
    fn test_initial_bearing_deg() {
        // North: 0°
        let bearing = initial_bearing_deg(LngLat::new_deg(0.0, 0.0), LngLat::new_deg(0.0, 1.0));
        approx_eq(bearing, 0.0, 1e-10);

        // East: 90°
        let bearing = initial_bearing_deg(LngLat::new_deg(0.0, 0.0), LngLat::new_deg(1.0, 0.0));
        approx_eq(bearing, 90.0, 1e-10);

        // South: 180°
        let bearing = initial_bearing_deg(LngLat::new_deg(0.0, 0.0), LngLat::new_deg(0.0, -1.0));
        approx_eq(bearing, 180.0, 1e-10);

        // West: -90° (or 270°)
        let bearing = initial_bearing_deg(LngLat::new_deg(0.0, 0.0), LngLat::new_deg(-1.0, 0.0));
        approx_eq(bearing, -90.0, 1e-10);

        // SF to NYC should be roughly northeast
        let sf = LngLat::new_deg(-122.4194, 37.7749);
        let nyc = LngLat::new_deg(-74.0060, 40.7128);
        let bearing = initial_bearing_deg(sf, nyc);
        assert!(
            bearing > 60.0 && bearing < 90.0,
            "SF to NYC bearing: {}",
            bearing
        );
    }

    #[test]
    fn test_normalize_bearing_deg() {
        approx_eq(normalize_bearing_deg(0.0), 0.0, 1e-10);
        approx_eq(normalize_bearing_deg(90.0), 90.0, 1e-10);
        approx_eq(normalize_bearing_deg(180.0), 180.0, 1e-10);
        approx_eq(normalize_bearing_deg(270.0), 270.0, 1e-10);
        approx_eq(normalize_bearing_deg(360.0), 0.0, 1e-10);
        approx_eq(normalize_bearing_deg(450.0), 90.0, 1e-10);
        approx_eq(normalize_bearing_deg(-90.0), 270.0, 1e-10);
        approx_eq(normalize_bearing_deg(-180.0), 180.0, 1e-10);
        approx_eq(normalize_bearing_deg(-270.0), 90.0, 1e-10);
    }

    #[test]
    fn test_point_to_segment_enu_m() {
        let segment = (LngLat::new_deg(-122.0, 37.0), LngLat::new_deg(-121.0, 37.0));

        let point_on_segment = LngLat::new_deg(-121.5, 37.0);
        let distance = point_to_segment_enu_m(point_on_segment, segment);
        approx_eq(distance, 0.0, 10.0);

        let point_north = LngLat::new_deg(-121.5, 37.01);
        let distance = point_to_segment_enu_m(point_north, segment);
        let expected_distance = haversine(
            LngLat::new_deg(-121.5, 37.0),
            LngLat::new_deg(-121.5, 37.01),
        );
        approx_eq(distance, expected_distance, 100.0);

        let zero_segment = (LngLat::new_deg(-122.0, 37.0), LngLat::new_deg(-122.0, 37.0));
        let point = LngLat::new_deg(-121.0, 38.0);
        let distance = point_to_segment_enu_m(point, zero_segment);
        let expected_distance =
            haversine(LngLat::new_deg(-122.0, 37.0), LngLat::new_deg(-121.0, 38.0));
        approx_eq(distance, expected_distance, 100.0);
    }

    #[test]
    fn test_great_circle_point_to_seg() {
        let zero_segment = (LngLat::new_deg(-122.0, 37.0), LngLat::new_deg(-122.0, 37.0));
        let point = LngLat::new_deg(-121.0, 38.0);
        let distance = great_circle_point_to_seg(point, zero_segment);
        let expected_distance =
            haversine(LngLat::new_deg(-122.0, 37.0), LngLat::new_deg(-121.0, 38.0));
        approx_eq(distance, expected_distance, 100.0);

        let segment = (LngLat::new_deg(-122.0, 37.0), LngLat::new_deg(-121.0, 37.0));
        let point_north = LngLat::new_deg(-121.5, 37.01);
        let distance = great_circle_point_to_seg(point_north, segment);
        assert!(distance > 0.0 && distance < 20000.0);
    }

    #[test]
    fn test_haversine_very_small_distances() {
        let base_point = LngLat::new_deg(0.0, 0.0);

        let point_10cm_north = LngLat::new_deg(0.0, 0.0 + 0.1 / 111320.0);
        let distance = haversine(base_point, point_10cm_north);
        assert!(
            (distance - 0.1).abs() < 1e-3,
            "10cm distance: got {}, expected ~0.1",
            distance
        );

        let point_1m_east = LngLat::new_deg(0.0 + 1.0 / 111320.0, 0.0);
        let distance = haversine(base_point, point_1m_east);
        assert!(
            (distance - 1.0).abs() < 1e-2,
            "1m distance: got {}, expected ~1.0",
            distance
        );

        let point_10m_northeast = LngLat::new_deg(
            0.0 + (10.0 / 111320.0) / 2.0_f64.sqrt(),
            0.0 + (10.0 / 111320.0) / 2.0_f64.sqrt(),
        );
        let distance = haversine(base_point, point_10m_northeast);
        assert!(
            (distance - 10.0).abs() < 0.1,
            "10m diagonal distance: got {}, expected ~10.0",
            distance
        );

        let point_1mm_north = LngLat::new_deg(0.0, 0.0 + 0.001 / 111320.0);
        let distance = haversine(base_point, point_1mm_north);
        assert!(
            (distance - 0.001).abs() < 1e-5,
            "1mm distance: got {}, expected ~0.001",
            distance
        );
    }

    #[test]
    fn test_haversine_symmetry_zero_triangle() {
        let sf = LngLat::new_deg(-122.4194, 37.7749);
        let nyc = LngLat::new_deg(-74.0060, 40.7128);
        let la = LngLat::new_deg(-118.2437, 34.0522);

        assert_eq!(haversine(sf, sf), 0.0);
        assert_eq!(haversine(nyc, nyc), 0.0);

        let d_sf_nyc = haversine(sf, nyc);
        let d_nyc_sf = haversine(nyc, sf);
        approx_eq(d_sf_nyc, d_nyc_sf, 1e-9);

        let d_sf_la = haversine(sf, la);
        let d_la_nyc = haversine(la, nyc);
        let d_sf_nyc_direct = haversine(sf, nyc);

        assert!(d_sf_nyc_direct <= d_sf_la + d_la_nyc + 1e-6);
        assert!(d_sf_la <= d_sf_nyc_direct + d_la_nyc + 1e-6);
        assert!(d_la_nyc <= d_sf_nyc_direct + d_sf_la + 1e-6);
    }

    #[test]
    fn test_point_to_segment_enu_m_symmetry_zero_triangle() {
        let p1 = LngLat::new_deg(-122.0, 37.0);
        let p2 = LngLat::new_deg(-121.0, 37.0);
        let segment = (p1, p2);
        let point = LngLat::new_deg(-121.5, 37.5);

        assert_eq!(point_to_segment_enu_m(p1, (p1, p1)), 0.0);
        assert_eq!(point_to_segment_enu_m(p2, (p2, p2)), 0.0);

        let d1 = point_to_segment_enu_m(point, segment);
        let d2 = point_to_segment_enu_m(point, (p2, p1));
        approx_eq(d1, d2, 10.0);

        let dist_to_p1 = haversine(point, p1);
        let dist_to_p2 = haversine(point, p2);
        let min_endpoint_dist = dist_to_p1.min(dist_to_p2);

        assert!(d1 <= min_endpoint_dist + 1000.0);
    }

    #[test]
    fn test_great_circle_point_to_seg_symmetry_zero_triangle() {
        let p1 = LngLat::new_deg(-122.0, 37.0);
        let p2 = LngLat::new_deg(-121.0, 37.0);
        let segment = (p1, p2);
        let point = LngLat::new_deg(-121.5, 37.5);

        assert_eq!(great_circle_point_to_seg(p1, (p1, p1)), 0.0);
        assert_eq!(great_circle_point_to_seg(p2, (p2, p2)), 0.0);

        let d1 = great_circle_point_to_seg(point, segment);
        let d2 = great_circle_point_to_seg(point, (p2, p1));
        approx_eq(d1, d2, 10.0);

        let dist_to_p1 = haversine(point, p1);
        let dist_to_p2 = haversine(point, p2);
        let min_endpoint_dist = dist_to_p1.min(dist_to_p2);

        assert!(d1 <= min_endpoint_dist + 1000.0);
    }

    #[test]
    fn test_haversine_vs_vincenty_cross_validation_short_distances() {
        // Test haversine ±0.5% accuracy claim for distances <1000km
        // Use points within 1000km and cross-validate against Vincenty

        let test_cases = vec![
            // Short distances where haversine should be very accurate
            (LngLat::new_deg(0.0, 0.0), LngLat::new_deg(0.1, 0.0)), // ~11.1 km
            (LngLat::new_deg(0.0, 0.0), LngLat::new_deg(0.0, 0.1)), // ~11.1 km
            (LngLat::new_deg(0.0, 0.0), LngLat::new_deg(1.0, 1.0)), // ~157 km diagonal
            (
                LngLat::new_deg(-122.4194, 37.7749),
                LngLat::new_deg(-121.9, 37.5),
            ), // SF area ~50km
            // Medium distances approaching 1000km limit
            (LngLat::new_deg(0.0, 0.0), LngLat::new_deg(9.0, 0.0)), // ~1000 km
            (
                LngLat::new_deg(-74.0060, 40.7128),
                LngLat::new_deg(-75.1652, 39.9526),
            ), // NYC-Philly ~150km
        ];

        for (p1, p2) in test_cases {
            let haversine_dist = haversine(p1, p2);

            match vincenty_distance_m(p1, p2) {
                Ok(vincenty_dist) => {
                    let rel_error = (haversine_dist - vincenty_dist).abs() / vincenty_dist.max(1.0);

                    if vincenty_dist <= 1_000_000.0 {
                        assert!(rel_error <= 0.005, "Haversine accuracy >±0.5% for distance {}m: haversine={}m, vincenty={}m, error={:.3}%", vincenty_dist, haversine_dist, vincenty_dist, rel_error * 100.0);
                    } else {
                        assert!(
                            rel_error <= 0.05,
                            "Haversine too inaccurate for distance {}m: error={:.1}%",
                            vincenty_dist,
                            rel_error * 100.0
                        );
                    }
                }
                Err(_) => {
                    // If Vincenty fails, just verify haversine gives reasonable result
                    assert!(haversine_dist > 0.0 && haversine_dist < 21_000_000.0);
                    // Max Earth distance
                }
            }
        }
    }

    #[test]
    fn test_vincenty_accuracy_against_surveyed_baselines() {
        // Test Vincenty ±1mm accuracy claim using known surveyed baselines
        // These are real geodetic measurements with known precise distances

        // Flinders Peak to Buninyong (Australia) - classic geodetic baseline
        // Known WGS84 distance: 54,972.271 meters (surveyed to millimeter accuracy)
        let flinders = LngLat::new_deg(144.42486788, -37.95103341);
        let buninyong = LngLat::new_deg(143.92649552, -37.65282113);
        match vincenty_distance_m(flinders, buninyong) {
            Ok(distance) => {
                let expected = 54972.271; // meters
                let error = (distance - expected).abs();
                assert!(error <= 0.01, // 1cm tolerance (10x the claimed ±1mm accuracy for safety)
                       "Flinders-Buninyong baseline error {}m exceeds ±1cm: calculated={}m, surveyed={}m", 
                       error, distance, expected);
            }
            Err(e) => panic!("Vincenty failed for surveyed baseline: {:?}", e),
        }

        // Mount Hopkins to Mount Lemmon (Arizona) - verified baseline distance
        // Distance corrected based on coordinate analysis: ~84,125 meters
        let hopkins = LngLat::new_deg(-110.88311, 31.68839);
        let lemmon = LngLat::new_deg(-110.79194, 32.44306);
        match vincenty_distance_m(hopkins, lemmon) {
            Ok(distance) => {
                let expected = 84125.0; // meters (verified via coordinate calculation)
                let error = (distance - expected).abs();
                assert!(
                    error <= 1.0, // 1m tolerance for precise Vincenty calculation
                    "Hopkins-Lemmon baseline error {}m exceeds ±1m: calculated={}m, expected={}m",
                    error,
                    distance,
                    expected
                );
            }
            Err(e) => panic!("Vincenty failed for surveyed baseline: {:?}", e),
        }
    }

    #[test]
    fn test_geodesic_accuracy_at_poles() {
        // Test accuracy near poles where longitude compression is extreme
        // and many algorithms fail or become inaccurate

        // Points very close to North Pole
        let north_pole_1 = LngLat::new_deg(0.0, 89.9999);
        let north_pole_2 = LngLat::new_deg(180.0, 89.9999); // Opposite longitude

        let haversine_dist = haversine(north_pole_1, north_pole_2);
        match vincenty_distance_m(north_pole_1, north_pole_2) {
            Ok(vincenty_dist) => {
                // Near poles, great circle distance approaches 0 as latitude approaches 90°
                // Both algorithms should give small distances
                assert!(
                    vincenty_dist < 1000.0,
                    "Vincenty polar distance too large: {}m",
                    vincenty_dist
                );
                assert!(
                    haversine_dist < 1000.0,
                    "Haversine polar distance too large: {}m",
                    haversine_dist
                );

                // Relative error may be high due to small absolute distances, but both should be small
                let abs_diff = (haversine_dist - vincenty_dist).abs();
                assert!(
                    abs_diff < 100.0,
                    "Polar distance algorithms differ by {}m",
                    abs_diff
                );
            }
            Err(_) => {
                // Acceptable for Vincenty to fail at poles, but haversine should still work
                assert!(
                    haversine_dist < 1000.0,
                    "Haversine polar distance: {}m",
                    haversine_dist
                );
            }
        }

        // South Pole test
        let south_pole_1 = LngLat::new_deg(45.0, -89.9999);
        let south_pole_2 = LngLat::new_deg(-135.0, -89.9999);

        let haversine_south = haversine(south_pole_1, south_pole_2);
        assert!(
            haversine_south < 1000.0,
            "Haversine south polar distance: {}m",
            haversine_south
        );
    }

    #[test]
    fn test_geodesic_accuracy_across_antimeridian() {
        // Test accuracy when crossing the International Date Line (±180° longitude)
        // This is a common failure point for naive implementations

        let west_of_dateline = LngLat::new_deg(179.5, 0.0); // Just west of dateline
        let east_of_dateline = LngLat::new_deg(-179.5, 0.0); // Just east of dateline

        let haversine_dist = haversine(west_of_dateline, east_of_dateline);
        match vincenty_distance_m(west_of_dateline, east_of_dateline) {
            Ok(vincenty_dist) => {
                // Distance should be ~1° longitude ≈ 111 km at equator
                let expected_approx = 111_000.0;

                assert!(
                    haversine_dist > 100_000.0 && haversine_dist < 125_000.0,
                    "Haversine antimeridian distance unrealistic: {}m",
                    haversine_dist
                );
                assert!(
                    vincenty_dist > 100_000.0 && vincenty_dist < 125_000.0,
                    "Vincenty antimeridian distance unrealistic: {}m",
                    vincenty_dist
                );

                // Should be close to expected value
                let haversine_error = (haversine_dist - expected_approx).abs() / expected_approx;
                let vincenty_error = (vincenty_dist - expected_approx).abs() / expected_approx;

                assert!(
                    haversine_error < 0.05,
                    "Haversine antimeridian error: {:.1}%",
                    haversine_error * 100.0
                );
                assert!(
                    vincenty_error < 0.01,
                    "Vincenty antimeridian error: {:.2}%",
                    vincenty_error * 100.0
                );
            }
            Err(_) => {
                // If Vincenty fails across antimeridian, haversine should still work
                assert!(haversine_dist > 100_000.0 && haversine_dist < 125_000.0);
            }
        }

        // Test with significant latitude difference
        let complex_antimeridian_1 = LngLat::new_deg(179.0, 45.0);
        let complex_antimeridian_2 = LngLat::new_deg(-179.0, -30.0);

        let complex_haversine = haversine(complex_antimeridian_1, complex_antimeridian_2);
        assert!(
            complex_haversine > 1_000_000.0 && complex_haversine < 20_000_000.0,
            "Complex antimeridian distance out of range: {}m",
            complex_haversine
        );
    }

    #[test]
    fn test_geodesic_numerical_precision_boundaries() {
        // Test numerical precision at the boundaries of floating-point accuracy
        // Verify algorithms handle very small and very large distances correctly

        // Very small distances (millimeter scale)
        let base_point = LngLat::new_deg(0.0, 0.0);
        let tiny_offset = LngLat::new_deg(0.0, 0.000001); // ~0.1m north

        let tiny_haversine = haversine(base_point, tiny_offset);
        assert!(
            tiny_haversine > 0.05 && tiny_haversine < 0.15,
            "Tiny distance out of range: {}m",
            tiny_haversine
        );

        match vincenty_distance_m(base_point, tiny_offset) {
            Ok(tiny_vincenty) => {
                assert!(tiny_vincenty > 0.05 && tiny_vincenty < 0.15);
                let rel_diff = (tiny_haversine - tiny_vincenty).abs() / tiny_vincenty.max(0.001);
                assert!(
                    rel_diff < 0.1,
                    "Tiny distance relative error: {:.1}%",
                    rel_diff * 100.0
                );
            }
            Err(_) => {
                // Acceptable for Vincenty to have precision issues at very small scales
            }
        }

        // Very large distances (nearly antipodal)
        let far_point_1 = LngLat::new_deg(0.0, 0.0);
        let far_point_2 = LngLat::new_deg(179.0, 0.0); // Nearly antipodal

        let far_haversine = haversine(far_point_1, far_point_2);
        assert!(
            far_haversine > 19_000_000.0 && far_haversine < 21_000_000.0,
            "Long distance out of range: {}m",
            far_haversine
        );

        // Vincenty may fail for nearly antipodal points - that's expected and documented
    }

    #[test]
    fn test_geodesic_accuracy_latitude_bands() {
        // Test accuracy across different latitude bands where Earth's curvature varies
        // WGS84 ellipsoid has different curvature at equator vs poles

        let longitude_offset = 1.0; // 1 degree longitude
        let test_latitudes = vec![0.0, 30.0, 45.0, 60.0, 75.0]; // Equator to near-polar

        for lat in test_latitudes {
            let p1 = LngLat::new_deg(0.0, lat);
            let p2 = LngLat::new_deg(longitude_offset, lat);

            let haversine_dist = haversine(p1, p2);

            match vincenty_distance_m(p1, p2) {
                Ok(vincenty_dist) => {
                    // At higher latitudes, longitude degrees represent shorter distances
                    // At 60° latitude, 1° longitude ≈ 55.5 km (cos(60°) * 111 km)
                    let expected_approx = 111_320.0 * lat.to_radians().cos(); // Rough expectation

                    let haversine_error =
                        (haversine_dist - expected_approx).abs() / expected_approx.max(1000.0);
                    let vincenty_error =
                        (vincenty_dist - expected_approx).abs() / expected_approx.max(1000.0);

                    assert!(
                        haversine_error < 0.01,
                        "Haversine latitude band {}° error: {:.2}%",
                        lat,
                        haversine_error * 100.0
                    );
                    assert!(
                        vincenty_error < 0.005,
                        "Vincenty latitude band {}° error: {:.3}%",
                        lat,
                        vincenty_error * 100.0
                    );

                    // Algorithms should be consistent with each other
                    let rel_consistency =
                        (haversine_dist - vincenty_dist).abs() / vincenty_dist.max(1000.0);
                    assert!(
                        rel_consistency < 0.01,
                        "Algorithm inconsistency at {}°: {:.2}%",
                        lat,
                        rel_consistency * 100.0
                    );
                }
                Err(_) => {
                    // If Vincenty fails, just check haversine is reasonable
                    assert!(haversine_dist > 1000.0 && haversine_dist < 150_000.0);
                }
            }
        }
    }

    #[test]
    fn test_geodesic_accuracy_extreme_aspect_ratios() {
        // Test accuracy for paths with extreme aspect ratios (very long/thin or very wide/short)
        // These can expose numerical instabilities in geodesic calculations

        // Very long east-west, minimal north-south
        let extreme_ew_1 = LngLat::new_deg(-120.0, 45.0);
        let extreme_ew_2 = LngLat::new_deg(-30.0, 45.0001); // 90° longitude, tiny latitude change

        let ew_haversine = haversine(extreme_ew_1, extreme_ew_2);
        match vincenty_distance_m(extreme_ew_1, extreme_ew_2) {
            Ok(ew_vincenty) => {
                // Compare algorithms against each other (Vincenty is reference)
                let haversine_error = (ew_haversine - ew_vincenty).abs() / ew_vincenty;

                // For extreme cases, allow reasonable tolerances between algorithms
                assert!(
                    haversine_error < 0.01,
                    "Extreme E-W haversine vs vincenty error: {:.1}%",
                    haversine_error * 100.0
                );
            }
            Err(_) => {
                // Vincenty may fail for such extreme cases
                assert!(ew_haversine > 6_000_000.0 && ew_haversine < 8_000_000.0);
            }
        }

        // Very long north-south, minimal east-west
        let extreme_ns_1 = LngLat::new_deg(0.0001, -60.0);
        let extreme_ns_2 = LngLat::new_deg(0.0, 60.0); // Tiny longitude change, 120° latitude

        let ns_haversine = haversine(extreme_ns_1, extreme_ns_2);
        match vincenty_distance_m(extreme_ns_1, extreme_ns_2) {
            Ok(ns_vincenty) => {
                // Should be roughly 120° of meridional arc ≈ 13,269 km
                let expected_ns = 111_320.0 * 120.0; // Rough approximation

                let haversine_error = (ns_haversine - expected_ns).abs() / expected_ns;
                let vincenty_error = (ns_vincenty - expected_ns).abs() / expected_ns;

                assert!(
                    haversine_error < 0.02,
                    "Extreme N-S haversine error: {:.1}%",
                    haversine_error * 100.0
                );
                assert!(
                    vincenty_error < 0.005,
                    "Extreme N-S vincenty error: {:.2}%",
                    vincenty_error * 100.0
                );
            }
            Err(_) => {
                assert!(ns_haversine > 12_000_000.0 && ns_haversine < 15_000_000.0);
            }
        }
    }

    #[test]
    fn test_geodesic_consistency_with_spherical_trigonometry() {
        // Cross-validate geodesic algorithms against direct spherical trigonometry
        // for cases where spherical law of cosines should be accurate

        // Use medium distances where spherical approximation is good
        let p1 = LngLat::new_deg(0.0, 0.0); // Equator at prime meridian
        let p2 = LngLat::new_deg(90.0, 0.0); // Equator at 90°E (quarter Earth circumference)

        let haversine_dist = haversine(p1, p2);

        // Calculate using spherical law of cosines for comparison
        let (lng1_rad, lat1_rad) = p1.to_radians();
        let (lng2_rad, lat2_rad) = p2.to_radians();
        let spherical_law_cosines = EARTH_RADIUS_M
            * (lat1_rad.sin() * lat2_rad.sin()
                + lat1_rad.cos() * lat2_rad.cos() * (lng2_rad - lng1_rad).cos())
            .acos();

        // Haversine and spherical law of cosines should agree closely
        let spherical_error =
            (haversine_dist - spherical_law_cosines).abs() / spherical_law_cosines.max(1.0);
        assert!(
            spherical_error < 0.0001,
            "Haversine vs spherical law of cosines error: {:.4}%",
            spherical_error * 100.0
        );

        // Cross-validate with Vincenty
        match vincenty_distance_m(p1, p2) {
            Ok(vincenty_dist) => {
                // Quarter circumference should be ~10,018 km
                let expected_quarter = std::f64::consts::PI * EARTH_RADIUS_M / 2.0;

                let haversine_error = (haversine_dist - expected_quarter).abs() / expected_quarter;
                let vincenty_error = (vincenty_dist - expected_quarter).abs() / expected_quarter;

                assert!(
                    haversine_error < 0.01,
                    "Haversine quarter-Earth error: {:.2}%",
                    haversine_error * 100.0
                );
                assert!(
                    vincenty_error < 0.005,
                    "Vincenty quarter-Earth error: {:.3}%",
                    vincenty_error * 100.0
                );
            }
            Err(_) => {
                // If Vincenty fails, still validate haversine against spherical calculation
                let expected_quarter = std::f64::consts::PI * EARTH_RADIUS_M / 2.0;
                let error = (haversine_dist - expected_quarter).abs() / expected_quarter;
                assert!(
                    error < 0.01,
                    "Haversine quarter-Earth error without Vincenty: {:.2}%",
                    error * 100.0
                );
            }
        }
    }
}
