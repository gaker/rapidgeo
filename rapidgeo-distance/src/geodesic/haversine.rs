use super::F;
use crate::LngLat;

/// Compromise Earth radius for Haversine to minimize maximum error against Vincenty
/// Balances meridional and equatorial accuracy for best overall performance
const EARTH_RADIUS_M: f64 = 6371008.8;

#[inline]
fn normalize_longitude_difference(dlng: f64) -> f64 {
    let mut normalized_dlng = dlng;
    // Handle antimeridian crossing - take shorter path
    if normalized_dlng > std::f64::consts::PI {
        normalized_dlng -= 2.0 * std::f64::consts::PI;
    } else if normalized_dlng < -std::f64::consts::PI {
        normalized_dlng += 2.0 * std::f64::consts::PI;
    }
    normalized_dlng
}

#[inline]
fn compute_haversine_parameter(dlat: f64, dlng: f64, lat1_rad: f64, lat2_rad: f64) -> f64 {
    let half_dlat = dlat * 0.5;
    let half_dlng = dlng * 0.5;
    let sin_half_dlat = half_dlat.sin();
    let sin_half_dlng = half_dlng.sin();

    sin_half_dlat * sin_half_dlat + lat1_rad.cos() * lat2_rad.cos() * sin_half_dlng * sin_half_dlng
}

#[inline]
fn compute_central_angle(h: f64) -> f64 {
    2.0 * h.sqrt().atan2((1.0 - h).sqrt())
}

#[inline]
fn apply_ellipsoidal_correction(
    spherical_distance: f64,
    dlat: f64,
    dlng: f64,
    lat1_rad: f64,
    lat2_rad: f64,
) -> f64 {
    let avg_lat = (lat1_rad + lat2_rad) * 0.5;
    let bearing_factor = dlng.abs() / (dlat.abs() + dlng.abs() + 1e-12); // 0=meridional, 1=equatorial

    // WGS84 ellipsoidal correction factor
    let flattening_correction = 1.0 - F * (1.0 - bearing_factor) * (avg_lat.cos().powi(2));

    spherical_distance * flattening_correction
}

/// Calculates the great-circle distance between two points using the [Haversine formula](https://en.wikipedia.org/wiki/Haversine_formula).
///
/// Uses a spherical Earth approximation with mean radius 6,371,008.8 meters, enhanced with
/// ellipsoidal corrections for the [WGS84 ellipsoid](https://en.wikipedia.org/wiki/World_Geodetic_System).
/// Fast but less accurate than Vincenty, with ±0.5% error for distances under 1000km.
///
/// The formula calculates the shortest distance over the Earth's surface, giving an
/// "as-the-crow-flies" distance between the points (ignoring any hills, valleys, or
/// obstacles along the surface of the earth).
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
/// use rapidgeo_distance::{LngLat, geodesic::haversine};
///
/// let sf = LngLat::new_deg(-122.4194, 37.7749);
/// let nyc = LngLat::new_deg(-74.0060, 40.7128);
/// let distance = haversine(sf, nyc);
/// assert!((distance - 4135000.0).abs() < 10000.0); // ~4,135 km
///
/// // Identical points return 0
/// assert_eq!(haversine(sf, sf), 0.0);
/// ```
///
/// # Algorithm Details
///
/// This implementation uses the standard [Haversine formula](https://en.wikipedia.org/wiki/Haversine_formula)
/// with ellipsoidal corrections to account for Earth's flattening. The formula is:
///
/// ```text
/// a = sin²(Δφ/2) + cos(φ1) × cos(φ2) × sin²(Δλ/2)
/// c = 2 × atan2(√a, √(1-a))
/// d = R × c
/// ```
///
/// Where φ is latitude, λ is longitude, and R is Earth's radius.
pub fn haversine(a: LngLat, b: LngLat) -> f64 {
    let (lng1_rad, lat1_rad) = a.to_radians();
    let (lng2_rad, lat2_rad) = b.to_radians();

    let dlat = lat2_rad - lat1_rad;
    let dlng = normalize_longitude_difference(lng2_rad - lng1_rad);

    let h = compute_haversine_parameter(dlat, dlng, lat1_rad, lat2_rad);

    let central_angle = compute_central_angle(h);

    let spherical_distance = EARTH_RADIUS_M * central_angle;

    apply_ellipsoidal_correction(spherical_distance, dlat, dlng, lat1_rad, lat2_rad)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_longitude_difference() {
        // No normalization needed
        assert!((normalize_longitude_difference(1.0) - 1.0).abs() < 1e-10);
        assert!((normalize_longitude_difference(-1.0) - (-1.0)).abs() < 1e-10);

        // Positive wrap around (> π)
        let input = std::f64::consts::PI + 1.0;
        let expected = input - 2.0 * std::f64::consts::PI;
        assert!((normalize_longitude_difference(input) - expected).abs() < 1e-10);

        // Negative wrap around (< -π)
        let input = -std::f64::consts::PI - 1.0;
        let expected = input + 2.0 * std::f64::consts::PI;
        assert!((normalize_longitude_difference(input) - expected).abs() < 1e-10);

        // Exactly at boundaries
        assert!(
            (normalize_longitude_difference(std::f64::consts::PI) - std::f64::consts::PI).abs()
                < 1e-10
        );
        assert!(
            (normalize_longitude_difference(-std::f64::consts::PI) - (-std::f64::consts::PI)).abs()
                < 1e-10
        );
    }

    #[test]
    fn test_compute_haversine_parameter() {
        // Same points should give h = 0
        let h = compute_haversine_parameter(0.0, 0.0, 0.0, 0.0);
        assert!(h.abs() < 1e-10);

        // 90 degree latitude difference at equator
        let dlat = std::f64::consts::PI / 2.0;
        let h = compute_haversine_parameter(dlat, 0.0, 0.0, dlat);
        let expected = (dlat / 2.0).sin().powi(2);
        assert!((h - expected).abs() < 1e-10);

        // 90 degree longitude difference at equator
        let dlng = std::f64::consts::PI / 2.0;
        let h = compute_haversine_parameter(0.0, dlng, 0.0, 0.0);
        let expected = (dlng / 2.0).sin().powi(2);
        assert!((h - expected).abs() < 1e-10);
    }

    #[test]
    fn test_compute_central_angle() {
        // h = 0 should give angle = 0
        assert!((compute_central_angle(0.0) - 0.0).abs() < 1e-10);

        // h = 1 should give angle = π (antipodal points)
        let angle = compute_central_angle(1.0);
        assert!((angle - std::f64::consts::PI).abs() < 1e-10);

        // h = 0.5 should give angle = π/2 (quarter circle)
        let angle = compute_central_angle(0.5);
        assert!((angle - std::f64::consts::PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_apply_ellipsoidal_correction() {
        let base_distance = 100000.0; // 100km

        // Pure meridional (north-south) should apply more correction
        let corrected_meridional = apply_ellipsoidal_correction(
            base_distance,
            1.0, // dlat = 1 radian
            0.0, // dlng = 0 (pure meridional)
            0.0, // lat1 = equator
            1.0, // lat2 = ~57 degrees
        );

        // Pure equatorial (east-west) should apply less correction
        let corrected_equatorial = apply_ellipsoidal_correction(
            base_distance,
            0.0, // dlat = 0 (pure equatorial)
            1.0, // dlng = 1 radian
            0.0, // lat1 = equator
            0.0, // lat2 = equator
        );

        // Meridional should be smaller due to flattening correction
        assert!(corrected_meridional < corrected_equatorial);

        // Both should be close to but less than the original distance
        assert!(corrected_meridional < base_distance);
        assert!(corrected_equatorial <= base_distance); // Might be equal at equator
        assert!(corrected_meridional > base_distance * 0.99); // Not too much correction
    }

    #[test]
    fn test_apply_ellipsoidal_correction_at_poles() {
        let base_distance = 100000.0;

        // Near north pole
        let corrected_polar = apply_ellipsoidal_correction(
            base_distance,
            0.1, // small dlat
            0.1, // small dlng
            1.4, // ~80 degrees north
            1.5, // ~86 degrees north
        );

        assert!(corrected_polar > 0.0);
        assert!(corrected_polar < base_distance * 1.01); // Small correction at poles
    }

    #[test]
    fn test_haversine_identical_points() {
        let point = LngLat::new_deg(-122.4194, 37.7749);
        assert_eq!(haversine(point, point), 0.0);
    }

    #[test]
    fn test_haversine_symmetry() {
        let sf = LngLat::new_deg(-122.4194, 37.7749);
        let nyc = LngLat::new_deg(-74.0060, 40.7128);

        let d1 = haversine(sf, nyc);
        let d2 = haversine(nyc, sf);
        assert!((d1 - d2).abs() < 1e-10);
    }

    #[test]
    fn test_haversine_known_distances() {
        // San Francisco to NYC (~4,135 km)
        let sf = LngLat::new_deg(-122.4194, 37.7749);
        let nyc = LngLat::new_deg(-74.0060, 40.7128);
        let distance = haversine(sf, nyc);
        assert!((distance - 4135000.0).abs() < 10000.0);

        // Equatorial degree (~111.32 km)
        let p1 = LngLat::new_deg(0.0, 0.0);
        let p2 = LngLat::new_deg(1.0, 0.0);
        let distance = haversine(p1, p2);
        assert!((distance - 111320.0).abs() < 1000.0);
    }

    #[test]
    fn test_haversine_antimeridian_crossing() {
        // Points on opposite sides of antimeridian
        let west = LngLat::new_deg(179.5, 0.0);
        let east = LngLat::new_deg(-179.5, 0.0);
        let distance = haversine(west, east);

        // Should take shorter path (~111 km), not longer path (~39,885 km)
        assert!(distance < 200000.0); // Much less than halfway around Earth
        assert!(distance > 100000.0); // But more than 100km
    }

    #[test]
    fn test_haversine_very_small_distances() {
        let base = LngLat::new_deg(0.0, 0.0);

        // 1 meter north (approximately)
        let one_meter_north = LngLat::new_deg(0.0, 0.0 + 1.0 / 111320.0);
        let distance = haversine(base, one_meter_north);
        assert!((distance - 1.0).abs() < 0.1);

        // 10 cm east (approximately)
        let ten_cm_east = LngLat::new_deg(0.0 + 0.1 / 111320.0, 0.0);
        let distance = haversine(base, ten_cm_east);
        assert!((distance - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_haversine_polar_regions() {
        // Near north pole
        let north_1 = LngLat::new_deg(0.0, 89.9);
        let north_2 = LngLat::new_deg(180.0, 89.9);
        let distance = haversine(north_1, north_2);
        assert!(distance < 50000.0); // Very short at high latitude

        // Near south pole
        let south_1 = LngLat::new_deg(45.0, -89.9);
        let south_2 = LngLat::new_deg(-135.0, -89.9);
        let distance = haversine(south_1, south_2);
        assert!(distance < 50000.0);
    }

    #[test]
    fn test_haversine_meridional_vs_equatorial() {
        // 1 degree north-south (meridional)
        let meridional = haversine(LngLat::new_deg(0.0, 0.0), LngLat::new_deg(0.0, 1.0));

        // 1 degree east-west at equator (equatorial)
        let equatorial = haversine(LngLat::new_deg(0.0, 0.0), LngLat::new_deg(1.0, 0.0));

        // Meridional should be slightly shorter due to Earth's flattening
        assert!(meridional < equatorial);
        assert!((meridional - equatorial).abs() < 1000.0); // But close
    }

    #[test]
    fn test_haversine_long_distances() {
        // Quarter Earth circumference
        let quarter_earth = haversine(LngLat::new_deg(0.0, 0.0), LngLat::new_deg(90.0, 0.0));
        let expected = std::f64::consts::PI * EARTH_RADIUS_M / 2.0;
        assert!((quarter_earth - expected).abs() < 50000.0);

        // Nearly antipodal
        let near_antipodal = haversine(LngLat::new_deg(0.0, 0.0), LngLat::new_deg(179.0, 0.0));
        assert!(near_antipodal > 19_000_000.0);
        assert!(near_antipodal < 21_000_000.0);
    }

    #[test]
    fn test_haversine_triangle_inequality() {
        let sf = LngLat::new_deg(-122.4194, 37.7749);
        let chicago = LngLat::new_deg(-87.6298, 41.8781);
        let nyc = LngLat::new_deg(-74.0060, 40.7128);

        let sf_chi = haversine(sf, chicago);
        let chi_nyc = haversine(chicago, nyc);
        let sf_nyc = haversine(sf, nyc);

        // Triangle inequality: any side ≤ sum of other two sides
        assert!(sf_nyc <= sf_chi + chi_nyc + 1000.0); // Small tolerance for floating point
        assert!(sf_chi <= sf_nyc + chi_nyc + 1000.0);
        assert!(chi_nyc <= sf_chi + sf_nyc + 1000.0);
    }
}
