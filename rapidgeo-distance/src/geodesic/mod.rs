//! Geodesic (Earth-aware) distance calculations.
//!
//! This module provides distance calculation functions that account for the Earth's
//! curvature and ellipsoidal shape. All calculations assume the [WGS84 ellipsoid](https://en.wikipedia.org/wiki/World_Geodetic_System).
//!
//! # Algorithms
//!
//! - [`haversine()`]: Fast spherical approximation using the [Haversine formula](https://en.wikipedia.org/wiki/Haversine_formula), ±0.5% accuracy for distances <1000km
//! - [`vincenty_distance_m`]: High-precision ellipsoidal calculation using [Vincenty's formulae](https://en.wikipedia.org/wiki/Vincenty%27s_formulae), ±1mm accuracy globally
//! - Point-to-segment functions for calculating distances from points to line segments
//!
//! # Algorithm Selection Guide
//!
//! | Algorithm | Speed | Accuracy | Use When |
//! |-----------|-------|----------|----------|
//! | [Haversine](https://en.wikipedia.org/wiki/Haversine_formula) | Fast | ±0.5% | Short distances (<1000km), performance critical |
//! | [Vincenty](https://en.wikipedia.org/wiki/Vincenty%27s_formulae) | Slow | ±1mm | High precision needed, any distance |
//!
//! # WGS84 Ellipsoid Parameters
//!
//! The geodesic calculations use the [World Geodetic System 1984 (WGS84)](https://en.wikipedia.org/wiki/World_Geodetic_System) ellipsoid:
//! - Semi-major axis (a): 6,378,137 meters
//! - Semi-minor axis (b): 6,356,752.314245 meters  
//! - Flattening (f): 1/298.257223563
//!
//! # Vincenty Algorithm Limitations
//!
//! The [Vincenty algorithm](https://en.wikipedia.org/wiki/Vincenty%27s_formulae) may fail to converge for nearly antipodal points (opposite
//! sides of Earth). When this occurs, [`VincentyError::DidNotConverge`] is returned.
//! Consider using [`haversine()`] as a fallback for such cases.

pub mod haversine;
pub mod point_to_segment;
pub mod vincenty;

pub use haversine::haversine;
pub use point_to_segment::{great_circle_point_to_seg, point_to_segment_enu_m};
pub use vincenty::{vincenty_distance_m, VincentyError};

/// WGS84 flattening factor
const F: f64 = 1.0 / 298.257_223_563;

/// Computes reduced latitudes and their trigonometric values for both coordinate points.
///
/// This shared computation is used by both Haversine and Vincenty algorithms to transform
/// geodetic latitudes to reduced latitudes on the auxiliary sphere, accounting for Earth's
/// ellipsoidal shape.
///
/// # Arguments
///
/// * `lat1_rad` - First latitude in radians
/// * `lat2_rad` - Second latitude in radians
///
/// # Returns
///
/// Tuple of (u1, u2, sin_u1, cos_u1, sin_u2, cos_u2) where:
/// - u1, u2 are the reduced latitudes in radians
/// - sin_u1, cos_u1, sin_u2, cos_u2 are the trigonometric values
pub(crate) fn compute_reduced_latitudes_and_trig(
    lat1_rad: f64,
    lat2_rad: f64,
) -> (f64, f64, f64, f64, f64, f64) {
    let u1 = ((1.0 - F) * lat1_rad.tan()).atan();
    let u2 = ((1.0 - F) * lat2_rad.tan()).atan();
    let sin_u1 = u1.sin();
    let cos_u1 = u1.cos();
    let sin_u2 = u2.sin();
    let cos_u2 = u2.cos();
    (u1, u2, sin_u1, cos_u1, sin_u2, cos_u2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_reduced_latitudes_and_trig() {
        let lat1_rad = 0.6588; // ~37.7 degrees
        let lat2_rad = 0.7084; // ~40.7 degrees

        let (u1, u2, sin_u1, cos_u1, sin_u2, cos_u2) =
            compute_reduced_latitudes_and_trig(lat1_rad, lat2_rad);

        // Verify reduced latitudes are computed correctly
        let expected_u1 = ((1.0 - F) * lat1_rad.tan()).atan();
        let expected_u2 = ((1.0 - F) * lat2_rad.tan()).atan();
        assert!((u1 - expected_u1).abs() < 1e-15);
        assert!((u2 - expected_u2).abs() < 1e-15);

        // Verify trigonometric values
        assert!((sin_u1 - u1.sin()).abs() < 1e-15);
        assert!((cos_u1 - u1.cos()).abs() < 1e-15);
        assert!((sin_u2 - u2.sin()).abs() < 1e-15);
        assert!((cos_u2 - u2.cos()).abs() < 1e-15);

        // Verify fundamental trig identity
        assert!((sin_u1 * sin_u1 + cos_u1 * cos_u1 - 1.0).abs() < 1e-15);
        assert!((sin_u2 * sin_u2 + cos_u2 * cos_u2 - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_compute_reduced_latitudes_equatorial() {
        let (u1, u2, sin_u1, cos_u1, sin_u2, cos_u2) = compute_reduced_latitudes_and_trig(0.0, 0.0);

        // At equator, reduced latitude should be 0
        assert!(u1.abs() < 1e-15);
        assert!(u2.abs() < 1e-15);
        assert!(sin_u1.abs() < 1e-15);
        assert!(sin_u2.abs() < 1e-15);
        assert!((cos_u1 - 1.0).abs() < 1e-15);
        assert!((cos_u2 - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_compute_reduced_latitudes_polar() {
        let lat_polar = std::f64::consts::PI / 2.0 - 1e-8; // Very close to north pole
        let (u1, _u2, sin_u1, cos_u1, _sin_u2, _cos_u2) =
            compute_reduced_latitudes_and_trig(lat_polar, 0.0);

        // Near pole, reduced latitude should be close to but less than π/2
        assert!(u1 < std::f64::consts::PI / 2.0);
        assert!(u1 > std::f64::consts::PI / 2.0 - 0.01);
        assert!(sin_u1 > 0.99);
        assert!(cos_u1 > 0.0 && cos_u1 < 0.1);
    }
}
