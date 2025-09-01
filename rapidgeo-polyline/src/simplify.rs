//! Polyline simplification using the [Douglas-Peucker algorithm](https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm).
//!
//! This module integrates `rapidgeo-simplify` to provide coordinate simplification
//! directly within polyline encoding workflows. Useful for reducing GPS track
//! complexity while preserving route shape.

use crate::{decode, encode, LngLat, PolylineResult};
use rapidgeo_simplify::{simplify_dp_into, SimplifyMethod};

/// Simplifies a polyline string using the [Douglas-Peucker algorithm](https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm).
///
/// This function decodes the polyline, applies simplification to reduce coordinate
/// density while preserving shape, then re-encodes it. Useful for reducing storage
/// requirements and transmission bandwidth for GPS tracks and routes.
///
/// The algorithm identifies points that can be removed without significantly
/// affecting the line's shape, based on their perpendicular distance to line segments.
///
/// # Arguments
///
/// * `polyline` - ASCII polyline string to simplify
/// * `tolerance_m` - Simplification tolerance in meters (larger = more aggressive)
/// * `method` - Distance calculation method ([`SimplifyMethod`])
/// * `precision` - Precision for encoding/decoding (typically 5 or 6)
///
/// # Returns
///
/// Returns a simplified polyline string with potentially fewer coordinate points.
/// The first and last coordinates are always preserved.
///
/// # Examples
///
/// ```rust
/// use rapidgeo_polyline::simplify_polyline;
/// use rapidgeo_simplify::SimplifyMethod;
///
/// // Simplify a detailed route with 1km tolerance
/// let detailed = "_p~iF~ps|U_ulLnnqC_mqNvxq`@";
/// let simplified = simplify_polyline(
///     detailed,
///     1000.0, // 1 kilometer tolerance
///     SimplifyMethod::GreatCircleMeters,
///     5
/// )?;
/// # Ok::<(), rapidgeo_polyline::PolylineError>(())
/// ```
///
/// # Distance Methods
///
/// - [`SimplifyMethod::GreatCircleMeters`] - Accurate for global routes
/// - [`SimplifyMethod::PlanarMeters`] - Fast approximation for small areas  
/// - [`SimplifyMethod::EuclidRaw`] - Fastest, coordinate units
pub fn simplify_polyline(
    polyline: &str,
    tolerance_m: f64,
    method: SimplifyMethod,
    precision: u8,
) -> PolylineResult<String> {
    let coordinates = decode(polyline, precision)?;
    let simplified = simplify_coordinates(&coordinates, tolerance_m, method);
    encode(&simplified, precision)
}

/// Simplifies a coordinate sequence using the [Douglas-Peucker algorithm](https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm).
///
/// Reduces the number of points in a coordinate sequence while preserving the
/// overall shape. This is the core simplification function used by other functions
/// in this module.
///
/// # Arguments
///
/// * `coordinates` - Coordinate sequence in longitude, latitude order
/// * `tolerance_m` - Simplification tolerance in meters (larger = more simplification)
/// * `method` - Distance calculation method for determining point significance
///
/// # Returns
///
/// Returns a simplified coordinate sequence. Empty inputs return empty vectors.
/// Single coordinates are returned unchanged. The first and last coordinates
/// are always preserved for multi-point sequences.
///
/// # Examples
///
/// ```rust
/// use rapidgeo_polyline::simplify_coordinates;
/// use rapidgeo_simplify::SimplifyMethod;
/// use rapidgeo_distance::LngLat;
///
/// // GPS track with redundant points
/// let gps_track = vec![
///     LngLat::new_deg(-120.0, 38.0),
///     LngLat::new_deg(-120.001, 38.001),  // Very close to previous
///     LngLat::new_deg(-120.002, 38.002),  // Very close to previous  
///     LngLat::new_deg(-121.0, 39.0),     // Significant change
/// ];
///
/// // Simplify with 100m tolerance
/// let simplified = simplify_coordinates(
///     &gps_track,
///     100.0,
///     SimplifyMethod::GreatCircleMeters
/// );
///
/// // Should remove redundant intermediate points
/// assert!(simplified.len() < gps_track.len());
/// assert_eq!(simplified[0], gps_track[0]); // First preserved
/// assert_eq!(simplified.last(), gps_track.last()); // Last preserved
/// ```
pub fn simplify_coordinates(
    coordinates: &[LngLat],
    tolerance_m: f64,
    method: SimplifyMethod,
) -> Vec<LngLat> {
    if coordinates.is_empty() {
        return Vec::new();
    }

    let mut simplified = Vec::new();
    simplify_dp_into(coordinates, tolerance_m, method, &mut simplified);
    simplified
}

/// Encodes coordinates directly to a simplified polyline string.
///
/// This is more efficient than the encode → decode → simplify → encode workflow
/// when you have raw coordinates and want a simplified polyline result.
///
/// Combines coordinate simplification with polyline encoding in a single operation.
///
/// # Arguments
///
/// * `coordinates` - Raw coordinate sequence in longitude, latitude order
/// * `tolerance_m` - Simplification tolerance in meters (larger = more aggressive)
/// * `method` - Distance calculation method for simplification
/// * `precision` - Decimal places to preserve in encoding (typically 5 or 6)
///
/// # Returns
///
/// Returns a simplified polyline string, or an error if encoding fails.
///
/// # Examples
///
/// ```rust
/// use rapidgeo_polyline::encode_simplified;
/// use rapidgeo_simplify::SimplifyMethod;
/// use rapidgeo_distance::LngLat;
///
/// // Detailed GPS route
/// let gps_route = vec![
///     LngLat::new_deg(-120.0, 38.0),
///     LngLat::new_deg(-120.01, 38.01),   // Close to previous
///     LngLat::new_deg(-120.02, 38.02),   // Close to previous
///     LngLat::new_deg(-120.1, 38.1),     // Significant change
///     LngLat::new_deg(-120.2, 38.0),     // End point
/// ];
///
/// // Encode with 50m simplification tolerance
/// let simplified_polyline = encode_simplified(
///     &gps_route,
///     50.0, // 50 meter tolerance
///     SimplifyMethod::GreatCircleMeters,
///     5
/// )?;
///
/// // Result is more compact than encoding all points
/// # Ok::<(), rapidgeo_polyline::PolylineError>(())
/// ```
///
/// # Use Case: GPS Track Processing
///
/// ```rust
/// use rapidgeo_polyline::encode_simplified;
/// use rapidgeo_simplify::SimplifyMethod;
/// use rapidgeo_distance::LngLat;
///
/// // High-frequency GPS data (1Hz sampling)
/// let high_res_track = vec![
///     LngLat::new_deg(-122.0, 37.0),
///     LngLat::new_deg(-122.0001, 37.0001), // 1 second later, minimal movement
///     LngLat::new_deg(-122.0002, 37.0002), // 2 seconds later, minimal movement
///     // ... many more points
///     LngLat::new_deg(-122.1, 37.1),       // Significant position change
/// ];
///
/// // Create storage-efficient polyline with 10m tolerance
/// let efficient_track = encode_simplified(
///     &high_res_track,
///     10.0, // Remove GPS noise within 10 meters
///     SimplifyMethod::GreatCircleMeters,
///     5
/// )?;
/// # Ok::<(), rapidgeo_polyline::PolylineError>(())
/// ```
pub fn encode_simplified(
    coordinates: &[LngLat],
    tolerance_m: f64,
    method: SimplifyMethod,
    precision: u8,
) -> PolylineResult<String> {
    let simplified = simplify_coordinates(coordinates, tolerance_m, method);
    encode(&simplified, precision)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_coordinates() -> Vec<LngLat> {
        vec![
            LngLat::new_deg(-122.0, 37.0),
            LngLat::new_deg(-122.1, 37.1),
            LngLat::new_deg(-122.2, 37.2),
            LngLat::new_deg(-122.3, 37.1),
            LngLat::new_deg(-122.4, 37.0),
        ]
    }

    #[test]
    fn test_simplify_coordinates() {
        let coords = create_test_coordinates();

        // Test with zero tolerance (should keep all points)
        let simplified_zero = simplify_coordinates(&coords, 0.0, SimplifyMethod::GreatCircleMeters);
        assert_eq!(simplified_zero.len(), coords.len());

        // Test with high tolerance (should keep only endpoints)
        let simplified_high =
            simplify_coordinates(&coords, 100000.0, SimplifyMethod::GreatCircleMeters);
        assert_eq!(simplified_high.len(), 2); // Only start and end points
        assert_eq!(simplified_high[0], coords[0]);
        assert_eq!(simplified_high[1], coords[coords.len() - 1]);
    }

    #[test]
    fn test_simplify_coordinates_empty() {
        let coords: Vec<LngLat> = vec![];
        let simplified = simplify_coordinates(&coords, 1000.0, SimplifyMethod::GreatCircleMeters);
        assert_eq!(simplified.len(), 0);
    }

    #[test]
    fn test_simplify_coordinates_single_point() {
        let coords = vec![LngLat::new_deg(-122.0, 37.0)];
        let simplified = simplify_coordinates(&coords, 1000.0, SimplifyMethod::GreatCircleMeters);
        assert_eq!(simplified.len(), 1);
        assert_eq!(simplified[0], coords[0]);
    }

    #[test]
    fn test_encode_simplified() {
        let coords = create_test_coordinates();

        let result = encode_simplified(&coords, 1000.0, SimplifyMethod::GreatCircleMeters, 5);
        assert!(result.is_ok());

        let simplified_polyline = result.unwrap();
        assert!(!simplified_polyline.is_empty());

        // Decode to verify it's a valid polyline
        let decoded = decode(&simplified_polyline, 5).unwrap();
        assert!(decoded.len() >= 2); // At least start and end points
    }

    #[test]
    fn test_simplify_polyline_roundtrip() {
        let coords = create_test_coordinates();

        // Encode original coordinates
        let original_polyline = encode(&coords, 5).unwrap();

        // Simplify the polyline
        let simplified_polyline = simplify_polyline(
            &original_polyline,
            1000.0,
            SimplifyMethod::GreatCircleMeters,
            5,
        )
        .unwrap();

        // Decode simplified polyline
        let decoded = decode(&simplified_polyline, 5).unwrap();

        // Should have fewer or equal points
        assert!(decoded.len() <= coords.len());
        // Should have at least 2 points (start and end)
        assert!(decoded.len() >= 2);
        // First and last points should be preserved
        assert!((decoded[0].lng_deg - coords[0].lng_deg).abs() < 0.00001);
        assert!((decoded[0].lat_deg - coords[0].lat_deg).abs() < 0.00001);
        assert!((decoded.last().unwrap().lng_deg - coords.last().unwrap().lng_deg).abs() < 0.00001);
        assert!((decoded.last().unwrap().lat_deg - coords.last().unwrap().lat_deg).abs() < 0.00001);
    }

    #[test]
    fn test_different_simplify_methods() {
        let coords = create_test_coordinates();

        for method in [
            SimplifyMethod::GreatCircleMeters,
            SimplifyMethod::PlanarMeters,
            SimplifyMethod::EuclidRaw,
        ] {
            let simplified = simplify_coordinates(&coords, 1000.0, method);
            assert!(simplified.len() >= 2); // At least endpoints
            assert!(simplified.len() <= coords.len());

            // Test encoding with different methods
            let encoded = encode_simplified(&coords, 1000.0, method, 5).unwrap();
            assert!(!encoded.is_empty());
        }
    }

    #[test]
    fn test_simplify_preserves_endpoints() {
        let coords = create_test_coordinates();

        let simplified = simplify_coordinates(&coords, 50000.0, SimplifyMethod::GreatCircleMeters);

        // Should preserve endpoints even with high tolerance
        assert_eq!(simplified[0], coords[0]);
        assert_eq!(simplified.last().unwrap(), coords.last().unwrap());
    }

    #[test]
    fn test_simplify_polyline_error_handling() {
        // Test with invalid polyline (using characters outside valid range)
        let result = simplify_polyline("invalid\x1f", 1000.0, SimplifyMethod::GreatCircleMeters, 5);
        assert!(result.is_err());

        // Test with invalid precision
        let valid_polyline = "_p~iF~ps|U";
        let result =
            simplify_polyline(valid_polyline, 1000.0, SimplifyMethod::GreatCircleMeters, 0);
        assert!(result.is_err());
    }
}
