//! Polyline simplification integration with Douglas-Peucker algorithm.

use crate::{decode, encode, LngLat, PolylineResult};
use rapidgeo_simplify::{simplify_dp_into, SimplifyMethod};

/// Simplifies a polyline string using Douglas-Peucker algorithm.
///
/// This function decodes the polyline, applies simplification, then re-encodes it.
/// This is a common workflow for reducing polyline complexity while preserving shape.
///
/// # Arguments
///
/// * `polyline` - The polyline string to simplify
/// * `tolerance_m` - Simplification tolerance in meters
/// * `method` - Distance calculation method for simplification
/// * `precision` - Precision for encoding/decoding (typically 5 or 6)
///
/// # Examples
///
/// ```
/// use rapidgeo_polyline::simplify_polyline;
/// use rapidgeo_simplify::SimplifyMethod;
///
/// let polyline = "_p~iF~ps|U_ulLnnqC_mqNvxq`@";
/// let simplified = simplify_polyline(polyline, 1000.0, SimplifyMethod::GreatCircleMeters, 5).unwrap();
/// ```
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

/// Simplifies a sequence of coordinates using Douglas-Peucker algorithm.
///
/// # Arguments
///
/// * `coordinates` - The coordinates to simplify
/// * `tolerance_m` - Simplification tolerance in meters
/// * `method` - Distance calculation method for simplification
///
/// # Examples
///
/// ```
/// use rapidgeo_polyline::simplify_coordinates;
/// use rapidgeo_simplify::SimplifyMethod;
/// use rapidgeo_distance::LngLat;
///
/// let coords = vec![
///     LngLat::new_deg(-120.2, 38.5),
///     LngLat::new_deg(-120.95, 40.7),
///     LngLat::new_deg(-126.453, 43.252),
/// ];
/// let simplified = simplify_coordinates(&coords, 1000.0, SimplifyMethod::GreatCircleMeters);
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

/// Encodes coordinates directly to a simplified polyline.
///
/// This function applies simplification during encoding, which can be more efficient
/// than encode -> decode -> simplify -> encode workflow.
///
/// # Arguments
///
/// * `coordinates` - The coordinates to encode and simplify
/// * `tolerance_m` - Simplification tolerance in meters
/// * `method` - Distance calculation method for simplification
/// * `precision` - Precision for encoding (typically 5 or 6)
///
/// # Examples
///
/// ```
/// use rapidgeo_polyline::encode_simplified;
/// use rapidgeo_simplify::SimplifyMethod;
/// use rapidgeo_distance::LngLat;
///
/// let coords = vec![
///     LngLat::new_deg(-120.2, 38.5),
///     LngLat::new_deg(-120.4, 38.6),
///     LngLat::new_deg(-120.6, 38.7),
///     LngLat::new_deg(-120.95, 40.7),
/// ];
/// let simplified_polyline = encode_simplified(&coords, 1000.0, SimplifyMethod::GreatCircleMeters, 5).unwrap();
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
