//! Polyline decoding functions.

use crate::{LngLat, PolylineError, PolylineResult};

/// Validates decoded coordinates for reasonable bounds and finite values
fn validate_decoded_coordinate(coord: &LngLat) -> PolylineResult<()> {
    if !coord.lng_deg.is_finite() || !coord.lat_deg.is_finite() {
        return Err(PolylineError::InvalidCoordinate(
            "Decoded coordinate is NaN or infinity".to_string(),
        ));
    }

    // Allow slightly larger bounds for decoded coordinates to handle precision artifacts
    if coord.lng_deg < -180.1 || coord.lng_deg > 180.1 {
        return Err(PolylineError::InvalidCoordinate(format!(
            "Decoded longitude {} is unreasonable",
            coord.lng_deg
        )));
    }

    if coord.lat_deg < -90.1 || coord.lat_deg > 90.1 {
        return Err(PolylineError::InvalidCoordinate(format!(
            "Decoded latitude {} is unreasonable",
            coord.lat_deg
        )));
    }

    Ok(())
}

/// Decodes a polyline string into a sequence of coordinates using [Google's Polyline Algorithm](https://developers.google.com/maps/documentation/utilities/polylinealgorithm).
///
/// Reverses the encoding process by parsing the ASCII string, extracting delta-encoded
/// coordinate values, and reconstructing the original coordinate sequence.
///
/// # Arguments
///
/// * `polyline` - ASCII polyline string (characters in range 63-126)
/// * `precision` - Decimal places the polyline was encoded with (1-11, typically 5 or 6)
///
/// # Returns
///
/// Returns a vector of coordinates in longitude, latitude order, or an error if
/// the polyline is malformed or precision is invalid.
///
/// # Errors
///
/// - [`PolylineError::InvalidCharacter`] for characters outside valid range (63-126)
/// - [`PolylineError::TruncatedData`] for incomplete or malformed polylines
/// - [`PolylineError::InvalidPrecision`] for precision outside 1-11 range
/// - [`PolylineError::CoordinateOverflow`] during reconstruction
/// - [`PolylineError::InvalidCoordinate`] for decoded coordinates outside reasonable bounds
///
/// # Examples
///
/// ```rust
/// use rapidgeo_polyline::decode;
///
/// // Decode Google's test vector
/// let decoded = decode("_p~iF~ps|U_ulLnnqC_mqNvxq`@", 5)?;
/// assert_eq!(decoded.len(), 3);
///
/// // Check first coordinate
/// assert!((decoded[0].lng_deg - (-120.2)).abs() < 0.00001);
/// assert!((decoded[0].lat_deg - 38.5).abs() < 0.00001);
///
/// // Empty polylines decode to empty vectors
/// let empty = decode("", 5)?;
/// assert_eq!(empty.len(), 0);
/// # Ok::<(), rapidgeo_polyline::PolylineError>(())
/// ```
///
/// # Round-trip with Encoding
///
/// ```rust
/// use rapidgeo_polyline::{encode, decode};
/// use rapidgeo_distance::LngLat;
///
/// let original = vec![
///     LngLat::new_deg(-122.4194, 37.7749),  // San Francisco
///     LngLat::new_deg(-74.0059, 40.7128),   // New York
/// ];
///
/// let encoded = encode(&original, 5)?;
/// let decoded = decode(&encoded, 5)?;
///
/// // Should round-trip with minimal precision loss
/// for (orig, dec) in original.iter().zip(decoded.iter()) {
///     assert!((orig.lng_deg - dec.lng_deg).abs() < 0.00001);
///     assert!((orig.lat_deg - dec.lat_deg).abs() < 0.00001);
/// }
/// # Ok::<(), rapidgeo_polyline::PolylineError>(())
/// ```
pub fn decode(polyline: &str, precision: u8) -> PolylineResult<Vec<LngLat>> {
    if precision == 0 || precision > 11 {
        return Err(PolylineError::InvalidPrecision(precision));
    }

    if polyline.is_empty() {
        return Ok(Vec::new());
    }

    let factor = 10_i64.pow(precision as u32) as f64;
    let bytes = polyline.as_bytes();
    let mut byte_iter = bytes.iter().enumerate();

    // Pre-allocate coordinates vector with estimated size
    let estimated_coords = polyline.len() / 6; // Rough estimate: ~6 chars per coordinate pair
    let mut coordinates = Vec::with_capacity(estimated_coords);

    let mut lat = 0i64;
    let mut lng = 0i64;

    loop {
        let delta_lat = match decode_signed_number_streaming(&mut byte_iter) {
            Ok(val) => val,
            Err(PolylineError::TruncatedData) => break, // End of data
            Err(e) => return Err(e),                    // Other errors
        };

        // If we got a latitude delta, we must have a longitude delta too
        let delta_lng = match decode_signed_number_streaming(&mut byte_iter) {
            Ok(val) => val,
            Err(PolylineError::TruncatedData) => {
                // We got latitude but not longitude - this is truncated data
                return Err(PolylineError::TruncatedData);
            }
            Err(e) => return Err(e),
        };

        lat = lat
            .checked_add(delta_lat)
            .ok_or(PolylineError::CoordinateOverflow)?;
        lng = lng
            .checked_add(delta_lng)
            .ok_or(PolylineError::CoordinateOverflow)?;

        let lat_deg = (lat as f64) / factor;
        let lng_deg = (lng as f64) / factor;

        let coord = LngLat::new_deg(lng_deg, lat_deg);
        validate_decoded_coordinate(&coord)?;
        coordinates.push(coord);
    }

    Ok(coordinates)
}

fn decode_signed_number_streaming(
    byte_iter: &mut std::iter::Enumerate<std::slice::Iter<u8>>,
) -> PolylineResult<i64> {
    let mut shift = 0;
    let mut result = 0i64;

    loop {
        let (position, &byte) = byte_iter.next().ok_or(PolylineError::TruncatedData)?;

        if !(63..=126).contains(&byte) {
            return Err(PolylineError::InvalidCharacter {
                character: char::from(byte),
                position,
            });
        }

        let value = (byte - 63) as i64;
        result |= (value & 0x1f) << shift;

        if value < 0x20 {
            break;
        }

        shift += 5;
        if shift >= 64 {
            return Err(PolylineError::CoordinateOverflow);
        }
    }

    let value = result >> 1;
    if (result & 1) != 0 {
        Ok(!value)
    } else {
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_empty() {
        let decoded = decode("", 5).unwrap();
        assert_eq!(decoded.len(), 0);
    }

    #[test]
    fn test_decode_google_example() {
        let decoded = decode("_p~iF~ps|U_ulLnnqC_mqNvxq`@", 5).unwrap();
        assert_eq!(decoded.len(), 3);

        assert!((decoded[0].lng_deg - (-120.2)).abs() < 0.00001);
        assert!((decoded[0].lat_deg - 38.5).abs() < 0.00001);

        assert!((decoded[1].lng_deg - (-120.95)).abs() < 0.00001);
        assert!((decoded[1].lat_deg - 40.7).abs() < 0.00001);

        assert!((decoded[2].lng_deg - (-126.453)).abs() < 0.00001);
        assert!((decoded[2].lat_deg - 43.252).abs() < 0.00001);
    }

    #[test]
    fn test_decode_invalid_character() {
        let result = decode("_p~iF~ps|U_ulLnnqC_mqN\x1f", 5);
        assert!(matches!(
            result,
            Err(PolylineError::InvalidCharacter { .. })
        ));
    }

    #[test]
    fn test_decode_truncated() {
        // Test a polyline that is truly truncated (ends in middle of a coordinate pair)
        // "_p~iF~ps|U_ulLnnqC" represents 2 complete coordinate pairs
        // Adding partial characters should cause truncation error
        let result = decode("_p~iF~ps|U_ulLnnqC_mqNvxq", 5); // Truncated in the middle
        assert!(matches!(result, Err(PolylineError::TruncatedData)));
    }

    #[test]
    fn test_decode_precision_validation() {
        assert!(decode("_p~iF~ps|U", 0).is_err());
        assert!(decode("_p~iF~ps|U", 12).is_err());
        assert!(decode("_p~iF~ps|U", 5).is_ok());
        assert!(decode("_p~iF~ps|U", 6).is_ok());
    }

    #[test]
    fn test_decode_single_point() {
        use crate::encode;

        let coords = vec![LngLat::new_deg(-122.4194, 37.7749)];
        let encoded = encode(&coords, 5).unwrap();
        let decoded = decode(&encoded, 5).unwrap();

        assert_eq!(decoded.len(), 1);
        assert!((decoded[0].lng_deg - coords[0].lng_deg).abs() < 0.00001);
        assert!((decoded[0].lat_deg - coords[0].lat_deg).abs() < 0.00001);
    }

    #[test]
    fn test_decode_precision_6() {
        use crate::encode;

        let coords = vec![
            LngLat::new_deg(-122.483696, 37.833818),
            LngLat::new_deg(-122.483482, 37.833174),
        ];

        let encoded = encode(&coords, 6).unwrap();
        let decoded = decode(&encoded, 6).unwrap();

        assert_eq!(decoded.len(), 2);
        for (original, decoded_coord) in coords.iter().zip(decoded.iter()) {
            assert!((original.lng_deg - decoded_coord.lng_deg).abs() < 0.000001);
            assert!((original.lat_deg - decoded_coord.lat_deg).abs() < 0.000001);
        }
    }

    #[test]
    fn test_decode_malicious_input_protection() {
        // Test malformed polyline that could cause issues
        let malicious_input = "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~";
        let result = decode(malicious_input, 5);
        // Should either decode successfully or fail gracefully (not panic)
        if result.is_ok() {
            // Valid decode
        }

        // Test input with invalid characters
        let invalid_chars = "_p~iF\x00\x01\x02";
        assert!(matches!(
            decode(invalid_chars, 5),
            Err(PolylineError::InvalidCharacter { .. })
        ));
    }

    #[test]
    fn test_decode_overflow_protection() {
        // Test that we handle coordinate overflow gracefully
        // This creates a very long polyline string that might cause integer overflow
        let long_polyline = "~".repeat(1000); // Character that represents continuation
        let result = decode(&long_polyline, 5);
        // Should not panic, either decode or error gracefully
        if result.is_ok() {
            // Graceful error handling
        }
    }

    #[test]
    fn test_decode_bounds_validation() {
        // Create a crafted polyline that would decode to out-of-bounds coordinates
        // This is hard to do without knowing the exact encoding, but we test the validation

        // Test that reasonable coordinates are accepted
        let valid = "_p~iF~ps|U"; // Should decode to reasonable coordinates
        assert!(decode(valid, 5).is_ok());
    }
}
