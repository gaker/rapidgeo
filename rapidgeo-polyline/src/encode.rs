//! Polyline encoding functions.

use crate::{LngLat, PolylineError, PolylineResult};

/// Validates coordinate bounds and checks for NaN/infinity values
fn validate_coordinate(coord: &LngLat) -> PolylineResult<()> {
    if !coord.lng_deg.is_finite() || !coord.lat_deg.is_finite() {
        return Err(PolylineError::InvalidCoordinate(
            "NaN or infinity coordinate".to_string(),
        ));
    }

    if coord.lng_deg < -180.0 || coord.lng_deg > 180.0 {
        return Err(PolylineError::InvalidCoordinate(format!(
            "Longitude {} out of bounds [-180, 180]",
            coord.lng_deg
        )));
    }

    if coord.lat_deg < -90.0 || coord.lat_deg > 90.0 {
        return Err(PolylineError::InvalidCoordinate(format!(
            "Latitude {} out of bounds [-90, 90]",
            coord.lat_deg
        )));
    }

    Ok(())
}

/// Safely converts coordinate to integer, checking for overflow
fn safe_coordinate_to_int(coord_deg: f64, factor: f64) -> PolylineResult<i64> {
    let scaled = coord_deg * factor;

    if scaled > (i64::MAX as f64) || scaled < (i64::MIN as f64) || !scaled.is_finite() {
        return Err(PolylineError::CoordinateOverflow);
    }

    Ok(scaled.round() as i64)
}

/// Encodes a sequence of coordinates into a polyline string using [Google's Polyline Algorithm](https://developers.google.com/maps/documentation/utilities/polylinealgorithm).
///
/// The algorithm uses delta encoding and variable-length encoding to compress
/// coordinate sequences. Coordinates are scaled by 10^precision, delta-encoded,
/// and compressed into printable ASCII characters.
///
/// # Arguments
///
/// * `coordinates` - Coordinate sequence in longitude, latitude order
/// * `precision` - Decimal places to preserve (1-11, typically 5 or 6)
///
/// # Returns
///
/// Returns a polyline string or an error if coordinates are invalid or precision is out of range.
///
/// # Errors
///
/// - [`PolylineError::InvalidCoordinate`] for NaN, infinite, or out-of-bounds coordinates
/// - [`PolylineError::InvalidPrecision`] for precision outside 1-11 range
/// - [`PolylineError::CoordinateOverflow`] for coordinates causing integer overflow
///
/// # Examples
///
/// ```rust
/// use rapidgeo_polyline::encode;
/// use rapidgeo_distance::LngLat;
///
/// // Standard precision 5 (1 meter accuracy)
/// let coords = vec![
///     LngLat::new_deg(-120.2, 38.5),
///     LngLat::new_deg(-120.95, 40.7),
/// ];
/// let encoded = encode(&coords, 5)?;
///
/// // High precision 6 (10 centimeter accuracy)
/// let precise_coords = vec![
///     LngLat::new_deg(-122.483696, 37.833818),
///     LngLat::new_deg(-122.483482, 37.833174),
/// ];
/// let precise_encoded = encode(&precise_coords, 6)?;
/// # Ok::<(), rapidgeo_polyline::PolylineError>(())
/// ```
///
/// # Google's Test Vector
///
/// ```rust
/// use rapidgeo_polyline::encode;
/// use rapidgeo_distance::LngLat;
///
/// // Google's official test case
/// let coords = vec![
///     LngLat::new_deg(-120.2, 38.5),
///     LngLat::new_deg(-120.95, 40.7),
///     LngLat::new_deg(-126.453, 43.252),
/// ];
/// let encoded = encode(&coords, 5)?;
/// assert_eq!(encoded, "_p~iF~ps|U_ulLnnqC_mqNvxq`@");
/// # Ok::<(), rapidgeo_polyline::PolylineError>(())
/// ```
pub fn encode(coordinates: &[LngLat], precision: u8) -> PolylineResult<String> {
    if precision == 0 || precision > 11 {
        return Err(PolylineError::InvalidPrecision(precision));
    }

    if coordinates.is_empty() {
        return Ok(String::new());
    }

    let factor = 10_i64.pow(precision as u32) as f64;

    // Better capacity estimation: each coordinate pair typically needs 6-8 characters
    // Account for higher precision and larger deltas with more generous estimate
    let estimated_capacity = coordinates.len() * 8 + 16; // +16 for safety margin
    let mut result = String::with_capacity(estimated_capacity);

    let mut prev_lat = 0i64;
    let mut prev_lng = 0i64;

    for coord in coordinates {
        validate_coordinate(coord)?;

        let lat = safe_coordinate_to_int(coord.lat_deg, factor)?;
        let lng = safe_coordinate_to_int(coord.lng_deg, factor)?;

        let delta_lat = lat
            .checked_sub(prev_lat)
            .ok_or(PolylineError::CoordinateOverflow)?;
        let delta_lng = lng
            .checked_sub(prev_lng)
            .ok_or(PolylineError::CoordinateOverflow)?;

        encode_signed_number(delta_lat, &mut result);
        encode_signed_number(delta_lng, &mut result);

        prev_lat = lat;
        prev_lng = lng;
    }

    Ok(result)
}

fn encode_signed_number(value: i64, result: &mut String) {
    let mut value = value << 1;
    if value < 0 {
        value = !value;
    }

    // Build bytes first, then append to string (avoids repeated UTF-8 validation)
    let mut bytes = Vec::with_capacity(12); // Max ~12 bytes for 64-bit number

    while value >= 0x20 {
        bytes.push((0x20 | (value & 0x1f)) as u8 + 63);
        value >>= 5;
    }
    bytes.push((value as u8) + 63);

    // Safe because we know all bytes are in valid ASCII range [63-126]
    let encoded_str = unsafe { std::str::from_utf8_unchecked(&bytes) };
    result.push_str(encoded_str);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_empty() {
        let coords: Vec<LngLat> = vec![];
        assert_eq!(encode(&coords, 5).unwrap(), "");
    }

    #[test]
    fn test_encode_single_point() {
        let coords = vec![LngLat::new_deg(-122.4194, 37.7749)];
        let encoded = encode(&coords, 5).unwrap();
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_encode_google_example() {
        let coords = vec![
            LngLat::new_deg(-120.2, 38.5),
            LngLat::new_deg(-120.95, 40.7),
            LngLat::new_deg(-126.453, 43.252),
        ];
        let encoded = encode(&coords, 5).unwrap();
        assert_eq!(encoded, "_p~iF~ps|U_ulLnnqC_mqNvxq`@");
    }

    #[test]
    fn test_encode_precision_validation() {
        let coords = vec![LngLat::new_deg(0.0, 0.0)];
        assert!(encode(&coords, 0).is_err());
        assert!(encode(&coords, 12).is_err());
        assert!(encode(&coords, 5).is_ok());
        assert!(encode(&coords, 6).is_ok());
    }

    #[test]
    fn test_encode_negative_coordinates() {
        let coords = vec![LngLat::new_deg(-1.0, -1.0), LngLat::new_deg(-2.0, -2.0)];
        let encoded = encode(&coords, 5).unwrap();
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_encode_coordinate_validation() {
        // Test NaN coordinates
        let nan_coord = vec![LngLat::new_deg(f64::NAN, 0.0)];
        assert!(matches!(
            encode(&nan_coord, 5),
            Err(PolylineError::InvalidCoordinate(_))
        ));

        let nan_coord = vec![LngLat::new_deg(0.0, f64::NAN)];
        assert!(matches!(
            encode(&nan_coord, 5),
            Err(PolylineError::InvalidCoordinate(_))
        ));

        // Test infinity coordinates
        let inf_coord = vec![LngLat::new_deg(f64::INFINITY, 0.0)];
        assert!(matches!(
            encode(&inf_coord, 5),
            Err(PolylineError::InvalidCoordinate(_))
        ));

        let neg_inf_coord = vec![LngLat::new_deg(0.0, f64::NEG_INFINITY)];
        assert!(matches!(
            encode(&neg_inf_coord, 5),
            Err(PolylineError::InvalidCoordinate(_))
        ));
    }

    #[test]
    fn test_encode_coordinate_bounds() {
        // Test longitude bounds
        let invalid_lng_high = vec![LngLat::new_deg(180.1, 0.0)];
        assert!(matches!(
            encode(&invalid_lng_high, 5),
            Err(PolylineError::InvalidCoordinate(_))
        ));

        let invalid_lng_low = vec![LngLat::new_deg(-180.1, 0.0)];
        assert!(matches!(
            encode(&invalid_lng_low, 5),
            Err(PolylineError::InvalidCoordinate(_))
        ));

        // Test latitude bounds
        let invalid_lat_high = vec![LngLat::new_deg(0.0, 90.1)];
        assert!(matches!(
            encode(&invalid_lat_high, 5),
            Err(PolylineError::InvalidCoordinate(_))
        ));

        let invalid_lat_low = vec![LngLat::new_deg(0.0, -90.1)];
        assert!(matches!(
            encode(&invalid_lat_low, 5),
            Err(PolylineError::InvalidCoordinate(_))
        ));

        // Test valid boundary coordinates
        let valid_bounds = vec![LngLat::new_deg(-180.0, -90.0), LngLat::new_deg(180.0, 90.0)];
        assert!(encode(&valid_bounds, 5).is_ok());
    }

    #[test]
    fn test_encode_overflow_protection() {
        // Test coordinates that exceed valid geographic bounds
        let invalid_lng = vec![LngLat::new_deg(1000.0, 0.0)];
        assert!(matches!(
            encode(&invalid_lng, 5),
            Err(PolylineError::InvalidCoordinate(_))
        ));

        // Test coordinates that would cause integer overflow during subtraction
        // Create a scenario where delta calculation would overflow
        let coords = vec![
            LngLat::new_deg(0.0, 0.0),
            LngLat::new_deg(180.0, 90.0), // Maximum valid coordinates
        ];
        // This should work fine since coordinates are valid
        assert!(encode(&coords, 5).is_ok());

        // Test with very high precision where arithmetic could overflow
        let high_precision_coords = vec![LngLat::new_deg(179.9999999, 89.9999999)];
        assert!(encode(&high_precision_coords, 11).is_ok());
    }
}
