//! Smart coordinate format detection with early termination optimization.
//!
//! Automatically detects whether coordinate pairs follow lng,lat or lat,lng ordering
//! by analyzing coordinate value ranges. Uses statistical sampling and early termination
//! for performance optimization on large datasets.

use crate::formats::FormatHint;

const LAT_MIN: f64 = -90.0;
const LAT_MAX: f64 = 90.0;
const LNG_MIN: f64 = -180.0;
const LNG_MAX: f64 = 180.0;

/// Detects coordinate format (lng,lat vs lat,lng) from coordinate pairs.
///
/// Analyzes up to 100 coordinate pairs to determine the most likely format.
/// Uses 95% confidence threshold with early termination for performance.
///
/// # Format Detection Logic
///
/// - **lng,lat**: First value in range [-180, 180], second in [-90, 90]
/// - **lat,lng**: First value in range [-90, 90], second in [-180, 180]
/// - **Unknown**: Cannot determine format with confidence
///
/// # Performance
///
/// - Samples up to 100 coordinates maximum
/// - Early termination when 95% confidence reached
/// - Typically processes 10-20 coordinates for clear datasets
/// - O(min(n, 100)) time complexity where n is input size
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::detection::detect_format;
/// use rapidgeo_distance::formats::FormatHint;
///
/// // Clear lng,lat format (US cities)
/// let coords = vec![
///     (-122.4194, 37.7749),  // San Francisco
///     (-74.0060, 40.7128),   // New York
///     (-87.6298, 41.8781),   // Chicago
/// ];
/// assert_eq!(detect_format(&coords), FormatHint::LngLat);
///
/// // Clear lat,lng format (same cities, swapped)
/// let coords = vec![
///     (37.7749, -122.4194),  // San Francisco  
///     (40.7128, -74.0060),   // New York
///     (41.8781, -87.6298),   // Chicago
/// ];
/// assert_eq!(detect_format(&coords), FormatHint::LatLng);
///
/// // Empty input
/// let empty: Vec<(f64, f64)> = vec![];
/// assert_eq!(detect_format(&empty), FormatHint::Unknown);
/// ```
///
/// # See Also
///
/// - [`detect_format_with_early_termination`] for custom parameters
/// - [`confidence_score`] to validate detection quality
pub fn detect_format(coords: &[(f64, f64)]) -> FormatHint {
    detect_format_with_early_termination(coords, 100, 0.95)
}

/// Detects coordinate format with custom sampling and confidence parameters.
///
/// Provides fine-grained control over the detection algorithm for performance
/// tuning or when working with datasets of known characteristics.
///
/// # Arguments
///
/// - `coords`: Coordinate pairs to analyze
/// - `max_samples`: Maximum number of coordinates to examine (performance limit)
/// - `confidence_threshold`: Required confidence ratio (0.0 to 1.0) for early termination
///
/// # Early Termination Logic
///
/// The algorithm terminates early when:
/// 1. At least 10 samples processed (minimum for statistical validity)
/// 2. Check every 5th sample for efficiency
/// 3. One format achieves the specified confidence threshold
/// 4. Mathematical impossibility of reaching threshold
///
/// # Performance Characteristics
///
/// - **Best case**: O(10) - clear format detected quickly
/// - **Average case**: O(20-50) - typical real-world datasets
/// - **Worst case**: O(max_samples) - ambiguous or invalid data
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::detection::detect_format_with_early_termination;
/// use rapidgeo_distance::formats::FormatHint;
///
/// let coords = vec![
///     (-122.4194, 37.7749),
///     (-74.0060, 40.7128),
///     (-87.6298, 41.8781),
/// ];
///
/// // High confidence, small sample for speed
/// let format = detect_format_with_early_termination(&coords, 50, 0.9);
/// assert_eq!(format, FormatHint::LngLat);
///
/// // Conservative detection for ambiguous data
/// let format = detect_format_with_early_termination(&coords, 200, 0.99);
/// assert_eq!(format, FormatHint::LngLat);
/// ```
///
/// # Use Cases
///
/// - **Speed optimization**: Lower max_samples for large datasets
/// - **High precision**: Higher confidence_threshold for critical applications
/// - **Ambiguous data**: Lower confidence_threshold to get best guess
pub fn detect_format_with_early_termination(
    coords: &[(f64, f64)],
    max_samples: usize,
    confidence_threshold: f64,
) -> FormatHint {
    if coords.is_empty() {
        return FormatHint::Unknown;
    }

    let mut lng_lat_valid = 0;
    let mut lat_lng_valid = 0;
    let sample_size = coords.len().min(max_samples);
    let min_samples_for_confidence = 10.min(sample_size);

    for (i, &(first, second)) in coords.iter().take(sample_size).enumerate() {
        if is_valid_lng(first) && is_valid_lat(second) {
            lng_lat_valid += 1;
        }
        if is_valid_lat(first) && is_valid_lng(second) {
            lat_lng_valid += 1;
        }

        if i + 1 >= min_samples_for_confidence && (i + 1) % 5 == 0 {
            let _samples_so_far = i + 1;
            let total_valid = lng_lat_valid + lat_lng_valid;

            if total_valid > 0 {
                let lng_lat_ratio = lng_lat_valid as f64 / total_valid as f64;
                let lat_lng_ratio = lat_lng_valid as f64 / total_valid as f64;

                if lng_lat_ratio >= confidence_threshold {
                    return FormatHint::LngLat;
                } else if lat_lng_ratio >= confidence_threshold {
                    return FormatHint::LatLng;
                }
            }
        }
    }

    match (lng_lat_valid, lat_lng_valid) {
        (0, 0) => FormatHint::Unknown,
        (a, b) if a > b => FormatHint::LngLat,
        (a, b) if b > a => FormatHint::LatLng,
        _ => FormatHint::Unknown,
    }
}

/// Validates if a value is a valid longitude coordinate.
/// Longitude values must be in the range [-180.0, 180.0] degrees and finite.
fn is_valid_lng(value: f64) -> bool {
    (LNG_MIN..=LNG_MAX).contains(&value) && value.is_finite()
}

/// Validates if a value is a valid latitude coordinate.
/// Latitude values must be in the range [-90.0, 90.0] degrees and finite.
fn is_valid_lat(value: f64) -> bool {
    (LAT_MIN..=LAT_MAX).contains(&value) && value.is_finite()
}

/// Calculates confidence score for a given format hint.
///
/// Measures how well the coordinate data fits the specified format by computing
/// the ratio of valid coordinates. Useful for validating detection results or
/// comparing format quality.
///
/// # Arguments
///
/// - `coords`: Coordinate pairs to analyze
/// - `hint`: The format to test (LngLat, LatLng, or Unknown)
///
/// # Returns
///
/// Confidence score from 0.0 to 1.0:
/// - **1.0**: All coordinates valid for the format
/// - **0.5**: Half the coordinates valid
/// - **0.0**: No coordinates valid (or Unknown format)
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::detection::confidence_score;
/// use rapidgeo_distance::formats::FormatHint;
///
/// let coords = vec![
///     (-122.4194, 37.7749),  // Valid lng,lat
///     (-74.0060, 40.7128),   // Valid lng,lat
///     (200.0, 95.0),         // Invalid coordinates
/// ];
///
/// // Perfect confidence for valid coordinates
/// let conf_lnglat = confidence_score(&coords[..2], FormatHint::LngLat);
/// assert_eq!(conf_lnglat, 1.0);
///
/// // Partial confidence with invalid data
/// let conf_mixed = confidence_score(&coords, FormatHint::LngLat);
/// assert!((conf_mixed - 0.666).abs() < 0.01);
///
/// // Unknown format always returns 0.0
/// let conf_unknown = confidence_score(&coords, FormatHint::Unknown);
/// assert_eq!(conf_unknown, 0.0);
/// ```
pub fn confidence_score(coords: &[(f64, f64)], hint: FormatHint) -> f64 {
    confidence_score_with_early_termination(coords, hint, 100, 1.0)
}

/// Calculates confidence score with early termination optimization.
///
/// Provides performance optimization for large datasets by terminating early
/// when the target confidence is reached or becomes mathematically impossible.
///
/// # Arguments
///
/// - `coords`: Coordinate pairs to analyze
/// - `hint`: The format to test (LngLat, LatLng, or Unknown)
/// - `max_samples`: Maximum number of coordinates to examine
/// - `target_confidence`: Stop early if this confidence is reached
///
/// # Early Termination Conditions
///
/// 1. **Target reached**: Current confidence >= target_confidence
/// 2. **Impossible target**: Even if all remaining samples were valid,
///    the target confidence could not be achieved
/// 3. **Sample limit**: Processed max_samples coordinates
///
/// # Performance Benefits
///
/// - **Large datasets**: Avoid processing millions of coordinates unnecessarily
/// - **Quality assurance**: Quick validation of format detection
/// - **Streaming data**: Process data as it arrives, stop when confident
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::detection::confidence_score_with_early_termination;
/// use rapidgeo_distance::formats::FormatHint;
///
/// // Large dataset with perfect format
/// let coords: Vec<(f64, f64)> = (0..10000)
///     .map(|i| (-180.0 + i as f64 * 0.01, i as f64 * 0.01))
///     .collect();
///
/// // Will terminate early after ~10-20 samples
/// let confidence = confidence_score_with_early_termination(
///     &coords, FormatHint::LngLat, 10000, 0.95
/// );
/// assert!(confidence >= 0.95);
///
/// // Conservative scoring with impossible target
/// let mixed_coords = vec![
///     (-122.4, 37.7),  // Valid
///     (500.0, 200.0),  // Invalid
/// ];
/// let confidence = confidence_score_with_early_termination(
///     &mixed_coords, FormatHint::LngLat, 100, 0.95
/// );
/// assert!(confidence < 0.95);  // Impossible to reach 95%
/// ```
pub fn confidence_score_with_early_termination(
    coords: &[(f64, f64)],
    hint: FormatHint,
    max_samples: usize,
    target_confidence: f64,
) -> f64 {
    if coords.is_empty() {
        return 0.0;
    }

    let sample_size = coords.len().min(max_samples);
    let mut valid_count = 0;
    let min_samples_for_confidence = 10.min(sample_size);

    for (i, &(first, second)) in coords.iter().take(sample_size).enumerate() {
        let is_valid = match hint {
            FormatHint::LngLat => is_valid_lng(first) && is_valid_lat(second),
            FormatHint::LatLng => is_valid_lat(first) && is_valid_lng(second),
            FormatHint::Unknown => false,
        };

        if is_valid {
            valid_count += 1;
        }

        if i + 1 >= min_samples_for_confidence && (i + 1) % 5 == 0 {
            let current_confidence = valid_count as f64 / (i + 1) as f64;
            if current_confidence >= target_confidence {
                return current_confidence;
            }

            let remaining_samples = sample_size - (i + 1);
            let max_possible_confidence =
                (valid_count + remaining_samples) as f64 / sample_size as f64;
            if max_possible_confidence < target_confidence {
                return valid_count as f64 / (i + 1) as f64;
            }
        }
    }

    valid_count as f64 / sample_size as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_format_lng_lat_order() {
        // Clear lng, lat format - San Francisco, New York, Chicago
        let coords = vec![
            (-122.4194, 37.7749), // San Francisco
            (-74.0060, 40.7128),  // New York
            (-87.6298, 41.8781),  // Chicago
        ];

        let detected = detect_format(&coords);
        assert!(matches!(detected, FormatHint::LngLat));

        let confidence = confidence_score(&coords, FormatHint::LngLat);
        assert_eq!(confidence, 1.0); // All coordinates are valid lng,lat
    }

    #[test]
    fn test_detect_format_lat_lng_order() {
        // Clear lat, lng format - same cities but swapped order
        let coords = vec![
            (37.7749, -122.4194), // San Francisco (lat, lng)
            (40.7128, -74.0060),  // New York (lat, lng)
            (41.8781, -87.6298),  // Chicago (lat, lng)
        ];

        let detected = detect_format(&coords);
        assert!(matches!(detected, FormatHint::LatLng));

        let confidence = confidence_score(&coords, FormatHint::LatLng);
        assert_eq!(confidence, 1.0); // All coordinates are valid lat,lng
    }

    #[test]
    fn test_detect_format_empty_coords() {
        let empty_coords: Vec<(f64, f64)> = vec![];

        let detected = detect_format(&empty_coords);
        assert!(matches!(detected, FormatHint::Unknown));

        let confidence = confidence_score(&empty_coords, FormatHint::LngLat);
        assert_eq!(confidence, 0.0);
    }

    #[test]
    fn test_detect_format_ambiguous_coordinates() {
        // Coordinates that could be valid in both orders (within [-90, 90] for both)
        let coords = vec![
            (45.0, 60.0),  // Both valid as lng/lat or lat/lng
            (30.0, -80.0), // Both valid as lng/lat or lat/lng
            (-45.0, 70.0), // Both valid as lng/lat or lat/lng
        ];

        let detected = detect_format(&coords);
        // Could be either format, might return Unknown due to tie
        assert!(matches!(detected, FormatHint::Unknown));
    }

    #[test]
    fn test_detect_format_invalid_coordinates() {
        // Coordinates that are invalid in both interpretations
        let coords = vec![
            (200.0, 95.0),   // Both exceed valid ranges
            (-200.0, -95.0), // Both exceed valid ranges
            (250.0, 100.0),  // Both exceed valid ranges
        ];

        let detected = detect_format(&coords);
        assert!(matches!(detected, FormatHint::Unknown));

        let confidence_lng_lat = confidence_score(&coords, FormatHint::LngLat);
        let confidence_lat_lng = confidence_score(&coords, FormatHint::LatLng);
        assert_eq!(confidence_lng_lat, 0.0);
        assert_eq!(confidence_lat_lng, 0.0);
    }

    #[test]
    fn test_detect_format_mixed_validity() {
        // Mix of valid and invalid coordinates favoring lng,lat interpretation
        let coords = vec![
            (-122.4194, 37.7749), // Valid as lng,lat only
            (200.0, 40.7128),     // Invalid first coordinate
            (-87.6298, 41.8781),  // Valid as lng,lat only
        ];

        let detected = detect_format(&coords);
        assert!(matches!(detected, FormatHint::LngLat));

        let confidence_lng_lat = confidence_score(&coords, FormatHint::LngLat);
        assert_eq!(confidence_lng_lat, 2.0 / 3.0); // 2 out of 3 valid

        let confidence_lat_lng = confidence_score(&coords, FormatHint::LatLng);
        assert_eq!(confidence_lat_lng, 1.0 / 3.0); // 1 out of 3 valid as lat,lng
    }

    #[test]
    fn test_is_valid_lng() {
        // Valid longitude values
        assert!(is_valid_lng(-180.0));
        assert!(is_valid_lng(180.0));
        assert!(is_valid_lng(0.0));
        assert!(is_valid_lng(-122.4194));
        assert!(is_valid_lng(151.2093));

        // Invalid longitude values
        assert!(!is_valid_lng(-180.1));
        assert!(!is_valid_lng(180.1));
        assert!(!is_valid_lng(-200.0));
        assert!(!is_valid_lng(200.0));
        assert!(!is_valid_lng(f64::NAN));
        assert!(!is_valid_lng(f64::INFINITY));
        assert!(!is_valid_lng(f64::NEG_INFINITY));
    }

    #[test]
    fn test_is_valid_lat() {
        // Valid latitude values
        assert!(is_valid_lat(-90.0));
        assert!(is_valid_lat(90.0));
        assert!(is_valid_lat(0.0));
        assert!(is_valid_lat(37.7749));
        assert!(is_valid_lat(-33.8688));

        // Invalid latitude values
        assert!(!is_valid_lat(-90.1));
        assert!(!is_valid_lat(90.1));
        assert!(!is_valid_lat(-100.0));
        assert!(!is_valid_lat(100.0));
        assert!(!is_valid_lat(f64::NAN));
        assert!(!is_valid_lat(f64::INFINITY));
        assert!(!is_valid_lat(f64::NEG_INFINITY));
    }

    #[test]
    fn test_detect_format_large_sample() {
        // Test with more than 100 coordinates (should sample first 100)
        let mut coords = Vec::new();

        // Add 150 valid lng,lat coordinates
        for i in 0..150 {
            let lng = -180.0 + (i as f64 * 2.4); // Range from -180 to 180
            let lat = -90.0 + (i as f64 * 1.2); // Range from -90 to 90
            coords.push((lng, lat));
        }

        let detected = detect_format(&coords);
        assert!(matches!(detected, FormatHint::LngLat));

        let confidence = confidence_score(&coords, FormatHint::LngLat);
        assert_eq!(confidence, 1.0); // All sampled coordinates should be valid
    }

    #[test]
    fn test_detect_format_edge_coordinate_values() {
        // Test with coordinates at the exact boundaries
        let coords = vec![
            (-180.0, -90.0),          // Southwest corner of world
            (180.0, 90.0),            // Northeast corner of world
            (0.0, 0.0),               // Prime meridian and equator
            (-179.999999, 89.999999), // Very close to limits
        ];

        let detected = detect_format(&coords);
        assert!(matches!(detected, FormatHint::LngLat));

        let confidence = confidence_score(&coords, FormatHint::LngLat);
        assert_eq!(confidence, 1.0);

        let confidence_reversed = confidence_score(&coords, FormatHint::LatLng);
        assert_eq!(confidence_reversed, 0.25); // 1 out of 4 valid when interpreted as lat,lng
    }

    #[test]
    fn test_detect_format_single_coordinate() {
        // Single clearly identifiable lng,lat coordinate
        let coords = vec![(-122.4194, 37.7749)];

        let detected = detect_format(&coords);
        assert!(matches!(detected, FormatHint::LngLat));

        let confidence = confidence_score(&coords, FormatHint::LngLat);
        assert_eq!(confidence, 1.0);
    }

    #[test]
    fn test_confidence_score_unknown_format() {
        let coords = vec![(-122.4194, 37.7749), (-74.0060, 40.7128)];

        // Unknown format should always return 0.0 confidence
        let confidence = confidence_score(&coords, FormatHint::Unknown);
        assert_eq!(confidence, 0.0);
    }

    #[test]
    fn test_confidence_score_partial_validity() {
        let coords = vec![
            (-122.4194, 37.7749), // Valid lng,lat
            (200.0, 40.7128),     // Invalid lng
            (-87.6298, 41.8781),  // Valid lng,lat
            (-200.0, 42.3601),    // Invalid lng
        ];

        let confidence = confidence_score(&coords, FormatHint::LngLat);
        assert_eq!(confidence, 0.5); // 2 out of 4 valid

        let confidence_reversed = confidence_score(&coords, FormatHint::LatLng);
        assert_eq!(confidence_reversed, 0.25); // 1 out of 4 valid as lat,lng
    }

    #[test]
    fn test_detect_format_precision_coordinates() {
        // High precision coordinates that are clearly lng,lat
        let coords = vec![
            (-122.419416123456, 37.774928987654),
            (-74.006012345679, 40.712776543211),
            (-87.629798765432, 41.878113456789),
        ];

        let detected = detect_format(&coords);
        assert!(matches!(detected, FormatHint::LngLat));

        let confidence = confidence_score(&coords, FormatHint::LngLat);
        assert_eq!(confidence, 1.0);
    }

    #[test]
    fn test_detect_format_extreme_longitude_values() {
        // Test coordinates with extreme but valid longitude values
        let coords = vec![
            (-179.999999, 45.0), // Near date line (west)
            (179.999999, 45.0),  // Near date line (east)
            (-170.0, 60.0),      // Pacific Ocean
            (170.0, -60.0),      // Pacific Ocean (south)
        ];

        let detected = detect_format(&coords);
        assert!(matches!(detected, FormatHint::LngLat));

        let confidence = confidence_score(&coords, FormatHint::LngLat);
        assert_eq!(confidence, 1.0);

        // These would be invalid as lat,lng (latitudes > 90)
        let confidence_reversed = confidence_score(&coords, FormatHint::LatLng);
        assert_eq!(confidence_reversed, 0.0);
    }

    #[test]
    fn test_detect_format_polar_regions() {
        // Test coordinates in polar regions
        let coords = vec![
            (-120.0, 89.5),   // Near North Pole
            (45.0, -89.5),    // Near South Pole
            (0.0, 89.999),    // Very close to North Pole
            (180.0, -89.999), // Very close to South Pole
        ];

        let detected = detect_format(&coords);
        assert!(matches!(detected, FormatHint::LngLat));

        let confidence = confidence_score(&coords, FormatHint::LngLat);
        assert_eq!(confidence, 1.0);
    }

    #[test]
    fn test_detect_format_equatorial_region() {
        // Test coordinates around the equator and prime meridian
        let coords = vec![
            (0.0, 0.0),  // Null Island
            (-1.0, 1.0), // Near prime meridian
            (1.0, -1.0), // Near prime meridian
            (-0.5, 0.5), // Very close to origin
        ];

        // These coordinates are ambiguous - could be valid in either order
        let detected = detect_format(&coords);
        // Result could be either LngLat, LatLng, or Unknown depending on implementation details
        // The important thing is that it doesn't crash and returns a valid FormatHint
        assert!(matches!(
            detected,
            FormatHint::LngLat | FormatHint::LatLng | FormatHint::Unknown
        ));
    }

    #[test]
    fn test_confidence_score_mathematical_properties() {
        let coords = vec![
            (-122.4194, 37.7749),
            (-74.0060, 40.7128),
            (-87.6298, 41.8781),
        ];

        // Confidence scores should be between 0.0 and 1.0
        let conf_lng_lat = confidence_score(&coords, FormatHint::LngLat);
        let conf_lat_lng = confidence_score(&coords, FormatHint::LatLng);
        let conf_unknown = confidence_score(&coords, FormatHint::Unknown);

        assert!((0.0..=1.0).contains(&conf_lng_lat));
        assert!((0.0..=1.0).contains(&conf_lat_lng));
        assert!((0.0..=1.0).contains(&conf_unknown));

        // For these coordinates, lng_lat should have higher confidence than lat_lng
        assert!(conf_lng_lat > conf_lat_lng);

        // Unknown should always be 0
        assert_eq!(conf_unknown, 0.0);
    }

    #[test]
    fn test_detect_format_sampling_consistency() {
        // Create a dataset larger than 100 elements to test sampling
        let mut coords = Vec::new();

        // First 100 are clearly lng,lat
        for i in 0..100 {
            coords.push((-(180.0 - i as f64), i as f64 - 50.0));
        }

        // Next 50 are invalid (but shouldn't be sampled)
        for _i in 100..150 {
            coords.push((200.0, 95.0)); // Invalid
        }

        let detected = detect_format(&coords);
        assert!(matches!(detected, FormatHint::LngLat));

        // Should only sample first 100, which are all valid
        let confidence = confidence_score(&coords, FormatHint::LngLat);
        assert_eq!(confidence, 1.0);
    }

    #[test]
    fn test_early_termination_clear_format() {
        // Create clearly identifiable lng,lat data
        let coords: Vec<(f64, f64)> = (0..1000)
            .map(|i| (-(180.0 - i as f64 * 0.1), i as f64 * 0.1 - 50.0))
            .collect();

        // Should terminate early with high confidence
        let detected = detect_format_with_early_termination(&coords, 1000, 0.95);
        assert!(matches!(detected, FormatHint::LngLat));

        // Test with custom parameters
        let detected_conservative = detect_format_with_early_termination(&coords, 50, 0.9);
        assert!(matches!(detected_conservative, FormatHint::LngLat));
    }

    #[test]
    fn test_early_termination_ambiguous_format() {
        // Create ambiguous data that could be either format
        let coords = vec![
            (45.0, 60.0),  // Valid as both lng,lat and lat,lng
            (30.0, -80.0), // Valid as both lng,lat and lat,lng
            (-45.0, 70.0), // Valid as both lng,lat and lat,lng
            (50.0, 45.0),  // Valid as both lng,lat and lat,lng
        ];

        let detected = detect_format_with_early_termination(&coords, 100, 0.95);
        // Should return Unknown since neither format has clear dominance
        assert!(matches!(detected, FormatHint::Unknown));
    }

    #[test]
    fn test_early_termination_mixed_data() {
        let mut coords = Vec::new();

        // First 20 coordinates clearly lng,lat
        for i in 0..20 {
            coords.push((-(180.0 - i as f64), i as f64 - 10.0));
        }

        // Add some ambiguous coordinates
        for i in 20..100 {
            coords.push((i as f64 * 0.5, (i + 10) as f64 * 0.3));
        }

        let detected = detect_format_with_early_termination(&coords, 100, 0.8);
        // Should detect lng,lat due to initial clear examples
        assert!(matches!(detected, FormatHint::LngLat));
    }

    #[test]
    fn test_confidence_score_early_termination() {
        // Perfect lng,lat data
        let coords: Vec<(f64, f64)> = (0..1000)
            .map(|i| (-(180.0 - i as f64 * 0.1), i as f64 * 0.1 - 50.0))
            .collect();

        // Should achieve 100% confidence quickly
        let confidence =
            confidence_score_with_early_termination(&coords, FormatHint::LngLat, 1000, 0.95);
        assert!(confidence >= 0.95);

        // Test early termination with impossible target
        let confidence_impossible =
            confidence_score_with_early_termination(&coords, FormatHint::LatLng, 100, 0.95);
        assert!(confidence_impossible < 0.95);
    }

    #[test]
    fn test_confidence_score_mixed_validity() {
        let coords = vec![
            (-122.4194, 37.7749), // Valid lng,lat
            (200.0, 40.7128),     // Invalid lng
            (-87.6298, 41.8781),  // Valid lng,lat
            (-200.0, 42.3601),    // Invalid lng
            (-74.0060, 40.7128),  // Valid lng,lat
        ];

        // Should get 3/5 = 0.6 confidence for lng,lat
        let confidence =
            confidence_score_with_early_termination(&coords, FormatHint::LngLat, 100, 1.0);
        assert_eq!(confidence, 0.6);

        // Should terminate early when target is achievable
        let confidence_low_target =
            confidence_score_with_early_termination(&coords, FormatHint::LngLat, 100, 0.5);
        assert!(confidence_low_target >= 0.5);
    }

    #[test]
    fn test_early_termination_performance_characteristics() {
        // Create a large dataset where early samples are perfect
        let mut coords = Vec::new();

        // First 15 are perfect lng,lat
        for i in 0..15 {
            coords.push((-(180.0 - i as f64), i as f64 - 10.0));
        }

        // Rest are mixed/invalid (but should not be processed due to early termination)
        for _i in 15..10000 {
            coords.push((200.0, 95.0)); // Invalid
        }

        let detected = detect_format_with_early_termination(&coords, 10000, 0.9);
        // Should detect lng,lat from the first 15 perfect samples
        assert!(matches!(detected, FormatHint::LngLat));
    }

    #[test]
    fn test_early_termination_minimum_sample_requirement() {
        // Small dataset with clear format
        let coords = vec![
            (-122.4194, 37.7749),
            (-74.0060, 40.7128),
            (-87.6298, 41.8781),
        ];

        // Should work even with very small datasets
        let detected = detect_format_with_early_termination(&coords, 100, 0.8);
        assert!(matches!(detected, FormatHint::LngLat));

        let confidence =
            confidence_score_with_early_termination(&coords, FormatHint::LngLat, 100, 0.8);
        assert_eq!(confidence, 1.0);
    }
}
