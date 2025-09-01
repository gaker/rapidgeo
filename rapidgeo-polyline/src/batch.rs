//! Batch operations for encoding and decoding multiple polylines in parallel.

use crate::{decode, encode, simplify_coordinates, LngLat, PolylineResult};
use rapidgeo_simplify::SimplifyMethod;
use rayon::prelude::*;

/// Encodes multiple coordinate sequences into polyline strings in parallel.
///
/// Uses parallel processing for improved performance when processing large numbers
/// of polylines (typically beneficial for >100 polylines).
///
/// # Arguments
///
/// * `coordinates_batch` - A slice of coordinate sequences to encode
/// * `precision` - Number of decimal places to preserve (typically 5 or 6)
///
/// # Examples
///
/// ```
/// use rapidgeo_polyline::batch::encode_batch;
/// use rapidgeo_distance::LngLat;
///
/// let batch = vec![
///     vec![LngLat::new_deg(-120.2, 38.5), LngLat::new_deg(-120.95, 35.6)],
///     vec![LngLat::new_deg(-126.453, 43.252), LngLat::new_deg(-122.4194, 37.7749)],
/// ];
///
/// let encoded_batch = encode_batch(&batch, 5).unwrap();
/// assert_eq!(encoded_batch.len(), 2);
/// ```
pub fn encode_batch(
    coordinates_batch: &[Vec<LngLat>],
    precision: u8,
) -> PolylineResult<Vec<String>> {
    if coordinates_batch.len() < 100 {
        coordinates_batch
            .iter()
            .map(|coords| encode(coords, precision))
            .collect()
    } else {
        coordinates_batch
            .par_iter()
            .map(|coords| encode(coords, precision))
            .collect()
    }
}

/// Decodes multiple polyline strings into coordinate sequences in parallel.
///
/// Uses parallel processing for improved performance when processing large numbers
/// of polylines (typically beneficial for >100 polylines).
///
/// # Arguments
///
/// * `polylines` - A slice of polyline strings to decode
/// * `precision` - Number of decimal places the polylines were encoded with (typically 5 or 6)
///
/// # Examples
///
/// ```
/// use rapidgeo_polyline::batch::decode_batch;
///
/// let polylines = vec![
///     "_p~iF~ps|U_ulLnnqC".to_string(),
///     "_mqNvxq`@".to_string(),
/// ];
///
/// let decoded_batch = decode_batch(&polylines, 5).unwrap();
/// assert_eq!(decoded_batch.len(), 2);
/// ```
pub fn decode_batch(polylines: &[String], precision: u8) -> PolylineResult<Vec<Vec<LngLat>>> {
    if polylines.len() < 100 {
        polylines
            .iter()
            .map(|polyline| decode(polyline, precision))
            .collect()
    } else {
        polylines
            .par_iter()
            .map(|polyline| decode(polyline, precision))
            .collect()
    }
}

/// Encodes coordinate sequences from string slices in parallel.
pub fn decode_batch_strs(polylines: &[&str], precision: u8) -> PolylineResult<Vec<Vec<LngLat>>> {
    if polylines.len() < 100 {
        polylines
            .iter()
            .map(|polyline| decode(polyline, precision))
            .collect()
    } else {
        polylines
            .par_iter()
            .map(|polyline| decode(polyline, precision))
            .collect()
    }
}

/// Simplifies multiple coordinate sequences using Douglas-Peucker algorithm in parallel.
///
/// Uses parallel processing for improved performance when processing large numbers
/// of coordinate sequences (typically beneficial for >50 sequences).
///
/// # Arguments
///
/// * `coordinates_batch` - A slice of coordinate sequences to simplify
/// * `tolerance_m` - Simplification tolerance in meters
/// * `method` - Distance calculation method for simplification
///
/// # Examples
///
/// ```
/// use rapidgeo_polyline::batch::simplify_coordinates_batch;
/// use rapidgeo_simplify::SimplifyMethod;
/// use rapidgeo_distance::LngLat;
///
/// let batch = vec![
///     vec![LngLat::new_deg(-120.2, 38.5), LngLat::new_deg(-120.4, 38.6), LngLat::new_deg(-120.95, 40.7)],
///     vec![LngLat::new_deg(-126.453, 43.252), LngLat::new_deg(-126.5, 43.3), LngLat::new_deg(-122.4194, 37.7749)],
/// ];
///
/// let simplified_batch = simplify_coordinates_batch(&batch, 1000.0, SimplifyMethod::GreatCircleMeters);
/// assert_eq!(simplified_batch.len(), 2);
/// ```
pub fn simplify_coordinates_batch(
    coordinates_batch: &[Vec<LngLat>],
    tolerance_m: f64,
    method: SimplifyMethod,
) -> Vec<Vec<LngLat>> {
    if coordinates_batch.len() < 50 {
        coordinates_batch
            .iter()
            .map(|coords| simplify_coordinates(coords, tolerance_m, method))
            .collect()
    } else {
        coordinates_batch
            .par_iter()
            .map(|coords| simplify_coordinates(coords, tolerance_m, method))
            .collect()
    }
}

/// Simplifies and encodes multiple coordinate sequences to polyline strings in parallel.
///
/// This combines simplification and encoding in a single operation for efficiency.
///
/// # Arguments
///
/// * `coordinates_batch` - A slice of coordinate sequences to simplify and encode
/// * `tolerance_m` - Simplification tolerance in meters
/// * `method` - Distance calculation method for simplification
/// * `precision` - Number of decimal places to preserve (typically 5 or 6)
///
/// # Examples
///
/// ```
/// use rapidgeo_polyline::batch::encode_simplified_batch;
/// use rapidgeo_simplify::SimplifyMethod;
/// use rapidgeo_distance::LngLat;
///
/// let batch = vec![
///     vec![LngLat::new_deg(-120.2, 38.5), LngLat::new_deg(-120.95, 40.7)],
///     vec![LngLat::new_deg(-126.453, 43.252), LngLat::new_deg(-122.4194, 37.7749)],
/// ];
///
/// let encoded_batch = encode_simplified_batch(&batch, 1000.0, SimplifyMethod::GreatCircleMeters, 5).unwrap();
/// assert_eq!(encoded_batch.len(), 2);
/// ```
pub fn encode_simplified_batch(
    coordinates_batch: &[Vec<LngLat>],
    tolerance_m: f64,
    method: SimplifyMethod,
    precision: u8,
) -> PolylineResult<Vec<String>> {
    if coordinates_batch.len() < 50 {
        coordinates_batch
            .iter()
            .map(|coords| {
                let simplified = simplify_coordinates(coords, tolerance_m, method);
                encode(&simplified, precision)
            })
            .collect()
    } else {
        coordinates_batch
            .par_iter()
            .map(|coords| {
                let simplified = simplify_coordinates(coords, tolerance_m, method);
                encode(&simplified, precision)
            })
            .collect()
    }
}

/// Decodes, simplifies, and re-encodes multiple polyline strings in parallel.
///
/// This is useful for bulk processing of existing polylines to reduce their complexity.
///
/// # Arguments
///
/// * `polylines` - A slice of polyline strings to process
/// * `tolerance_m` - Simplification tolerance in meters
/// * `method` - Distance calculation method for simplification
/// * `precision` - Precision for decoding/encoding (typically 5 or 6)
///
/// # Examples
///
/// ```
/// use rapidgeo_polyline::batch::simplify_polylines_batch;
/// use rapidgeo_simplify::SimplifyMethod;
///
/// let polylines = vec![
///     "_p~iF~ps|U_ulLnnqC".to_string(),
///     "_mqNvxq`@".to_string(),
/// ];
///
/// let simplified_batch = simplify_polylines_batch(&polylines, 1000.0, SimplifyMethod::GreatCircleMeters, 5).unwrap();
/// assert_eq!(simplified_batch.len(), 2);
/// ```
pub fn simplify_polylines_batch(
    polylines: &[String],
    tolerance_m: f64,
    method: SimplifyMethod,
    precision: u8,
) -> PolylineResult<Vec<String>> {
    if polylines.len() < 50 {
        polylines
            .iter()
            .map(|polyline| {
                let coords = decode(polyline, precision)?;
                let simplified = simplify_coordinates(&coords, tolerance_m, method);
                encode(&simplified, precision)
            })
            .collect()
    } else {
        polylines
            .par_iter()
            .map(|polyline| {
                let coords = decode(polyline, precision)?;
                let simplified = simplify_coordinates(&coords, tolerance_m, method);
                encode(&simplified, precision)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_batch_small() {
        let batch = vec![
            vec![
                LngLat::new_deg(-120.2, 38.5),
                LngLat::new_deg(-120.95, 35.6),
            ],
            vec![LngLat::new_deg(-126.453, 43.252)],
            vec![],
        ];

        let encoded = encode_batch(&batch, 5).unwrap();
        assert_eq!(encoded.len(), 3);
        assert!(!encoded[0].is_empty());
        assert!(!encoded[1].is_empty());
        assert_eq!(encoded[2], "");
    }

    #[test]
    fn test_decode_batch_small() {
        let polylines = vec![
            "_p~iF~ps|U_ulLnnqC".to_string(),
            "_mqNvxq`@".to_string(),
            "".to_string(),
        ];

        let decoded = decode_batch(&polylines, 5).unwrap();
        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded[0].len(), 2);
        assert_eq!(decoded[1].len(), 1);
        assert_eq!(decoded[2].len(), 0);
    }

    #[test]
    fn test_decode_batch_strs() {
        let polylines = vec!["_p~iF~ps|U_ulLnnqC", "_mqNvxq`@", ""];

        let decoded = decode_batch_strs(&polylines, 5).unwrap();
        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded[0].len(), 2);
        assert_eq!(decoded[1].len(), 1);
        assert_eq!(decoded[2].len(), 0);
    }

    #[test]
    fn test_large_batch_parallel() {
        let coords = vec![
            LngLat::new_deg(-120.2, 38.5),
            LngLat::new_deg(-120.95, 35.6),
        ];
        let large_batch: Vec<Vec<LngLat>> = (0..150).map(|_| coords.clone()).collect();

        let encoded = encode_batch(&large_batch, 5).unwrap();
        assert_eq!(encoded.len(), 150);

        let decoded = decode_batch(&encoded, 5).unwrap();
        assert_eq!(decoded.len(), 150);

        for decoded_coords in decoded {
            assert_eq!(decoded_coords.len(), 2);
            assert!((decoded_coords[0].lng_deg - coords[0].lng_deg).abs() < 0.00001);
            assert!((decoded_coords[0].lat_deg - coords[0].lat_deg).abs() < 0.00001);
        }
    }

    #[test]
    fn test_batch_roundtrip() {
        let batch = vec![
            vec![
                LngLat::new_deg(-120.2, 38.5),
                LngLat::new_deg(-120.95, 35.6),
                LngLat::new_deg(-126.453, 43.252),
            ],
            vec![LngLat::new_deg(-122.4194, 37.7749)],
            vec![],
        ];

        let encoded = encode_batch(&batch, 5).unwrap();
        let decoded = decode_batch(&encoded, 5).unwrap();

        assert_eq!(batch.len(), decoded.len());

        for (original_coords, decoded_coords) in batch.iter().zip(decoded.iter()) {
            assert_eq!(original_coords.len(), decoded_coords.len());
            for (original, decoded_coord) in original_coords.iter().zip(decoded_coords.iter()) {
                assert!((original.lng_deg - decoded_coord.lng_deg).abs() < 0.00001);
                assert!((original.lat_deg - decoded_coord.lat_deg).abs() < 0.00001);
            }
        }
    }

    #[test]
    fn test_simplify_coordinates_batch_small() {
        use rapidgeo_simplify::SimplifyMethod;

        let batch = vec![
            vec![
                LngLat::new_deg(-122.0, 37.0),
                LngLat::new_deg(-122.1, 37.1),
                LngLat::new_deg(-122.2, 37.0),
            ],
            vec![
                LngLat::new_deg(-120.0, 38.0),
                LngLat::new_deg(-120.5, 38.5),
                LngLat::new_deg(-121.0, 38.0),
            ],
            vec![], // Empty sequence
        ];

        let simplified =
            simplify_coordinates_batch(&batch, 1000.0, SimplifyMethod::GreatCircleMeters);
        assert_eq!(simplified.len(), 3);

        // Each non-empty sequence should have at least 2 points (start/end)
        assert!(simplified[0].len() >= 2);
        assert!(simplified[1].len() >= 2);
        assert_eq!(simplified[2].len(), 0); // Empty stays empty

        // Endpoints should be preserved
        assert_eq!(simplified[0][0], batch[0][0]);
        assert_eq!(simplified[0].last().unwrap(), batch[0].last().unwrap());
    }

    #[test]
    fn test_encode_simplified_batch() {
        use rapidgeo_simplify::SimplifyMethod;

        let batch = vec![
            vec![
                LngLat::new_deg(-122.0, 37.0),
                LngLat::new_deg(-122.1, 37.1),
                LngLat::new_deg(-122.2, 37.0),
            ],
            vec![LngLat::new_deg(-120.0, 38.0)], // Single point
        ];

        let encoded =
            encode_simplified_batch(&batch, 1000.0, SimplifyMethod::GreatCircleMeters, 5).unwrap();
        assert_eq!(encoded.len(), 2);

        // All encoded strings should be valid
        for polyline in &encoded {
            if !polyline.is_empty() {
                assert!(decode(polyline, 5).is_ok());
            }
        }
    }

    #[test]
    fn test_simplify_polylines_batch() {
        use rapidgeo_simplify::SimplifyMethod;

        // Create some test polylines first
        let coords1 = vec![
            LngLat::new_deg(-122.0, 37.0),
            LngLat::new_deg(-122.1, 37.1),
            LngLat::new_deg(-122.2, 37.0),
        ];
        let coords2 = vec![
            LngLat::new_deg(-120.0, 38.0),
            LngLat::new_deg(-120.5, 38.5),
            LngLat::new_deg(-121.0, 38.0),
        ];

        let polylines = vec![encode(&coords1, 5).unwrap(), encode(&coords2, 5).unwrap()];

        let simplified =
            simplify_polylines_batch(&polylines, 1000.0, SimplifyMethod::GreatCircleMeters, 5)
                .unwrap();
        assert_eq!(simplified.len(), 2);

        // All simplified polylines should be valid and non-empty
        for polyline in &simplified {
            assert!(!polyline.is_empty());
            let decoded = decode(polyline, 5).unwrap();
            assert!(decoded.len() >= 2); // At least start and end
        }
    }

    #[test]
    fn test_batch_simplification_parallel() {
        use rapidgeo_simplify::SimplifyMethod;

        // Create a large batch to trigger parallel processing
        let coord_template = vec![
            LngLat::new_deg(-122.0, 37.0),
            LngLat::new_deg(-122.1, 37.1),
            LngLat::new_deg(-122.2, 37.2),
            LngLat::new_deg(-122.3, 37.1),
            LngLat::new_deg(-122.4, 37.0),
        ];
        let large_batch: Vec<Vec<LngLat>> = (0..75).map(|_| coord_template.clone()).collect();

        // Test parallel coordinate simplification
        let simplified =
            simplify_coordinates_batch(&large_batch, 1000.0, SimplifyMethod::GreatCircleMeters);
        assert_eq!(simplified.len(), 75);

        for coords in &simplified {
            assert!(coords.len() >= 2); // At least endpoints
            assert!(coords.len() <= coord_template.len()); // Not more than original
        }

        // Test parallel encode with simplification
        let encoded =
            encode_simplified_batch(&large_batch, 1000.0, SimplifyMethod::GreatCircleMeters, 5)
                .unwrap();
        assert_eq!(encoded.len(), 75);

        // Verify all are valid polylines
        for polyline in &encoded {
            assert!(!polyline.is_empty());
            assert!(decode(polyline, 5).is_ok());
        }
    }

    #[test]
    fn test_different_simplification_methods() {
        use rapidgeo_simplify::SimplifyMethod;

        let batch = vec![vec![
            LngLat::new_deg(-122.0, 37.0),
            LngLat::new_deg(-122.1, 37.1),
            LngLat::new_deg(-122.2, 37.0),
        ]];

        for method in [
            SimplifyMethod::GreatCircleMeters,
            SimplifyMethod::PlanarMeters,
            SimplifyMethod::EuclidRaw,
        ] {
            let simplified = simplify_coordinates_batch(&batch, 1000.0, method);
            assert_eq!(simplified.len(), 1);
            assert!(simplified[0].len() >= 2);

            let encoded = encode_simplified_batch(&batch, 1000.0, method, 5).unwrap();
            assert_eq!(encoded.len(), 1);
            assert!(!encoded[0].is_empty());
        }
    }
}
