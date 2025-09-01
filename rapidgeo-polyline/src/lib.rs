//! Fast [Google Polyline Algorithm](https://developers.google.com/maps/documentation/utilities/polylinealgorithm) encoding and decoding.
//!
//! Encodes sequences of geographic coordinates into compact ASCII strings using Google's
//! Polyline Algorithm. Commonly used for route storage, GPS track compression, and
//! mapping applications.
//!
//! Uses the `LngLat` type from `rapidgeo-distance` and integrates with `rapidgeo-simplify`
//! for [Douglas-Peucker](https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm)
//! simplification.
//!
//! # Coordinate System
//!
//! All coordinates use **longitude, latitude** ordering (x, y convention).
//!
//! # Examples
//!
//! ## Basic Encoding and Decoding
//!
//! ```rust
//! use rapidgeo_polyline::{encode, decode};
//! use rapidgeo_distance::LngLat;
//!
//! // Google's test vector
//! let coords = vec![
//!     LngLat::new_deg(-120.2, 38.5),
//!     LngLat::new_deg(-120.95, 40.7),
//!     LngLat::new_deg(-126.453, 43.252),
//! ];
//!
//! let encoded = encode(&coords, 5)?;
//! assert_eq!(encoded, "_p~iF~ps|U_ulLnnqC_mqNvxq`@");
//!
//! let decoded = decode(&encoded, 5)?;
//! assert_eq!(coords.len(), decoded.len());
//! # Ok::<(), rapidgeo_polyline::PolylineError>(())
//! ```
//!
//! ## Polyline Simplification with Douglas-Peucker
//!
//! ```rust
//! use rapidgeo_polyline::{encode, encode_simplified, simplify_polyline};
//! use rapidgeo_simplify::SimplifyMethod;
//! use rapidgeo_distance::LngLat;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//!
//! // Detailed GPS track with close points
//! let gps_track = vec![
//!     LngLat::new_deg(-122.0, 37.0),
//!     LngLat::new_deg(-122.001, 37.001),  // Very close point
//!     LngLat::new_deg(-122.002, 37.002),  // Very close point
//!     LngLat::new_deg(-122.1, 37.1),     // Significant change
//! ];
//!
//! // Encode with 10m simplification tolerance
//! let simplified = encode_simplified(
//!     &gps_track,
//!     10.0, // meters tolerance
//!     SimplifyMethod::GreatCircleMeters,
//!     5
//! )?;
//!
//! // Or simplify an existing polyline
//! let original = encode(&gps_track, 5)?;
//! let simplified_existing = simplify_polyline(
//!     &original,
//!     10.0,
//!     SimplifyMethod::GreatCircleMeters,
//!     5
//! )?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Batch Operations (Parallel Processing)
//!
//! ```rust
//! #[cfg(feature = "batch")]
//! use rapidgeo_polyline::batch::{encode_batch, encode_simplified_batch};
//! use rapidgeo_simplify::SimplifyMethod;
//! use rapidgeo_distance::LngLat;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//!
//! let routes = vec![
//!     vec![LngLat::new_deg(-120.2, 38.5), LngLat::new_deg(-120.95, 40.7)],
//!     vec![LngLat::new_deg(-126.453, 43.252), LngLat::new_deg(-122.4194, 37.7749)],
//! ];
//!
//! #[cfg(feature = "batch")]
//! {
//!     // Parallel encoding (beneficial for >100 routes)
//!     let encoded_routes = encode_batch(&routes, 5)?;
//!     
//!     // Parallel encoding with simplification
//!     let simplified_routes = encode_simplified_batch(
//!         &routes,
//!         1000.0, // 1km tolerance
//!         SimplifyMethod::GreatCircleMeters,
//!         5
//!     )?;
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Precision Levels
//!
//! - **5**: ~1 meter accuracy, standard for most mapping applications
//! - **6**: ~10 centimeter accuracy, used for high-precision applications
//!
//! Valid range: 1-11

pub use rapidgeo_distance::LngLat;

#[cfg(feature = "batch")]
pub mod batch;

mod decode;
mod encode;
mod error;
mod simplify;

pub use decode::*;
pub use encode::*;
pub use error::*;
pub use simplify::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_encode_decode() {
        let coords = vec![
            LngLat::new_deg(-120.2, 38.5),
            LngLat::new_deg(-120.95, 35.6),
            LngLat::new_deg(-126.453, 43.252),
        ];

        let encoded = encode(&coords, 5).unwrap();
        let decoded = decode(&encoded, 5).unwrap();

        assert_eq!(coords.len(), decoded.len());

        for (original, decoded_coord) in coords.iter().zip(decoded.iter()) {
            assert!((original.lng_deg - decoded_coord.lng_deg).abs() < 0.00001);
            assert!((original.lat_deg - decoded_coord.lat_deg).abs() < 0.00001);
        }
    }

    #[test]
    fn test_precision_6() {
        let coords = vec![
            LngLat::new_deg(-122.483696, 37.833818),
            LngLat::new_deg(-122.483482, 37.833174),
        ];

        let encoded = encode(&coords, 6).unwrap();
        let decoded = decode(&encoded, 6).unwrap();

        assert_eq!(coords.len(), decoded.len());

        for (original, decoded_coord) in coords.iter().zip(decoded.iter()) {
            assert!((original.lng_deg - decoded_coord.lng_deg).abs() < 0.000001);
            assert!((original.lat_deg - decoded_coord.lat_deg).abs() < 0.000001);
        }
    }

    #[test]
    fn test_empty_coords() {
        let coords: Vec<LngLat> = vec![];
        let encoded = encode(&coords, 5).unwrap();
        assert_eq!(encoded, "");

        let decoded = decode("", 5).unwrap();
        assert_eq!(decoded.len(), 0);
    }

    #[test]
    fn test_single_coordinate() {
        let coords = vec![LngLat::new_deg(-122.4194, 37.7749)];

        let encoded = encode(&coords, 5).unwrap();
        let decoded = decode(&encoded, 5).unwrap();

        assert_eq!(decoded.len(), 1);
        assert!((coords[0].lng_deg - decoded[0].lng_deg).abs() < 0.00001);
        assert!((coords[0].lat_deg - decoded[0].lat_deg).abs() < 0.00001);
    }

    #[test]
    fn test_known_vectors() {
        // Google's test vector (note: coordinates are in lng, lat order)
        let coords = vec![
            LngLat::new_deg(-120.2, 38.5),
            LngLat::new_deg(-120.95, 40.7),
            LngLat::new_deg(-126.453, 43.252),
        ];

        let encoded = encode(&coords, 5).unwrap();
        assert_eq!(encoded, "_p~iF~ps|U_ulLnnqC_mqNvxq`@");

        let decoded = decode("_p~iF~ps|U_ulLnnqC_mqNvxq`@", 5).unwrap();
        assert_eq!(decoded.len(), 3);

        for (original, decoded_coord) in coords.iter().zip(decoded.iter()) {
            assert!((original.lng_deg - decoded_coord.lng_deg).abs() < 0.00001);
            assert!((original.lat_deg - decoded_coord.lat_deg).abs() < 0.00001);
        }
    }
}
