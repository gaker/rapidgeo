//! Fast Google Polyline Algorithm encoding and decoding.
//!
//! This crate provides efficient encoding and decoding of geographic coordinate sequences
//! using Google's Polyline Algorithm. It uses the `LngLat` type from rapidgeo-distance
//! and supports configurable precision and batch operations.
//!
//! # Coordinate System
//!
//! All coordinates use the **lng, lat** ordering convention (longitude first, latitude second).
//!
//! # Examples
//!
//! ## Basic Encoding and Decoding
//!
//! ```
//! use rapidgeo_polyline::{encode, decode};
//! use rapidgeo_distance::LngLat;
//!
//! let coords = vec![
//!     LngLat::new_deg(-120.2, 38.5),
//!     LngLat::new_deg(-120.95, 40.7),
//!     LngLat::new_deg(-126.453, 43.252),
//! ];
//!
//! let encoded = encode(&coords, 5).unwrap();
//! let decoded = decode(&encoded, 5).unwrap();
//!
//! assert_eq!(coords.len(), decoded.len());
//! ```
//!
//! ## Polyline Simplification
//!
//! ```
//! use rapidgeo_polyline::{encode, encode_simplified, simplify_polyline};
//! use rapidgeo_simplify::SimplifyMethod;
//! use rapidgeo_distance::LngLat;
//!
//! // Create a detailed route with many points
//! let detailed_route = vec![
//!     LngLat::new_deg(-122.0, 37.0),
//!     LngLat::new_deg(-122.01, 37.01),
//!     LngLat::new_deg(-122.02, 37.02),
//!     LngLat::new_deg(-122.1, 37.1),
//!     LngLat::new_deg(-122.2, 37.0),
//! ];
//!
//! // Encode with simplification (removes intermediate points within tolerance)
//! let simplified_polyline = encode_simplified(
//!     &detailed_route,
//!     1000.0, // 1km tolerance
//!     SimplifyMethod::GreatCircleMeters,
//!     5
//! ).unwrap();
//!
//! // Or simplify an existing polyline string
//! let original_polyline = encode(&detailed_route, 5).unwrap();
//! let simplified = simplify_polyline(
//!     &original_polyline,
//!     1000.0,
//!     SimplifyMethod::GreatCircleMeters,
//!     5
//! ).unwrap();
//! ```
//!
//! ## Batch Operations
//!
//! ```
//! #[cfg(feature = "batch")]
//! use rapidgeo_polyline::batch::{encode_batch, encode_simplified_batch};
//! use rapidgeo_simplify::SimplifyMethod;
//! use rapidgeo_distance::LngLat;
//!
//! let routes = vec![
//!     vec![LngLat::new_deg(-120.2, 38.5), LngLat::new_deg(-120.95, 40.7)],
//!     vec![LngLat::new_deg(-126.453, 43.252), LngLat::new_deg(-122.4194, 37.7749)],
//! ];
//!
//! #[cfg(feature = "batch")]
//! {
//!     // Encode multiple routes in parallel
//!     let encoded_routes = encode_batch(&routes, 5).unwrap();
//!     
//!     // Encode and simplify multiple routes in parallel
//!     let simplified_routes = encode_simplified_batch(
//!         &routes,
//!         1000.0,
//!         SimplifyMethod::GreatCircleMeters,
//!         5
//!     ).unwrap();
//! }
//! ```

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
