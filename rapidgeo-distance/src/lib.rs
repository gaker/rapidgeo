//! Fast, accurate geographic and planar distance calculations.
//!
//! This crate provides distance calculation functions for geographic coordinates
//! using both geodesic (Earth-aware) and Euclidean (flat-plane) algorithms.
//!
//! # Quick Start
//!
//! ```
//! use rapidgeo_distance::{LngLat, geodesic, euclid};
//!
//! let sf = LngLat::new_deg(-122.4194, 37.7749);   // San Francisco
//! let nyc = LngLat::new_deg(-74.0060, 40.7128);   // New York City
//!
//! // Haversine: Fast, ±0.5% accuracy for distances <1000km
//! let distance = geodesic::haversine(sf, nyc);
//! println!("Distance: {:.1} km", distance / 1000.0);
//!
//! // Euclidean: Very fast but inaccurate for large distances
//! let euclidean = euclid::distance_euclid(sf, nyc);
//! println!("Euclidean: {:.6}°", euclidean);
//! ```
//!
//! # Algorithm Selection
//!
//! | Algorithm | Speed | Accuracy | Best For |
//! |-----------|-------|----------|----------|
//! | [Haversine](https://en.wikipedia.org/wiki/Haversine_formula) | Fast | ±0.5% | General use, distances <1000km |
//! | [Vincenty](https://en.wikipedia.org/wiki/Vincenty%27s_formulae) | Slow | ±1mm | High precision, any distance |
//! | [Euclidean](https://en.wikipedia.org/wiki/Euclidean_distance) | Fastest | Poor | Small areas, relative comparisons |
//!
//! # Coordinate System
//!
//! All coordinates use the **lng, lat** ordering convention (longitude first, latitude second).
//! Coordinates are stored in decimal degrees and converted to radians internally as needed.
//! The geodesic calculations assume the [WGS84 ellipsoid](https://en.wikipedia.org/wiki/World_Geodetic_System).
//!
//! # Modules
//!
//! - [`geodesic`]: Earth-aware distance calculations using [geodesic algorithms](https://en.wikipedia.org/wiki/Geodesic)
//! - [`euclid`]: Fast planar distance calculations using [Euclidean geometry](https://en.wikipedia.org/wiki/Euclidean_geometry)
//! - [`batch`]: Parallel batch operations (requires `batch` feature)

pub mod detection;
pub mod euclid;
pub mod formats;
pub mod geodesic;

#[cfg(feature = "batch")]
pub mod format_batch;

#[cfg(feature = "batch")]
pub mod batch;

/// A geographic coordinate in decimal degrees.
///
/// Represents a point on Earth's surface using longitude and latitude in decimal degrees.
/// Follows the **lng, lat** ordering convention (longitude first, latitude second).
///
/// # Coordinate Bounds
///
/// - Longitude: -180.0 to +180.0 degrees (West to East)
/// - Latitude: -90.0 to +90.0 degrees (South to North)
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::LngLat;
///
/// // San Francisco coordinates (lng, lat order)
/// let sf = LngLat::new_deg(-122.4194, 37.7749);
///
/// // Convert from radians
/// let coord = LngLat::new_rad(-2.1364, 0.6588);
///
/// // Convert to radians for calculations
/// let (lng_rad, lat_rad) = sf.to_radians();
/// ```
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LngLat {
    /// Longitude in decimal degrees (-180.0 to +180.0)
    pub lng_deg: f64,
    /// Latitude in decimal degrees (-90.0 to +90.0)
    pub lat_deg: f64,
}

impl LngLat {
    /// Creates a new coordinate from decimal degrees.
    ///
    /// # Arguments
    ///
    /// * `lng_deg` - Longitude in decimal degrees (-180.0 to +180.0)
    /// * `lat_deg` - Latitude in decimal degrees (-90.0 to +90.0)
    ///
    /// # Examples
    ///
    /// ```
    /// use rapidgeo_distance::LngLat;
    ///
    /// let nyc = LngLat::new_deg(-74.0060, 40.7128);
    /// ```
    pub fn new_deg(lng_deg: f64, lat_deg: f64) -> Self {
        Self { lng_deg, lat_deg }
    }

    /// Creates a new coordinate from radians.
    ///
    /// Converts the input radians to decimal degrees for storage.
    ///
    /// # Arguments
    ///
    /// * `lng_rad` - Longitude in radians
    /// * `lat_rad` - Latitude in radians
    ///
    /// # Examples
    ///
    /// ```
    /// use rapidgeo_distance::LngLat;
    ///
    /// let coord = LngLat::new_rad(-1.2916484, 0.7084);
    /// assert!((coord.lng_deg - (-74.0060)).abs() < 0.001);
    /// ```
    pub fn new_rad(lng_rad: f64, lat_rad: f64) -> Self {
        Self {
            lng_deg: lng_rad.to_degrees(),
            lat_deg: lat_rad.to_degrees(),
        }
    }

    /// Converts the coordinate to radians.
    ///
    /// Returns a tuple of (longitude_radians, latitude_radians) for use in
    /// trigonometric calculations. Many distance algorithms require radians.
    ///
    /// # Examples
    ///
    /// ```
    /// use rapidgeo_distance::LngLat;
    ///
    /// let sf = LngLat::new_deg(-122.4194, 37.7749);
    /// let (lng_rad, lat_rad) = sf.to_radians();
    /// ```
    pub fn to_radians(self) -> (f64, f64) {
        (self.lng_deg.to_radians(), self.lat_deg.to_radians())
    }
}

impl From<(f64, f64)> for LngLat {
    fn from((lng_deg, lat_deg): (f64, f64)) -> Self {
        Self { lng_deg, lat_deg }
    }
}

impl From<LngLat> for (f64, f64) {
    fn from(coord: LngLat) -> Self {
        (coord.lng_deg, coord.lat_deg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lnglat_conversions() {
        let coord = LngLat::new_deg(-122.4194, 37.7749);

        let tuple: (f64, f64) = coord.into();
        assert_eq!(tuple, (-122.4194, 37.7749));

        let coord2: LngLat = tuple.into();
        assert_eq!(coord2, coord);

        let coord3 = LngLat::from((-74.0060, 40.7128));
        assert_eq!(coord3.lng_deg, -74.0060);
        assert_eq!(coord3.lat_deg, 40.7128);
    }

    #[test]
    fn test_new_rad() {
        let lng_rad = -2.1364;
        let lat_rad = 0.6588;

        let coord = LngLat::new_rad(lng_rad, lat_rad);

        let expected_lng_deg = lng_rad.to_degrees();
        let expected_lat_deg = lat_rad.to_degrees();

        assert!((coord.lng_deg - expected_lng_deg).abs() < 1e-10);
        assert!((coord.lat_deg - expected_lat_deg).abs() < 1e-10);

        let (back_lng_rad, back_lat_rad) = coord.to_radians();
        assert!((back_lng_rad - lng_rad).abs() < 1e-15);
        assert!((back_lat_rad - lat_rad).abs() < 1e-15);
    }
}
