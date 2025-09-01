//! Douglas-Peucker polyline simplification with pluggable distance backends.
//!
//! This crate implements the [Douglas-Peucker algorithm](https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm)
//! for simplifying polylines while preserving their essential shape characteristics.
//! The algorithm reduces the number of points in a polyline by removing points that
//! don't significantly contribute to the overall shape, based on a distance threshold.
//!
//! # Quick Start
//!
//! ```rust
//! use rapidgeo_distance::LngLat;
//! use rapidgeo_simplify::{simplify_dp_into, SimplifyMethod};
//!
//! let points = vec![
//!     LngLat::new_deg(-122.4194, 37.7749), // San Francisco
//!     LngLat::new_deg(-122.0, 37.5),       // Detour
//!     LngLat::new_deg(-121.5, 37.0),       // Another point
//!     LngLat::new_deg(-118.2437, 34.0522), // Los Angeles
//! ];
//!
//! let mut simplified = Vec::new();
//! let count = simplify_dp_into(
//!     &points,
//!     50000.0, // 50km tolerance
//!     SimplifyMethod::GreatCircleMeters,
//!     &mut simplified,
//! );
//!
//! assert_eq!(count, simplified.len());
//! // Endpoints are always preserved
//! assert_eq!(simplified[0], points[0]);
//! assert_eq!(simplified[count - 1], points[points.len() - 1]);
//! ```
//!
//! # Distance Methods
//!
//! - [`SimplifyMethod::GreatCircleMeters`]: Spherical distance for global accuracy
//! - [`SimplifyMethod::PlanarMeters`]: ENU projection for regional work
//! - [`SimplifyMethod::EuclidRaw`]: Raw coordinates for non-geographic data
//!
//! # Batch Processing
//!
//! Enable the `batch` feature for parallel processing of multiple polylines:
//!
//! ```toml
//! [dependencies]
//! rapidgeo-simplify = { version = "0.1", features = ["batch"] }
//! ```

pub mod dp;
pub mod xt;

#[cfg(feature = "batch")]
pub mod batch;

use rapidgeo_distance::LngLat;

/// Distance calculation method for Douglas-Peucker simplification.
///
/// Each method trades off between accuracy and performance for different use cases.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimplifyMethod {
    /// ENU projection around the polyline's midpoint.
    ///
    /// Best for regional datasets where reasonable accuracy is needed with better
    /// performance than great circle calculations. Accuracy decreases for very
    /// large geographic regions.
    PlanarMeters,

    /// Spherical distance calculations using the great circle method.
    ///
    /// Best for global datasets requiring high accuracy. Uses more computation
    /// than planar methods but works correctly at any scale.
    GreatCircleMeters,

    /// Direct Euclidean distance between raw coordinates.
    ///
    /// Best for non-geographic coordinate systems, screen coordinates, or
    /// already-projected data where coordinates represent planar distances.
    EuclidRaw,
}

/// Simplify a polyline using the Douglas-Peucker algorithm.
///
/// Returns the simplified points in the output vector and the count of points kept.
/// The output vector is cleared before simplification begins.
///
/// # Arguments
///
/// * `pts` - Input polyline points in longitude, latitude order
/// * `tolerance_m` - Maximum perpendicular distance threshold in meters (or coordinate units for EuclidRaw)
/// * `method` - Distance calculation method to use
/// * `out` - Output vector to store simplified points (will be cleared)
///
/// # Returns
///
/// Number of points in the simplified polyline (same as `out.len()` after the call)
///
/// # Examples
///
/// ```rust
/// use rapidgeo_distance::LngLat;
/// use rapidgeo_simplify::{simplify_dp_into, SimplifyMethod};
///
/// let points = vec![
///     LngLat::new_deg(-122.0, 37.0),
///     LngLat::new_deg(-121.5, 37.5), // This might be removed if tolerance is high
///     LngLat::new_deg(-121.0, 37.0),
/// ];
///
/// let mut simplified = Vec::new();
/// let count = simplify_dp_into(
///     &points,
///     1000.0, // 1km tolerance
///     SimplifyMethod::GreatCircleMeters,
///     &mut simplified,
/// );
///
/// assert!(count >= 2); // At least endpoints preserved
/// assert_eq!(simplified[0], points[0]); // First point preserved
/// assert_eq!(simplified[count - 1], points[points.len() - 1]); // Last point preserved
/// ```
///
/// # Algorithm
///
/// Implements the [Douglas-Peucker algorithm](https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm):
///
/// 1. Draw line between first and last points
/// 2. Find point with maximum perpendicular distance to this line
/// 3. If distance > tolerance, keep the point and recurse on both segments
/// 4. Otherwise, drop all intermediate points
pub fn simplify_dp_into(
    pts: &[LngLat],
    tolerance_m: f64,
    method: SimplifyMethod,
    out: &mut Vec<LngLat>,
) -> usize {
    out.clear();

    let mut mask = vec![false; pts.len()];
    simplify_dp_mask(pts, tolerance_m, method, &mut mask);

    for (i, &keep) in mask.iter().enumerate() {
        if keep {
            out.push(pts[i]);
        }
    }

    out.len()
}

/// Generate a boolean mask indicating which points to keep during simplification.
///
/// This is useful when you need to know which original points were kept,
/// or when you want to apply the same simplification to multiple related datasets.
///
/// # Arguments
///
/// * `pts` - Input polyline points
/// * `tolerance_m` - Maximum perpendicular distance threshold in meters (or coordinate units for EuclidRaw)
/// * `method` - Distance calculation method to use
/// * `mask` - Output boolean vector (will be resized to match input length)
///
/// # Examples
///
/// ```rust
/// use rapidgeo_distance::LngLat;
/// use rapidgeo_simplify::{simplify_dp_mask, SimplifyMethod};
///
/// let points = vec![
///     LngLat::new_deg(-122.0, 37.0),
///     LngLat::new_deg(-121.5, 37.5),
///     LngLat::new_deg(-121.0, 37.0),
/// ];
///
/// let mut mask = Vec::new();
/// simplify_dp_mask(
///     &points,
///     1000.0,
///     SimplifyMethod::GreatCircleMeters,
///     &mut mask,
/// );
///
/// assert_eq!(mask.len(), points.len());
/// assert!(mask[0]); // First endpoint always kept
/// assert!(mask[2]); // Last endpoint always kept
///
/// // Apply mask to get simplified points
/// let simplified: Vec<_> = points
///     .iter()
///     .zip(mask.iter())
///     .filter_map(|(point, &keep)| if keep { Some(*point) } else { None })
///     .collect();
/// ```
pub fn simplify_dp_mask(
    pts: &[LngLat],
    tolerance_m: f64,
    method: SimplifyMethod,
    mask: &mut Vec<bool>,
) {
    use xt::*;

    match method {
        SimplifyMethod::GreatCircleMeters => {
            let backend = XtGreatCircle;
            dp::simplify_mask(pts, tolerance_m, &backend, mask);
        }
        SimplifyMethod::PlanarMeters => {
            let midpoint = compute_midpoint(pts);
            let backend = XtEnu { origin: midpoint };
            dp::simplify_mask(pts, tolerance_m, &backend, mask);
        }
        SimplifyMethod::EuclidRaw => {
            let backend = XtEuclid;
            dp::simplify_mask(pts, tolerance_m, &backend, mask);
        }
    }
}

/// Compute the arithmetic midpoint of a set of points.
///
/// Used internally for ENU projection origin when using [`SimplifyMethod::PlanarMeters`].
/// Returns (0,0) for empty input.
pub(crate) fn compute_midpoint(pts: &[LngLat]) -> LngLat {
    if pts.is_empty() {
        return LngLat::new_deg(0.0, 0.0);
    }

    let mut sum_lng = 0.0;
    let mut sum_lat = 0.0;

    for pt in pts {
        sum_lng += pt.lng_deg;
        sum_lat += pt.lat_deg;
    }

    LngLat::new_deg(sum_lng / pts.len() as f64, sum_lat / pts.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_midpoint_with_empty_points() {
        let pts = vec![];
        let res = compute_midpoint(&pts);
        assert_eq!(res, LngLat::new_deg(0.0, 0.0));
    }

    #[test]
    fn test_endpoints_always_preserved() {
        let pts = vec![
            LngLat::new_deg(-122.0, 37.0),
            LngLat::new_deg(-121.5, 37.5),
            LngLat::new_deg(-121.0, 37.0),
        ];

        let mut mask = Vec::new();
        simplify_dp_mask(&pts, 1000.0, SimplifyMethod::GreatCircleMeters, &mut mask);

        assert_eq!(mask.len(), 3);
        assert!(mask[0]); // first endpoint
        assert!(mask[2]); // last endpoint
    }

    #[test]
    fn test_zero_length_segments() {
        let pts = vec![
            LngLat::new_deg(-122.0, 37.0),
            LngLat::new_deg(-122.0, 37.0),
            LngLat::new_deg(-122.0, 37.0),
        ];

        let mut mask = Vec::new();
        simplify_dp_mask(&pts, 10.0, SimplifyMethod::GreatCircleMeters, &mut mask);

        for &keep in &mask {
            assert!(keep);
        }
    }

    #[test]
    fn test_tolerance_zero_returns_original() {
        let pts = vec![
            LngLat::new_deg(-122.0, 37.0),
            LngLat::new_deg(-121.9, 37.1),
            LngLat::new_deg(-121.8, 37.2),
            LngLat::new_deg(-121.7, 37.0),
        ];

        let mut mask = Vec::new();
        simplify_dp_mask(&pts, 0.0, SimplifyMethod::GreatCircleMeters, &mut mask);

        for &keep in &mask {
            assert!(keep);
        }
    }

    #[test]
    fn test_very_large_tolerance_returns_endpoints() {
        let pts = vec![
            LngLat::new_deg(-122.0, 37.0),
            LngLat::new_deg(-121.9, 37.1),
            LngLat::new_deg(-121.8, 37.2),
            LngLat::new_deg(-121.7, 37.0),
        ];

        let mut mask = Vec::new();
        simplify_dp_mask(
            &pts,
            1_000_000.0,
            SimplifyMethod::GreatCircleMeters,
            &mut mask,
        );

        assert!(mask[0]); // first endpoint
        assert!(mask[3]); // last endpoint
        assert!(!mask[1] || !mask[2]); // at least one middle point should be removed
    }

    #[test]
    fn test_antimeridian_crossing() {
        let pts = vec![
            LngLat::new_deg(179.0, 0.0),
            LngLat::new_deg(179.5, 0.1),
            LngLat::new_deg(-179.5, 0.2),
            LngLat::new_deg(-179.0, 0.0),
        ];

        let mut mask = Vec::new();
        simplify_dp_mask(&pts, 1000.0, SimplifyMethod::GreatCircleMeters, &mut mask);

        assert!(mask[0]); // first endpoint
        assert!(mask[3]); // last endpoint
    }

    #[test]
    fn test_high_latitude_longitude_squeeze() {
        let pts = vec![
            LngLat::new_deg(-1.0, 89.0),
            LngLat::new_deg(0.0, 89.1),
            LngLat::new_deg(1.0, 89.0),
        ];

        let mut mask = Vec::new();
        simplify_dp_mask(&pts, 1000.0, SimplifyMethod::GreatCircleMeters, &mut mask);

        assert!(mask[0]); // first endpoint
        assert!(mask[2]); // last endpoint
    }

    #[test]
    fn test_simplify_dp_into() {
        let pts = vec![
            LngLat::new_deg(-122.0, 37.0),
            LngLat::new_deg(-121.5, 37.5),
            LngLat::new_deg(-121.0, 37.0),
        ];

        let mut out = Vec::new();
        let count = simplify_dp_into(&pts, 1000.0, SimplifyMethod::GreatCircleMeters, &mut out);

        assert_eq!(count, out.len());
        assert!(count >= 2); // at least endpoints
        assert_eq!(out[0], pts[0]); // first endpoint preserved
        assert_eq!(out[out.len() - 1], pts[pts.len() - 1]); // last endpoint preserved
    }

    #[test]
    fn test_different_methods() {
        let pts = vec![
            LngLat::new_deg(-122.0, 37.0),
            LngLat::new_deg(-121.5, 37.5),
            LngLat::new_deg(-121.0, 37.0),
        ];

        for method in [
            SimplifyMethod::GreatCircleMeters,
            SimplifyMethod::PlanarMeters,
            SimplifyMethod::EuclidRaw,
        ] {
            let mut mask = Vec::new();
            simplify_dp_mask(&pts, 1000.0, method, &mut mask);

            assert!(mask[0]);
            assert!(mask[2]);
        }
    }

    #[test]
    fn test_single_point() {
        let pts = vec![LngLat::new_deg(-122.0, 37.0)];

        let mut mask = Vec::new();
        simplify_dp_mask(&pts, 10.0, SimplifyMethod::GreatCircleMeters, &mut mask);

        assert_eq!(mask.len(), 1);
        assert!(mask[0]);
    }

    #[test]
    fn test_two_points() {
        let pts = vec![LngLat::new_deg(-122.0, 37.0), LngLat::new_deg(-121.0, 37.0)];

        let mut mask = Vec::new();
        simplify_dp_mask(&pts, 10.0, SimplifyMethod::GreatCircleMeters, &mut mask);

        assert_eq!(mask.len(), 2);
        assert!(mask[0]);
        assert!(mask[1]);
    }

    #[test]
    fn test_empty_points() {
        let pts = vec![];

        let mut mask = Vec::new();
        simplify_dp_mask(&pts, 10.0, SimplifyMethod::GreatCircleMeters, &mut mask);

        assert_eq!(mask.len(), 0);
    }
}
