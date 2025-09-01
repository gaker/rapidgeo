//! Batch processing for high-performance distance calculations.
//!
//! This module provides functions for calculating distances on collections of coordinates,
//! with both serial and parallel implementations. Parallel functions require the `batch` feature.
//!
//! # Performance Considerations
//!
//! **Serial vs Parallel:**
//! - **Serial**: Better for small datasets (< 1,000 points) due to no threading overhead
//! - **Parallel**: Better for large datasets (> 10,000 points) when multiple CPU cores available
//! - **Breakeven point**: Usually around 1,000-5,000 points depending on CPU and calculation type
//!
//! **Memory Allocation:**
//! - Functions ending in `_into` write to pre-allocated buffers (faster, no allocation)
//! - Functions without `_into` allocate and return new vectors (convenient but slower)
//! - Use `_into` variants for hot paths and high-frequency calculations
//!
//! **Feature Requirements:**
//! - Basic functions: No features required
//! - `*_par*` functions: Require `batch` feature (enables Rayon parallel processing)
//! - `*_vincenty*` functions: Require `vincenty` feature
//! - Combined functions: Require both features (`batch` + `vincenty`)
//!
//! # Examples
//!
//! ```no_run
//! use rapidgeo_distance::{LngLat, batch::*};
//!
//! let points = vec![
//!     LngLat::new_deg(-122.4194, 37.7749), // San Francisco  
//!     LngLat::new_deg(-74.0060, 40.7128),  // New York
//!     LngLat::new_deg(-87.6298, 41.8781),  // Chicago
//! ];
//!
//! // Serial path calculation
//! let total_distance = path_length_haversine(&points);
//!
//! // Parallel calculation (requires batch feature)
//! #[cfg(feature = "batch")]
//! let distances = pairwise_haversine_par(&points);
//!
//! // Memory-efficient calculation
//! let mut buffer = vec![0.0; points.len() - 1];
//! pairwise_haversine_into(&points, &mut buffer);
//! ```

use crate::{geodesic, LngLat};

#[cfg(feature = "batch")]
use rayon::prelude::*;

/// Calculates haversine distances between consecutive points in a path.
///
/// Returns an iterator over the distances between each pair of consecutive points.
/// Memory-efficient as it processes points lazily without allocating a result vector.
///
/// # Arguments
///
/// * `pts` - Slice of coordinates representing a path
///
/// # Returns
///
/// Iterator yielding distances in meters. Length will be `pts.len() - 1`.
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::{LngLat, batch::pairwise_haversine};
///
/// let path = [
///     LngLat::new_deg(0.0, 0.0),
///     LngLat::new_deg(1.0, 0.0),
///     LngLat::new_deg(1.0, 1.0),
/// ];
///
/// let distances: Vec<f64> = pairwise_haversine(&path).collect();
/// assert_eq!(distances.len(), 2);
/// // Each distance is roughly 111 km (1 degree)
/// assert!(distances[0] > 110_000.0 && distances[0] < 112_000.0);
/// ```
pub fn pairwise_haversine(pts: &[LngLat]) -> impl Iterator<Item = f64> + '_ {
    pts.windows(2)
        .map(|pair| geodesic::haversine(pair[0], pair[1]))
}

/// Calculates the total haversine distance along a path.
///
/// Sums all consecutive point-to-point distances using the haversine formula.
/// Equivalent to `pairwise_haversine(pts).sum()` but more convenient.
///
/// # Arguments
///
/// * `pts` - Slice of coordinates representing a path
///
/// # Returns
///
/// Total path length in meters
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::{LngLat, batch::path_length_haversine};
///
/// let path = [
///     LngLat::new_deg(0.0, 0.0),
///     LngLat::new_deg(1.0, 0.0),
///     LngLat::new_deg(1.0, 1.0),
/// ];
///
/// let total_distance = path_length_haversine(&path);
/// // Two 1-degree segments ≈ 222 km total
/// assert!(total_distance > 220_000.0 && total_distance < 224_000.0);
/// ```
pub fn path_length_haversine(pts: &[LngLat]) -> f64 {
    pairwise_haversine(pts).sum()
}

/// Parallel version of [`pairwise_haversine`] that returns a vector.
///
/// Uses Rayon for parallel processing. More efficient for large datasets (>1,000 points)
/// but has overhead for small datasets. Requires the `batch` feature.
///
/// # Arguments
///
/// * `pts` - Slice of coordinates representing a path
///
/// # Returns
///
/// Vector of distances in meters. Length will be `pts.len() - 1`.
///
/// # Examples
///
/// ```no_run
/// # #[cfg(feature = "batch")]
/// # {
/// use rapidgeo_distance::{LngLat, batch::pairwise_haversine_par};
///
/// let path: Vec<LngLat> = (0..10000)
///     .map(|i| LngLat::new_deg(i as f64 * 0.001, 0.0))
///     .collect();
///
/// // Parallel processing beneficial for large datasets
/// let distances = pairwise_haversine_par(&path);
/// assert_eq!(distances.len(), path.len() - 1);
/// # }
/// ```
#[cfg(feature = "batch")]
pub fn pairwise_haversine_par(pts: &[LngLat]) -> Vec<f64> {
    pts.windows(2)
        .collect::<Vec<_>>()
        .par_iter()
        .map(|pair| geodesic::haversine(pair[0], pair[1]))
        .collect()
}

/// Parallel version of [`path_length_haversine`] that processes using [Rayon](https://docs.rs/rayon/).
///
/// Uses [Rayon](https://docs.rs/rayon/) for parallel processing of path segments across multiple CPU cores.
/// More efficient for large datasets (>10,000 points) but has threading overhead for small datasets.
/// Requires the `batch` feature.
///
/// # Arguments
///
/// * `pts` - Slice of coordinates representing a path
///
/// # Returns
///
/// Total path length in meters
///
/// # Examples
///
/// ```no_run
/// # #[cfg(feature = "batch")]
/// # {
/// use rapidgeo_distance::{LngLat, batch::path_length_haversine_par};
///
/// let path: Vec<LngLat> = (0..10000)
///     .map(|i| LngLat::new_deg(i as f64 * 0.001, 0.0))
///     .collect();
///
/// // Parallel processing beneficial for large datasets
/// let total_distance = path_length_haversine_par(&path);
/// assert!(total_distance > 0.0);
/// # }
/// ```
///
/// # Performance Notes
///
/// Parallel processing is most beneficial when:
/// - Dataset size > 1,000 points
/// - Multiple CPU cores are available
/// - CPU is not already saturated with other work
#[cfg(feature = "batch")]
pub fn path_length_haversine_par(pts: &[LngLat]) -> f64 {
    pts.windows(2)
        .collect::<Vec<_>>()
        .par_iter()
        .map(|pair| geodesic::haversine(pair[0], pair[1]))
        .sum()
}

/// Calculates distances from multiple points to a single target point in parallel.
///
/// Uses [Rayon](https://docs.rs/rayon/) to compute haversine distances from each point
/// to the target in parallel. Beneficial for large point sets (>1,000 points).
///
/// # Arguments
///
/// * `points` - Slice of coordinates to measure from
/// * `target` - Target coordinate to measure to
///
/// # Returns
///
/// Vector of distances in meters, same length as input points
///
/// # Examples
///
/// ```no_run
/// # #[cfg(feature = "batch")]
/// # {
/// use rapidgeo_distance::{LngLat, batch::distances_to_point_par};
///
/// let points: Vec<LngLat> = vec![
///     LngLat::new_deg(-122.4194, 37.7749), // San Francisco
///     LngLat::new_deg(-74.0060, 40.7128),  // New York
///     LngLat::new_deg(-87.6298, 41.8781),  // Chicago
/// ];
/// let target = LngLat::new_deg(0.0, 0.0); // Prime Meridian
///
/// let distances = distances_to_point_par(&points, target);
/// assert_eq!(distances.len(), points.len());
/// # }
/// ```
#[cfg(feature = "batch")]
pub fn distances_to_point_par(points: &[LngLat], target: LngLat) -> Vec<f64> {
    points
        .par_iter()
        .map(|&p| geodesic::haversine(p, target))
        .collect()
}

/// Calculates high-precision distances from multiple points to a target using Vincenty in parallel.
///
/// Uses [Rayon](https://docs.rs/rayon/) to compute [Vincenty distances](https://en.wikipedia.org/wiki/Vincenty%27s_formulae)
/// from each point to the target in parallel. Provides ±1mm accuracy but slower than haversine.
/// Requires both `batch` and `vincenty` features.
///
/// # Arguments
///
/// * `points` - Slice of coordinates to measure from  
/// * `target` - Target coordinate to measure to
///
/// # Returns
///
/// - `Ok(Vec<f64>)` - Vector of distances in meters, same length as input points
/// - `Err(VincentyError)` - If any calculation fails (antipodal points, invalid coordinates)
///
/// # Examples
///
/// ```no_run
/// # #[cfg(all(feature = "batch", feature = "vincenty"))]
/// # {
/// use rapidgeo_distance::{LngLat, batch::distances_to_point_vincenty_par};
///
/// let points = vec![
///     LngLat::new_deg(-122.4194, 37.7749), // San Francisco  
///     LngLat::new_deg(-74.0060, 40.7128),  // New York
/// ];
/// let target = LngLat::new_deg(0.0, 0.0);
///
/// match distances_to_point_vincenty_par(&points, target) {
///     Ok(distances) => println!("Precise distances: {:?}", distances),
///     Err(e) => eprintln!("Calculation failed: {:?}", e),
/// }
/// # }
/// ```
#[cfg(feature = "batch")]
pub fn distances_to_point_vincenty_par(
    points: &[LngLat],
    target: LngLat,
) -> Result<Vec<f64>, geodesic::VincentyError> {
    points
        .par_iter()
        .map(|&p| geodesic::vincenty_distance_m(p, target))
        .collect()
}

/// Calculates haversine distances between consecutive points, writing to a pre-allocated buffer.
///
/// Memory-efficient version of [`pairwise_haversine`] that writes results to an existing buffer
/// instead of allocating a new vector. Useful for high-frequency calculations or when
/// memory allocation should be avoided.
///
/// # Arguments
///
/// * `pts` - Slice of coordinates representing a path
/// * `output` - Mutable slice to write distances to (must be at least `pts.len() - 1` long)
///
/// # Panics
///
/// Panics if output buffer is too small to hold all results.
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::{LngLat, batch::pairwise_haversine_into};
///
/// let path = [
///     LngLat::new_deg(0.0, 0.0),
///     LngLat::new_deg(1.0, 0.0),
///     LngLat::new_deg(1.0, 1.0),
/// ];
///
/// let mut distances = vec![0.0; path.len() - 1];
/// pairwise_haversine_into(&path, &mut distances);
///
/// // distances now contains the calculated values
/// assert_eq!(distances.len(), 2);
/// assert!(distances[0] > 100_000.0); // ~1 degree
/// ```
pub fn pairwise_haversine_into(pts: &[LngLat], output: &mut [f64]) {
    assert!(
        output.len() >= pts.len().saturating_sub(1),
        "Output buffer too small: need {}, got {}",
        pts.len().saturating_sub(1),
        output.len()
    );

    for (i, pair) in pts.windows(2).enumerate() {
        output[i] = geodesic::haversine(pair[0], pair[1]);
    }
}

/// Calculates distances between consecutive point pairs using parallel processing.
///
/// Computes the [Haversine distance](https://en.wikipedia.org/wiki/Haversine_formula) between each
/// consecutive pair of points in the input slice, writing results directly to the output buffer.
/// Uses [Rayon](https://docs.rs/rayon/) for parallel computation when processing large datasets.
///
/// # Arguments
///
/// * `pts` - Slice of points to process
/// * `output` - Mutable buffer to write distances (must have length ≥ pts.len() - 1)
///
/// # Panics
///
/// Panics if output buffer is too small
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::{LngLat, batch::pairwise_haversine_par_into};
///
/// let points = [
///     LngLat::new_deg(-122.4194, 37.7749), // San Francisco
///     LngLat::new_deg(-87.6298, 41.8781),  // Chicago  
///     LngLat::new_deg(-74.0060, 40.7128),  // New York
/// ];
///
/// let mut distances = vec![0.0; points.len() - 1];
/// pairwise_haversine_par_into(&points, &mut distances);
///
/// assert!(distances[0] > 2900000.0); // SF to Chicago ~2900km
/// assert!(distances[1] > 1100000.0); // Chicago to NYC ~1100km
/// ```
///
/// # Performance
///
/// Uses parallel processing for better performance on large datasets.
/// Consider using [`pairwise_haversine_into`] for small datasets (< 1000 points) to avoid parallelization overhead.
#[cfg(feature = "batch")]
pub fn pairwise_haversine_par_into(pts: &[LngLat], output: &mut [f64]) {
    assert!(
        output.len() >= pts.len().saturating_sub(1),
        "Output buffer too small: need {}, got {}",
        pts.len().saturating_sub(1),
        output.len()
    );

    let windows: Vec<_> = pts.windows(2).collect();
    output[..pts.len().saturating_sub(1)]
        .par_iter_mut()
        .zip(windows.par_iter())
        .for_each(|(out, pair)| {
            *out = geodesic::haversine(pair[0], pair[1]);
        });
}

/// Calculates distances from multiple points to a single target point.
///
/// Computes the [Haversine distance](https://en.wikipedia.org/wiki/Haversine_formula) from each point
/// in the input slice to the target point, writing results directly to the output buffer.
/// Uses sequential processing - see [`distances_to_point_par_into`] for parallel version.
///
/// # Arguments
///
/// * `points` - Slice of points to measure from
/// * `target` - Target point to measure distances to
/// * `output` - Mutable buffer to write distances (must have length ≥ points.len())
///
/// # Panics
///
/// Panics if output buffer is too small
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::{LngLat, batch::distances_to_point_into};
///
/// let points = [
///     LngLat::new_deg(-122.4194, 37.7749), // San Francisco
///     LngLat::new_deg(-87.6298, 41.8781),  // Chicago
///     LngLat::new_deg(-0.1278, 51.5074),   // London
/// ];
/// let target = LngLat::new_deg(-74.0060, 40.7128); // New York
///
/// let mut distances = vec![0.0; points.len()];
/// distances_to_point_into(&points, target, &mut distances);
///
/// assert!(distances[0] > 4100000.0); // SF to NYC ~4100km
/// assert!(distances[1] > 1100000.0); // Chicago to NYC ~1100km  
/// assert!(distances[2] > 5500000.0); // London to NYC ~5500km
/// ```
///
/// # Performance
///
/// Sequential processing suitable for small to medium datasets.
/// Use [`distances_to_point_par_into`] for better performance on large datasets.
pub fn distances_to_point_into(points: &[LngLat], target: LngLat, output: &mut [f64]) {
    assert!(
        output.len() >= points.len(),
        "Output buffer too small: need {}, got {}",
        points.len(),
        output.len()
    );

    for (i, &point) in points.iter().enumerate() {
        output[i] = geodesic::haversine(point, target);
    }
}

/// Calculates distances from multiple points to a single target point using parallel processing.
///
/// Computes the [Haversine distance](https://en.wikipedia.org/wiki/Haversine_formula) from each point
/// in the input slice to the target point, writing results directly to the output buffer.
/// Uses [Rayon](https://docs.rs/rayon/) for parallel computation when processing large datasets.
///
/// # Arguments
///
/// * `points` - Slice of points to measure from
/// * `target` - Target point to measure distances to  
/// * `output` - Mutable buffer to write distances (must have length ≥ points.len())
///
/// # Panics
///
/// Panics if output buffer is too small
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::{LngLat, batch::distances_to_point_par_into};
///
/// let points = [
///     LngLat::new_deg(-122.4194, 37.7749), // San Francisco
///     LngLat::new_deg(-87.6298, 41.8781),  // Chicago
///     LngLat::new_deg(-0.1278, 51.5074),   // London
/// ];
/// let target = LngLat::new_deg(-74.0060, 40.7128); // New York
///
/// let mut distances = vec![0.0; points.len()];
/// distances_to_point_par_into(&points, target, &mut distances);
///
/// assert!(distances[0] > 4100000.0); // SF to NYC ~4100km
/// assert!(distances[1] > 1100000.0); // Chicago to NYC ~1100km
/// assert!(distances[2] > 5500000.0); // London to NYC ~5500km
/// ```
///
/// # Performance
///
/// Uses parallel processing for better performance on large datasets.
/// Consider using [`distances_to_point_into`] for small datasets (< 1000 points) to avoid parallelization overhead.
#[cfg(feature = "batch")]
pub fn distances_to_point_par_into(points: &[LngLat], target: LngLat, output: &mut [f64]) {
    assert!(
        output.len() >= points.len(),
        "Output buffer too small: need {}, got {}",
        points.len(),
        output.len()
    );

    output[..points.len()]
        .par_iter_mut()
        .zip(points.par_iter())
        .for_each(|(out, &point)| {
            *out = geodesic::haversine(point, target);
        });
}

/// Calculates high-precision distances from multiple points to a single target point.
///
/// Computes distances using [Vincenty's formulae](https://en.wikipedia.org/wiki/Vincenty%27s_formulae)
/// for the [WGS84 ellipsoid](https://en.wikipedia.org/wiki/World_Geodetic_System),
/// providing millimeter accuracy at the cost of slower computation.
/// Uses sequential processing - see [`distances_to_point_vincenty_par_into`] for parallel version.
///
/// # Arguments
///
/// * `points` - Slice of points to measure from
/// * `target` - Target point to measure distances to
/// * `output` - Mutable buffer to write distances (must have length ≥ points.len())
///
/// # Returns
///
/// `Ok(())` on success, `Err(VincentyError)` if algorithm fails to converge for any point pair
///
/// # Panics
///
/// Panics if output buffer is too small
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::{LngLat, batch::distances_to_point_vincenty_into};
///
/// let points = [
///     LngLat::new_deg(-122.4194, 37.7749), // San Francisco
///     LngLat::new_deg(-87.6298, 41.8781),  // Chicago
/// ];
/// let target = LngLat::new_deg(-74.0060, 40.7128); // New York
///
/// let mut distances = vec![0.0; points.len()];
/// distances_to_point_vincenty_into(&points, target, &mut distances).unwrap();
///
/// // Vincenty provides millimeter precision
/// assert!(distances[0] > 4100000.0 && distances[0] < 4200000.0); // SF to NYC ~4150km
/// assert!(distances[1] > 1100000.0 && distances[1] < 1200000.0); // Chicago to NYC ~1150km
/// ```
///
/// # Errors
///
/// Returns [`geodesic::VincentyError::DidNotConverge`] for nearly antipodal points (opposite sides of Earth).
/// Consider using [`distances_to_point_into`] with Haversine as a fallback for such cases.
///
/// # Performance
///
/// Slower than Haversine but much more accurate. Sequential processing suitable for small to medium datasets.
#[cfg(feature = "vincenty")]
pub fn distances_to_point_vincenty_into(
    points: &[LngLat],
    target: LngLat,
    output: &mut [f64],
) -> Result<(), geodesic::VincentyError> {
    assert!(
        output.len() >= points.len(),
        "Output buffer too small: need {}, got {}",
        points.len(),
        output.len()
    );

    for (i, &point) in points.iter().enumerate() {
        output[i] = geodesic::vincenty_distance_m(point, target)?;
    }
    Ok(())
}

/// Calculates high-precision distances from multiple points to a single target point using parallel processing.
///
/// Computes distances using [Vincenty's formulae](https://en.wikipedia.org/wiki/Vincenty%27s_formulae)
/// for the [WGS84 ellipsoid](https://en.wikipedia.org/wiki/World_Geodetic_System),
/// providing millimeter accuracy. Uses [Rayon](https://docs.rs/rayon/) for parallel computation.
///
/// # Arguments
///
/// * `points` - Slice of points to measure from
/// * `target` - Target point to measure distances to
/// * `output` - Mutable buffer to write distances (must have length ≥ points.len())
///
/// # Returns
///
/// `Ok(())` on success, `Err(VincentyError)` if algorithm fails to converge for any point pair
///
/// # Panics
///
/// Panics if output buffer is too small
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::{LngLat, batch::distances_to_point_vincenty_par_into};
///
/// let points = [
///     LngLat::new_deg(-122.4194, 37.7749), // San Francisco
///     LngLat::new_deg(-87.6298, 41.8781),  // Chicago
///     LngLat::new_deg(-0.1278, 51.5074),   // London
/// ];
/// let target = LngLat::new_deg(-74.0060, 40.7128); // New York
///
/// let mut distances = vec![0.0; points.len()];
/// distances_to_point_vincenty_par_into(&points, target, &mut distances).unwrap();
///
/// // Vincenty provides millimeter precision
/// assert!(distances[0] > 4100000.0 && distances[0] < 4200000.0); // SF to NYC ~4150km
/// ```
///
/// # Errors
///
/// Returns [`geodesic::VincentyError::DidNotConverge`] for nearly antipodal points.
/// All points are processed in parallel, but if any fail, the entire operation fails.
///
/// # Performance
///
/// Uses parallel processing for better performance on large datasets requiring high precision.
/// Consider [`distances_to_point_vincenty_into`] for small datasets to avoid parallelization overhead.
#[cfg(all(feature = "batch", feature = "vincenty"))]
pub fn distances_to_point_vincenty_par_into(
    points: &[LngLat],
    target: LngLat,
    output: &mut [f64],
) -> Result<(), geodesic::VincentyError> {
    assert!(
        output.len() >= points.len(),
        "Output buffer too small: need {}, got {}",
        points.len(),
        output.len()
    );

    // For Vincenty, we need to handle errors properly, so we can't use par_iter_mut easily
    // We'll collect Results first, then handle the error
    let results: Result<Vec<_>, _> = points
        .par_iter()
        .map(|&point| geodesic::vincenty_distance_m(point, target))
        .collect();

    match results {
        Ok(distances) => {
            output[..points.len()].copy_from_slice(&distances);
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Calculates the total length of a path using high-precision geodesic distances.
///
/// Computes the sum of distances between consecutive points using [Vincenty's formulae](https://en.wikipedia.org/wiki/Vincenty%27s_formulae)
/// for the [WGS84 ellipsoid](https://en.wikipedia.org/wiki/World_Geodetic_System).
/// Provides millimeter accuracy for measuring GPS tracks, routes, and geographic paths.
///
/// # Arguments
///
/// * `pts` - Slice of points defining the path (minimum 2 points)
///
/// # Returns
///
/// `Ok(total_length_meters)` on success, `Err(VincentyError)` if algorithm fails to converge for any segment
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::{LngLat, batch::path_length_vincenty_m};
///
/// // GPS track from San Francisco to NYC via Chicago
/// let path = [
///     LngLat::new_deg(-122.4194, 37.7749), // San Francisco
///     LngLat::new_deg(-87.6298, 41.8781),  // Chicago
///     LngLat::new_deg(-74.0060, 40.7128),  // New York City
/// ];
///
/// let total_length = path_length_vincenty_m(&path).unwrap();
/// assert!(total_length > 4000000.0); // > 4000km total
/// ```
///
/// # Use Cases
///
/// - GPS track analysis with millimeter precision
/// - Route planning and optimization
/// - Geographic survey measurements
/// - Research requiring geodetic accuracy
///
/// # Errors
///
/// Returns [`geodesic::VincentyError::DidNotConverge`] for paths containing nearly antipodal segments.
/// Consider using `path_length_haversine_m` as a fallback for such cases.
///
/// # Performance
///
/// Slower than Haversine-based path length calculation but provides surveyor-grade accuracy.
/// For paths with thousands of points, consider chunking or using approximate methods for real-time applications.
#[cfg(feature = "vincenty")]
pub fn path_length_vincenty_m(pts: &[LngLat]) -> Result<f64, geodesic::VincentyError> {
    let mut total = 0.0;
    for pair in pts.windows(2) {
        total += geodesic::vincenty_distance_m(pair[0], pair[1])?;
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pairwise_haversine() {
        let pts = [
            LngLat::new_deg(0.0, 0.0),
            LngLat::new_deg(1.0, 0.0),
            LngLat::new_deg(1.0, 1.0),
        ];

        let distances: Vec<f64> = pairwise_haversine(&pts).collect();
        assert_eq!(distances.len(), 2);
        assert!(distances[0] > 110000.0 && distances[0] < 112000.0); // ~1° longitude at equator
        assert!(distances[1] > 110000.0 && distances[1] < 112000.0); // ~1° latitude
    }

    #[test]
    fn test_path_length_haversine() {
        let pts = [
            LngLat::new_deg(0.0, 0.0),
            LngLat::new_deg(1.0, 0.0),
            LngLat::new_deg(1.0, 1.0),
        ];

        let total = path_length_haversine(&pts);
        assert!(total > 220000.0 && total < 224000.0); // Sum of two ~111km segments
    }

    #[test]
    #[cfg(feature = "vincenty")]
    fn test_path_length_vincenty_m() {
        let pts = [
            LngLat::new_deg(0.0, 0.0),
            LngLat::new_deg(1.0, 0.0),
            LngLat::new_deg(1.0, 1.0),
        ];

        let total = path_length_vincenty_m(&pts).unwrap();
        assert!(total > 220000.0 && total < 224000.0); // Sum of two ~111km segments
    }

    #[test]
    #[cfg(feature = "batch")]
    fn test_pairwise_haversine_par() {
        let pts = [
            LngLat::new_deg(0.0, 0.0),
            LngLat::new_deg(1.0, 0.0),
            LngLat::new_deg(1.0, 1.0),
        ];

        let distances = pairwise_haversine_par(&pts);
        assert_eq!(distances.len(), 2);
        assert!(distances[0] > 110000.0 && distances[0] < 112000.0);
        assert!(distances[1] > 110000.0 && distances[1] < 112000.0);

        let serial_distances: Vec<f64> = pairwise_haversine(&pts).collect();
        assert_eq!(distances, serial_distances);
    }

    #[test]
    #[cfg(feature = "batch")]
    fn test_path_length_haversine_par() {
        let pts = [
            LngLat::new_deg(0.0, 0.0),
            LngLat::new_deg(1.0, 0.0),
            LngLat::new_deg(1.0, 1.0),
        ];

        let total_par = path_length_haversine_par(&pts);
        let total_serial = path_length_haversine(&pts);

        assert!((total_par - total_serial).abs() < 1e-6);
        assert!(total_par > 220000.0 && total_par < 224000.0);
    }

    #[test]
    #[cfg(feature = "batch")]
    fn test_distances_to_point_par() {
        let points = [
            LngLat::new_deg(0.0, 0.0),
            LngLat::new_deg(1.0, 0.0),
            LngLat::new_deg(0.0, 1.0),
        ];
        let target = LngLat::new_deg(0.5, 0.5);

        let distances = distances_to_point_par(&points, target);
        assert_eq!(distances.len(), 3);

        for (i, &distance) in distances.iter().enumerate() {
            let expected = geodesic::haversine(points[i], target);
            assert!((distance - expected).abs() < 1e-6);
        }
    }

    #[test]
    #[cfg(all(feature = "batch", feature = "vincenty"))]
    fn test_distances_to_point_vincenty_par() {
        let points = [
            LngLat::new_deg(0.0, 0.0),
            LngLat::new_deg(1.0, 0.0),
            LngLat::new_deg(0.0, 1.0),
        ];
        let target = LngLat::new_deg(0.5, 0.5);

        let distances = distances_to_point_vincenty_par(&points, target).unwrap();
        assert_eq!(distances.len(), 3);

        for (i, &distance) in distances.iter().enumerate() {
            let expected = geodesic::vincenty_distance_m(points[i], target).unwrap();
            assert!((distance - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_pairwise_haversine_into() {
        let pts = [
            LngLat::new_deg(0.0, 0.0),
            LngLat::new_deg(1.0, 0.0),
            LngLat::new_deg(1.0, 1.0),
        ];

        let mut output = vec![0.0; 2];
        pairwise_haversine_into(&pts, &mut output);

        let expected: Vec<f64> = pairwise_haversine(&pts).collect();
        assert_eq!(output, expected);

        assert!(output[0] > 110000.0 && output[0] < 112000.0);
        assert!(output[1] > 110000.0 && output[1] < 112000.0);
    }

    #[test]
    #[should_panic(expected = "Output buffer too small")]
    fn test_pairwise_haversine_into_buffer_too_small() {
        let pts = [
            LngLat::new_deg(0.0, 0.0),
            LngLat::new_deg(1.0, 0.0),
            LngLat::new_deg(1.0, 1.0),
        ];
        let mut output = vec![0.0; 1]; // Too small!
        pairwise_haversine_into(&pts, &mut output);
    }

    #[test]
    #[cfg(feature = "batch")]
    fn test_pairwise_haversine_par_into() {
        let pts = [
            LngLat::new_deg(0.0, 0.0),
            LngLat::new_deg(1.0, 0.0),
            LngLat::new_deg(1.0, 1.0),
        ];

        let mut output_par = vec![0.0; 2];
        let mut output_serial = vec![0.0; 2];

        pairwise_haversine_par_into(&pts, &mut output_par);
        pairwise_haversine_into(&pts, &mut output_serial);

        assert_eq!(output_par, output_serial);
    }

    #[test]
    fn test_distances_to_point_into() {
        let points = [
            LngLat::new_deg(0.0, 0.0),
            LngLat::new_deg(1.0, 0.0),
            LngLat::new_deg(0.0, 1.0),
        ];
        let target = LngLat::new_deg(0.5, 0.5);

        let mut output = vec![0.0; 3];
        distances_to_point_into(&points, target, &mut output);

        for (i, &distance) in output.iter().enumerate() {
            let expected = geodesic::haversine(points[i], target);
            assert!((distance - expected).abs() < 1e-6);
        }
    }

    #[test]
    #[cfg(feature = "batch")]
    fn test_distances_to_point_par_into() {
        let points = [
            LngLat::new_deg(0.0, 0.0),
            LngLat::new_deg(1.0, 0.0),
            LngLat::new_deg(0.0, 1.0),
        ];
        let target = LngLat::new_deg(0.5, 0.5);

        let mut output_par = vec![0.0; 3];
        let mut output_serial = vec![0.0; 3];

        distances_to_point_par_into(&points, target, &mut output_par);
        distances_to_point_into(&points, target, &mut output_serial);

        for (par, serial) in output_par.iter().zip(output_serial.iter()) {
            assert!((par - serial).abs() < 1e-6);
        }
    }

    #[test]
    #[cfg(feature = "vincenty")]
    fn test_distances_to_point_vincenty_into() {
        let points = [
            LngLat::new_deg(0.0, 0.0),
            LngLat::new_deg(1.0, 0.0),
            LngLat::new_deg(0.0, 1.0),
        ];
        let target = LngLat::new_deg(0.5, 0.5);

        let mut output = vec![0.0; 3];
        distances_to_point_vincenty_into(&points, target, &mut output).unwrap();

        for (i, &distance) in output.iter().enumerate() {
            let expected = geodesic::vincenty_distance_m(points[i], target).unwrap();
            assert!((distance - expected).abs() < 1e-6);
        }
    }

    #[test]
    #[cfg(all(feature = "batch", feature = "vincenty"))]
    fn test_distances_to_point_vincenty_par_into() {
        let points = [
            LngLat::new_deg(0.0, 0.0),
            LngLat::new_deg(1.0, 0.0),
            LngLat::new_deg(0.0, 1.0),
        ];
        let target = LngLat::new_deg(0.5, 0.5);

        let mut output_par = vec![0.0; 3];
        let mut output_serial = vec![0.0; 3];

        distances_to_point_vincenty_par_into(&points, target, &mut output_par).unwrap();
        distances_to_point_vincenty_into(&points, target, &mut output_serial).unwrap();

        for (par, serial) in output_par.iter().zip(output_serial.iter()) {
            assert!((par - serial).abs() < 1e-6);
        }
    }

    #[test]
    fn test_into_functions_with_larger_buffers() {
        let points = [LngLat::new_deg(0.0, 0.0), LngLat::new_deg(1.0, 0.0)];
        let target = LngLat::new_deg(0.5, 0.5);

        let mut output = vec![f64::NAN; 5]; // Larger than needed
        distances_to_point_into(&points, target, &mut output);

        assert!(!output[0].is_nan());
        assert!(!output[1].is_nan());
        assert!(output[2].is_nan()); // Unchanged
        assert!(output[3].is_nan()); // Unchanged
        assert!(output[4].is_nan()); // Unchanged
    }
}
