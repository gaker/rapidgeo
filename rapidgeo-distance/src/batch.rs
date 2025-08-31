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

#[cfg(feature = "batch")]
pub fn path_length_haversine_par(pts: &[LngLat]) -> f64 {
    pts.windows(2)
        .collect::<Vec<_>>()
        .par_iter()
        .map(|pair| geodesic::haversine(pair[0], pair[1]))
        .sum()
}

#[cfg(feature = "batch")]
pub fn distances_to_point_par(points: &[LngLat], target: LngLat) -> Vec<f64> {
    points
        .par_iter()
        .map(|&p| geodesic::haversine(p, target))
        .collect()
}

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
