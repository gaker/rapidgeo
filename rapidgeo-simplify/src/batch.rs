//! Batch and parallel processing for multiple polylines.
//!
//! This module provides functions to simplify multiple polylines efficiently,
//! with optional parallel processing using [Rayon](https://docs.rs/rayon/).
//!
//! # Features
//!
//! - Batch processing of multiple polylines
//! - Parallel processing with automatic work stealing
//! - Memory-efficient "into" variants that reuse output buffers
//! - Hybrid serial/parallel processing based on segment size
//!
//! # Examples
//!
//! ```rust
//! use rapidgeo_distance::LngLat;
//! use rapidgeo_simplify::{batch::simplify_batch, SimplifyMethod};
//!
//! let polylines = vec![
//!     vec![
//!         LngLat::new_deg(-122.0, 37.0),
//!         LngLat::new_deg(-121.5, 37.5),
//!         LngLat::new_deg(-121.0, 37.0),
//!     ],
//!     vec![
//!         LngLat::new_deg(-74.0, 40.0),
//!         LngLat::new_deg(-73.5, 40.5),
//!         LngLat::new_deg(-73.0, 40.0),
//!     ],
//! ];
//!
//! let simplified = simplify_batch(
//!     &polylines,
//!     1000.0, // 1km tolerance
//!     SimplifyMethod::GreatCircleMeters,
//! );
//!
//! assert_eq!(simplified.len(), polylines.len());
//! ```

use crate::{xt::PerpDistance, SimplifyMethod};
use rapidgeo_distance::LngLat;

#[cfg(feature = "batch")]
use rayon::prelude::*;

/// Threshold for switching between serial and parallel distance calculations.
///
/// Segments with fewer candidate points use serial processing to avoid
/// the overhead of parallel task creation.
const PARALLEL_DISTANCE_THRESHOLD: usize = 100;

/// Errors that can occur during batch processing operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchError {
    /// The provided output buffer is too small for the input data.
    ///
    /// This occurs when using the "_into" variants with pre-allocated buffers
    /// that don't have enough capacity for all input polylines.
    BufferTooSmall {
        /// Number of slots needed in the output buffer
        needed: usize,
        /// Number of slots actually provided
        provided: usize,
    },
}

/// Simplify multiple polylines in parallel.
///
/// Uses Rayon to process polylines across multiple threads with automatic
/// work stealing for optimal load balancing.
///
/// # Arguments
///
/// * `polylines` - Slice of input polylines to simplify
/// * `tolerance_m` - Distance threshold for all polylines
/// * `method` - Distance calculation method to use
///
/// # Returns
///
/// Vector of simplified polylines in the same order as input
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "batch")]
/// # {
/// use rapidgeo_distance::LngLat;
/// use rapidgeo_simplify::{batch::simplify_batch_par, SimplifyMethod};
///
/// let polylines = vec![
///     vec![
///         LngLat::new_deg(-122.0, 37.0),
///         LngLat::new_deg(-121.0, 37.0),
///     ],
///     // ... more polylines
/// ];
///
/// let simplified = simplify_batch_par(
///     &polylines,
///     1000.0,
///     SimplifyMethod::GreatCircleMeters,
/// );
/// # }
/// ```
///
/// # Performance
///
/// Parallel processing provides the most benefit when:
/// - Processing many polylines (>= 10)
/// - Polylines have many points (>= 1000 each)
/// - Using computationally expensive distance methods (GreatCircleMeters)
#[cfg(feature = "batch")]
pub fn simplify_batch_par(
    polylines: &[Vec<LngLat>],
    tolerance_m: f64,
    method: SimplifyMethod,
) -> Vec<Vec<LngLat>> {
    polylines
        .par_iter()
        .map(|polyline| {
            let mut simplified = Vec::new();
            crate::simplify_dp_into(polyline, tolerance_m, method, &mut simplified);
            simplified
        })
        .collect()
}

/// Simplify multiple polylines in parallel into pre-allocated output buffers.
///
/// This variant avoids allocating new vectors for the output, instead reusing
/// the provided output slice. Each output vector is cleared before use.
///
/// # Arguments
///
/// * `polylines` - Input polylines to simplify
/// * `tolerance_m` - Distance threshold
/// * `method` - Distance calculation method
/// * `output` - Pre-allocated output vectors (must have at least `polylines.len()` capacity)
///
/// # Errors
///
/// Returns [`BatchError::BufferTooSmall`] if the output slice has fewer
/// elements than the input polylines slice.
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "batch")]
/// # {
/// use rapidgeo_distance::LngLat;
/// use rapidgeo_simplify::{batch::simplify_batch_par_into, SimplifyMethod};
///
/// let polylines = vec![
///     vec![LngLat::new_deg(-122.0, 37.0), LngLat::new_deg(-121.0, 37.0)],
/// ];
///
/// let mut output = vec![Vec::new(); polylines.len()];
/// let result = simplify_batch_par_into(
///     &polylines,
///     1000.0,
///     SimplifyMethod::GreatCircleMeters,
///     &mut output,
/// );
///
/// assert!(result.is_ok());
/// assert_eq!(output[0].len(), 2); // Both endpoints kept
/// # }
/// ```
#[cfg(feature = "batch")]
pub fn simplify_batch_par_into(
    polylines: &[Vec<LngLat>],
    tolerance_m: f64,
    method: SimplifyMethod,
    output: &mut [Vec<LngLat>],
) -> Result<(), BatchError> {
    if output.len() < polylines.len() {
        return Err(BatchError::BufferTooSmall {
            needed: polylines.len(),
            provided: output.len(),
        });
    }

    output[..polylines.len()]
        .par_iter_mut()
        .zip(polylines.par_iter())
        .for_each(|(out_polyline, in_polyline)| {
            crate::simplify_dp_into(in_polyline, tolerance_m, method, out_polyline);
        });

    Ok(())
}

/// Simplify multiple polylines sequentially.
///
/// Processes polylines one at a time in a single thread. Use this when:
/// - Processing few polylines
/// - Polylines are small
/// - Memory usage is more important than speed
/// - The `batch` feature is not enabled
///
/// # Examples
///
/// ```rust
/// use rapidgeo_distance::LngLat;
/// use rapidgeo_simplify::{batch::simplify_batch, SimplifyMethod};
///
/// let polylines = vec![
///     vec![
///         LngLat::new_deg(-122.0, 37.0),
///         LngLat::new_deg(-121.5, 37.5),
///         LngLat::new_deg(-121.0, 37.0),
///     ],
/// ];
///
/// let simplified = simplify_batch(
///     &polylines,
///     1000.0,
///     SimplifyMethod::GreatCircleMeters,
/// );
///
/// assert_eq!(simplified.len(), 1);
/// assert!(simplified[0].len() >= 2); // Endpoints preserved
/// ```
pub fn simplify_batch(
    polylines: &[Vec<LngLat>],
    tolerance_m: f64,
    method: SimplifyMethod,
) -> Vec<Vec<LngLat>> {
    polylines
        .iter()
        .map(|polyline| {
            let mut simplified = Vec::new();
            crate::simplify_dp_into(polyline, tolerance_m, method, &mut simplified);
            simplified
        })
        .collect()
}

/// Simplify multiple polylines sequentially into pre-allocated output buffers.
///
/// Sequential version of [`simplify_batch_par_into`]. Use when parallel
/// processing is not needed or the `batch` feature is disabled.
///
/// # Examples
///
/// ```rust
/// use rapidgeo_distance::LngLat;
/// use rapidgeo_simplify::{batch::{simplify_batch_into, BatchError}, SimplifyMethod};
///
/// let polylines = vec![
///     vec![LngLat::new_deg(-122.0, 37.0), LngLat::new_deg(-121.0, 37.0)],
/// ];
///
/// let mut output = vec![Vec::new(); 2]; // More capacity than needed
/// let result = simplify_batch_into(
///     &polylines,
///     1000.0,
///     SimplifyMethod::GreatCircleMeters,
///     &mut output,
/// );
///
/// assert!(result.is_ok());
/// assert_eq!(output[0].len(), 2); // Both endpoints kept
/// assert_eq!(output[1].len(), 0); // Second slot unused
/// ```
pub fn simplify_batch_into(
    polylines: &[Vec<LngLat>],
    tolerance_m: f64,
    method: SimplifyMethod,
    output: &mut [Vec<LngLat>],
) -> Result<(), BatchError> {
    if output.len() < polylines.len() {
        return Err(BatchError::BufferTooSmall {
            needed: polylines.len(),
            provided: output.len(),
        });
    }

    for (out_polyline, in_polyline) in output[..polylines.len()].iter_mut().zip(polylines.iter()) {
        crate::simplify_dp_into(in_polyline, tolerance_m, method, out_polyline);
    }

    Ok(())
}

/// Generate a simplification mask using parallel processing.
///
/// Parallel version of [`crate::simplify_dp_mask`] that uses multiple threads
/// for distance calculations on large line segments.
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "batch")]
/// # {
/// use rapidgeo_distance::LngLat;
/// use rapidgeo_simplify::{batch::simplify_dp_mask_par, SimplifyMethod};
///
/// let points = vec![
///     LngLat::new_deg(-122.0, 37.0),
///     LngLat::new_deg(-121.5, 37.5),
///     LngLat::new_deg(-121.0, 37.0),
/// ];
///
/// let mut mask = Vec::new();
/// simplify_dp_mask_par(
///     &points,
///     1000.0,
///     SimplifyMethod::GreatCircleMeters,
///     &mut mask,
/// );
///
/// assert_eq!(mask.len(), points.len());
/// # }
/// ```
///
/// # Performance Notes
///
/// - Switches to parallel processing when segments have > 100 candidate points
/// - For smaller segments, uses serial processing to avoid overhead
/// - Results are identical to the serial version
#[cfg(feature = "batch")]
pub fn simplify_dp_mask_par(
    pts: &[LngLat],
    tolerance_m: f64,
    method: SimplifyMethod,
    mask: &mut Vec<bool>,
) {
    use crate::xt::*;

    match method {
        SimplifyMethod::GreatCircleMeters => {
            let backend = XtGreatCircle;
            simplify_mask_par(pts, tolerance_m, &backend, mask);
        }
        SimplifyMethod::PlanarMeters => {
            let midpoint = crate::compute_midpoint(pts);
            let backend = XtEnu { origin: midpoint };
            simplify_mask_par(pts, tolerance_m, &backend, mask);
        }
        SimplifyMethod::EuclidRaw => {
            let backend = XtEuclid;
            simplify_mask_par(pts, tolerance_m, &backend, mask);
        }
    }
}

/// Simplify a single polyline using parallel processing.
///
/// Parallel version of [`crate::simplify_dp_into`] that can provide better
/// performance for very large polylines (thousands of points).
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "batch")]
/// # {
/// use rapidgeo_distance::LngLat;
/// use rapidgeo_simplify::{batch::simplify_dp_into_par, SimplifyMethod};
///
/// let points = vec![
///     LngLat::new_deg(-122.0, 37.0),
///     LngLat::new_deg(-121.5, 37.5),
///     LngLat::new_deg(-121.0, 37.0),
/// ];
///
/// let mut simplified = Vec::new();
/// let count = simplify_dp_into_par(
///     &points,
///     1000.0,
///     SimplifyMethod::GreatCircleMeters,
///     &mut simplified,
/// );
///
/// assert_eq!(count, simplified.len());
/// assert!(count >= 2); // Endpoints preserved
/// # }
/// ```
#[cfg(feature = "batch")]
pub fn simplify_dp_into_par(
    pts: &[LngLat],
    tolerance_m: f64,
    method: SimplifyMethod,
    out: &mut Vec<LngLat>,
) -> usize {
    out.clear();

    let mut mask = vec![false; pts.len()];
    simplify_dp_mask_par(pts, tolerance_m, method, &mut mask);

    for (i, &keep) in mask.iter().enumerate() {
        if keep {
            out.push(pts[i]);
        }
    }

    out.len()
}

/// Internal parallel implementation of the Douglas-Peucker algorithm.
///
/// This function implements the same algorithm as [`crate::dp::simplify_mask`]
/// but uses parallel processing for distance calculations when processing
/// large line segments.
///
/// # Hybrid Processing
///
/// - Segments with ≤ 100 candidate points: Serial processing
/// - Segments with > 100 candidate points: Parallel processing with Rayon
///
/// This hybrid approach avoids the overhead of parallel task creation for
/// small segments while gaining benefits for large segments.
#[cfg(feature = "batch")]
fn simplify_mask_par<D: PerpDistance + Sync>(
    pts: &[LngLat],
    tolerance_m: f64,
    backend: &D,
    mask: &mut Vec<bool>,
) {
    let n = pts.len();

    mask.clear();
    mask.resize(n, false);

    if n <= 2 {
        for item in mask.iter_mut().take(n) {
            *item = true;
        }
        return;
    }

    // Check if all points are identical
    let first_point = pts[0];
    let all_identical = pts
        .iter()
        .all(|&p| p.lng_deg == first_point.lng_deg && p.lat_deg == first_point.lat_deg);

    if all_identical {
        for item in mask.iter_mut().take(n) {
            *item = true;
        }
        return;
    }

    mask[0] = true;
    mask[n - 1] = true;

    let mut stack = Vec::new();
    stack.push((0, n - 1));

    while let Some((i, j)) = stack.pop() {
        if j <= i + 1 {
            continue;
        }

        let candidate_range = (i + 1)..j;
        let num_candidates = candidate_range.len();

        let (max_distance, max_index) = if num_candidates > PARALLEL_DISTANCE_THRESHOLD {
            // Use parallel processing for large segments
            candidate_range
                .into_par_iter()
                .map(|k| (backend.d_perp_m(pts[i], pts[j], pts[k]), k))
                .reduce_with(|(max_dist, max_idx), (dist, idx)| {
                    if dist > max_dist {
                        (dist, idx)
                    } else {
                        (max_dist, max_idx)
                    }
                })
                .unwrap_or((0.0, i + 1))
        } else {
            // Use serial processing for small segments
            let mut max_distance = 0.0;
            let mut max_index = i + 1;

            for k in candidate_range {
                let distance = backend.d_perp_m(pts[i], pts[j], pts[k]);
                if distance > max_distance {
                    max_distance = distance;
                    max_index = k;
                }
            }

            (max_distance, max_index)
        };

        if max_distance > tolerance_m {
            mask[max_index] = true;
            stack.push((i, max_index));
            stack.push((max_index, j));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rapidgeo_distance::LngLat;

    fn create_test_polylines() -> Vec<Vec<LngLat>> {
        vec![
            vec![
                LngLat::new_deg(-122.4194, 37.7749), // SF
                LngLat::new_deg(-121.5, 37.0),
                LngLat::new_deg(-118.2437, 34.0522), // LA
            ],
            vec![
                LngLat::new_deg(-74.0060, 40.7128), // NYC
                LngLat::new_deg(-75.0, 40.0),
                LngLat::new_deg(-87.6298, 41.8781), // Chicago
            ],
        ]
    }

    #[test]
    fn test_simplify_batch() {
        let polylines = create_test_polylines();
        let simplified = simplify_batch(&polylines, 1000.0, SimplifyMethod::GreatCircleMeters);

        assert_eq!(simplified.len(), 2);
        // Each simplified polyline should have at least endpoints
        for simplified_line in &simplified {
            assert!(simplified_line.len() >= 2);
        }
    }

    #[test]
    fn test_simplify_batch_into() {
        let polylines = create_test_polylines();
        let mut output = vec![Vec::new(); 3]; // Larger than needed

        let result = simplify_batch_into(
            &polylines,
            1000.0,
            SimplifyMethod::GreatCircleMeters,
            &mut output,
        );

        assert!(result.is_ok());
        assert!(output[0].len() >= 2); // First polyline simplified
        assert!(output[1].len() >= 2); // Second polyline simplified
        assert_eq!(output[2].len(), 0); // Third slot unused
    }

    #[test]
    fn test_simplify_batch_into_too_small() {
        let polylines = create_test_polylines();
        let mut output = vec![Vec::new(); 1]; // Too small!

        let result = simplify_batch_into(
            &polylines,
            1000.0,
            SimplifyMethod::GreatCircleMeters,
            &mut output,
        );

        assert!(result.is_err());
        match result.unwrap_err() {
            BatchError::BufferTooSmall { needed, provided } => {
                assert_eq!(needed, 2);
                assert_eq!(provided, 1);
            }
        }
    }

    #[test]
    #[cfg(feature = "batch")]
    fn test_simplify_batch_par() {
        let polylines = create_test_polylines();
        let simplified = simplify_batch_par(&polylines, 1000.0, SimplifyMethod::GreatCircleMeters);

        assert_eq!(simplified.len(), 2);
        // Each simplified polyline should have at least endpoints
        for simplified_line in &simplified {
            assert!(simplified_line.len() >= 2);
        }

        // Compare with serial version
        let serial_simplified =
            simplify_batch(&polylines, 1000.0, SimplifyMethod::GreatCircleMeters);
        assert_eq!(simplified, serial_simplified);
    }

    #[test]
    #[cfg(feature = "batch")]
    fn test_simplify_batch_par_into_too_small() {
        let polylines = create_test_polylines();
        let mut par_output = vec![Vec::new(); 1]; // Too small!

        let result = simplify_batch_par_into(
            &polylines,
            1000.0,
            SimplifyMethod::GreatCircleMeters,
            &mut par_output,
        );

        assert!(result.is_err());
        match result.unwrap_err() {
            BatchError::BufferTooSmall { needed, provided } => {
                assert_eq!(needed, 2);
                assert_eq!(provided, 1);
            }
        }
    }

    #[test]
    #[cfg(feature = "batch")]
    fn test_simplify_batch_par_into() {
        let polylines = create_test_polylines();
        let mut par_output = vec![Vec::new(); 2];
        let mut serial_output = vec![Vec::new(); 2];

        let par_result = simplify_batch_par_into(
            &polylines,
            1000.0,
            SimplifyMethod::GreatCircleMeters,
            &mut par_output,
        );
        let serial_result = simplify_batch_into(
            &polylines,
            1000.0,
            SimplifyMethod::GreatCircleMeters,
            &mut serial_output,
        );

        assert!(par_result.is_ok());
        assert!(serial_result.is_ok());
        assert_eq!(par_output, serial_output);
    }

    #[test]
    #[cfg(feature = "batch")]
    fn test_simplify_dp_mask_par() {
        let points = vec![
            LngLat::new_deg(-122.0, 37.0),
            LngLat::new_deg(-121.5, 37.5),
            LngLat::new_deg(-121.0, 37.0),
        ];

        let mut par_mask = Vec::new();
        let mut serial_mask = Vec::new();

        simplify_dp_mask_par(
            &points,
            1000.0,
            SimplifyMethod::GreatCircleMeters,
            &mut par_mask,
        );
        crate::simplify_dp_mask(
            &points,
            1000.0,
            SimplifyMethod::GreatCircleMeters,
            &mut serial_mask,
        );

        assert_eq!(par_mask, serial_mask);
        assert!(par_mask[0]); // First endpoint
        assert!(par_mask[2]); // Last endpoint
    }

    #[test]
    #[cfg(feature = "batch")]
    fn test_simplify_dp_mask_par_planar_meters() {
        let points = vec![
            LngLat::new_deg(-122.0, 37.0),
            LngLat::new_deg(-121.5, 37.5),
            LngLat::new_deg(-121.0, 37.0),
        ];

        let mut par_mask = Vec::new();
        let mut serial_mask = Vec::new();

        simplify_dp_mask_par(&points, 1000.0, SimplifyMethod::PlanarMeters, &mut par_mask);
        crate::simplify_dp_mask(
            &points,
            1000.0,
            SimplifyMethod::PlanarMeters,
            &mut serial_mask,
        );

        assert_eq!(par_mask, serial_mask);
        assert!(par_mask[0]); // First endpoint
        assert!(par_mask[2]); // Last endpoint
    }

    #[test]
    #[cfg(feature = "batch")]
    fn test_simplify_dp_mask_par_euclid_raw() {
        let points = vec![
            LngLat::new_deg(-122.0, 37.0),
            LngLat::new_deg(-121.5, 37.5),
            LngLat::new_deg(-121.0, 37.0),
        ];

        let mut par_mask = Vec::new();
        let mut serial_mask = Vec::new();

        simplify_dp_mask_par(&points, 0.5, SimplifyMethod::EuclidRaw, &mut par_mask);
        crate::simplify_dp_mask(&points, 0.5, SimplifyMethod::EuclidRaw, &mut serial_mask);

        assert_eq!(par_mask, serial_mask);
        assert!(par_mask[0]); // First endpoint
        assert!(par_mask[2]); // Last endpoint
    }

    #[test]
    #[cfg(feature = "batch")]
    fn test_simplify_dp_into_par() {
        let points = vec![
            LngLat::new_deg(-122.0, 37.0),
            LngLat::new_deg(-121.5, 37.5),
            LngLat::new_deg(-121.0, 37.0),
        ];

        let mut par_output = Vec::new();
        let mut serial_output = Vec::new();

        let par_count = simplify_dp_into_par(
            &points,
            1000.0,
            SimplifyMethod::GreatCircleMeters,
            &mut par_output,
        );
        let serial_count = crate::simplify_dp_into(
            &points,
            1000.0,
            SimplifyMethod::GreatCircleMeters,
            &mut serial_output,
        );

        assert_eq!(par_count, serial_count);
        assert_eq!(par_output, serial_output);
        assert_eq!(par_count, par_output.len());
    }

    #[test]
    #[cfg(feature = "batch")]
    fn test_parallel_threshold_behavior() {
        // Create a large polyline to test parallel behavior
        let mut large_polyline = Vec::new();
        for i in 0..1000 {
            large_polyline.push(LngLat::new_deg(
                -122.0 + i as f64 * 0.001,
                37.0 + (i as f64 * 0.1).sin() * 0.01,
            ));
        }

        let mut par_mask = Vec::new();
        let mut serial_mask = Vec::new();

        simplify_dp_mask_par(
            &large_polyline,
            50.0,
            SimplifyMethod::GreatCircleMeters,
            &mut par_mask,
        );
        crate::simplify_dp_mask(
            &large_polyline,
            50.0,
            SimplifyMethod::GreatCircleMeters,
            &mut serial_mask,
        );

        // Results should be identical regardless of parallel/serial execution
        assert_eq!(par_mask, serial_mask);
        assert!(par_mask[0]); // First endpoint always kept
        assert!(par_mask[par_mask.len() - 1]); // Last endpoint always kept
    }

    #[test]
    fn test_different_methods_batch() {
        let polylines = vec![vec![
            LngLat::new_deg(-122.0, 37.0),
            LngLat::new_deg(-121.5, 37.5),
            LngLat::new_deg(-121.0, 37.0),
        ]];

        for method in [
            SimplifyMethod::GreatCircleMeters,
            SimplifyMethod::PlanarMeters,
            SimplifyMethod::EuclidRaw,
        ] {
            let simplified = simplify_batch(&polylines, 1000.0, method);
            assert_eq!(simplified.len(), 1);
            assert!(simplified[0].len() >= 2); // At least endpoints preserved
        }
    }

    #[test]
    fn test_empty_and_small_polylines() {
        let polylines = vec![
            vec![],                                                             // Empty
            vec![LngLat::new_deg(-122.0, 37.0)],                                // Single point
            vec![LngLat::new_deg(-122.0, 37.0), LngLat::new_deg(-121.0, 37.0)], // Two points
        ];

        let simplified = simplify_batch(&polylines, 1000.0, SimplifyMethod::GreatCircleMeters);

        assert_eq!(simplified.len(), 3);
        assert_eq!(simplified[0].len(), 0); // Empty stays empty
        assert_eq!(simplified[1].len(), 1); // Single point stays single
        assert_eq!(simplified[2].len(), 2); // Two points stay two
    }
}
