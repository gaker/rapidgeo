//! Core Douglas-Peucker simplification algorithm implementation.
//!
//! This module contains the recursive algorithm that performs the actual
//! simplification work, independent of the distance calculation method.

use crate::xt::PerpDistance;
use rapidgeo_distance::LngLat;

/// Simplify a polyline using the Douglas-Peucker algorithm with a custom distance backend.
///
/// This is the core implementation that performs the recursive simplification.
/// It uses a stack-based approach to avoid potential stack overflow with very
/// large polylines.
///
/// # Arguments
///
/// * `pts` - Input points to simplify
/// * `tolerance_m` - Distance threshold in the backend's units
/// * `backend` - Distance calculation implementation
/// * `mask` - Output mask indicating which points to keep
///
/// # Algorithm Details
///
/// The algorithm follows these steps:
/// 1. Handle edge cases (≤2 points, all identical points)
/// 2. Mark endpoints as always kept
/// 3. Use a stack to process line segments recursively:
///    - Find the point with maximum perpendicular distance to the current segment
///    - If distance exceeds tolerance, keep the point and add two new segments to process
///    - Otherwise, discard all points in the current segment
///
/// # Performance
///
/// - Time: O(n log n) average case, O(n²) worst case
/// - Space: O(log n) stack depth average, O(n) worst case
/// - The `#[cfg_attr(test, inline(never))]` prevents inlining during tests for better profiling
#[cfg_attr(test, inline(never))]
pub fn simplify_mask<D: PerpDistance>(
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

        let mut max_distance = 0.0;
        let mut max_index = i + 1;

        for k in (i + 1)..j {
            let distance = backend.d_perp_m(pts[i], pts[j], pts[k]);
            if distance > max_distance {
                max_distance = distance;
                max_index = k;
            }
        }

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
    use crate::xt::XtEuclid;
    use rapidgeo_distance::LngLat;

    #[test]
    fn test_simplify_mask_all_identical_large() {
        // Test with a larger number of identical points
        let pts = vec![LngLat::new_deg(0.0, 0.0); 100];

        let mut mask = Vec::new();
        let backend = XtEuclid;
        simplify_mask(&pts, 5.0, &backend, &mut mask);

        assert_eq!(mask.len(), 100);
        for (i, &keep) in mask.iter().enumerate() {
            assert!(
                keep,
                "Point {} should be kept when all points are identical",
                i
            );
        }
    }

    #[test]
    fn test_simplify_mask_two_identical_points() {
        let pts = vec![LngLat::new_deg(-122.0, 37.0), LngLat::new_deg(-122.0, 37.0)];

        let mut mask = Vec::new();
        let backend = XtEuclid;
        simplify_mask(&pts, 10.0, &backend, &mut mask);

        assert_eq!(mask.len(), 2);
        assert!(mask[0], "First point should be kept");
        assert!(mask[1], "Second point should be kept");
    }

    #[test]
    fn mask_clear_and_resize_happens() {
        let pts = vec![
            LngLat::new_deg(0.0, 0.0),
            LngLat::new_deg(1.0, 0.0),
            LngLat::new_deg(2.0, 0.0),
        ];
        // Pre-fill with junk to ensure clear/resize are not optimized away.
        let mut mask = vec![true, true, true, true, true];
        simplify_mask(&pts, 0.0, &XtEuclid, &mut mask);
        assert_eq!(mask.len(), 3);
        assert_eq!(mask, vec![true, false, true]);
    }

    // n == 0, 1, 2 early exits (hits the early return path)
    #[test]
    fn n_zero_one_two() {
        let mut m = Vec::new();
        simplify_mask(&[], 1.0, &XtEuclid, &mut m);
        assert!(m.is_empty());

        let pts1 = vec![LngLat::new_deg(0.0, 0.0)];
        simplify_mask(&pts1, 1.0, &XtEuclid, &mut m);
        assert_eq!(m, vec![true]);

        let pts2 = vec![LngLat::new_deg(0.0, 0.0), LngLat::new_deg(1.0, 0.0)];
        simplify_mask(&pts2, 1.0, &XtEuclid, &mut m);
        assert_eq!(m, vec![true, true]);
    }

    // all_identical branch with n >= 3 (forces that path)
    #[test]
    fn all_identical_three_points() {
        let pts = vec![
            LngLat::new_deg(7.0, -3.0),
            LngLat::new_deg(7.0, -3.0),
            LngLat::new_deg(7.0, -3.0),
        ];
        let mut mask = Vec::new();
        simplify_mask(&pts, 5.0, &XtEuclid, &mut mask);
        assert_eq!(mask, vec![true, true, true]);
    }

    // Force one split, then children hit the `j <= i + 1` continue path
    #[test]
    fn single_split_then_children_continue() {
        // Middle is far off the baseline so it must be kept for small tol
        let pts = vec![
            LngLat::new_deg(0.0, 0.0), // A
            LngLat::new_deg(1.0, 5.0), // P (offset)
            LngLat::new_deg(2.0, 0.0), // B
        ];
        let mut mask = Vec::new();
        simplify_mask(&pts, 0.0001, &XtEuclid, &mut mask);
        // After split: (0,1) and (1,2) pop and hit the continue branch
        assert_eq!(mask, vec![true, true, true]);
    }

    // High tolerance: main while loop runs, but no split; only endpoints kept
    #[test]
    fn high_tolerance_endpoints_only() {
        let pts = vec![
            LngLat::new_deg(0.0, 0.0),
            LngLat::new_deg(0.2, 0.5),
            LngLat::new_deg(0.4, -0.5),
            LngLat::new_deg(0.6, 0.5),
            LngLat::new_deg(0.8, -0.5),
            LngLat::new_deg(1.0, 0.0),
        ];
        let mut mask = Vec::new();
        simplify_mask(&pts, 1_000_000.0, &XtEuclid, &mut mask);
        assert_eq!(mask, vec![true, false, false, false, false, true]);
    }

    // Boundary: max_distance == tolerance should NOT split (uses >, not >=)
    #[test]
    fn tolerance_equality_boundary_no_split() {
        // Segment AB along x-axis, P at (1,1) so d_perp == 1
        let pts = vec![
            LngLat::new_deg(0.0, 0.0), // A
            LngLat::new_deg(1.0, 1.0), // P
            LngLat::new_deg(2.0, 0.0), // B
        ];
        let mut mask = Vec::new();
        simplify_mask(&pts, 1.0, &XtEuclid, &mut mask);
        // No split at equality -> middle dropped
        assert_eq!(mask, vec![true, false, true]);
    }

    // Just-below boundary forces a split
    #[test]
    fn tolerance_just_below_splits() {
        let pts = vec![
            LngLat::new_deg(0.0, 0.0), // A
            LngLat::new_deg(1.0, 1.0), // P (perp 1)
            LngLat::new_deg(2.0, 0.0), // B
        ];
        let mut mask = Vec::new();
        simplify_mask(&pts, 0.9999, &XtEuclid, &mut mask);
        assert_eq!(mask, vec![true, true, true]);
    }

    // Negative tolerance forces maximal splitting (exercises deep stack)
    #[test]
    fn negative_tolerance_keeps_every_point() {
        // Sawtooth that will keep splitting until all points are kept.
        let mut pts = Vec::new();
        for i in 0..50 {
            let x = i as f64 * 0.1;
            let y = if i % 2 == 0 { 0.0 } else { 1.0 };
            pts.push(LngLat::new_deg(x, y));
        }
        let mut mask = Vec::new();
        simplify_mask(&pts, -1.0, &XtEuclid, &mut mask);
        assert_eq!(mask.len(), pts.len());
        assert!(mask.iter().all(|&b| b));
    }
}
