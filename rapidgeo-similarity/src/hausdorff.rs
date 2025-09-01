use crate::{LngLat, SimilarityError, SimilarityMeasure};
use rapidgeo_distance::geodesic;

/// Implementation of Hausdorff distance for the SimilarityMeasure trait.
///
/// The Hausdorff distance measures the greatest distance from any point in one
/// set to the closest point in another set. For polylines, this gives you the
/// maximum "worst-case" distance between the two curves.
///
/// Unlike Fréchet distance, Hausdorff distance doesn't consider the order of
/// points, making it suitable for comparing shapes regardless of how they were
/// traced.
///
/// # Example
///
/// ```rust
/// use rapidgeo_similarity::{Hausdorff, SimilarityMeasure, LngLat};
///
/// let measure = Hausdorff;
/// let a = vec![LngLat::new_deg(-122.0, 37.0), LngLat::new_deg(-122.1, 37.1)];
/// let b = vec![LngLat::new_deg(-122.0, 37.0), LngLat::new_deg(-122.2, 37.2)];
/// let distance = measure.distance(&a, &b).unwrap();
/// ```
pub struct Hausdorff;

impl SimilarityMeasure for Hausdorff {
    fn distance(&self, a: &[LngLat], b: &[LngLat]) -> Result<f64, SimilarityError> {
        hausdorff_distance(a, b)
    }
}

/// Calculate the Hausdorff distance between two polylines.
///
/// The Hausdorff distance finds the greatest distance from any point in one
/// polyline to the closest point in the other polyline. This gives you the
/// "worst-case" similarity - how far apart the most dissimilar parts are.
///
/// The function computes the symmetric Hausdorff distance, which is the maximum
/// of the directed distances in both directions.
///
/// # Arguments
///
/// * `a` - First polyline as longitude/latitude points
/// * `b` - Second polyline as longitude/latitude points
///
/// # Returns
///
/// The Hausdorff distance in meters.
///
/// # Errors
///
/// Returns `SimilarityError::EmptyInput` if either polyline is empty.
///
/// # Example
///
/// ```rust
/// use rapidgeo_similarity::{hausdorff_distance, LngLat};
///
/// let shape1 = vec![
///     LngLat::new_deg(-122.0, 37.0),
///     LngLat::new_deg(-122.1, 37.1),
/// ];
/// let shape2 = vec![
///     LngLat::new_deg(-122.0, 37.0),
///     LngLat::new_deg(-122.2, 37.2),
/// ];
///
/// let distance = hausdorff_distance(&shape1, &shape2).unwrap();
/// println!("Maximum deviation: {} meters", distance);
/// ```
pub fn hausdorff_distance(a: &[LngLat], b: &[LngLat]) -> Result<f64, SimilarityError> {
    hausdorff_distance_with_threshold(a, b, f64::INFINITY)
}

/// Calculate the Hausdorff distance with early termination.
///
/// This function computes the Hausdorff distance but stops early if the distance
/// exceeds the given threshold. This optimization can save significant computation
/// time when you only need to know if two shapes are "close enough".
///
/// # Arguments
///
/// * `a` - First polyline as longitude/latitude points
/// * `b` - Second polyline as longitude/latitude points
/// * `threshold` - Maximum distance to compute; returns early if exceeded
///
/// # Returns
///
/// The Hausdorff distance in meters, or a value greater than the threshold
/// if the actual distance exceeds it.
///
/// # Errors
///
/// Returns `SimilarityError::EmptyInput` if either polyline is empty.
///
/// # Example
///
/// ```rust
/// use rapidgeo_similarity::{hausdorff_distance_with_threshold, LngLat};
///
/// let shape1 = vec![LngLat::new_deg(-122.0, 37.0)];
/// let shape2 = vec![LngLat::new_deg(-122.5, 37.5)];
///
/// // Only compute if distance is less than 5km
/// let result = hausdorff_distance_with_threshold(&shape1, &shape2, 5000.0);
/// if result.unwrap() > 5000.0 {
///     println!("Shapes are too different");
/// }
/// ```
pub fn hausdorff_distance_with_threshold(
    a: &[LngLat],
    b: &[LngLat],
    threshold: f64,
) -> Result<f64, SimilarityError> {
    if a.is_empty() || b.is_empty() {
        return Err(SimilarityError::EmptyInput);
    }

    let h_ab = directed_hausdorff_with_threshold(a, b, threshold);
    if h_ab > threshold {
        return Ok(threshold + 1.0);
    }

    let h_ba = directed_hausdorff_with_threshold(b, a, threshold);

    let result = h_ab.max(h_ba);
    if result > threshold {
        Ok(threshold + 1.0)
    } else {
        Ok(result)
    }
}

fn directed_hausdorff_with_threshold(a: &[LngLat], b: &[LngLat], threshold: f64) -> f64 {
    let mut max_dist = 0.0f64;

    for point_a in a {
        let min_dist = min_distance_to_set(point_a, b);
        max_dist = max_dist.max(min_dist);

        if max_dist > threshold {
            return threshold + 1.0;
        }
    }

    max_dist
}

fn min_distance_to_set(point: &LngLat, set: &[LngLat]) -> f64 {
    set.iter()
        .map(|other| geodesic::haversine(*point, *other))
        .fold(f64::INFINITY, f64::min)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_input() {
        let a = vec![];
        let b = vec![LngLat::new_deg(0.0, 0.0)];

        assert_eq!(hausdorff_distance(&a, &b), Err(SimilarityError::EmptyInput));
        assert_eq!(hausdorff_distance(&b, &a), Err(SimilarityError::EmptyInput));
    }

    #[test]
    fn test_identical_single_points() {
        let a = vec![LngLat::new_deg(0.0, 0.0)];
        let b = vec![LngLat::new_deg(0.0, 0.0)];

        let result = hausdorff_distance(&a, &b).unwrap();
        assert!(result < 1e-10);
    }

    #[test]
    fn test_different_single_points() {
        let a = vec![LngLat::new_deg(0.0, 0.0)];
        let b = vec![LngLat::new_deg(1.0, 1.0)];

        let result = hausdorff_distance(&a, &b).unwrap();
        let expected = geodesic::haversine(a[0], b[0]);
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_identical_curves() {
        let curve = vec![
            LngLat::new_deg(0.0, 0.0),
            LngLat::new_deg(1.0, 1.0),
            LngLat::new_deg(2.0, 0.0),
        ];

        let result = hausdorff_distance(&curve, &curve).unwrap();
        assert!(result < 1e-10);
    }

    #[test]
    fn test_simple_hausdorff() {
        let a = vec![LngLat::new_deg(0.0, 0.0), LngLat::new_deg(1.0, 0.0)];
        let b = vec![LngLat::new_deg(0.0, 1.0), LngLat::new_deg(1.0, 1.0)];

        let result = hausdorff_distance(&a, &b).unwrap();
        assert!(result > 0.0);

        let expected_distance = geodesic::haversine(a[0], b[0]);
        assert!((result - expected_distance).abs() < 1e-10);
    }

    #[test]
    fn test_trait_implementation() {
        let hausdorff = Hausdorff;
        let a = vec![LngLat::new_deg(0.0, 0.0)];
        let b = vec![LngLat::new_deg(1.0, 1.0)];

        let result1 = hausdorff.distance(&a, &b).unwrap();
        let result2 = hausdorff_distance(&a, &b).unwrap();

        assert!((result1 - result2).abs() < 1e-10);
    }

    #[test]
    fn test_symmetry() {
        let a = vec![LngLat::new_deg(0.0, 0.0), LngLat::new_deg(1.0, 0.0)];
        let b = vec![LngLat::new_deg(0.0, 1.0), LngLat::new_deg(1.0, 1.0)];

        let result_ab = hausdorff_distance(&a, &b).unwrap();
        let result_ba = hausdorff_distance(&b, &a).unwrap();

        assert!((result_ab - result_ba).abs() < 1e-10);
    }

    #[test]
    fn test_triangle_inequality() {
        let a = vec![LngLat::new_deg(0.0, 0.0), LngLat::new_deg(1.0, 0.0)];
        let b = vec![LngLat::new_deg(0.5, 0.5), LngLat::new_deg(1.5, 0.5)];
        let c = vec![LngLat::new_deg(1.0, 1.0), LngLat::new_deg(2.0, 1.0)];

        let d_ab = hausdorff_distance(&a, &b).unwrap();
        let d_bc = hausdorff_distance(&b, &c).unwrap();
        let d_ac = hausdorff_distance(&a, &c).unwrap();

        assert!(d_ac <= d_ab + d_bc + 1e-10);
    }

    #[test]
    fn test_subset_property() {
        let a = vec![LngLat::new_deg(0.0, 0.0), LngLat::new_deg(1.0, 0.0)];
        let b = vec![
            LngLat::new_deg(0.0, 0.0),
            LngLat::new_deg(0.5, 0.0),
            LngLat::new_deg(1.0, 0.0),
        ];

        let result = hausdorff_distance(&a, &b).unwrap();

        let h_ab = directed_hausdorff_with_threshold(&a, &b, f64::INFINITY);
        let h_ba = directed_hausdorff_with_threshold(&b, &a, f64::INFINITY);

        assert!(h_ab < 1e-10, "h(A,B) should be ~0 when A ⊆ B, got {}", h_ab);
        assert!(h_ba >= 0.0, "h(B,A) should be >= 0, got {}", h_ba);
        assert!(
            result >= 0.0,
            "Hausdorff distance should be >= 0, got {}",
            result
        );
    }

    #[test]
    fn test_threshold_early_termination() {
        let a = vec![LngLat::new_deg(0.0, 0.0), LngLat::new_deg(1.0, 0.0)];
        let b = vec![LngLat::new_deg(0.0, 1.0), LngLat::new_deg(1.0, 1.0)];

        let full_distance = hausdorff_distance(&a, &b).unwrap();

        let low_threshold = full_distance / 2.0;
        let high_threshold = full_distance * 2.0;

        let result_low = hausdorff_distance_with_threshold(&a, &b, low_threshold).unwrap();
        let result_high = hausdorff_distance_with_threshold(&a, &b, high_threshold).unwrap();

        assert!(result_low > low_threshold);
        assert!((result_high - full_distance).abs() < 1e-10);
    }
}
