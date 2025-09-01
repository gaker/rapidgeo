use crate::{LngLat, SimilarityError, SimilarityMeasure};
use rapidgeo_distance::geodesic;

/// Implementation of discrete Fréchet distance for the SimilarityMeasure trait.
///
/// The Fréchet distance measures the similarity between two curves by finding
/// the shortest "leash" needed for a person and dog to walk along their respective
/// curves. This struct provides a convenient way to use Fréchet distance with
/// the trait-based API.
///
/// # Example
///
/// ```rust
/// use rapidgeo_similarity::{DiscreteFrechet, SimilarityMeasure, LngLat};
///
/// let measure = DiscreteFrechet;
/// let a = vec![LngLat::new_deg(-122.0, 37.0), LngLat::new_deg(-122.1, 37.1)];
/// let b = vec![LngLat::new_deg(-122.0, 37.0), LngLat::new_deg(-122.2, 37.2)];
/// let distance = measure.distance(&a, &b).unwrap();
/// ```
pub struct DiscreteFrechet;

impl SimilarityMeasure for DiscreteFrechet {
    fn distance(&self, a: &[LngLat], b: &[LngLat]) -> Result<f64, SimilarityError> {
        discrete_frechet_distance(a, b)
    }
}

/// Calculate the discrete Fréchet distance between two polylines.
///
/// The Fréchet distance measures how similar two curves are by considering
/// their shape and the order of points. It finds the minimum distance needed
/// to traverse both curves simultaneously, never going backwards.
///
/// This implementation uses dynamic programming and computes distances using
/// the haversine formula for great-circle distance on Earth.
///
/// # Arguments
///
/// * `a` - First polyline as longitude/latitude points
/// * `b` - Second polyline as longitude/latitude points
///
/// # Returns
///
/// The Fréchet distance in meters.
///
/// # Errors
///
/// Returns `SimilarityError::EmptyInput` if either polyline is empty.
///
/// # Example
///
/// ```rust
/// use rapidgeo_similarity::{discrete_frechet_distance, LngLat};
///
/// let route1 = vec![
///     LngLat::new_deg(-122.0, 37.0),
///     LngLat::new_deg(-122.1, 37.1),
/// ];
/// let route2 = vec![
///     LngLat::new_deg(-122.0, 37.0),
///     LngLat::new_deg(-122.2, 37.2),
/// ];
///
/// let distance = discrete_frechet_distance(&route1, &route2).unwrap();
/// println!("Fréchet distance: {} meters", distance);
/// ```
pub fn discrete_frechet_distance(a: &[LngLat], b: &[LngLat]) -> Result<f64, SimilarityError> {
    discrete_frechet_distance_with_threshold(a, b, f64::INFINITY)
}

/// Calculate the discrete Fréchet distance with early termination.
///
/// This function computes the Fréchet distance but stops early if the distance
/// exceeds the given threshold. This can save computation time when you only
/// need to know if two curves are "close enough" rather than the exact distance.
///
/// # Arguments
///
/// * `a` - First polyline as longitude/latitude points
/// * `b` - Second polyline as longitude/latitude points  
/// * `threshold` - Maximum distance to compute; returns early if exceeded
///
/// # Returns
///
/// The Fréchet distance in meters, or a value greater than the threshold
/// if the actual distance exceeds it.
///
/// # Errors
///
/// Returns `SimilarityError::EmptyInput` if either polyline is empty.
///
/// # Example
///
/// ```rust
/// use rapidgeo_similarity::{discrete_frechet_distance_with_threshold, LngLat};
///
/// let route1 = vec![LngLat::new_deg(-122.0, 37.0)];
/// let route2 = vec![LngLat::new_deg(-122.5, 37.5)];
///
/// // Only compute if distance is less than 10km
/// let result = discrete_frechet_distance_with_threshold(&route1, &route2, 10000.0);
/// if result.unwrap() > 10000.0 {
///     println!("Routes are too far apart");
/// }
/// ```
pub fn discrete_frechet_distance_with_threshold(
    a: &[LngLat],
    b: &[LngLat],
    threshold: f64,
) -> Result<f64, SimilarityError> {
    if a.is_empty() || b.is_empty() {
        return Err(SimilarityError::EmptyInput);
    }

    let mut dp = create_dp_matrix(a, b);
    fill_frechet_dp_with_threshold(&mut dp, a, b, threshold);

    let result = dp[a.len() - 1][b.len() - 1];
    if result > threshold {
        Ok(threshold + 1.0)
    } else {
        Ok(result)
    }
}

fn create_dp_matrix(a: &[LngLat], b: &[LngLat]) -> Vec<Vec<f64>> {
    vec![vec![f64::INFINITY; b.len()]; a.len()]
}

fn compute_distance(a: &LngLat, b: &LngLat) -> f64 {
    geodesic::haversine(*a, *b)
}

fn fill_frechet_dp_with_threshold(dp: &mut [Vec<f64>], a: &[LngLat], b: &[LngLat], threshold: f64) {
    dp[0][0] = compute_distance(&a[0], &b[0]);

    fill_first_row_dp_with_threshold(&mut dp[0], &a[0], b, threshold);
    fill_first_column_dp_with_threshold(dp, a, &b[0], threshold);
    fill_remaining_cells_dp_with_threshold(dp, a, b, threshold);
}

fn fill_first_row_dp_with_threshold(
    dp_row: &mut [f64],
    a_point: &LngLat,
    b: &[LngLat],
    threshold: f64,
) {
    for j in 1..dp_row.len() {
        if dp_row[j - 1] > threshold {
            dp_row[j] = dp_row[j - 1];
            continue;
        }

        let distance = compute_distance(a_point, &b[j]);
        dp_row[j] = dp_row[j - 1].max(distance);
    }
}

fn fill_first_column_dp_with_threshold(
    dp: &mut [Vec<f64>],
    a: &[LngLat],
    b_point: &LngLat,
    threshold: f64,
) {
    for i in 1..dp.len() {
        if dp[i - 1][0] > threshold {
            dp[i][0] = dp[i - 1][0];
            continue;
        }

        let distance = compute_distance(&a[i], b_point);
        dp[i][0] = dp[i - 1][0].max(distance);
    }
}

fn fill_remaining_cells_dp_with_threshold(
    dp: &mut [Vec<f64>],
    a: &[LngLat],
    b: &[LngLat],
    threshold: f64,
) {
    for i in 1..dp.len() {
        for (j, b_point) in b.iter().enumerate().skip(1) {
            let min_prev = dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1]);

            if min_prev > threshold {
                dp[i][j] = min_prev;
                continue;
            }

            let distance = compute_distance(&a[i], b_point);
            dp[i][j] = min_prev.max(distance);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_input() {
        let a = vec![];
        let b = vec![LngLat::new_deg(0.0, 0.0)];

        assert_eq!(
            discrete_frechet_distance(&a, &b),
            Err(SimilarityError::EmptyInput)
        );
        assert_eq!(
            discrete_frechet_distance(&b, &a),
            Err(SimilarityError::EmptyInput)
        );
    }

    #[test]
    fn test_identical_single_points() {
        let a = vec![LngLat::new_deg(0.0, 0.0)];
        let b = vec![LngLat::new_deg(0.0, 0.0)];

        let result = discrete_frechet_distance(&a, &b).unwrap();
        assert!(result < 1e-10);
    }

    #[test]
    fn test_different_single_points() {
        let a = vec![LngLat::new_deg(0.0, 0.0)];
        let b = vec![LngLat::new_deg(1.0, 1.0)];

        let result = discrete_frechet_distance(&a, &b).unwrap();
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

        let result = discrete_frechet_distance(&curve, &curve).unwrap();
        assert!(result < 1e-10);
    }

    #[test]
    fn test_simple_curves() {
        let a = vec![LngLat::new_deg(0.0, 0.0), LngLat::new_deg(1.0, 0.0)];
        let b = vec![LngLat::new_deg(0.0, 1.0), LngLat::new_deg(1.0, 1.0)];

        let result = discrete_frechet_distance(&a, &b).unwrap();
        assert!(result > 0.0);

        let expected_min = geodesic::haversine(a[0], b[0]).max(geodesic::haversine(a[1], b[1]));
        assert!(result >= expected_min - 1e-10);
    }

    #[test]
    fn test_trait_implementation() {
        let frechet = DiscreteFrechet;
        let a = vec![LngLat::new_deg(0.0, 0.0)];
        let b = vec![LngLat::new_deg(1.0, 1.0)];

        let result1 = frechet.distance(&a, &b).unwrap();
        let result2 = discrete_frechet_distance(&a, &b).unwrap();

        assert!((result1 - result2).abs() < 1e-10);
    }

    #[test]
    fn test_symmetry() {
        let a = vec![LngLat::new_deg(0.0, 0.0), LngLat::new_deg(1.0, 0.0)];
        let b = vec![LngLat::new_deg(0.0, 1.0), LngLat::new_deg(1.0, 1.0)];

        let result_ab = discrete_frechet_distance(&a, &b).unwrap();
        let result_ba = discrete_frechet_distance(&b, &a).unwrap();

        assert!((result_ab - result_ba).abs() < 1e-10);
    }

    #[test]
    fn test_triangle_inequality() {
        let a = vec![LngLat::new_deg(0.0, 0.0), LngLat::new_deg(1.0, 0.0)];
        let b = vec![LngLat::new_deg(0.5, 0.5), LngLat::new_deg(1.5, 0.5)];
        let c = vec![LngLat::new_deg(1.0, 1.0), LngLat::new_deg(2.0, 1.0)];

        let d_ab = discrete_frechet_distance(&a, &b).unwrap();
        let d_bc = discrete_frechet_distance(&b, &c).unwrap();
        let d_ac = discrete_frechet_distance(&a, &c).unwrap();

        assert!(d_ac <= d_ab + d_bc + 1e-10);
    }

    #[test]
    fn test_threshold_early_termination() {
        let a = vec![LngLat::new_deg(0.0, 0.0), LngLat::new_deg(1.0, 0.0)];
        let b = vec![LngLat::new_deg(0.0, 1.0), LngLat::new_deg(1.0, 1.0)];

        let full_distance = discrete_frechet_distance(&a, &b).unwrap();

        let low_threshold = full_distance / 2.0;
        let high_threshold = full_distance * 2.0;

        let result_low = discrete_frechet_distance_with_threshold(&a, &b, low_threshold).unwrap();
        let result_high = discrete_frechet_distance_with_threshold(&a, &b, high_threshold).unwrap();

        assert!(result_low > low_threshold);
        assert!((result_high - full_distance).abs() < 1e-10);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn valid_coord() -> impl Strategy<Value = f64> {
        -180.0..=180.0f64
    }

    fn valid_lat() -> impl Strategy<Value = f64> {
        -90.0..=90.0f64
    }

    fn polyline() -> impl Strategy<Value = Vec<LngLat>> {
        prop::collection::vec(
            (valid_coord(), valid_lat()).prop_map(|(lng, lat)| LngLat::new_deg(lng, lat)),
            1..20,
        )
    }

    proptest! {
        #[test]
        fn test_frechet_non_negative(a in polyline(), b in polyline()) {
            let result = discrete_frechet_distance(&a, &b).unwrap();
            prop_assert!(result >= 0.0);
        }

        #[test]
        fn test_frechet_identity(curve in polyline()) {
            let result = discrete_frechet_distance(&curve, &curve).unwrap();
            prop_assert!(result < 1e-10);
        }

        #[test]
        fn test_frechet_symmetry(a in polyline(), b in polyline()) {
            let result_ab = discrete_frechet_distance(&a, &b).unwrap();
            let result_ba = discrete_frechet_distance(&b, &a).unwrap();
            prop_assert!((result_ab - result_ba).abs() < 1e-10);
        }

        #[test]
        fn test_single_point_frechet(
            lng1 in valid_coord(), lat1 in valid_lat(),
            lng2 in valid_coord(), lat2 in valid_lat()
        ) {
            let a = vec![LngLat::new_deg(lng1, lat1)];
            let b = vec![LngLat::new_deg(lng2, lat2)];

            let result = discrete_frechet_distance(&a, &b).unwrap();
            let expected = geodesic::haversine(a[0], b[0]);
            prop_assert!((result - expected).abs() < 1e-10);
        }
    }
}
