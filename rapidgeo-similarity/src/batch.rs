use crate::{frechet::discrete_frechet_distance, LngLat, SimilarityError};
use rayon::prelude::*;

/// Calculate Fréchet distances for multiple polyline pairs in parallel.
///
/// This function processes multiple polyline pairs simultaneously using Rayon
/// for parallel computation. Each pair is processed independently, making this
/// efficient for large batches of similarity computations.
///
/// # Arguments
///
/// * `polylines` - Slice of polyline pairs to compare
///
/// # Returns
///
/// Vector of Fréchet distances in meters, in the same order as input pairs.
///
/// # Errors
///
/// Returns an error if any polyline pair contains empty polylines.
///
/// # Example
///
/// ```rust
/// use rapidgeo_similarity::{batch_frechet_distance, LngLat};
///
/// let route1 = vec![LngLat::new_deg(-122.0, 37.0), LngLat::new_deg(-122.1, 37.1)];
/// let route2 = vec![LngLat::new_deg(-122.0, 37.0), LngLat::new_deg(-122.2, 37.2)];
/// let route3 = vec![LngLat::new_deg(-122.0, 37.0), LngLat::new_deg(-122.3, 37.3)];
///
/// let pairs = vec![
///     (route1.clone(), route2.clone()),
///     (route2, route3),
/// ];
///
/// let distances = batch_frechet_distance(&pairs).unwrap();
/// println!("Computed {} distances", distances.len());
/// ```
pub fn batch_frechet_distance(
    polylines: &[(Vec<LngLat>, Vec<LngLat>)],
) -> Result<Vec<f64>, SimilarityError> {
    if polylines.is_empty() {
        return Ok(Vec::new());
    }

    let results: Result<Vec<f64>, SimilarityError> = polylines
        .par_iter()
        .map(|(a, b)| discrete_frechet_distance(a, b))
        .collect();

    results
}

/// Check if multiple polyline pairs are within a distance threshold.
///
/// This function processes multiple polyline pairs in parallel and returns
/// boolean results indicating whether each pair's Fréchet distance is within
/// the given threshold. This is more efficient than computing exact distances
/// when you only need to know if pairs are "similar enough".
///
/// # Arguments
///
/// * `polylines` - Slice of polyline pairs to compare
/// * `threshold` - Maximum distance in meters for pairs to be considered similar
///
/// # Returns
///
/// Vector of boolean values indicating which pairs are within the threshold.
///
/// # Errors
///
/// Returns an error if any polyline pair contains empty polylines.
///
/// # Example
///
/// ```rust
/// use rapidgeo_similarity::{batch_frechet_distance_threshold, LngLat};
///
/// let route1 = vec![LngLat::new_deg(-122.0, 37.0)];
/// let route2 = vec![LngLat::new_deg(-122.0, 37.0)];
/// let route3 = vec![LngLat::new_deg(-125.0, 40.0)];
///
/// let pairs = vec![
///     (route1.clone(), route2),  // Very similar
///     (route1, route3),          // Very different
/// ];
///
/// let results = batch_frechet_distance_threshold(&pairs, 1000.0).unwrap();
/// println!("Similar pairs: {:?}", results);  // [true, false]
/// ```
pub fn batch_frechet_distance_threshold(
    polylines: &[(Vec<LngLat>, Vec<LngLat>)],
    threshold: f64,
) -> Result<Vec<bool>, SimilarityError> {
    if polylines.is_empty() {
        return Ok(Vec::new());
    }

    let results: Result<Vec<bool>, SimilarityError> = polylines
        .par_iter()
        .map(|(a, b)| discrete_frechet_distance(a, b).map(|distance| distance <= threshold))
        .collect();

    results
}

/// Compute a distance matrix of Fréchet distances between all polyline pairs.
///
/// This function creates a symmetric matrix where each cell (i,j) contains the
/// Fréchet distance between polyline i and polyline j. The diagonal contains
/// zeros (distance from each polyline to itself).
///
/// The computation is parallelized and takes advantage of matrix symmetry
/// to avoid redundant calculations.
///
/// # Arguments
///
/// * `polylines` - Slice of polylines to compare pairwise
///
/// # Returns
///
/// A square matrix (Vec<Vec<f64>>) where matrix[i][j] is the Fréchet distance
/// between polylines[i] and polylines[j] in meters.
///
/// # Errors
///
/// Returns an error if any polyline is empty.
///
/// # Example
///
/// ```rust
/// use rapidgeo_similarity::{pairwise_frechet_matrix, LngLat};
///
/// let routes = vec![
///     vec![LngLat::new_deg(-122.0, 37.0), LngLat::new_deg(-122.1, 37.1)],
///     vec![LngLat::new_deg(-122.0, 37.0), LngLat::new_deg(-122.2, 37.2)],
///     vec![LngLat::new_deg(-122.0, 37.0), LngLat::new_deg(-122.3, 37.3)],
/// ];
///
/// let matrix = pairwise_frechet_matrix(&routes).unwrap();
/// println!("Distance between route 0 and 1: {} meters", matrix[0][1]);
/// ```
pub fn pairwise_frechet_matrix(
    polylines: &[Vec<LngLat>],
) -> Result<Vec<Vec<f64>>, SimilarityError> {
    if polylines.is_empty() {
        return Ok(Vec::new());
    }

    let n = polylines.len();
    let mut results = vec![vec![0.0; n]; n];

    let indices: Vec<(usize, usize)> = (0..n).flat_map(|i| (i..n).map(move |j| (i, j))).collect();

    let distances: Result<Vec<f64>, SimilarityError> = indices
        .par_iter()
        .map(|(i, j)| {
            if i == j {
                Ok(0.0)
            } else {
                discrete_frechet_distance(&polylines[*i], &polylines[*j])
            }
        })
        .collect();

    let distances = distances?;

    for (idx, (i, j)) in indices.iter().enumerate() {
        results[*i][*j] = distances[idx];
        if i != j {
            results[*j][*i] = distances[idx];
        }
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_polylines() -> Vec<Vec<LngLat>> {
        vec![
            vec![LngLat::new_deg(-122.0, 37.0), LngLat::new_deg(-122.1, 37.1)],
            vec![LngLat::new_deg(-122.0, 37.0), LngLat::new_deg(-122.2, 37.2)],
            vec![LngLat::new_deg(-122.0, 37.0), LngLat::new_deg(-122.3, 37.3)],
        ]
    }

    #[test]
    fn test_batch_frechet_distance_empty() {
        let polylines: Vec<(Vec<LngLat>, Vec<LngLat>)> = vec![];
        let result = batch_frechet_distance(&polylines).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_batch_frechet_distance() {
        let polylines = create_test_polylines();
        let pairs = vec![
            (polylines[0].clone(), polylines[1].clone()),
            (polylines[1].clone(), polylines[2].clone()),
        ];

        let results = batch_frechet_distance(&pairs).unwrap();
        assert_eq!(results.len(), 2);

        for distance in &results {
            assert!(distance >= &0.0);
        }

        let expected_0_1 = discrete_frechet_distance(&polylines[0], &polylines[1]).unwrap();
        let expected_1_2 = discrete_frechet_distance(&polylines[1], &polylines[2]).unwrap();

        assert!((results[0] - expected_0_1).abs() < 1e-10);
        assert!((results[1] - expected_1_2).abs() < 1e-10);
    }

    #[test]
    fn test_batch_frechet_distance_threshold() {
        let polylines = create_test_polylines();
        let pairs = vec![
            (polylines[0].clone(), polylines[0].clone()),
            (polylines[0].clone(), polylines[1].clone()),
        ];

        let actual_distance = discrete_frechet_distance(&polylines[0], &polylines[1]).unwrap();

        let results = batch_frechet_distance_threshold(&pairs, actual_distance + 1000.0).unwrap();
        assert_eq!(results.len(), 2);

        assert!(results[0]);
        assert!(results[1]);

        let results_strict =
            batch_frechet_distance_threshold(&pairs, actual_distance - 1.0).unwrap();
        assert!(results_strict[0]);
        assert!(!results_strict[1]);
    }

    #[test]
    fn test_pairwise_frechet_matrix_empty() {
        let polylines: Vec<Vec<LngLat>> = vec![];
        let result = pairwise_frechet_matrix(&polylines).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_pairwise_frechet_matrix() {
        let polylines = create_test_polylines();
        let matrix = pairwise_frechet_matrix(&polylines).unwrap();

        assert_eq!(matrix.len(), 3);
        for row in &matrix {
            assert_eq!(row.len(), 3);
        }

        for (i, _) in matrix.iter().enumerate() {
            assert!(matrix[i][i] < 1e-10);
        }

        for (i, row) in matrix.iter().enumerate() {
            for (j, _) in row.iter().enumerate() {
                assert!((matrix[i][j] - matrix[j][i]).abs() < 1e-10);
            }
        }

        for row in &matrix {
            for &value in row {
                assert!(value >= 0.0);
            }
        }
    }

    #[test]
    fn test_pairwise_frechet_matrix_single() {
        let polylines = vec![vec![
            LngLat::new_deg(-122.0, 37.0),
            LngLat::new_deg(-122.1, 37.1),
        ]];
        let matrix = pairwise_frechet_matrix(&polylines).unwrap();

        assert_eq!(matrix.len(), 1);
        assert_eq!(matrix[0].len(), 1);
        assert!(matrix[0][0] < 1e-10);
    }

    #[test]
    fn test_error_propagation() {
        let pairs = vec![(vec![], vec![LngLat::new_deg(0.0, 0.0)])];

        let result = batch_frechet_distance(&pairs);
        assert!(result.is_err());

        let result = batch_frechet_distance_threshold(&pairs, 1.0);
        assert!(result.is_err());

        let polylines = vec![vec![], vec![LngLat::new_deg(0.0, 0.0)]];
        let result = pairwise_frechet_matrix(&polylines);
        assert!(result.is_err());
    }
}
