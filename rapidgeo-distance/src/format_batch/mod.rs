pub mod buffer_pool;
pub mod sequential;

#[cfg(feature = "batch")]
pub mod parallel;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LngLat;

    #[test]
    fn test_pairwise_haversine_iter_consistency() {
        let coords = vec![
            LngLat::new_deg(-122.4194, 37.7749), // San Francisco
            LngLat::new_deg(-74.0060, 40.7128),  // New York
            LngLat::new_deg(-87.6298, 41.8781),  // Chicago
        ];

        let iter_result = sequential::pairwise_haversine_iter(coords.iter().copied());
        let any_result = sequential::pairwise_haversine_any(&coords);

        assert_eq!(iter_result.len(), any_result.len());
        for (iter_dist, any_dist) in iter_result.iter().zip(any_result.iter()) {
            assert!((iter_dist - any_dist).abs() < 1e-10_f64);
        }
    }

    #[test]
    fn test_pairwise_haversine_iter_into_consistency() {
        let coords = vec![
            LngLat::new_deg(-122.4194, 37.7749),
            LngLat::new_deg(-74.0060, 40.7128),
        ];

        let mut iter_output = vec![0.0; 1];
        let written =
            sequential::pairwise_haversine_iter_into(coords.iter().copied(), &mut iter_output);

        let mut any_output = vec![0.0; 1];
        sequential::pairwise_haversine_into_any(&coords, &mut any_output);

        assert_eq!(written, 1);
        assert!((iter_output[0] - any_output[0]).abs() < 1e-10_f64);
    }

    #[test]
    fn test_path_length_haversine_iter_consistency() {
        let coords = vec![
            LngLat::new_deg(-122.4194, 37.7749),
            LngLat::new_deg(-74.0060, 40.7128),
            LngLat::new_deg(-87.6298, 41.8781),
        ];

        let iter_result = sequential::path_length_haversine_iter(coords.iter().copied());
        let any_result = sequential::path_length_haversine_any(&coords);

        assert!((iter_result - any_result).abs() < 1e-10_f64);
    }

    #[cfg(feature = "batch")]
    #[test]
    fn test_pairwise_haversine_par_iter_consistency() {
        let coords = [
            LngLat::new_deg(-122.4194, 37.7749), // San Francisco
            LngLat::new_deg(-74.0060, 40.7128),  // New York
            LngLat::new_deg(-87.6298, 41.8781),  // Chicago
        ];

        let par_result = parallel::pairwise_haversine_par_iter(coords.iter().copied());
        let seq_result = sequential::pairwise_haversine_iter(coords.iter().copied());

        assert_eq!(par_result.len(), seq_result.len());
        for (par_dist, seq_dist) in par_result.iter().zip(seq_result.iter()) {
            assert!((par_dist - seq_dist).abs() < 1e-6_f64);
        }
    }

    #[test]
    fn test_high_performance_functions_empty_input() {
        let empty_coords: Vec<LngLat> = vec![];

        assert_eq!(
            sequential::pairwise_haversine_iter(empty_coords.iter().copied()).len(),
            0
        );
        assert_eq!(sequential::pairwise_haversine_any(&empty_coords).len(), 0);
        assert_eq!(
            sequential::path_length_haversine_iter(empty_coords.iter().copied()),
            0.0
        );
        assert_eq!(sequential::path_length_haversine_any(&empty_coords), 0.0);
    }

    #[test]
    fn test_high_performance_functions_single_point() {
        let single_coord = vec![LngLat::new_deg(-122.4194, 37.7749)];

        assert_eq!(
            sequential::pairwise_haversine_iter(single_coord.iter().copied()).len(),
            0
        );
        assert_eq!(sequential::pairwise_haversine_any(&single_coord).len(), 0);
        assert_eq!(
            sequential::path_length_haversine_iter(single_coord.iter().copied()),
            0.0
        );
        assert_eq!(sequential::path_length_haversine_any(&single_coord), 0.0);
    }

    #[test]
    fn test_pairwise_haversine_iter_extend() {
        let coords = [LngLat::new_deg(0.0, 0.0), LngLat::new_deg(1.0, 0.0)];

        let mut result = vec![999.0];
        sequential::pairwise_haversine_iter_extend(coords.iter().copied(), &mut result);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 999.0);
        assert!(result[1] > 110_000.0);
    }

    #[test]
    fn test_pairwise_haversine_any_extend() {
        let coords = vec![LngLat::new_deg(0.0, 0.0), LngLat::new_deg(1.0, 0.0)];

        let mut result = vec![777.0];
        sequential::pairwise_haversine_any_extend(&coords, &mut result);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 777.0);
        assert!(result[1] > 110_000.0);
    }

    #[cfg(feature = "batch")]
    #[test]
    fn test_pairwise_haversine_par_iter_extend() {
        let coords = [
            LngLat::new_deg(0.0, 0.0),
            LngLat::new_deg(1.0, 0.0),
            LngLat::new_deg(1.0, 1.0),
        ];

        let mut par_result = vec![111.0];
        let mut seq_result = vec![111.0];

        parallel::pairwise_haversine_par_iter_extend(coords.iter().copied(), &mut par_result);
        sequential::pairwise_haversine_iter_extend(coords.iter().copied(), &mut seq_result);

        assert_eq!(par_result.len(), seq_result.len());
        assert_eq!(par_result[0], seq_result[0]);

        for (par_dist, seq_dist) in par_result.iter().skip(1).zip(seq_result.iter().skip(1)) {
            assert!((par_dist - seq_dist).abs() < 1e-6_f64);
        }
    }

    #[cfg(feature = "batch")]
    #[test]
    fn test_pairwise_haversine_par_any_extend() {
        let coords = vec![
            LngLat::new_deg(0.0, 0.0),
            LngLat::new_deg(1.0, 0.0),
            LngLat::new_deg(1.0, 1.0),
        ];

        let mut par_result = vec![222.0];
        let mut seq_result = vec![222.0];

        parallel::pairwise_haversine_par_any_extend(&coords, &mut par_result);
        sequential::pairwise_haversine_any_extend(&coords, &mut seq_result);

        assert_eq!(par_result.len(), seq_result.len());
        assert_eq!(par_result[0], seq_result[0]);

        for (par_dist, seq_dist) in par_result.iter().skip(1).zip(seq_result.iter().skip(1)) {
            assert!((par_dist - seq_dist).abs() < 1e-6_f64);
        }
    }

    #[test]
    fn test_pairwise_haversine_any_vec_lnglat() {
        let coords = vec![
            LngLat::new_deg(-122.4194, 37.7749), // San Francisco
            LngLat::new_deg(-74.0060, 40.7128),  // New York
        ];

        let distances = sequential::pairwise_haversine_any(&coords);
        assert_eq!(distances.len(), 1);
        assert!(distances[0] > 4_000_000.0 && distances[0] < 4_200_000.0);
    }

    #[test]
    fn test_pairwise_haversine_any_vec_tuples() {
        let coords = vec![
            (-122.4194, 37.7749), // San Francisco (lng, lat)
            (-74.0060, 40.7128),  // New York (lng, lat)
        ];

        let distances = sequential::pairwise_haversine_any(&coords);
        assert_eq!(distances.len(), 1);
        assert!(distances[0] > 4_000_000.0 && distances[0] < 4_200_000.0);
    }

    #[test]
    fn test_pairwise_haversine_any_flat_array() {
        let coords = vec![
            -122.4194, 37.7749, // San Francisco
            -74.0060, 40.7128, // New York
        ];

        let distances = sequential::pairwise_haversine_any(&coords);
        assert_eq!(distances.len(), 1);
        assert!(distances[0] > 4_000_000.0 && distances[0] < 4_200_000.0);
    }

    #[test]
    fn test_pairwise_haversine_any_empty_coords() {
        let empty_coords: Vec<LngLat> = vec![];
        let distances = sequential::pairwise_haversine_any(&empty_coords);
        assert_eq!(distances.len(), 0);
    }

    #[test]
    fn test_pairwise_haversine_any_single_coord() {
        let coords = vec![LngLat::new_deg(-122.4194, 37.7749)];
        let distances = sequential::pairwise_haversine_any(&coords);
        assert_eq!(distances.len(), 0);
    }

    #[test]
    fn test_buffer_pool_basic_functionality() {
        let mut pool = buffer_pool::BufferPool::new(100);

        assert_eq!(pool.pool_size(), 0);

        let buffer = pool.get_buffer();
        assert_eq!(buffer.len(), 0);
        assert!(buffer.capacity() >= 100);

        pool.return_buffer(buffer);
        assert_eq!(pool.pool_size(), 1);

        let result = pool.with_buffer(|buffer| {
            buffer.push(42.0);
            buffer.len()
        });
        assert_eq!(result, 1);
        assert_eq!(pool.pool_size(), 1);
    }

    #[test]
    fn test_buffer_pool_pairwise_haversine() {
        let mut pool = buffer_pool::BufferPool::new(10);

        let coords = vec![LngLat::new_deg(0.0, 0.0), LngLat::new_deg(1.0, 0.0)];

        let distances = pool.pairwise_haversine_any(&coords);
        assert_eq!(distances.len(), 1);
        assert!(distances[0] > 110_000.0);

        // Pool should still have a buffer
        assert_eq!(pool.pool_size(), 0); // Buffer is consumed by the function
    }

    #[test]
    fn test_buffer_pool_max_size() {
        let mut pool = buffer_pool::BufferPool::with_max_size(50, 2);

        // Get two different buffers and return them both
        let buffer1 = pool.get_buffer();
        let buffer2 = pool.get_buffer();
        pool.return_buffer(buffer1);
        pool.return_buffer(buffer2);
        assert_eq!(pool.pool_size(), 2);

        // Third buffer should be dropped
        let buffer3 = pool.get_buffer();
        pool.return_buffer(buffer3);
        assert_eq!(pool.pool_size(), 2);
    }

    #[test]
    fn test_buffer_pool_clear() {
        let mut pool = buffer_pool::BufferPool::new(50);

        // Get two different buffers and return them both
        let buffer1 = pool.get_buffer();
        let buffer2 = pool.get_buffer();
        pool.return_buffer(buffer1);
        pool.return_buffer(buffer2);
        assert_eq!(pool.pool_size(), 2);

        pool.clear_pool();
        assert_eq!(pool.pool_size(), 0);
    }

    #[test]
    fn test_buffer_pool_pairwise_haversine_iter() {
        let mut pool = buffer_pool::BufferPool::new(10);

        let coords = [
            LngLat::new_deg(0.0, 0.0),
            LngLat::new_deg(1.0, 0.0),
            LngLat::new_deg(1.0, 1.0),
        ];

        let distances = pool.pairwise_haversine_iter(coords.iter().copied());
        assert_eq!(distances.len(), 2);
        assert!(distances[0] > 110_000.0); // ~111km for 1° longitude
        assert!(distances[1] > 110_000.0); // ~111km for 1° latitude
    }

    #[cfg(feature = "batch")]
    #[test]
    fn test_buffer_pool_pairwise_haversine_par_iter() {
        let mut pool = buffer_pool::BufferPool::new(10);

        let coords = [
            LngLat::new_deg(0.0, 0.0),
            LngLat::new_deg(1.0, 0.0),
            LngLat::new_deg(1.0, 1.0),
        ];

        let distances = pool.pairwise_haversine_par_iter(coords.iter().copied());
        assert_eq!(distances.len(), 2);
        assert!(distances[0] > 110_000.0);
        assert!(distances[1] > 110_000.0);
    }

    #[cfg(feature = "batch")]
    #[test]
    fn test_buffer_pool_pairwise_haversine_par_any() {
        let mut pool = buffer_pool::BufferPool::new(10);

        let coords = vec![
            LngLat::new_deg(0.0, 0.0),
            LngLat::new_deg(1.0, 0.0),
            LngLat::new_deg(1.0, 1.0),
        ];

        let distances = pool.pairwise_haversine_par_any(&coords);
        assert_eq!(distances.len(), 2);
        assert!(distances[0] > 110_000.0);
        assert!(distances[1] > 110_000.0);
    }

    #[cfg(feature = "batch")]
    #[test]
    fn test_parallel_pairwise_haversine_par_any() {
        let coords = vec![
            LngLat::new_deg(-122.4194, 37.7749), // San Francisco
            LngLat::new_deg(-74.0060, 40.7128),  // New York
        ];

        let distances = parallel::pairwise_haversine_par_any(&coords);
        assert_eq!(distances.len(), 1);
        assert!(distances[0] > 4_000_000.0 && distances[0] < 4_200_000.0);
    }

    #[test]
    fn test_sequential_edge_cases() {
        // Test empty iterator for pairwise_haversine_iter_extend
        let mut output = vec![];
        sequential::pairwise_haversine_iter_extend(std::iter::empty::<LngLat>(), &mut output);
        assert_eq!(output.len(), 0);

        // Test empty iterator for path_length_haversine_iter
        let length = sequential::path_length_haversine_iter(std::iter::empty::<LngLat>());
        assert_eq!(length, 0.0);
    }

    #[test]
    fn test_sequential_pairwise_haversine_iter_into_buffer_overflow() {
        let coords = [
            LngLat::new_deg(0.0, 0.0),
            LngLat::new_deg(1.0, 0.0),
            LngLat::new_deg(1.0, 1.0),
        ];

        // Buffer too small - should fill what it can and zero the rest
        let mut small_buffer = [f64::NAN; 1];
        let count =
            sequential::pairwise_haversine_iter_into(coords.iter().copied(), &mut small_buffer);
        assert_eq!(count, 1); // Only wrote 1 value
        assert!(!small_buffer[0].is_nan()); // First value written

        // Larger buffer - should zero unused elements
        let mut large_buffer = [f64::NAN; 5];
        let count =
            sequential::pairwise_haversine_iter_into(coords.iter().copied(), &mut large_buffer);
        assert_eq!(count, 5); // Function zeroes remaining elements and returns buffer length
        assert!(!large_buffer[0].is_nan()); // First value written
        assert!(!large_buffer[1].is_nan()); // Second value written
        assert_eq!(large_buffer[2], 0.0); // Third value zeroed
        assert_eq!(large_buffer[3], 0.0); // Fourth value zeroed
        assert_eq!(large_buffer[4], 0.0); // Fifth value zeroed
    }
}
