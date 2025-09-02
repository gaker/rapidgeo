use crate::formats::CoordSource;
use crate::geodesic;
use crate::LngLat;

pub struct BufferPool {
    buffers: Vec<Vec<f64>>,
    initial_capacity: usize,
    max_pool_size: usize,
}

impl BufferPool {
    pub fn new(initial_capacity: usize) -> Self {
        Self {
            buffers: Vec::new(),
            initial_capacity,
            max_pool_size: 8,
        }
    }

    pub fn with_max_size(initial_capacity: usize, max_pool_size: usize) -> Self {
        Self {
            buffers: Vec::new(),
            initial_capacity,
            max_pool_size,
        }
    }

    pub fn get_buffer(&mut self) -> Vec<f64> {
        self.buffers
            .pop()
            .unwrap_or_else(|| Vec::with_capacity(self.initial_capacity))
    }

    pub fn return_buffer(&mut self, mut buffer: Vec<f64>) {
        if self.buffers.len() < self.max_pool_size {
            buffer.clear();
            self.buffers.push(buffer);
        }
    }

    pub fn with_buffer<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut Vec<f64>) -> R,
    {
        let mut buffer = self.get_buffer();
        let result = f(&mut buffer);
        self.return_buffer(buffer);
        result
    }

    pub fn pairwise_haversine_iter<I>(&mut self, iter: I) -> Vec<f64>
    where
        I: Iterator<Item = LngLat>,
    {
        let mut result = self.get_buffer();
        pairwise_haversine_iter_extend(iter, &mut result);
        result
    }

    pub fn pairwise_haversine_any<T: CoordSource>(&mut self, coords: &T) -> Vec<f64> {
        let mut result = self.get_buffer();
        pairwise_haversine_any_extend(coords, &mut result);
        result
    }

    #[cfg(feature = "batch")]
    pub fn pairwise_haversine_par_iter<I>(&mut self, iter: I) -> Vec<f64>
    where
        I: Iterator<Item = LngLat>,
    {
        let mut result = self.get_buffer();
        parallel::pairwise_haversine_par_iter_extend(iter, &mut result);
        result
    }

    #[cfg(feature = "batch")]
    pub fn pairwise_haversine_par_any<T: CoordSource + Sync>(&mut self, coords: &T) -> Vec<f64> {
        let mut result = self.get_buffer();
        parallel::pairwise_haversine_par_any_extend(coords, &mut result);
        result
    }

    pub fn pool_size(&self) -> usize {
        self.buffers.len()
    }

    pub fn clear_pool(&mut self) {
        self.buffers.clear();
    }
}

pub fn pairwise_haversine_iter<I>(iter: I) -> Vec<f64>
where
    I: Iterator<Item = LngLat>,
{
    let mut result = Vec::new();
    pairwise_haversine_iter_extend(iter, &mut result);
    result
}

pub fn pairwise_haversine_iter_extend<I>(iter: I, output: &mut Vec<f64>)
where
    I: Iterator<Item = LngLat>,
{
    let mut iter = iter;

    if let Some(mut prev) = iter.next() {
        for current in iter {
            output.push(geodesic::haversine(prev, current));
            prev = current;
        }
    }
}

pub fn pairwise_haversine_iter_into<I>(iter: I, output: &mut [f64]) -> usize
where
    I: Iterator<Item = LngLat>,
{
    let mut iter = iter;
    let mut index = 0;

    if let Some(mut prev) = iter.next() {
        for current in iter {
            if index < output.len() {
                output[index] = geodesic::haversine(prev, current);
                index += 1;
                prev = current;
            } else {
                break;
            }
        }
    }

    while index < output.len() {
        output[index] = 0.0;
        index += 1;
    }

    index
}

pub fn path_length_haversine_iter<I>(iter: I) -> f64
where
    I: Iterator<Item = LngLat>,
{
    let mut iter = iter;
    let mut total = 0.0;

    if let Some(mut prev) = iter.next() {
        for current in iter {
            total += geodesic::haversine(prev, current);
            prev = current;
        }
    }

    total
}

pub fn pairwise_haversine_any<T: CoordSource>(coords: &T) -> Vec<f64> {
    pairwise_haversine_iter(coords.get_coords())
}

pub fn pairwise_haversine_any_extend<T: CoordSource>(coords: &T, output: &mut Vec<f64>) {
    pairwise_haversine_iter_extend(coords.get_coords(), output);
}

pub fn pairwise_haversine_into_any<T: CoordSource>(coords: &T, output: &mut [f64]) {
    pairwise_haversine_iter_into(coords.get_coords(), output);
}

pub fn path_length_haversine_any<T: CoordSource>(coords: &T) -> f64 {
    path_length_haversine_iter(coords.get_coords())
}

#[cfg(feature = "batch")]
pub mod parallel {
    use super::*;
    use rayon::prelude::*;

    pub fn pairwise_haversine_par_iter<I>(iter: I) -> Vec<f64>
    where
        I: Iterator<Item = LngLat>,
    {
        let points: Vec<_> = iter.collect();

        points
            .par_windows(2)
            .map(|window| geodesic::haversine(window[0], window[1]))
            .collect()
    }

    pub fn pairwise_haversine_par_iter_extend<I>(iter: I, output: &mut Vec<f64>)
    where
        I: Iterator<Item = LngLat>,
    {
        let points: Vec<_> = iter.collect();

        let distances: Vec<f64> = points
            .par_windows(2)
            .map(|window| geodesic::haversine(window[0], window[1]))
            .collect();

        output.extend(distances);
    }

    pub fn pairwise_haversine_par_any<T: CoordSource + Sync>(coords: &T) -> Vec<f64> {
        pairwise_haversine_par_iter(coords.get_coords())
    }

    pub fn pairwise_haversine_par_any_extend<T: CoordSource + Sync>(
        coords: &T,
        output: &mut Vec<f64>,
    ) {
        pairwise_haversine_par_iter_extend(coords.get_coords(), output);
    }
}

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

        let iter_result = pairwise_haversine_iter(coords.iter().copied());
        let any_result = pairwise_haversine_any(&coords);

        assert_eq!(iter_result.len(), any_result.len());
        for (iter_dist, any_dist) in iter_result.iter().zip(any_result.iter()) {
            assert!((iter_dist - any_dist).abs() < 1e-10);
        }
    }

    #[test]
    fn test_pairwise_haversine_iter_into_consistency() {
        let coords = vec![
            LngLat::new_deg(-122.4194, 37.7749),
            LngLat::new_deg(-74.0060, 40.7128),
        ];

        let mut iter_output = vec![0.0; 1];
        let written = pairwise_haversine_iter_into(coords.iter().copied(), &mut iter_output);

        let mut any_output = vec![0.0; 1];
        pairwise_haversine_into_any(&coords, &mut any_output);

        assert_eq!(written, 1);
        assert!((iter_output[0] - any_output[0]).abs() < 1e-10);
    }

    #[test]
    fn test_path_length_haversine_iter_consistency() {
        let coords = vec![
            LngLat::new_deg(-122.4194, 37.7749),
            LngLat::new_deg(-74.0060, 40.7128),
            LngLat::new_deg(-87.6298, 41.8781),
        ];

        let iter_result = path_length_haversine_iter(coords.iter().copied());
        let any_result = path_length_haversine_any(&coords);

        assert!((iter_result - any_result).abs() < 1e-10);
    }

    #[cfg(feature = "batch")]
    #[test]
    fn test_pairwise_haversine_par_iter_consistency() {
        let coords = vec![
            LngLat::new_deg(-122.4194, 37.7749),
            LngLat::new_deg(-74.0060, 40.7128),
            LngLat::new_deg(-87.6298, 41.8781),
        ];

        let par_iter_result = parallel::pairwise_haversine_par_iter(coords.iter().copied());
        let par_any_result = parallel::pairwise_haversine_par_any(&coords);
        let serial_result = pairwise_haversine_any(&coords);

        assert_eq!(par_iter_result.len(), par_any_result.len());
        assert_eq!(par_iter_result.len(), serial_result.len());

        for ((par_iter, par_any), serial) in par_iter_result
            .iter()
            .zip(par_any_result.iter())
            .zip(serial_result.iter())
        {
            assert!((par_iter - par_any).abs() < 1e-10);
            assert!((par_iter - serial).abs() < 1e-6);
        }
    }

    #[test]
    fn test_high_performance_functions_empty_input() {
        let empty: Vec<LngLat> = vec![];

        let iter_result = pairwise_haversine_iter(empty.iter().copied());
        assert_eq!(iter_result.len(), 0);

        let mut output = vec![42.0; 3];
        pairwise_haversine_iter_into(empty.iter().copied(), &mut output);
        assert_eq!(output, vec![0.0, 0.0, 0.0]);

        let path_length = path_length_haversine_iter(empty.iter().copied());
        assert_eq!(path_length, 0.0);
    }

    #[test]
    fn test_high_performance_functions_single_point() {
        let single = [LngLat::new_deg(0.0, 0.0)];

        let iter_result = pairwise_haversine_iter(single.iter().copied());
        assert_eq!(iter_result.len(), 0);

        let mut output = vec![42.0; 2];
        pairwise_haversine_iter_into(single.iter().copied(), &mut output);
        assert_eq!(output, vec![0.0, 0.0]);

        let path_length = path_length_haversine_iter(single.iter().copied());
        assert_eq!(path_length, 0.0);
    }

    #[test]
    fn test_pairwise_haversine_iter_extend() {
        let coords = [
            LngLat::new_deg(-122.4194, 37.7749),
            LngLat::new_deg(-74.0060, 40.7128),
            LngLat::new_deg(-87.6298, 41.8781),
        ];

        let mut output = vec![100.0];
        pairwise_haversine_iter_extend(coords.iter().copied(), &mut output);

        assert_eq!(output.len(), 3);
        assert_eq!(output[0], 100.0);
        assert!(output[1] > 4_000_000.0 && output[1] < 4_200_000.0);
        assert!(output[2] > 1_100_000.0 && output[2] < 1_200_000.0);

        let direct_result = pairwise_haversine_iter(coords.iter().copied());
        assert_eq!(output[1], direct_result[0]);
        assert_eq!(output[2], direct_result[1]);
    }

    #[test]
    fn test_pairwise_haversine_any_extend() {
        let coords = vec![
            LngLat::new_deg(-122.4194, 37.7749),
            LngLat::new_deg(-74.0060, 40.7128),
        ];

        let mut output = vec![999.0];
        pairwise_haversine_any_extend(&coords, &mut output);

        assert_eq!(output.len(), 2);
        assert_eq!(output[0], 999.0);
        assert!(output[1] > 4_000_000.0 && output[1] < 4_200_000.0);

        let direct_result = pairwise_haversine_any(&coords);
        assert_eq!(output[1], direct_result[0]);
    }

    #[cfg(feature = "batch")]
    #[test]
    fn test_pairwise_haversine_par_iter_extend() {
        let coords = [
            LngLat::new_deg(-122.4194, 37.7749),
            LngLat::new_deg(-74.0060, 40.7128),
            LngLat::new_deg(-87.6298, 41.8781),
        ];

        let mut output = vec![555.0];
        parallel::pairwise_haversine_par_iter_extend(coords.iter().copied(), &mut output);

        assert_eq!(output.len(), 3);
        assert_eq!(output[0], 555.0);

        let direct_result = parallel::pairwise_haversine_par_iter(coords.iter().copied());
        let serial_result = pairwise_haversine_iter(coords.iter().copied());

        assert_eq!(output[1], direct_result[0]);
        assert_eq!(output[2], direct_result[1]);

        assert!((output[1] - serial_result[0]).abs() < 1e-6);
        assert!((output[2] - serial_result[1]).abs() < 1e-6);
    }

    #[cfg(feature = "batch")]
    #[test]
    fn test_pairwise_haversine_par_any_extend() {
        let coords = vec![
            LngLat::new_deg(-122.4194, 37.7749),
            LngLat::new_deg(-74.0060, 40.7128),
        ];

        let mut output = vec![777.0];
        parallel::pairwise_haversine_par_any_extend(&coords, &mut output);

        assert_eq!(output.len(), 2);
        assert_eq!(output[0], 777.0);

        let direct_result = parallel::pairwise_haversine_par_any(&coords);
        let serial_result = pairwise_haversine_any(&coords);

        assert_eq!(output[1], direct_result[0]);
        assert!((output[1] - serial_result[0]).abs() < 1e-6);
    }

    #[test]
    fn test_pairwise_haversine_any_vec_lnglat() {
        let coords = vec![
            LngLat::new_deg(-122.4194, 37.7749), // San Francisco
            LngLat::new_deg(-74.0060, 40.7128),  // New York
            LngLat::new_deg(-87.6298, 41.8781),  // Chicago
        ];

        let distances = pairwise_haversine_any(&coords);

        assert_eq!(distances.len(), 2);
        // SF to NYC is approximately 4100km
        assert!(distances[0] > 4_000_000.0 && distances[0] < 4_200_000.0);
        // NYC to Chicago is approximately 1150km
        assert!(distances[1] > 1_100_000.0 && distances[1] < 1_200_000.0);
    }

    #[test]
    fn test_pairwise_haversine_any_vec_tuples() {
        let coords = vec![
            (-122.4194, 37.7749), // San Francisco
            (-74.0060, 40.7128),  // New York
            (-87.6298, 41.8781),  // Chicago
        ];

        let distances = pairwise_haversine_any(&coords);

        assert_eq!(distances.len(), 2);
        assert!(distances[0] > 4_000_000.0 && distances[0] < 4_200_000.0);
        assert!(distances[1] > 1_100_000.0 && distances[1] < 1_200_000.0);
    }

    #[test]
    fn test_pairwise_haversine_any_flat_array() {
        let flat_coords = vec![
            -122.4194, 37.7749, // San Francisco
            -74.0060, 40.7128, // New York
            -87.6298, 41.8781, // Chicago
        ];

        let distances = pairwise_haversine_any(&flat_coords);

        assert_eq!(distances.len(), 2);
        assert!(distances[0] > 4_000_000.0 && distances[0] < 4_200_000.0);
        assert!(distances[1] > 1_100_000.0 && distances[1] < 1_200_000.0);
    }

    #[test]
    fn test_pairwise_haversine_any_empty_coords() {
        let empty_coords: Vec<LngLat> = vec![];
        let distances = pairwise_haversine_any(&empty_coords);
        assert_eq!(distances.len(), 0);

        let empty_tuples: Vec<(f64, f64)> = vec![];
        let distances = pairwise_haversine_any(&empty_tuples);
        assert_eq!(distances.len(), 0);

        let empty_flat: Vec<f64> = vec![];
        let distances = pairwise_haversine_any(&empty_flat);
        assert_eq!(distances.len(), 0);
    }

    #[test]
    fn test_pairwise_haversine_any_single_coord() {
        let single_coord = vec![LngLat::new_deg(0.0, 0.0)];
        let distances = pairwise_haversine_any(&single_coord);
        assert_eq!(distances.len(), 0);

        let single_tuple = vec![(0.0, 0.0)];
        let distances = pairwise_haversine_any(&single_tuple);
        assert_eq!(distances.len(), 0);

        let single_flat = vec![0.0, 0.0];
        let distances = pairwise_haversine_any(&single_flat);
        assert_eq!(distances.len(), 0);
    }

    #[test]
    fn test_pairwise_haversine_any_two_coords() {
        let two_coords = vec![LngLat::new_deg(0.0, 0.0), LngLat::new_deg(1.0, 0.0)];

        let distances = pairwise_haversine_any(&two_coords);
        assert_eq!(distances.len(), 1);
        // 1 degree longitude at equator is approximately 111km
        assert!(distances[0] > 110_000.0 && distances[0] < 112_000.0);
    }

    #[test]
    fn test_pairwise_haversine_into_any_vec_lnglat() {
        let coords = vec![
            LngLat::new_deg(-122.4194, 37.7749),
            LngLat::new_deg(-74.0060, 40.7128),
            LngLat::new_deg(-87.6298, 41.8781),
        ];

        let mut output = vec![0.0; 2];
        pairwise_haversine_into_any(&coords, &mut output);

        assert!(output[0] > 4_000_000.0 && output[0] < 4_200_000.0);
        assert!(output[1] > 1_100_000.0 && output[1] < 1_200_000.0);

        // Compare with non-into version
        let distances = pairwise_haversine_any(&coords);
        assert_eq!(output, distances);
    }

    #[test]
    fn test_pairwise_haversine_into_any_larger_buffer() {
        let coords = vec![LngLat::new_deg(0.0, 0.0), LngLat::new_deg(1.0, 0.0)];

        let mut output = vec![f64::NAN; 5]; // Larger than needed
        pairwise_haversine_into_any(&coords, &mut output);

        assert!(!output[0].is_nan());
        assert!(output[0] > 110_000.0 && output[0] < 112_000.0);
        // Remaining elements should be cleared for security
        assert_eq!(output[1], 0.0);
        assert_eq!(output[2], 0.0);
        assert_eq!(output[3], 0.0);
        assert_eq!(output[4], 0.0);
    }

    #[test]
    fn test_pairwise_haversine_into_any_exact_buffer() {
        let coords = vec![
            LngLat::new_deg(0.0, 0.0),
            LngLat::new_deg(1.0, 0.0),
            LngLat::new_deg(1.0, 1.0),
        ];

        let mut output = vec![0.0; 2]; // Exact size needed
        pairwise_haversine_into_any(&coords, &mut output);

        assert_eq!(output.len(), 2);
        assert!(output[0] > 110_000.0 && output[0] < 112_000.0);
        assert!(output[1] > 110_000.0 && output[1] < 112_000.0);
    }

    #[test]
    fn test_pairwise_haversine_into_any_small_buffer() {
        let coords = vec![
            LngLat::new_deg(0.0, 0.0),
            LngLat::new_deg(1.0, 0.0),
            LngLat::new_deg(1.0, 1.0),
            LngLat::new_deg(0.0, 1.0),
        ];

        let mut output = vec![0.0; 2]; // Only space for 2 results
        pairwise_haversine_into_any(&coords, &mut output);

        assert_eq!(output.len(), 2);
        assert!(output[0] > 110_000.0 && output[0] < 112_000.0);
        assert!(output[1] > 110_000.0 && output[1] < 112_000.0);
    }

    #[test]
    fn test_pairwise_haversine_into_any_empty_coords() {
        let empty_coords: Vec<LngLat> = vec![];
        let mut output = vec![f64::NAN; 3];
        pairwise_haversine_into_any(&empty_coords, &mut output);

        // All elements should be cleared for security
        assert_eq!(output[0], 0.0);
        assert_eq!(output[1], 0.0);
        assert_eq!(output[2], 0.0);
    }

    #[test]
    fn test_path_length_haversine_any_vec_lnglat() {
        let coords = vec![
            LngLat::new_deg(-122.4194, 37.7749), // San Francisco
            LngLat::new_deg(-74.0060, 40.7128),  // New York
            LngLat::new_deg(-87.6298, 41.8781),  // Chicago
        ];

        let total_distance = path_length_haversine_any(&coords);

        // Should be sum of SF->NYC + NYC->Chicago (approximately 5250km total)
        assert!(total_distance > 5_100_000.0 && total_distance < 5_400_000.0);

        // Verify it matches sum of pairwise distances
        let pairwise = pairwise_haversine_any(&coords);
        let pairwise_sum: f64 = pairwise.iter().sum();
        assert!((total_distance - pairwise_sum).abs() < 1e-6);
    }

    #[test]
    fn test_path_length_haversine_any_vec_tuples() {
        let coords = vec![
            (-122.4194, 37.7749),
            (-74.0060, 40.7128),
            (-87.6298, 41.8781),
        ];

        let total_distance = path_length_haversine_any(&coords);
        assert!(total_distance > 5_100_000.0 && total_distance < 5_400_000.0);
    }

    #[test]
    fn test_path_length_haversine_any_flat_array() {
        let flat_coords = vec![
            -122.4194, 37.7749, // San Francisco
            -74.0060, 40.7128, // New York
            -87.6298, 41.8781, // Chicago
        ];

        let total_distance = path_length_haversine_any(&flat_coords);
        assert!(total_distance > 5_100_000.0 && total_distance < 5_400_000.0);
    }

    #[test]
    fn test_path_length_haversine_any_empty_coords() {
        let empty_coords: Vec<LngLat> = vec![];
        let total_distance = path_length_haversine_any(&empty_coords);
        assert_eq!(total_distance, 0.0);

        let empty_tuples: Vec<(f64, f64)> = vec![];
        let total_distance = path_length_haversine_any(&empty_tuples);
        assert_eq!(total_distance, 0.0);

        let empty_flat: Vec<f64> = vec![];
        let total_distance = path_length_haversine_any(&empty_flat);
        assert_eq!(total_distance, 0.0);
    }

    #[test]
    fn test_path_length_haversine_any_single_coord() {
        let single_coord = vec![LngLat::new_deg(0.0, 0.0)];
        let total_distance = path_length_haversine_any(&single_coord);
        assert_eq!(total_distance, 0.0);

        let single_tuple = vec![(0.0, 0.0)];
        let total_distance = path_length_haversine_any(&single_tuple);
        assert_eq!(total_distance, 0.0);
    }

    #[test]
    fn test_path_length_haversine_any_two_coords() {
        let two_coords = vec![LngLat::new_deg(0.0, 0.0), LngLat::new_deg(1.0, 0.0)];

        let total_distance = path_length_haversine_any(&two_coords);
        assert!(total_distance > 110_000.0 && total_distance < 112_000.0);
    }

    #[test]
    fn test_consistency_across_coord_source_types() {
        // Same coordinates in different formats should produce identical results
        let test_coords = [
            (-122.4194, 37.7749),
            (-74.0060, 40.7128),
            (-87.6298, 41.8781),
        ];

        // Convert to different CoordSource types
        let lnglat_vec: Vec<LngLat> = test_coords
            .iter()
            .map(|&(lng, lat)| LngLat::new_deg(lng, lat))
            .collect();

        let tuple_vec = test_coords.to_vec();

        let flat_array: Vec<f64> = test_coords
            .iter()
            .flat_map(|&(lng, lat)| vec![lng, lat])
            .collect();

        // Test pairwise distances
        let distances_from_lnglat = pairwise_haversine_any(&lnglat_vec);
        let distances_from_tuples = pairwise_haversine_any(&tuple_vec);
        let distances_from_flat = pairwise_haversine_any(&flat_array);

        assert_eq!(distances_from_lnglat.len(), 2);
        assert_eq!(distances_from_tuples.len(), 2);
        assert_eq!(distances_from_flat.len(), 2);

        for i in 0..2 {
            assert!((distances_from_lnglat[i] - distances_from_tuples[i]).abs() < 1e-10);
            assert!((distances_from_lnglat[i] - distances_from_flat[i]).abs() < 1e-10);
            assert!((distances_from_tuples[i] - distances_from_flat[i]).abs() < 1e-10);
        }

        // Test path lengths
        let path_length_lnglat = path_length_haversine_any(&lnglat_vec);
        let path_length_tuples = path_length_haversine_any(&tuple_vec);
        let path_length_flat = path_length_haversine_any(&flat_array);

        assert!((path_length_lnglat - path_length_tuples).abs() < 1e-10);
        assert!((path_length_lnglat - path_length_flat).abs() < 1e-10);
        assert!((path_length_tuples - path_length_flat).abs() < 1e-10);
    }

    #[cfg(feature = "batch")]
    #[test]
    fn test_pairwise_haversine_par_any_vec_lnglat() {
        let coords = vec![
            LngLat::new_deg(-122.4194, 37.7749),
            LngLat::new_deg(-74.0060, 40.7128),
            LngLat::new_deg(-87.6298, 41.8781),
        ];

        let distances_par = parallel::pairwise_haversine_par_any(&coords);
        let distances_serial = pairwise_haversine_any(&coords);

        assert_eq!(distances_par.len(), distances_serial.len());
        for (par, serial) in distances_par.iter().zip(distances_serial.iter()) {
            assert!((par - serial).abs() < 1e-6);
        }
    }

    #[cfg(feature = "batch")]
    #[test]
    fn test_pairwise_haversine_par_any_vec_tuples() {
        let coords = vec![
            (-122.4194, 37.7749),
            (-74.0060, 40.7128),
            (-87.6298, 41.8781),
        ];

        let distances_par = parallel::pairwise_haversine_par_any(&coords);
        let distances_serial = pairwise_haversine_any(&coords);

        assert_eq!(distances_par.len(), distances_serial.len());
        for (par, serial) in distances_par.iter().zip(distances_serial.iter()) {
            assert!((par - serial).abs() < 1e-6);
        }
    }

    #[cfg(feature = "batch")]
    #[test]
    fn test_pairwise_haversine_par_any_flat_array() {
        let flat_coords = vec![-122.4194, 37.7749, -74.0060, 40.7128, -87.6298, 41.8781];

        let distances_par = parallel::pairwise_haversine_par_any(&flat_coords);
        let distances_serial = pairwise_haversine_any(&flat_coords);

        assert_eq!(distances_par.len(), distances_serial.len());
        for (par, serial) in distances_par.iter().zip(distances_serial.iter()) {
            assert!((par - serial).abs() < 1e-6);
        }
    }

    #[cfg(feature = "batch")]
    #[test]
    fn test_pairwise_haversine_par_any_empty_coords() {
        let empty_coords: Vec<LngLat> = vec![];
        let distances = parallel::pairwise_haversine_par_any(&empty_coords);
        assert_eq!(distances.len(), 0);
    }

    #[cfg(feature = "batch")]
    #[test]
    fn test_pairwise_haversine_par_any_single_coord() {
        let single_coord = vec![LngLat::new_deg(0.0, 0.0)];
        let distances = parallel::pairwise_haversine_par_any(&single_coord);
        assert_eq!(distances.len(), 0);
    }

    #[cfg(feature = "batch")]
    #[test]
    fn test_pairwise_haversine_par_any_large_dataset() {
        // Test with larger dataset to verify parallel processing works correctly
        let coords: Vec<LngLat> = (0..100)
            .map(|i| LngLat::new_deg(i as f64 * 0.01, (i + 1) as f64 * 0.01))
            .collect();

        let distances_par = parallel::pairwise_haversine_par_any(&coords);
        let distances_serial = pairwise_haversine_any(&coords);

        assert_eq!(distances_par.len(), 99);
        assert_eq!(distances_serial.len(), 99);

        for (i, (par, serial)) in distances_par
            .iter()
            .zip(distances_serial.iter())
            .enumerate()
        {
            assert!(
                (par - serial).abs() < 1e-6,
                "Mismatch at index {}: par={}, serial={}",
                i,
                par,
                serial
            );
        }
    }

    #[test]
    fn test_functions_match_batch_module_results() {
        use crate::batch;

        // Test that format_batch functions produce identical results to batch module functions
        let coords = vec![
            LngLat::new_deg(-122.4194, 37.7749),
            LngLat::new_deg(-74.0060, 40.7128),
            LngLat::new_deg(-87.6298, 41.8781),
        ];

        // Test pairwise distances
        let format_batch_distances = pairwise_haversine_any(&coords);
        let batch_distances: Vec<f64> = batch::pairwise_haversine(&coords).collect();

        assert_eq!(format_batch_distances.len(), batch_distances.len());
        for (fb, b) in format_batch_distances.iter().zip(batch_distances.iter()) {
            assert!((fb - b).abs() < 1e-10);
        }

        // Test path lengths
        let format_batch_length = path_length_haversine_any(&coords);
        let batch_length = batch::path_length_haversine(&coords);

        assert!((format_batch_length - batch_length).abs() < 1e-10);

        // Test into function
        let mut format_batch_output = vec![0.0; 2];
        pairwise_haversine_into_any(&coords, &mut format_batch_output);

        let mut batch_output = vec![0.0; 2];
        batch::pairwise_haversine_into(&coords, &mut batch_output);

        for (fb, b) in format_batch_output.iter().zip(batch_output.iter()) {
            assert!((fb - b).abs() < 1e-10);
        }
    }

    #[cfg(feature = "batch")]
    #[test]
    fn test_parallel_functions_match_batch_module_results() {
        use crate::batch;

        let coords = vec![
            LngLat::new_deg(-122.4194, 37.7749),
            LngLat::new_deg(-74.0060, 40.7128),
            LngLat::new_deg(-87.6298, 41.8781),
        ];

        // Test parallel pairwise distances
        let format_batch_par_distances = parallel::pairwise_haversine_par_any(&coords);
        let batch_par_distances = batch::pairwise_haversine_par(&coords);

        assert_eq!(format_batch_par_distances.len(), batch_par_distances.len());
        for (fb, b) in format_batch_par_distances
            .iter()
            .zip(batch_par_distances.iter())
        {
            assert!((fb - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_format_batch_extreme_coordinates() {
        // Test with extreme but valid coordinates
        let extreme_coords = vec![
            LngLat::new_deg(-180.0, -90.0), // Southwest corner
            LngLat::new_deg(180.0, 90.0),   // Northeast corner
            LngLat::new_deg(0.0, 0.0),      // Origin
        ];

        let distances = pairwise_haversine_any(&extreme_coords);
        assert_eq!(distances.len(), 2);

        // Distance from southwest to northeast corner should be large
        assert!(distances[0] > 15_000_000.0); // More than 15,000 km

        // All distances should be finite and positive
        for &distance in &distances {
            assert!(distance.is_finite());
            assert!(distance > 0.0);
        }

        let total_length = path_length_haversine_any(&extreme_coords);
        assert!(total_length.is_finite());
        assert!(total_length > 15_000_000.0);
    }

    #[test]
    fn test_format_batch_high_precision_coordinates() {
        let high_precision = vec![
            LngLat::new_deg(-122.419416123456, 37.774928987654),
            LngLat::new_deg(-74.006012345679, 40.712776543211),
        ];

        let distances = pairwise_haversine_any(&high_precision);
        assert_eq!(distances.len(), 1);
        assert!(distances[0] > 4_000_000.0 && distances[0] < 4_200_000.0);

        let path_length = path_length_haversine_any(&high_precision);
        assert!((path_length - distances[0]).abs() < 1e-10);
    }

    #[test]
    fn test_pairwise_haversine_any_multiple_iterations() {
        let coords = vec![
            LngLat::new_deg(-122.4194, 37.7749), // SF
            LngLat::new_deg(-74.0060, 40.7128),  // NYC
            LngLat::new_deg(-87.6298, 41.8781),  // Chicago
            LngLat::new_deg(-118.2437, 34.0522), // LA
        ];

        let distances = pairwise_haversine_any(&coords);
        assert_eq!(distances.len(), 3);

        for distance in distances {
            assert!(distance > 0.0);
            assert!(distance < 5_000_000.0); // Reasonable upper bound for US distances
        }
    }

    #[test]
    fn test_path_length_haversine_any_multiple_iterations() {
        let coords = vec![
            LngLat::new_deg(0.0, 0.0), // Start
            LngLat::new_deg(1.0, 0.0), // Move east
            LngLat::new_deg(1.0, 1.0), // Move north
            LngLat::new_deg(0.0, 1.0), // Move west
            LngLat::new_deg(0.0, 0.0), // Return to start
        ];

        let total_distance = path_length_haversine_any(&coords);
        assert!(total_distance > 400_000.0); // Should be around 4 * ~111km
        assert!(total_distance < 500_000.0);

        // Verify it's sum of individual segments
        let pairwise_distances = pairwise_haversine_any(&coords);
        let sum_of_segments: f64 = pairwise_distances.iter().sum();
        assert!((total_distance - sum_of_segments).abs() < 1e-10);
    }

    #[test]
    fn test_coord_source_consistency_with_many_points() {
        let coords_lnglat = vec![
            LngLat::new_deg(-122.4194, 37.7749),
            LngLat::new_deg(-74.0060, 40.7128),
            LngLat::new_deg(-87.6298, 41.8781),
            LngLat::new_deg(-118.2437, 34.0522),
            LngLat::new_deg(-95.3698, 29.7604), // Houston
        ];

        let coords_tuples: Vec<(f64, f64)> = coords_lnglat
            .iter()
            .map(|c| (c.lng_deg, c.lat_deg))
            .collect();

        let coords_flat: Vec<f64> = coords_lnglat
            .iter()
            .flat_map(|c| vec![c.lng_deg, c.lat_deg])
            .collect();

        let distances_lnglat = pairwise_haversine_any(&coords_lnglat);
        let distances_tuples = pairwise_haversine_any(&coords_tuples);
        let distances_flat = pairwise_haversine_any(&coords_flat);

        assert_eq!(distances_lnglat.len(), 4);
        assert_eq!(distances_tuples.len(), 4);
        assert_eq!(distances_flat.len(), 4);

        for ((d1, d2), d3) in distances_lnglat
            .iter()
            .zip(distances_tuples.iter())
            .zip(distances_flat.iter())
        {
            assert!((d1 - d2).abs() < 1e-10);
            assert!((d1 - d3).abs() < 1e-10);
        }
    }

    #[test]
    fn test_buffer_pool_creation() {
        let pool = BufferPool::new(100);
        assert_eq!(pool.pool_size(), 0);

        let pool = BufferPool::with_max_size(50, 4);
        assert_eq!(pool.pool_size(), 0);
    }

    #[test]
    fn test_buffer_pool_get_return() {
        let mut pool = BufferPool::new(10);

        let buffer1 = pool.get_buffer();
        assert!(buffer1.capacity() >= 10);
        assert_eq!(buffer1.len(), 0);

        pool.return_buffer(buffer1);
        assert_eq!(pool.pool_size(), 1);

        let buffer2 = pool.get_buffer();
        assert_eq!(pool.pool_size(), 0);

        pool.return_buffer(buffer2);
        assert_eq!(pool.pool_size(), 1);
    }

    #[test]
    fn test_buffer_pool_max_size() {
        let mut pool = BufferPool::with_max_size(10, 2);

        let buf1 = pool.get_buffer();
        let buf2 = pool.get_buffer();
        let buf3 = pool.get_buffer();

        pool.return_buffer(buf1);
        pool.return_buffer(buf2);
        pool.return_buffer(buf3);

        assert_eq!(pool.pool_size(), 2);
    }

    #[test]
    fn test_buffer_pool_with_buffer() {
        let mut pool = BufferPool::new(10);

        let result = pool.with_buffer(|buffer| {
            buffer.push(1.0);
            buffer.push(2.0);
            buffer.len()
        });

        assert_eq!(result, 2);
        assert_eq!(pool.pool_size(), 1);
    }

    #[test]
    fn test_buffer_pool_pairwise_haversine() {
        let mut pool = BufferPool::new(10);

        let coords = [
            LngLat::new_deg(-122.4194, 37.7749),
            LngLat::new_deg(-74.0060, 40.7128),
        ];

        let result1 = pool.pairwise_haversine_iter(coords.iter().copied());
        assert_eq!(result1.len(), 1);
        assert!(result1[0] > 4_000_000.0 && result1[0] < 4_200_000.0);

        assert_eq!(pool.pool_size(), 0);

        let result2 = pool.pairwise_haversine_any(&coords.to_vec());
        assert_eq!(result2.len(), 1);
        assert!((result1[0] - result2[0]).abs() < 1e-10);
    }

    #[cfg(feature = "batch")]
    #[test]
    fn test_buffer_pool_parallel_functions() {
        let mut pool = BufferPool::new(20);

        let coords = [
            LngLat::new_deg(-122.4194, 37.7749),
            LngLat::new_deg(-74.0060, 40.7128),
            LngLat::new_deg(-87.6298, 41.8781),
        ];

        let par_result = pool.pairwise_haversine_par_iter(coords.iter().copied());
        assert_eq!(par_result.len(), 2);

        let serial_result = pool.pairwise_haversine_iter(coords.iter().copied());
        assert_eq!(serial_result.len(), 2);

        for (par, serial) in par_result.iter().zip(serial_result.iter()) {
            assert!((par - serial).abs() < 1e-6);
        }
    }

    #[test]
    fn test_buffer_pool_reuse_efficiency() {
        let mut pool = BufferPool::new(100);

        let coords1 = [LngLat::new_deg(0.0, 0.0), LngLat::new_deg(1.0, 0.0)];
        let coords2 = [LngLat::new_deg(0.0, 0.0), LngLat::new_deg(1.0, 1.0)];

        let result1 = pool.with_buffer(|buffer| {
            pairwise_haversine_iter_extend(coords1.iter().copied(), buffer);
            buffer.clone()
        });

        assert_eq!(pool.pool_size(), 1);

        let result2 = pool.with_buffer(|buffer| {
            pairwise_haversine_iter_extend(coords2.iter().copied(), buffer);
            buffer.clone()
        });

        assert_eq!(pool.pool_size(), 1);
        assert_eq!(result1.len(), 1);
        assert_eq!(result2.len(), 1);
    }

    #[test]
    fn test_buffer_pool_clear() {
        let mut pool = BufferPool::new(10);

        let buf1 = pool.get_buffer();
        let buf2 = pool.get_buffer();

        pool.return_buffer(buf1);
        pool.return_buffer(buf2);

        assert_eq!(pool.pool_size(), 2);

        pool.clear_pool();
        assert_eq!(pool.pool_size(), 0);
    }

    #[test]
    fn test_buffer_pool_memory_efficiency() {
        let mut pool = BufferPool::new(1000);

        let coords: Vec<LngLat> = (0..100)
            .map(|i| LngLat::new_deg(i as f64 * 0.01, (i + 1) as f64 * 0.01))
            .collect();

        let result1 = pool.with_buffer(|buffer| {
            pairwise_haversine_any_extend(&coords, buffer);
            buffer.len()
        });

        assert_eq!(result1, 99);
        assert_eq!(pool.pool_size(), 1);

        let result2 = pool.with_buffer(|buffer| {
            pairwise_haversine_any_extend(&coords, buffer);
            buffer.capacity() >= 99
        });

        assert!(result2);
        assert_eq!(pool.pool_size(), 1);
    }
}
