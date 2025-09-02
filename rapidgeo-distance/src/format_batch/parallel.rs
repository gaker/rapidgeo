use crate::formats::CoordSource;
use crate::geodesic;
use crate::LngLat;
use rayon::prelude::*;

/// Computes pairwise Haversine distances using parallel processing.
///
/// Calculates distances between consecutive coordinate pairs using multiple CPU cores.
/// The iterator is first collected into a Vec, then processed in parallel using
/// sliding windows of size 2.
///
/// # Arguments
///
/// * `iter` - Iterator over `LngLat` coordinates
///
/// # Returns
///
/// Vector of distances in meters between consecutive coordinate pairs
///
/// # Performance
///
/// - **Time complexity**: O(n/p) where n = coordinates, p = CPU cores
/// - **Space complexity**: O(n) for coordinate collection + O(n) for results
/// - **Parallel overhead**: Iterator collection + work distribution
/// - **Break-even point**: ~100-500 coordinates depending on CPU
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "batch")]
/// # {
/// use rapidgeo_distance::format_batch::parallel::pairwise_haversine_par_iter;
/// use rapidgeo_distance::LngLat;
///
/// let coords = [
///     LngLat::new_deg(-122.4194, 37.7749), // San Francisco
///     LngLat::new_deg(-74.0060, 40.7128),  // New York
///     LngLat::new_deg(-87.6298, 41.8781),  // Chicago
/// ];
///
/// let distances = pairwise_haversine_par_iter(coords.iter().copied());
/// assert_eq!(distances.len(), 2);
/// assert!(distances[0] > 0.0);
/// assert!(distances[1] > 0.0);
/// # }
/// ```
///
/// # When to Use
///
/// - **Large datasets**: >1000 coordinate pairs
/// - **Multi-core systems**: 4+ CPU cores available
/// - **CPU-bound workloads**: Distance calculation is the bottleneck
/// - **Batch processing**: Processing multiple datasets
///
/// For smaller datasets or I/O-bound operations, use [`crate::sequential::pairwise_haversine_iter`]
/// to avoid parallel overhead.
///
/// # Thread Safety
///
/// The function is thread-safe and the returned Vec can be used across threads.
/// However, the parallel processing happens within this function call.
///
/// # See Also
///
/// - [`crate::sequential::pairwise_haversine_iter`] for serial version
/// - [`pairwise_haversine_par_any`] for `CoordSource` input
/// - [Rayon documentation](https://docs.rs/rayon/) for parallel processing details
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

/// Extends a vector with pairwise distances using parallel processing.
///
/// Computes pairwise Haversine distances in parallel and appends them to an
/// existing vector. This enables efficient composition of parallel results
/// while maintaining the performance benefits of parallel processing.
///
/// # Arguments
///
/// * `iter` - Iterator over `LngLat` coordinates
/// * `output` - Vector to extend with computed distances
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "batch")]
/// # {
/// use rapidgeo_distance::format_batch::parallel::pairwise_haversine_par_iter_extend;
/// use rapidgeo_distance::LngLat;
///
/// let coords = [
///     LngLat::new_deg(0.0, 0.0),
///     LngLat::new_deg(1.0, 0.0),
///     LngLat::new_deg(1.0, 1.0),
/// ];
///
/// let mut results = vec![999.0]; // Existing data
/// pairwise_haversine_par_iter_extend(coords.iter().copied(), &mut results);
///
/// assert_eq!(results.len(), 3); // 1 existing + 2 computed
/// assert_eq!(results[0], 999.0); // Original data preserved
/// # }
/// ```
///
/// # Performance vs Serial
///
/// Parallel version has additional overhead for:
/// - Iterator collection into temporary Vec
/// - Parallel work distribution and coordination
/// - Result collection and vector extension
///
/// Consider using [`crate::sequential::pairwise_haversine_iter_extend`] for smaller datasets.
///
/// # Memory Usage
///
/// - **Input collection**: O(n) temporary storage for coordinates
/// - **Parallel results**: O(n) temporary storage for distances  
/// - **Final extension**: Results moved into output vector
///
/// Total peak usage is ~3x the final output size during processing.
///
/// # Use Cases
///
/// - **Multi-segment routes**: Parallel processing of route segments
/// - **Batch composition**: Combining parallel results from multiple sources
/// - **Pipeline processing**: Parallel stage in larger processing pipeline
///
/// # See Also
///
/// - [`crate::sequential::pairwise_haversine_iter_extend`] for serial version
/// - [`pairwise_haversine_par_iter`] for new vector creation
/// - [`pairwise_haversine_par_any_extend`] for `CoordSource` input
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

/// Computes pairwise distances from any coordinate source using parallel processing.
///
/// Accepts any coordinate source implementing [`CoordSource + Sync`] and computes
/// pairwise Haversine distances in parallel. This combines the flexibility of
/// the coordinate source abstraction with the performance of parallel processing.
///
/// # Arguments
///
/// * `coords` - Any thread-safe coordinate source (Vec<LngLat>, Vec<(f64,f64)>, etc.)
///
/// # Returns
///
/// Vector of distances in meters between consecutive coordinate pairs
///
/// # Thread Safety Requirement
///
/// The coordinate source must implement `Sync` for parallel processing. This is
/// automatically satisfied by standard types like `Vec<LngLat>`, `Vec<(f64,f64)>`,
/// and `Vec<f64>`, but may require attention for custom coordinate types.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "batch")]
/// # {
/// use rapidgeo_distance::format_batch::parallel::pairwise_haversine_par_any;
/// use rapidgeo_distance::LngLat;
///
/// // Works with different coordinate formats
/// let coords_lnglat: Vec<LngLat> = (0..1000)
///     .map(|i| LngLat::new_deg(i as f64 * 0.1, (i + 1) as f64 * 0.1))
///     .collect();
///
/// let distances1 = pairwise_haversine_par_any(&coords_lnglat);
///
/// let coords_tuples: Vec<(f64, f64)> = coords_lnglat
///     .iter()
///     .map(|c| (c.lng_deg, c.lat_deg))
///     .collect();
///
/// let distances2 = pairwise_haversine_par_any(&coords_tuples);
///
/// // Results should be nearly identical
/// assert_eq!(distances1.len(), distances2.len());
/// for (d1, d2) in distances1.iter().zip(distances2.iter()) {
///     assert!((d1 - d2).abs() < 1e-10);
/// }
/// # }
/// ```
///
/// # Format Support
///
/// All standard coordinate formats that implement `Sync`:
/// - `Vec<LngLat>` - Native format, optimal performance
/// - `Vec<(f64, f64)>` - Tuples with format detection
/// - `Vec<f64>` - Flat arrays (chunked into pairs)
/// - `&[f64]` - Array slices
///
/// # Performance Considerations
///
/// - **Format detection overhead**: Some formats require detection analysis
/// - **Iterator boxing**: `CoordSource` uses boxed iterators (small overhead)
/// - **Collection cost**: Iterator must be collected before parallel processing
/// - **Parallel benefit**: Most significant with >1000 coordinate pairs
///
/// # Sync Requirement Details
///
/// The `+ Sync` bound ensures that:
/// - Coordinate data can be safely accessed from multiple threads
/// - No data races occur during parallel processing
/// - Standard types (Vec, slices) automatically satisfy this
///
/// Custom coordinate types may need explicit `Sync` implementation.
///
/// # Use Cases
///
/// - **Large GPS tracks**: Processing thousands of GPS coordinates
/// - **Batch route analysis**: Multiple routes processed in parallel
/// - **Scientific computing**: Large-scale geospatial data processing
/// - **Performance testing**: Comparing serial vs parallel processing
///
/// # See Also
///
/// - [`crate::sequential::pairwise_haversine_any`] for serial version
/// - [`pairwise_haversine_par_iter`] for iterator input
/// - [`CoordSource`] trait for supported input types
pub fn pairwise_haversine_par_any<T: CoordSource + Sync>(coords: &T) -> Vec<f64> {
    pairwise_haversine_par_iter(coords.get_coords())
}

/// Extends a vector with parallel pairwise distances from any coordinate source.
///
/// Combines the flexibility of [`CoordSource`] input formats with parallel processing
/// and efficient vector extension. This is ideal for building results from multiple
/// coordinate sources or processing data in chunks.
///
/// # Arguments
///
/// * `coords` - Any thread-safe coordinate source implementing [`CoordSource + Sync`]
/// * `output` - Vector to extend with computed distances
///
/// # Thread Safety
///
/// Requires `CoordSource + Sync` to ensure safe parallel access to coordinate data.
/// All standard coordinate types satisfy this requirement.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "batch")]
/// # {
/// use rapidgeo_distance::format_batch::parallel::pairwise_haversine_par_any_extend;
/// use rapidgeo_distance::LngLat;
///
/// let mut all_distances = vec![]; // Accumulator
///
/// // Process first route segment in parallel
/// let route1: Vec<LngLat> = (0..500)
///     .map(|i| LngLat::new_deg(i as f64 * 0.01, 0.0))
///     .collect();
/// pairwise_haversine_par_any_extend(&route1, &mut all_distances);
///
/// // Process second route segment in parallel
/// let route2: Vec<(f64, f64)> = (500..1000)
///     .map(|i| (i as f64 * 0.01, 0.0))
///     .collect();
/// pairwise_haversine_par_any_extend(&route2, &mut all_distances);
///
/// assert_eq!(all_distances.len(), 998); // (499) + (499) distances
/// # }
/// ```
///
/// # Performance Benefits
///
/// - **Parallel computation**: Leverages multiple CPU cores for distance calculations
/// - **Efficient extension**: Single vector extension operation per call
/// - **Format flexibility**: Automatic handling of different coordinate formats
/// - **Memory reuse**: Extends existing vector rather than creating new ones
///
/// # Memory Efficiency
///
/// The function internally:
/// 1. Collects coordinates into temporary Vec (O(n) space)
/// 2. Computes distances in parallel (O(n) space)
/// 3. Extends output vector (amortized O(1) per element)
/// 4. Releases temporary allocations
///
/// Peak memory usage is ~3x the final output size during processing.
///
/// # Format Detection Overhead
///
/// Some coordinate formats require detection analysis:
/// - `Vec<(f64, f64)>`: Requires lng,lat vs lat,lng detection
/// - `Vec<f64>`: Requires format detection after chunking
/// - `Vec<LngLat>`: No detection needed (known format)
///
/// Detection overhead is typically negligible compared to parallel distance computation.
///
/// # Use Cases
///
/// - **Multi-format datasets**: Processing mixed coordinate formats in parallel
/// - **Streaming processing**: Accumulating results as data arrives
/// - **Batch composition**: Combining parallel results from multiple sources
/// - **Pipeline stages**: Parallel processing stage in data pipelines
///
/// # Comparison with Serial Version
///
/// ```
/// # #[cfg(feature = "batch")]
/// # {
/// use rapidgeo_distance::format_batch::sequential::pairwise_haversine_any_extend;
/// use rapidgeo_distance::format_batch::parallel::pairwise_haversine_par_any_extend;
/// use rapidgeo_distance::LngLat;
///
/// let coords: Vec<LngLat> = (0..1000)
///     .map(|i| LngLat::new_deg(i as f64 * 0.01, 0.0))
///     .collect();
///
/// let mut serial_result = vec![];
/// let mut parallel_result = vec![];
///
/// pairwise_haversine_any_extend(&coords, &mut serial_result);
/// pairwise_haversine_par_any_extend(&coords, &mut parallel_result);
///
/// // Results should be nearly identical
/// assert_eq!(serial_result.len(), parallel_result.len());
/// for (s, p) in serial_result.iter().zip(parallel_result.iter()) {
///     assert!((s - p).abs() < 1e-6);
/// }
/// # }
/// ```
///
/// # See Also
///
/// - [`crate::sequential::pairwise_haversine_any_extend`] for serial version
/// - [`pairwise_haversine_par_iter_extend`] for iterator input
/// - [`pairwise_haversine_par_any`] for new vector creation
pub fn pairwise_haversine_par_any_extend<T: CoordSource + Sync>(coords: &T, output: &mut Vec<f64>) {
    pairwise_haversine_par_iter_extend(coords.get_coords(), output);
}
