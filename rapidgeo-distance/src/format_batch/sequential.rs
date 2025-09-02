use crate::formats::CoordSource;
use crate::geodesic;
use crate::LngLat;

/// Computes pairwise Haversine distances between consecutive coordinates.
///
/// Calculates the great-circle distance between each consecutive pair of coordinates
/// using the Haversine formula. For n input coordinates, returns n-1 distances.
///
/// # Arguments
///
/// * `iter` - Iterator over `LngLat` coordinates
///
/// # Returns
///
/// Vector of distances in meters between consecutive coordinate pairs
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::format_batch::sequential::pairwise_haversine_iter;
/// use rapidgeo_distance::LngLat;
///
/// let coords = [
///     LngLat::new_deg(-122.4194, 37.7749), // San Francisco
///     LngLat::new_deg(-74.0060, 40.7128),  // New York
///     LngLat::new_deg(-87.6298, 41.8781),  // Chicago
/// ];
///
/// let distances = pairwise_haversine_iter(coords.iter().copied());
/// assert_eq!(distances.len(), 2); // n-1 distances for n points
///
/// // SF to NYC is approximately 4100km
/// assert!(distances[0] > 4_000_000.0 && distances[0] < 4_200_000.0);
/// // NYC to Chicago is approximately 1150km  
/// assert!(distances[1] > 1_100_000.0 && distances[1] < 1_200_000.0);
/// ```
///
/// # Performance
///
/// - **Time complexity**: O(n) where n is the number of coordinates
/// - **Space complexity**: O(n) for the result vector
/// - **Accuracy**: ±0.5% for distances under 1000km using Haversine formula
///
/// # Edge Cases
///
/// - **Empty iterator**: Returns empty vector
/// - **Single coordinate**: Returns empty vector (no pairs to compute)
/// - **Identical coordinates**: Returns 0.0 distance
///
/// # See Also
///
/// - [`pairwise_haversine_any`] for `CoordSource` input
/// - [`path_length_haversine_iter`] for total path length
/// - [`crate::buffer_pool::BufferPool::pairwise_haversine_iter`] for pooled buffer version
pub fn pairwise_haversine_iter<I>(iter: I) -> Vec<f64>
where
    I: Iterator<Item = LngLat>,
{
    let mut result = Vec::new();
    pairwise_haversine_iter_extend(iter, &mut result);
    result
}

/// Extends a vector with pairwise Haversine distances between consecutive coordinates.
///
/// This is the building block for other pairwise distance functions. It appends
/// distances to an existing vector rather than creating a new one, which enables
/// efficient composition and reuse of allocated buffers.
///
/// # Arguments
///
/// * `iter` - Iterator over `LngLat` coordinates
/// * `output` - Vector to extend with computed distances
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::format_batch::sequential::pairwise_haversine_iter_extend;
/// use rapidgeo_distance::LngLat;
///
/// let coords = [
///     LngLat::new_deg(0.0, 0.0),
///     LngLat::new_deg(1.0, 0.0),
///     LngLat::new_deg(1.0, 1.0),
/// ];
///
/// let mut distances = vec![100.0]; // Existing data
/// pairwise_haversine_iter_extend(coords.iter().copied(), &mut distances);
///
/// assert_eq!(distances.len(), 3); // 1 existing + 2 new
/// assert_eq!(distances[0], 100.0); // Original data preserved
/// assert!(distances[1] > 110_000.0); // ~111km for 1° longitude at equator
/// ```
///
/// # Performance Benefits
///
/// - **No allocation**: Reuses existing vector capacity
/// - **Composable**: Can combine results from multiple sources
/// - **Buffer reuse**: Works efficiently with buffer pools
///
/// # Algorithm Details
///
/// Uses a stateful iteration pattern:
/// 1. Takes first coordinate as starting point
/// 2. Computes distance to each subsequent coordinate
/// 3. Updates the "previous" coordinate for next iteration
/// 4. Continues until iterator is exhausted
///
/// # See Also
///
/// - [`pairwise_haversine_iter`] for standalone vector creation
/// - [`pairwise_haversine_iter_into`] for pre-allocated output buffers
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

/// Computes pairwise Haversine distances into a pre-allocated buffer.
///
/// Fills a fixed-size buffer with pairwise distances, stopping when either the
/// iterator is exhausted or the buffer is full. Remaining buffer elements are
/// zeroed for security and consistency.
///
/// # Arguments
///
/// * `iter` - Iterator over `LngLat` coordinates
/// * `output` - Pre-allocated buffer to fill with distances
///
/// # Returns
///
/// Number of distances actually computed (before buffer exhaustion)
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::format_batch::sequential::pairwise_haversine_iter_into;
/// use rapidgeo_distance::LngLat;
///
/// let coords = [
///     LngLat::new_deg(0.0, 0.0),
///     LngLat::new_deg(1.0, 0.0),
///     LngLat::new_deg(1.0, 1.0),
/// ];
///
/// let mut buffer = [f64::NAN; 2]; // Buffer for 2 distances
/// let count = pairwise_haversine_iter_into(coords.iter().copied(), &mut buffer);
///
/// assert_eq!(count, 2); // 2 distances computed
/// assert!(!buffer[0].is_nan()); // First distance valid
/// assert!(!buffer[1].is_nan()); // Second distance valid
/// ```
///
/// # Buffer Management
///
/// - **Exact size**: Buffer exactly sized for expected output
/// - **Larger buffer**: Excess elements are zeroed
/// - **Smaller buffer**: Computation stops when buffer is full
///
/// ```
/// use rapidgeo_distance::format_batch::sequential::pairwise_haversine_iter_into;
/// use rapidgeo_distance::LngLat;
///
/// let coords = [LngLat::new_deg(0.0, 0.0), LngLat::new_deg(1.0, 0.0), LngLat::new_deg(2.0, 0.0)];
///
/// // Small buffer - computation truncated
/// let mut small_buffer = [0.0; 1];
/// let count = pairwise_haversine_iter_into(coords.iter().copied(), &mut small_buffer);
/// assert_eq!(count, 1); // Only first distance computed
/// ```
///
/// # Security Considerations
///
/// All unused buffer elements are explicitly zeroed to prevent information
/// leakage from previous buffer contents.
///
/// # Use Cases
///
/// - **Fixed-size processing**: When output size is known in advance
/// - **Streaming computation**: Process coordinates in fixed-size chunks
/// - **Memory-constrained environments**: Avoid dynamic allocation
/// - **Interfacing with C/FFI**: Pre-allocated buffers for external APIs
///
/// # See Also
///
/// - [`pairwise_haversine_iter`] for dynamic vector allocation
/// - [`pairwise_haversine_iter_extend`] for extending existing vectors
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

/// Computes the total path length along a sequence of coordinates.
///
/// Calculates the sum of all pairwise Haversine distances between consecutive
/// coordinates, representing the total length of the path that connects them.
///
/// # Arguments
///
/// * `iter` - Iterator over `LngLat` coordinates defining the path
///
/// # Returns
///
/// Total path length in meters
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::format_batch::sequential::path_length_haversine_iter;
/// use rapidgeo_distance::LngLat;
///
/// // Round trip around a 1° square
/// let square_path = [
///     LngLat::new_deg(0.0, 0.0),   // Origin
///     LngLat::new_deg(1.0, 0.0),   // East 1°
///     LngLat::new_deg(1.0, 1.0),   // North 1°  
///     LngLat::new_deg(0.0, 1.0),   // West 1°
///     LngLat::new_deg(0.0, 0.0),   // Back to origin
/// ];
///
/// let total_distance = path_length_haversine_iter(square_path.iter().copied());
///
/// // Should be approximately 4 × 111km = ~444km
/// assert!(total_distance > 440_000.0 && total_distance < 450_000.0);
/// ```
///
/// # Relationship to Pairwise Distances
///
/// The path length is exactly the sum of all pairwise distances:
///
/// ```
/// use rapidgeo_distance::format_batch::sequential::{path_length_haversine_iter, pairwise_haversine_iter};
/// use rapidgeo_distance::LngLat;
///
/// let coords = [
///     LngLat::new_deg(-122.4194, 37.7749), // San Francisco
///     LngLat::new_deg(-74.0060, 40.7128),  // New York
///     LngLat::new_deg(-87.6298, 41.8781),  // Chicago
/// ];
///
/// let path_length = path_length_haversine_iter(coords.iter().copied());
/// let pairwise_distances = pairwise_haversine_iter(coords.iter().copied());
/// let sum_of_distances: f64 = pairwise_distances.iter().sum();
///
/// assert!((path_length - sum_of_distances).abs() < 1e-10);
/// ```
///
/// # Performance
///
/// - **Time complexity**: O(n) where n is the number of coordinates
/// - **Space complexity**: O(1) - only accumulates total
/// - **Memory efficiency**: No intermediate vector allocation
///
/// # Edge Cases
///
/// - **Empty path**: Returns 0.0
/// - **Single point**: Returns 0.0 (no distance to travel)
/// - **Identical points**: Individual segments contribute 0.0 to total
///
/// # Use Cases
///
/// - **Route planning**: Total travel distance estimation
/// - **GPS tracking**: Track total distance covered  
/// - **Performance metrics**: Distance-based comparisons
/// - **Geometric analysis**: Perimeter calculations
///
/// # See Also
///
/// - [`pairwise_haversine_iter`] for individual segment distances
/// - [`path_length_haversine_any`] for `CoordSource` input
/// - [`crate::geodesic::haversine`] for individual coordinate pair distance
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

/// Computes pairwise Haversine distances from any coordinate source.
///
/// Accepts any type implementing [`CoordSource`] and computes distances between
/// consecutive coordinates. This provides a unified interface for distance calculation
/// regardless of the input coordinate format.
///
/// # Arguments
///
/// * `coords` - Any coordinate source (Vec<LngLat>, Vec<(f64,f64)>, Vec<f64>, etc.)
///
/// # Returns
///
/// Vector of distances in meters between consecutive coordinate pairs
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::format_batch::sequential::pairwise_haversine_any;
/// use rapidgeo_distance::LngLat;
///
/// // Works with LngLat vectors
/// let coords_lnglat = vec![
///     LngLat::new_deg(-122.4194, 37.7749),
///     LngLat::new_deg(-74.0060, 40.7128),
/// ];
/// let distances1 = pairwise_haversine_any(&coords_lnglat);
///
/// // Works with tuple vectors (automatic format detection)
/// let coords_tuples = vec![
///     (-122.4194, 37.7749),
///     (-74.0060, 40.7128),
/// ];
/// let distances2 = pairwise_haversine_any(&coords_tuples);
///
/// // Works with flat arrays
/// let coords_flat = vec![
///     -122.4194, 37.7749,  // San Francisco
///     -74.0060, 40.7128,   // New York
/// ];
/// let distances3 = pairwise_haversine_any(&coords_flat);
///
/// // All should produce identical results
/// assert!((distances1[0] - distances2[0]).abs() < 1e-10);
/// assert!((distances1[0] - distances3[0]).abs() < 1e-10);
/// ```
///
/// # Supported Input Types
///
/// All types implementing [`CoordSource`]:
/// - `Vec<LngLat>` - Native coordinate type
/// - `Vec<(f64, f64)>` - Tuples with automatic format detection
/// - `Vec<f64>` - Flat arrays (pairs of coordinates)
/// - `&[f64]` - Array slices
/// - Custom types implementing the trait
///
/// # Format Detection
///
/// For ambiguous formats (tuples, flat arrays), automatic detection:
/// - Analyzes coordinate value ranges
/// - Determines lng,lat vs lat,lng ordering
/// - Applies 95% confidence threshold
/// - Falls back to lng,lat for ambiguous cases
///
/// # Performance
///
/// - **Time complexity**: O(min(n, 100)) for format detection + O(n) for calculation
/// - **Space complexity**: O(n) for result vector
/// - **Iterator overhead**: Uses boxed iterator for flexibility
///
/// For maximum performance with known formats, consider using format-specific functions directly.
///
/// # See Also
///
/// - [`pairwise_haversine_iter`] for iterator-based input
/// - [`CoordSource`] trait for supported input types  
/// - [`path_length_haversine_any`] for total path length
pub fn pairwise_haversine_any<T: CoordSource>(coords: &T) -> Vec<f64> {
    pairwise_haversine_iter(coords.get_coords())
}

/// Extends a vector with pairwise distances from any coordinate source.
///
/// Appends pairwise Haversine distances to an existing vector, enabling efficient
/// composition of results from multiple coordinate sources or reuse of allocated buffers.
///
/// # Arguments
///
/// * `coords` - Any coordinate source implementing [`CoordSource`]
/// * `output` - Vector to extend with computed distances
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::format_batch::sequential::pairwise_haversine_any_extend;
/// use rapidgeo_distance::LngLat;
///
/// // Start with some existing data
/// let mut distances = vec![999.0];
///
/// // Add distances from first route segment
/// let coords1 = vec![
///     LngLat::new_deg(0.0, 0.0),
///     LngLat::new_deg(1.0, 0.0),
/// ];
/// pairwise_haversine_any_extend(&coords1, &mut distances);
///
/// // Add distances from second route segment  
/// let coords2 = vec![
///     LngLat::new_deg(1.0, 0.0),
///     LngLat::new_deg(1.0, 1.0),
/// ];
/// pairwise_haversine_any_extend(&coords2, &mut distances);
///
/// assert_eq!(distances.len(), 3); // 1 original + 2 computed
/// assert_eq!(distances[0], 999.0); // Original data preserved
/// ```
///
/// # Composition Benefits
///
/// - **Multi-source**: Combine distances from different coordinate sources
/// - **Incremental**: Build results progressively
/// - **Buffer reuse**: Efficient with pre-allocated or pooled buffers
/// - **Memory efficiency**: Single allocation for all results
///
/// # Use Cases
///
/// - **Multi-segment routes**: Combine distances from route segments
/// - **Batch processing**: Accumulate results from multiple datasets
/// - **Buffer pools**: Efficiently reuse allocated memory
/// - **Streaming data**: Process coordinates as they arrive
///
/// # See Also
///
/// - [`pairwise_haversine_any`] for creating new result vectors
/// - [`pairwise_haversine_iter_extend`] for iterator input
/// - [`crate::buffer_pool::BufferPool`] for efficient buffer management
pub fn pairwise_haversine_any_extend<T: CoordSource>(coords: &T, output: &mut Vec<f64>) {
    pairwise_haversine_iter_extend(coords.get_coords(), output);
}

/// Computes pairwise distances from any coordinate source into a pre-allocated buffer.
///
/// Fills a fixed-size buffer with pairwise Haversine distances from any coordinate
/// source. This provides memory-efficient processing when the output size is known
/// in advance or when interfacing with external systems.
///
/// # Arguments
///
/// * `coords` - Any coordinate source implementing [`CoordSource`]
/// * `output` - Pre-allocated buffer to fill with distances
///
/// # Buffer Behavior
///
/// - **Exact fit**: Buffer sized exactly for expected output
/// - **Oversized**: Excess elements are zeroed for security
/// - **Undersized**: Computation stops when buffer is full
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::format_batch::sequential::pairwise_haversine_into_any;
/// use rapidgeo_distance::LngLat;
///
/// let coords = vec![
///     LngLat::new_deg(0.0, 0.0),
///     LngLat::new_deg(1.0, 0.0),
///     LngLat::new_deg(1.0, 1.0),
/// ];
///
/// // Exact-sized buffer
/// let mut buffer = [0.0; 2]; // Expect 2 distances for 3 points
/// pairwise_haversine_into_any(&coords, &mut buffer);
///
/// assert!(buffer[0] > 110_000.0); // ~111km for 1° longitude
/// assert!(buffer[1] > 110_000.0); // ~111km for 1° latitude
///
/// // Oversized buffer (excess elements zeroed)
/// let mut large_buffer = [f64::NAN; 5];
/// pairwise_haversine_into_any(&coords, &mut large_buffer);
///
/// assert!(!large_buffer[0].is_nan()); // Valid distance
/// assert!(!large_buffer[1].is_nan()); // Valid distance
/// assert_eq!(large_buffer[2], 0.0);   // Zeroed
/// assert_eq!(large_buffer[3], 0.0);   // Zeroed
/// assert_eq!(large_buffer[4], 0.0);   // Zeroed
/// ```
///
/// # Performance Benefits
///
/// - **No allocation**: Uses pre-allocated memory
/// - **Cache efficiency**: Better memory locality for fixed buffers
/// - **Predictable memory**: No dynamic allocation in processing loop
/// - **FFI friendly**: Compatible with C interfaces and external libraries
///
/// # Use Cases
///
/// - **Embedded systems**: Memory-constrained environments
/// - **Real-time processing**: Avoid allocation latency
/// - **Interfacing with C**: Fixed-size buffers for FFI
/// - **Batch processing**: Process fixed-size chunks efficiently
///
/// # Security
///
/// Unused buffer elements are explicitly zeroed to prevent information
/// leakage from previous buffer contents.
///
/// # See Also
///
/// - [`pairwise_haversine_any`] for dynamic vector allocation
/// - [`pairwise_haversine_iter_into`] for iterator input
/// - [`pairwise_haversine_any_extend`] for extending existing vectors
pub fn pairwise_haversine_into_any<T: CoordSource>(coords: &T, output: &mut [f64]) {
    pairwise_haversine_iter_into(coords.get_coords(), output);
}

/// Computes the total path length from any coordinate source.
///
/// Calculates the sum of all pairwise Haversine distances between consecutive
/// coordinates from any coordinate source. This provides a unified interface for
/// path length calculation regardless of input format.
///
/// # Arguments
///
/// * `coords` - Any coordinate source implementing [`CoordSource`]
///
/// # Returns
///
/// Total path length in meters
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::format_batch::sequential::path_length_haversine_any;
/// use rapidgeo_distance::LngLat;
///
/// // Works with different coordinate formats
/// let coords_lnglat = vec![
///     LngLat::new_deg(0.0, 0.0),
///     LngLat::new_deg(1.0, 0.0),
///     LngLat::new_deg(1.0, 1.0),
/// ];
/// let length1 = path_length_haversine_any(&coords_lnglat);
///
/// let coords_tuples = vec![
///     (0.0, 0.0),
///     (1.0, 0.0),
///     (1.0, 1.0),
/// ];
/// let length2 = path_length_haversine_any(&coords_tuples);
///
/// let coords_flat = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0];
/// let length3 = path_length_haversine_any(&coords_flat);
///
/// // All formats should produce identical results
/// assert!((length1 - length2).abs() < 1e-10);
/// assert!((length1 - length3).abs() < 1e-10);
/// ```
///
/// # Format Support
///
/// Supports all coordinate formats via the [`CoordSource`] trait:
/// - `Vec<LngLat>` - Native format (no conversion needed)
/// - `Vec<(f64, f64)>` - Tuples with automatic format detection
/// - `Vec<f64>` - Flat arrays chunked into coordinate pairs
/// - `&[f64]` - Array slices
/// - Custom coordinate types
///
/// # Format Detection
///
/// For ambiguous formats, uses automatic detection:
/// - Samples coordinate values to determine ordering
/// - Applies statistical analysis with confidence thresholds
/// - Falls back to lng,lat ordering for truly ambiguous cases
///
/// # Performance
///
/// - **Time complexity**: O(min(n, 100)) for detection + O(n) for calculation
/// - **Space complexity**: O(1) - only accumulates running total
/// - **Memory efficiency**: No intermediate storage for individual distances
///
/// # Use Cases
///
/// - **Route analysis**: Total distance for GPS tracks or planned routes
/// - **Performance metrics**: Compare path efficiency across different formats
/// - **Geometric calculations**: Perimeter or boundary length measurements
/// - **Data validation**: Verify coordinate data integrity through distance checks
///
/// # Consistency Guarantee
///
/// The path length equals the sum of all pairwise distances:
///
/// ```
/// use rapidgeo_distance::format_batch::sequential::{path_length_haversine_any, pairwise_haversine_any};
///
/// let coords = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)];
///
/// let total_length = path_length_haversine_any(&coords);
/// let distances = pairwise_haversine_any(&coords);
/// let sum_of_distances: f64 = distances.iter().sum();
///
/// assert!((total_length - sum_of_distances).abs() < 1e-10);
/// ```
///
/// # See Also
///
/// - [`path_length_haversine_iter`] for iterator input
/// - [`pairwise_haversine_any`] for individual segment distances
/// - [`CoordSource`] trait for supported input types
pub fn path_length_haversine_any<T: CoordSource>(coords: &T) -> f64 {
    path_length_haversine_iter(coords.get_coords())
}
