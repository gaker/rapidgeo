use crate::formats::CoordSource;
use crate::LngLat;

#[cfg(feature = "batch")]
use super::parallel;
use super::sequential::{pairwise_haversine_any_extend, pairwise_haversine_iter_extend};

/// Buffer pool for coordinate processing operations.
///
/// Manages a pool of reusable `Vec<f64>` buffers to minimize memory allocations
/// during repeated coordinate calculations. This is particularly beneficial for
/// iterative processing of large datasets or real-time applications.
///
/// # Performance Benefits
///
/// - **Reduced allocations**: Reuses buffers instead of allocating new ones
/// - **Memory locality**: Keeps buffer capacity to avoid repeated growth
/// - **Pool management**: Limits pool size to prevent unbounded memory growth
/// - **RAII safety**: Automatic buffer return via scoped operations
///
/// # Usage Patterns
///
/// The pool supports two usage patterns:
/// 1. **Manual management**: `get_buffer()` and `return_buffer()`
/// 2. **Scoped operations**: `with_buffer()` for automatic lifecycle management
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::format_batch::buffer_pool::BufferPool;
/// use rapidgeo_distance::LngLat;
///
/// // Create pool with initial buffer capacity of 1000 elements
/// let mut pool = BufferPool::new(1000);
///
/// // Scoped operation (recommended)
/// let result = pool.with_buffer(|buffer| {
///     // Use buffer for calculations
///     buffer.extend([1.0, 2.0, 3.0]);
///     buffer.len()
/// }); // Buffer automatically returned to pool
///
/// assert_eq!(result, 3);
/// assert_eq!(pool.pool_size(), 1); // Buffer was returned
/// ```
///
/// # Memory Management
///
/// - Buffers are cleared (length set to 0) when returned, but capacity is preserved
/// - Pool size is capped to prevent unbounded growth
/// - Dropped buffers are not returned to the pool once capacity is reached
///
/// # Thread Safety
///
/// This pool is **not** thread-safe. Use separate pools per thread or add
/// synchronization for concurrent access.
///
/// # See Also
///
/// - [`with_buffer`](BufferPool::with_buffer) for scoped buffer operations
/// - [`pairwise_haversine_any`](BufferPool::pairwise_haversine_any) for coordinate-specific operations
pub struct BufferPool {
    buffers: Vec<Vec<f64>>,
    initial_capacity: usize,
    max_pool_size: usize,
}

impl BufferPool {
    /// Creates a new buffer pool with the specified initial buffer capacity.
    ///
    /// # Arguments
    ///
    /// * `initial_capacity` - The initial capacity (in elements) for new buffers
    ///
    /// # Examples
    ///
    /// ```
    /// use rapidgeo_distance::format_batch::buffer_pool::BufferPool;
    ///
    /// // Pool for processing up to 1000 coordinate pairs
    /// let pool = BufferPool::new(1000);
    /// assert_eq!(pool.pool_size(), 0); // No buffers initially
    /// ```
    ///
    /// # Default Settings
    ///
    /// - **Maximum pool size**: 8 buffers
    /// - **Initial pool size**: 0 buffers (created on demand)
    pub fn new(initial_capacity: usize) -> Self {
        Self {
            buffers: Vec::new(),
            initial_capacity,
            max_pool_size: 8,
        }
    }

    /// Creates a new buffer pool with custom capacity and pool size limits.
    ///
    /// # Arguments
    ///
    /// * `initial_capacity` - The initial capacity (in elements) for new buffers
    /// * `max_pool_size` - Maximum number of buffers to keep in the pool
    ///
    /// # Examples
    ///
    /// ```
    /// use rapidgeo_distance::format_batch::buffer_pool::BufferPool;
    ///
    /// // Pool for memory-constrained environments
    /// let pool = BufferPool::with_max_size(500, 4);
    /// assert_eq!(pool.pool_size(), 0);
    /// ```
    ///
    /// # Pool Size Considerations
    ///
    /// - **Small pools (1-4)**: Lower memory usage, more allocations
    /// - **Large pools (8-16)**: Higher memory usage, fewer allocations
    /// - **Very large pools (>16)**: Diminishing returns, potential memory waste
    pub fn with_max_size(initial_capacity: usize, max_pool_size: usize) -> Self {
        Self {
            buffers: Vec::new(),
            initial_capacity,
            max_pool_size,
        }
    }

    /// Gets a buffer from the pool, creating a new one if the pool is empty.
    ///
    /// The returned buffer is empty (length 0) but may have existing capacity
    /// from previous use. You must call [`return_buffer`](Self::return_buffer)
    /// when finished to return it to the pool.
    ///
    /// # Returns
    ///
    /// An empty `Vec<f64>` ready for use
    ///
    /// # Examples
    ///
    /// ```
    /// use rapidgeo_distance::format_batch::buffer_pool::BufferPool;
    ///
    /// let mut pool = BufferPool::new(100);
    ///
    /// let mut buffer = pool.get_buffer();
    /// assert_eq!(buffer.len(), 0);
    /// assert!(buffer.capacity() >= 100);
    ///
    /// buffer.push(42.0);
    /// pool.return_buffer(buffer);
    /// ```
    ///
    /// # Performance Notes
    ///
    /// - Reused buffers retain their capacity from previous use
    /// - New buffers are allocated with the pool's initial capacity
    /// - Consider using [`with_buffer`](Self::with_buffer) for automatic management
    pub fn get_buffer(&mut self) -> Vec<f64> {
        self.buffers
            .pop()
            .unwrap_or_else(|| Vec::with_capacity(self.initial_capacity))
    }

    /// Returns a buffer to the pool for reuse.
    ///
    /// The buffer is cleared (length set to 0) but capacity is preserved.
    /// If the pool is full, the buffer is dropped instead of being stored.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The buffer to return (will be cleared)
    ///
    /// # Examples
    ///
    /// ```
    /// use rapidgeo_distance::format_batch::buffer_pool::BufferPool;
    ///
    /// let mut pool = BufferPool::new(50);
    ///
    /// let mut buffer = pool.get_buffer();
    /// buffer.extend([1.0, 2.0, 3.0]);
    ///
    /// pool.return_buffer(buffer);
    /// assert_eq!(pool.pool_size(), 1);
    ///
    /// // Buffer is cleared but capacity preserved
    /// let buffer2 = pool.get_buffer();
    /// assert_eq!(buffer2.len(), 0);
    /// ```
    ///
    /// # Pool Capacity
    ///
    /// Buffers are only stored if there's room in the pool:
    ///
    /// ```
    /// use rapidgeo_distance::format_batch::buffer_pool::BufferPool;
    ///
    /// let mut pool = BufferPool::with_max_size(50, 2); // Max 2 buffers
    ///
    /// let buf1 = pool.get_buffer();
    /// let buf2 = pool.get_buffer();
    /// pool.return_buffer(buf1);
    /// pool.return_buffer(buf2);
    /// assert_eq!(pool.pool_size(), 2);
    ///
    /// // Third buffer is dropped, not stored
    /// let buf3 = pool.get_buffer();
    /// pool.return_buffer(buf3);
    /// assert_eq!(pool.pool_size(), 2); // Still 2
    /// ```
    pub fn return_buffer(&mut self, mut buffer: Vec<f64>) {
        if self.buffers.len() < self.max_pool_size {
            buffer.clear();
            self.buffers.push(buffer);
        }
    }

    /// Executes a closure with a temporary buffer, automatically managing its lifecycle.
    ///
    /// This is the recommended way to use the buffer pool as it ensures the buffer
    /// is always returned, even if the closure panics or returns early.
    ///
    /// # Arguments
    ///
    /// * `f` - Closure that receives a mutable buffer reference
    ///
    /// # Returns
    ///
    /// The result of the closure
    ///
    /// # Examples
    ///
    /// ```
    /// use rapidgeo_distance::format_batch::buffer_pool::BufferPool;
    ///
    /// let mut pool = BufferPool::new(100);
    ///
    /// let sum = pool.with_buffer(|buffer| {
    ///     buffer.extend([1.0, 2.0, 3.0, 4.0, 5.0]);
    ///     buffer.iter().sum::<f64>()
    /// });
    ///
    /// assert_eq!(sum, 15.0);
    /// assert_eq!(pool.pool_size(), 1); // Buffer was returned
    /// ```
    ///
    /// # Error Safety
    ///
    /// The buffer is returned to the pool even if the closure panics:
    ///
    /// ```should_panic
    /// use rapidgeo_distance::format_batch::buffer_pool::BufferPool;
    ///
    /// let mut pool = BufferPool::new(100);
    ///
    /// pool.with_buffer(|_buffer| {
    ///     panic!("Something went wrong!");
    /// });
    /// ```
    ///
    /// # Performance Benefits
    ///
    /// - **No manual tracking**: Impossible to forget buffer return
    /// - **Exception safety**: Buffer returned even on panic
    /// - **Zero overhead**: Inlined closure execution
    pub fn with_buffer<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut Vec<f64>) -> R,
    {
        let mut buffer = self.get_buffer();
        let result = f(&mut buffer);
        self.return_buffer(buffer);
        result
    }

    /// Computes pairwise Haversine distances using a pooled buffer.
    ///
    /// Calculates the distance between consecutive coordinate pairs using the
    /// Haversine formula. The result buffer is obtained from the pool but
    /// **not** returned automatically - you own the returned vector.
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
    /// use rapidgeo_distance::format_batch::buffer_pool::BufferPool;
    /// use rapidgeo_distance::LngLat;
    ///
    /// let mut pool = BufferPool::new(100);
    ///
    /// let coords = [
    ///     LngLat::new_deg(-122.4194, 37.7749), // San Francisco
    ///     LngLat::new_deg(-74.0060, 40.7128),  // New York
    ///     LngLat::new_deg(-87.6298, 41.8781),  // Chicago
    /// ];
    ///
    /// let distances = pool.pairwise_haversine_iter(coords.iter().copied());
    /// assert_eq!(distances.len(), 2); // n-1 distances for n points
    ///
    /// // SF to NYC is approximately 4100km
    /// assert!(distances[0] > 4_000_000.0 && distances[0] < 4_200_000.0);
    /// ```
    ///
    /// # Performance
    ///
    /// - **Buffer reuse**: Uses pooled buffer for intermediate calculations
    /// - **Single allocation**: Result vector allocated once with appropriate capacity
    /// - **Lazy evaluation**: Iterator is consumed on-demand
    ///
    /// # See Also
    ///
    /// - [`pairwise_haversine_any`](Self::pairwise_haversine_any) for `CoordSource` input
    /// - [`pairwise_haversine_iter`](super::sequential::pairwise_haversine_iter) for non-pooled version
    pub fn pairwise_haversine_iter<I>(&mut self, iter: I) -> Vec<f64>
    where
        I: Iterator<Item = LngLat>,
    {
        let mut result = self.get_buffer();
        pairwise_haversine_iter_extend(iter, &mut result);
        result
    }

    /// Computes pairwise Haversine distances from any coordinate source using a pooled buffer.
    ///
    /// Accepts any type implementing [`CoordSource`] (tuples, arrays, etc.) and computes
    /// distances between consecutive coordinates. Automatically handles format detection
    /// and conversion as needed.
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
    /// use rapidgeo_distance::format_batch::buffer_pool::BufferPool;
    /// use rapidgeo_distance::LngLat;
    ///
    /// let mut pool = BufferPool::new(100);
    ///
    /// // Works with various coordinate formats
    /// let coords_lnglat = vec![
    ///     LngLat::new_deg(-122.4194, 37.7749),
    ///     LngLat::new_deg(-74.0060, 40.7128),
    /// ];
    /// let distances1 = pool.pairwise_haversine_any(&coords_lnglat);
    ///
    /// let coords_tuples = vec![
    ///     (-122.4194, 37.7749),
    ///     (-74.0060, 40.7128),
    /// ];
    /// let distances2 = pool.pairwise_haversine_any(&coords_tuples);
    ///
    /// // Results should be identical
    /// assert!((distances1[0] - distances2[0]).abs() < 1e-10);
    /// ```
    ///
    /// # Format Support
    ///
    /// Supports all coordinate formats:
    /// - `Vec<LngLat>` - Native format
    /// - `Vec<(f64, f64)>` - Tuples with format detection
    /// - `Vec<f64>` - Flat arrays (chunked into pairs)
    /// - `&[f64]` - Array slices
    ///
    /// # See Also
    ///
    /// - [`pairwise_haversine_iter`](Self::pairwise_haversine_iter) for iterator input
    /// - [`CoordSource`] trait for supported input types
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

    /// Returns the number of buffers currently stored in the pool.
    ///
    /// This count represents available buffers ready for reuse. It will be
    /// between 0 and the maximum pool size configured during construction.
    ///
    /// # Examples
    ///
    /// ```
    /// use rapidgeo_distance::format_batch::buffer_pool::BufferPool;
    ///
    /// let mut pool = BufferPool::new(100);
    /// assert_eq!(pool.pool_size(), 0); // Initially empty
    ///
    /// let buffer = pool.get_buffer();
    /// assert_eq!(pool.pool_size(), 0); // Buffer checked out
    ///
    /// pool.return_buffer(buffer);
    /// assert_eq!(pool.pool_size(), 1); // Buffer returned
    /// ```
    ///
    /// # Use Cases
    ///
    /// - **Debugging**: Verify buffers are being returned properly
    /// - **Monitoring**: Track pool utilization in long-running applications
    /// - **Testing**: Ensure proper resource management in tests
    pub fn pool_size(&self) -> usize {
        self.buffers.len()
    }

    /// Removes all buffers from the pool, freeing their memory.
    ///
    /// This is useful for releasing memory when the pool won't be used for
    /// an extended period, or for cleanup in tests and benchmarks.
    ///
    /// # Examples
    ///
    /// ```
    /// use rapidgeo_distance::format_batch::buffer_pool::BufferPool;
    ///
    /// let mut pool = BufferPool::new(100);
    ///
    /// // Use some buffers
    /// let buf1 = pool.get_buffer();
    /// let buf2 = pool.get_buffer();
    /// pool.return_buffer(buf1);
    /// pool.return_buffer(buf2);
    /// assert_eq!(pool.pool_size(), 2);
    ///
    /// // Clear all buffers
    /// pool.clear_pool();
    /// assert_eq!(pool.pool_size(), 0);
    /// ```
    ///
    /// # Memory Impact
    ///
    /// After clearing, subsequent `get_buffer()` calls will allocate new buffers
    /// with the pool's configured initial capacity. This may cause temporary
    /// performance degradation until the pool is rebuilt.
    ///
    /// # Use Cases
    ///
    /// - **Memory pressure**: Free memory when pool is idle
    /// - **Test cleanup**: Reset pool state between test cases
    /// - **Capacity changes**: Clear before changing buffer sizing strategy
    pub fn clear_pool(&mut self) {
        self.buffers.clear();
    }
}
