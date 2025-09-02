use crate::{detection::detect_format, LngLat};

/// Coordinate format hint indicating the ordering of coordinate pairs.
///
/// Used by format detection algorithms to determine whether coordinate pairs
/// follow lng,lat or lat,lng ordering. This enables automatic correction of
/// coordinate data from various sources.
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::formats::FormatHint;
///
/// // Check format equality
/// assert_eq!(FormatHint::LngLat, FormatHint::LngLat);
/// assert_ne!(FormatHint::LngLat, FormatHint::LatLng);
///
/// // Debug formatting
/// println!("Detected format: {:?}", FormatHint::LngLat);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FormatHint {
    /// Longitude-first format: (longitude, latitude)
    ///
    /// This is the standard GIS format where the first value is longitude (-180 to +180)
    /// and the second value is latitude (-90 to +90).
    LngLat,

    /// Latitude-first format: (latitude, longitude)
    ///
    /// This is common in some APIs and user interfaces where latitude comes first.
    /// The first value is latitude (-90 to +90) and the second is longitude (-180 to +180).
    LatLng,

    /// Format cannot be determined with confidence
    ///
    /// Either the data is ambiguous (all values valid for both formats),
    /// invalid (out of coordinate bounds), or empty.
    Unknown,
}

/// Trait for types that can provide coordinate data in a uniform way.
///
/// This trait abstracts over different coordinate storage formats, allowing functions
/// to accept coordinates from various sources (tuples, arrays, custom types) without
/// needing separate implementations for each format.
///
/// The trait is thread-safe (`Send + Sync`) to support parallel processing.
///
/// # Performance Considerations
///
/// - `get_coords()` returns a boxed iterator for flexibility but adds allocation overhead
/// - `hint_size()` helps pre-allocate output buffers for better performance  
/// - `hint_format()` enables format-specific optimizations when possible
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::formats::{CoordSource, FormatHint};
/// use rapidgeo_distance::LngLat;
///
/// let coords = vec![
///     LngLat::new_deg(-122.4194, 37.7749),
///     LngLat::new_deg(-74.0060, 40.7128),
/// ];
///
/// // Access coordinate data uniformly
/// let mut iter = coords.get_coords();
/// assert_eq!(iter.next(), Some(LngLat::new_deg(-122.4194, 37.7749)));
///
/// // Get size hint for buffer allocation
/// assert_eq!(coords.hint_size(), Some(2));
///
/// // Get format information
/// assert_eq!(coords.hint_format(), FormatHint::LngLat);
/// ```
///
/// # Thread Safety
///
/// All implementations must be `Send + Sync` to support parallel processing:
///
/// ```
/// use rapidgeo_distance::formats::CoordSource;
///
/// fn process_coords_parallel<T: CoordSource>(coords: &T) {
///     // Can safely use in parallel contexts
/// }
/// ```
pub trait CoordSource: Send + Sync {
    /// Returns an iterator over the coordinates as `LngLat` values.
    ///
    /// The iterator performs any necessary format conversion (e.g., lat,lng → lng,lat)
    /// and coordinate validation. Invalid coordinates may be skipped or cause panics
    /// depending on the implementation.
    ///
    /// # Performance
    ///
    /// Returns a boxed iterator for flexibility, but this adds heap allocation overhead.
    /// For performance-critical code, consider using format-specific functions directly.
    fn get_coords(&self) -> Box<dyn Iterator<Item = LngLat> + '_>;

    /// Returns an estimate of the number of coordinates, if known.
    ///
    /// Used for buffer pre-allocation to improve performance. Returns `None` if the
    /// size cannot be determined efficiently (e.g., for lazy iterators).
    ///
    /// # Examples
    ///
    /// ```
    /// use rapidgeo_distance::formats::CoordSource;
    ///
    /// let coords = vec![(1.0, 2.0), (3.0, 4.0)];
    /// assert_eq!(coords.hint_size(), Some(2));
    ///
    /// let empty: Vec<(f64, f64)> = vec![];
    /// assert_eq!(empty.hint_size(), Some(0));
    /// ```
    fn hint_size(&self) -> Option<usize>;

    /// Returns a hint about the coordinate format, if detectable.
    ///
    /// This enables format-specific optimizations and helps with automatic
    /// format detection. Returns `FormatHint::Unknown` for ambiguous data.
    ///
    /// # Examples
    ///
    /// ```
    /// use rapidgeo_distance::formats::{CoordSource, FormatHint};
    /// use rapidgeo_distance::LngLat;
    ///
    /// // LngLat vectors have known format
    /// let coords = vec![LngLat::new_deg(0.0, 0.0)];
    /// assert_eq!(coords.hint_format(), FormatHint::LngLat);
    ///
    /// // Raw tuples require detection
    /// let tuples = vec![(0.0, 0.0)];
    /// assert_eq!(tuples.hint_format(), FormatHint::Unknown);
    /// ```
    fn hint_format(&self) -> FormatHint;
}

impl CoordSource for Vec<LngLat> {
    fn get_coords(&self) -> Box<dyn Iterator<Item = LngLat> + '_> {
        Box::new(self.iter().copied())
    }

    fn hint_size(&self) -> Option<usize> {
        Some(self.len())
    }

    fn hint_format(&self) -> FormatHint {
        FormatHint::LngLat
    }
}

impl CoordSource for Vec<(f64, f64)> {
    fn get_coords(&self) -> Box<dyn Iterator<Item = LngLat> + '_> {
        Box::new(self.iter().map(|&(lng, lat)| LngLat::new_deg(lng, lat)))
    }

    fn hint_size(&self) -> Option<usize> {
        Some(self.len())
    }

    fn hint_format(&self) -> FormatHint {
        FormatHint::Unknown
    }
}

impl CoordSource for [f64] {
    fn get_coords(&self) -> Box<dyn Iterator<Item = LngLat> + '_> {
        Box::new(
            self.chunks_exact(2)
                .map(|chunk| LngLat::new_deg(chunk[0], chunk[1])),
        )
    }

    fn hint_size(&self) -> Option<usize> {
        Some(self.len() / 2)
    }

    fn hint_format(&self) -> FormatHint {
        FormatHint::Unknown
    }
}

impl CoordSource for Vec<f64> {
    fn get_coords(&self) -> Box<dyn Iterator<Item = LngLat> + '_> {
        Box::new(
            self.chunks_exact(2)
                .map(|chunk| LngLat::new_deg(chunk[0], chunk[1])),
        )
    }

    fn hint_size(&self) -> Option<usize> {
        Some(self.len() / 2)
    }

    fn hint_format(&self) -> FormatHint {
        FormatHint::Unknown
    }
}

/// GeoJSON-style point geometry with [longitude, latitude] coordinates.
///
/// Represents a point in GeoJSON format where coordinates are stored as
/// `[longitude, latitude]` in decimal degrees. This matches the GeoJSON
/// specification and GIS conventions.
///
/// # GeoJSON Standard
///
/// According to [RFC 7946](https://tools.ietf.org/html/rfc7946), GeoJSON
/// coordinates are always in `[longitude, latitude]` order, regardless of
/// local conventions that might prefer latitude first.
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::formats::GeoPoint;
///
/// // San Francisco in GeoJSON format
/// let sf = GeoPoint {
///     coordinates: [-122.4194, 37.7749], // [lng, lat]
/// };
///
/// assert_eq!(sf.coordinates[0], -122.4194); // longitude
/// assert_eq!(sf.coordinates[1], 37.7749);   // latitude
/// ```
///
/// # Coordinate Bounds
///
/// - `coordinates[0]`: Longitude in range [-180.0, +180.0]
/// - `coordinates[1]`: Latitude in range [-90.0, +90.0]
///
/// # See Also
///
/// - [GeoJSON Specification](https://geojson.org/)
/// - [`LngLat`](crate::LngLat) for the native coordinate type
#[derive(Debug, Clone)]
pub struct GeoPoint {
    /// Coordinates array in [longitude, latitude] order (GeoJSON standard)
    pub coordinates: [f64; 2],
}

impl CoordSource for Vec<GeoPoint> {
    fn get_coords(&self) -> Box<dyn Iterator<Item = LngLat> + '_> {
        Box::new(
            self.iter()
                .map(|point| LngLat::new_deg(point.coordinates[0], point.coordinates[1])),
        )
    }

    fn hint_size(&self) -> Option<usize> {
        Some(self.len())
    }

    fn hint_format(&self) -> FormatHint {
        FormatHint::LngLat
    }
}

/// Converts various coordinate formats to a standardized `Vec<LngLat>` with automatic format detection.
///
/// This is the main entry point for coordinate format conversion. It handles multiple
/// input formats and automatically detects coordinate ordering (lng,lat vs lat,lng)
/// when the format is ambiguous.
///
/// # Supported Input Formats
///
/// - **Tuples**: `Vec<(f64, f64)>` with automatic lng,lat vs lat,lng detection
/// - **Flat arrays**: `Vec<f64>` chunked into coordinate pairs
/// - **GeoJSON**: `Vec<GeoPoint>` following GeoJSON [lng, lat] specification
/// - **Already converted**: `Vec<LngLat>` (returned as clone)
///
/// # Format Detection Logic
///
/// For ambiguous formats (tuples, flat arrays), the function:
/// 1. Samples up to 100 coordinate pairs for performance
/// 2. Tests which interpretation (lng,lat or lat,lng) has more valid coordinates
/// 3. Applies 95% confidence threshold for early termination
/// 4. Falls back to lng,lat ordering for truly ambiguous data
///
/// # Performance
///
/// - **Time complexity**: O(min(n, 100)) for format detection + O(n) for conversion
/// - **Space complexity**: O(n) for output vector
/// - **Early termination**: Typically processes 10-20 samples for clear datasets
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::formats::{coords_to_lnglat_vec, CoordinateInput};
/// use rapidgeo_distance::LngLat;
///
/// // Automatic lng,lat detection (US cities)
/// let us_cities = vec![
///     (-122.4194, 37.7749), // San Francisco
///     (-74.0060, 40.7128),  // New York
/// ];
/// let input = CoordinateInput::Tuples(us_cities);
/// let coords = coords_to_lnglat_vec(&input);
/// assert_eq!(coords[0], LngLat::new_deg(-122.4194, 37.7749));
///
/// // Automatic lat,lng detection (same cities, swapped)
/// let swapped_cities = vec![
///     (37.7749, -122.4194), // San Francisco (lat, lng)
///     (40.7128, -74.0060),  // New York (lat, lng)
/// ];
/// let input = CoordinateInput::Tuples(swapped_cities);
/// let coords = coords_to_lnglat_vec(&input);
/// // Automatically corrected to lng,lat order
/// assert_eq!(coords[0], LngLat::new_deg(-122.4194, 37.7749));
///
/// // Flat array format
/// let flat = vec![-122.4194, 37.7749, -74.0060, 40.7128];
/// let input = CoordinateInput::FlatArray(flat);
/// let coords = coords_to_lnglat_vec(&input);
/// assert_eq!(coords.len(), 2);
/// ```
///
/// # Error Handling
///
/// - **Empty input**: Returns empty vector
/// - **Odd-length arrays**: Incomplete pairs are ignored
/// - **Invalid coordinates**: Out-of-bounds values affect format detection but are preserved
/// - **Ambiguous format**: Defaults to lng,lat ordering
///
/// # See Also
///
/// - [`detect_format`](crate::detection::detect_format) for format detection details
/// - [`CoordinateInput`] for input format options
/// - [`tuples_to_lnglat_vec_auto`] for tuple-specific conversion
pub fn coords_to_lnglat_vec(input: &CoordinateInput) -> Vec<LngLat> {
    match input {
        CoordinateInput::Tuples(tuples) => tuples_to_lnglat_vec_auto(tuples),
        CoordinateInput::FlatArray(arr) => flat_array_to_lnglat_vec_auto(arr),
        CoordinateInput::GeoJson(points) => points
            .iter()
            .map(|p| LngLat::new_deg(p.coordinates[0], p.coordinates[1]))
            .collect(),
        CoordinateInput::Already(coords) => coords.clone(),
    }
}

/// Converts coordinate tuples to `Vec<LngLat>` with automatic format detection.
///
/// Detects whether the input tuples follow lng,lat or lat,lng ordering and converts
/// them to the standard lng,lat format. Uses statistical analysis of coordinate
/// bounds to determine the most likely format.
///
/// # Format Detection Process
///
/// 1. **Empty check**: Returns empty vector immediately
/// 2. **Format detection**: Uses [`detect_format`](crate::detection::detect_format) to analyze coordinate bounds
/// 3. **Conversion**: Applies appropriate coordinate swapping based on detected format
/// 4. **Fallback**: Assumes lng,lat for ambiguous cases
///
/// # Performance
///
/// - **Time complexity**: O(min(n, 100)) for detection + O(n) for conversion
/// - **Space complexity**: O(n) for output vector with pre-allocation
/// - **Early termination**: Detection stops early when confidence threshold is reached
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::formats::tuples_to_lnglat_vec_auto;
/// use rapidgeo_distance::LngLat;
///
/// // Clear lng,lat format (US cities with distinctive longitudes)
/// let lng_lat_data = vec![
///     (-122.4194, 37.7749), // San Francisco
///     (-74.0060, 40.7128),  // New York
/// ];
/// let coords = tuples_to_lnglat_vec_auto(&lng_lat_data);
/// assert_eq!(coords[0], LngLat::new_deg(-122.4194, 37.7749));
///
/// // Clear lat,lng format (same cities, swapped)
/// let lat_lng_data = vec![
///     (37.7749, -122.4194), // San Francisco (lat, lng)
///     (40.7128, -74.0060),  // New York (lat, lng)
/// ];
/// let coords = tuples_to_lnglat_vec_auto(&lat_lng_data);
/// // Automatically detects and swaps to lng,lat
/// assert_eq!(coords[0], LngLat::new_deg(-122.4194, 37.7749));
///
/// // Empty input
/// let empty: Vec<(f64, f64)> = vec![];
/// let coords = tuples_to_lnglat_vec_auto(&empty);
/// assert_eq!(coords.len(), 0);
/// ```
///
/// # Detection Accuracy
///
/// The detection algorithm is highly accurate for:
/// - **Clear cases**: Coordinates with values outside ±90° range
/// - **Large datasets**: More samples improve confidence
/// - **Geographically diverse data**: Mixed coordinate ranges
///
/// Less reliable for:
/// - **Equatorial regions**: Values within ±90° for both coordinates
/// - **Small datasets**: Less statistical significance
/// - **Single coordinates**: No redundancy for validation
///
/// # See Also
///
/// - [`detect_format`](crate::detection::detect_format) for detection algorithm details
/// - [`coords_to_lnglat_vec`] for multi-format conversion
/// - [`flat_array_to_lnglat_vec_auto`] for flat array conversion
pub fn tuples_to_lnglat_vec_auto(tuples: &[(f64, f64)]) -> Vec<LngLat> {
    if tuples.is_empty() {
        return Vec::new();
    }

    let format = detect_format(tuples);
    let mut result = Vec::with_capacity(tuples.len());

    match format {
        FormatHint::LngLat => {
            result.extend(tuples.iter().map(|&(lng, lat)| LngLat::new_deg(lng, lat)));
        }
        FormatHint::LatLng => {
            result.extend(tuples.iter().map(|&(lat, lng)| LngLat::new_deg(lng, lat)));
        }
        FormatHint::Unknown => {
            result.extend(
                tuples
                    .iter()
                    .map(|&(first, second)| LngLat::new_deg(first, second)),
            );
        }
    }

    result
}

/// Converts flat coordinate arrays to `Vec<LngLat>` with automatic format detection.
///
/// Takes a flat array of coordinates in the format `[x1, y1, x2, y2, ...]` and converts
/// them to `LngLat` coordinates. Automatically detects whether the pairs represent
/// lng,lat or lat,lng ordering.
///
/// # Array Format
///
/// The input array is interpreted as consecutive coordinate pairs:
/// - `[lng1, lat1, lng2, lat2, ...]` or
/// - `[lat1, lng1, lat2, lng2, ...]` (detected automatically)
///
/// # Processing Steps
///
/// 1. **Length validation**: Arrays with less than 2 elements return empty vector
/// 2. **Pair extraction**: Chunks array into `(f64, f64)` pairs using `chunks_exact(2)`
/// 3. **Format detection**: Analyzes pairs to determine coordinate ordering
/// 4. **Conversion**: Applies appropriate coordinate swapping
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::formats::flat_array_to_lnglat_vec_auto;
/// use rapidgeo_distance::LngLat;
///
/// // Lng,lat format
/// let lng_lat_array = vec![
///     -122.4194, 37.7749,  // San Francisco
///     -74.0060, 40.7128,   // New York
/// ];
/// let coords = flat_array_to_lnglat_vec_auto(&lng_lat_array);
/// assert_eq!(coords.len(), 2);
/// assert_eq!(coords[0], LngLat::new_deg(-122.4194, 37.7749));
///
/// // Lat,lng format (automatically detected and corrected)
/// let lat_lng_array = vec![
///     37.7749, -122.4194,  // San Francisco (lat, lng)
///     40.7128, -74.0060,   // New York (lat, lng)
/// ];
/// let coords = flat_array_to_lnglat_vec_auto(&lat_lng_array);
/// assert_eq!(coords[0], LngLat::new_deg(-122.4194, 37.7749));
///
/// // Odd-length array (last element ignored)
/// let odd_array = vec![-122.4194, 37.7749, -74.0060];
/// let coords = flat_array_to_lnglat_vec_auto(&odd_array);
/// assert_eq!(coords.len(), 1); // Only complete pairs are processed
/// ```
///
/// # Edge Cases
///
/// - **Empty arrays**: Return empty vector
/// - **Single element**: Return empty vector (incomplete pair)
/// - **Odd length**: Incomplete final pair is ignored
/// - **Invalid coordinates**: Preserved in output, affect format detection
///
/// # Performance
///
/// - **Time complexity**: O(n) for chunking + O(min(n/2, 100)) for detection
/// - **Space complexity**: O(n) for coordinate pairs + output vector
/// - **Memory efficiency**: Single allocation for output vector
///
/// # See Also
///
/// - [`tuples_to_lnglat_vec_auto`] for tuple-based conversion
/// - [`coords_to_lnglat_vec`] for multi-format conversion
/// - [`detect_format`](crate::detection::detect_format) for detection details
pub fn flat_array_to_lnglat_vec_auto(arr: &[f64]) -> Vec<LngLat> {
    if arr.len() < 2 {
        return Vec::new();
    }

    let pairs: Vec<(f64, f64)> = arr
        .chunks_exact(2)
        .map(|chunk| (chunk[0], chunk[1]))
        .collect();

    tuples_to_lnglat_vec_auto(&pairs)
}

/// Enumeration of supported coordinate input formats for batch conversion.
///
/// This enum unifies different coordinate representations to enable consistent
/// processing through a single interface. Each variant handles a specific
/// coordinate format with appropriate conversion logic.
///
/// # Design Rationale
///
/// Different systems and APIs represent coordinates in various formats:
/// - **GIS systems**: Often use lng,lat ordering
/// - **Web APIs**: May use lat,lng for user familiarity  
/// - **Databases**: Might store as flat arrays for efficiency
/// - **GeoJSON**: Standardized as [lng, lat] arrays
///
/// This enum allows handling all formats uniformly while preserving format-specific
/// optimizations.
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::formats::{CoordinateInput, GeoPoint, coords_to_lnglat_vec};
/// use rapidgeo_distance::LngLat;
///
/// // From tuple pairs
/// let tuples = vec![(-122.4194, 37.7749), (-74.0060, 40.7128)];
/// let input = CoordinateInput::Tuples(tuples);
/// let coords = coords_to_lnglat_vec(&input);
///
/// // From flat array
/// let flat = vec![-122.4194, 37.7749, -74.0060, 40.7128];
/// let input = CoordinateInput::FlatArray(flat);
/// let coords = coords_to_lnglat_vec(&input);
///
/// // From GeoJSON-style points
/// let geojson = vec![
///     GeoPoint { coordinates: [-122.4194, 37.7749] },
///     GeoPoint { coordinates: [-74.0060, 40.7128] },
/// ];
/// let input = CoordinateInput::GeoJson(geojson);
/// let coords = coords_to_lnglat_vec(&input);
/// ```
///
/// # Conversion Trait Implementations
///
/// All variants implement `From<T>` for convenient conversion:
///
/// ```
/// use rapidgeo_distance::formats::CoordinateInput;
///
/// let tuples = vec![(1.0, 2.0), (3.0, 4.0)];
/// let input: CoordinateInput = tuples.into();
/// ```
///
/// # See Also
///
/// - [`coords_to_lnglat_vec`] for format conversion
/// - [`GeoPoint`] for GeoJSON-style coordinates
/// - [`LngLat`](crate::LngLat) for the target coordinate type
#[derive(Debug)]
pub enum CoordinateInput {
    /// Coordinate pairs as tuples with automatic format detection.
    ///
    /// Each tuple represents a coordinate pair that could be either (lng, lat)
    /// or (lat, lng). The format is automatically detected by analyzing the
    /// value ranges and determining which interpretation has more valid coordinates.
    ///
    /// Format detection considers:
    /// - Longitude values must be in range [-180, +180]
    /// - Latitude values must be in range [-90, +90]
    /// - Statistical confidence based on sample size
    Tuples(Vec<(f64, f64)>),

    /// Flat array of coordinates as `[x1, y1, x2, y2, ...]`.
    ///
    /// The array is chunked into coordinate pairs using `chunks_exact(2)`,
    /// then format detection is applied to determine if pairs represent
    /// lng,lat or lat,lng ordering. Odd-length arrays ignore the final element.
    ///
    /// This format is common in:
    /// - Database storage (efficient packing)
    /// - Graphics APIs (vertex arrays)
    /// - Scientific computing (NumPy arrays)
    FlatArray(Vec<f64>),

    /// GeoJSON-style point geometries with [longitude, latitude] coordinates.
    ///
    /// Following the GeoJSON specification (RFC 7946), coordinates are always
    /// stored as [longitude, latitude] regardless of local conventions.
    /// No format detection is needed as the format is standardized.
    ///
    /// This format is used by:
    /// - GeoJSON files and APIs
    /// - PostGIS and other spatial databases
    /// - Web mapping libraries (Leaflet, OpenLayers)
    GeoJson(Vec<GeoPoint>),

    /// Coordinates already in the target `LngLat` format.
    ///
    /// No conversion is needed - the vector is simply cloned.
    /// This variant exists for API consistency and to avoid
    /// special-case handling in generic code.
    Already(Vec<LngLat>),
}

impl From<Vec<(f64, f64)>> for CoordinateInput {
    fn from(tuples: Vec<(f64, f64)>) -> Self {
        CoordinateInput::Tuples(tuples)
    }
}

impl From<Vec<f64>> for CoordinateInput {
    fn from(arr: Vec<f64>) -> Self {
        CoordinateInput::FlatArray(arr)
    }
}

impl From<Vec<GeoPoint>> for CoordinateInput {
    fn from(points: Vec<GeoPoint>) -> Self {
        CoordinateInput::GeoJson(points)
    }
}

impl From<Vec<LngLat>> for CoordinateInput {
    fn from(coords: Vec<LngLat>) -> Self {
        CoordinateInput::Already(coords)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_hint_debug() {
        // Verify FormatHint implements Debug and displays correctly
        assert_eq!(format!("{:?}", FormatHint::LngLat), "LngLat");
        assert_eq!(format!("{:?}", FormatHint::LatLng), "LatLng");
        assert_eq!(format!("{:?}", FormatHint::Unknown), "Unknown");
    }

    #[test]
    fn test_coord_source_vec_lnglat_iteration() {
        let coords = vec![
            LngLat::new_deg(-122.4194, 37.7749), // San Francisco
            LngLat::new_deg(-74.0060, 40.7128),  // New York
            LngLat::new_deg(-87.6298, 41.8781),  // Chicago
        ];

        let mut iter = coords.get_coords();

        assert_eq!(iter.next(), Some(LngLat::new_deg(-122.4194, 37.7749)));
        assert_eq!(iter.next(), Some(LngLat::new_deg(-74.0060, 40.7128)));
        assert_eq!(iter.next(), Some(LngLat::new_deg(-87.6298, 41.8781)));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None); // Should remain None
    }

    #[test]
    fn test_coord_source_tuples_iteration() {
        let tuples = vec![
            (-122.4194, 37.7749),
            (-74.0060, 40.7128),
            (-87.6298, 41.8781),
        ];

        let mut iter = tuples.get_coords();

        assert_eq!(iter.next(), Some(LngLat::new_deg(-122.4194, 37.7749)));
        assert_eq!(iter.next(), Some(LngLat::new_deg(-74.0060, 40.7128)));
        assert_eq!(iter.next(), Some(LngLat::new_deg(-87.6298, 41.8781)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_coord_source_flat_vec_iteration() {
        let flat_array = vec![
            -122.4194, 37.7749, // San Francisco
            -74.0060, 40.7128, // New York
            -87.6298, 41.8781, // Chicago
        ];

        let mut iter = flat_array.get_coords();

        assert_eq!(iter.next(), Some(LngLat::new_deg(-122.4194, 37.7749)));
        assert_eq!(iter.next(), Some(LngLat::new_deg(-74.0060, 40.7128)));
        assert_eq!(iter.next(), Some(LngLat::new_deg(-87.6298, 41.8781)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_coord_source_flat_vec_odd_length() {
        // Odd length should ignore the last element
        let flat_array = vec![
            -122.4194, 37.7749, // San Francisco
            -74.0060, 40.7128,  // New York
            -87.6298, // Incomplete coordinate
        ];

        let mut iter = flat_array.get_coords();

        assert_eq!(iter.next(), Some(LngLat::new_deg(-122.4194, 37.7749)));
        assert_eq!(iter.next(), Some(LngLat::new_deg(-74.0060, 40.7128)));
        assert_eq!(iter.next(), None); // Incomplete coordinate is ignored
    }

    #[test]
    fn test_coord_source_empty_arrays() {
        // Test empty Vec<LngLat>
        let empty_coords: Vec<LngLat> = vec![];
        let mut iter = empty_coords.get_coords();
        assert_eq!(iter.next(), None);

        // Test empty tuples
        let empty_tuples: Vec<(f64, f64)> = vec![];
        let mut iter = empty_tuples.get_coords();
        assert_eq!(iter.next(), None);

        // Test empty flat array
        let empty_flat: Vec<f64> = vec![];
        let mut iter = empty_flat.get_coords();
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_coord_source_vec_lnglat() {
        let coords = vec![
            LngLat::new_deg(-122.4194, 37.7749),
            LngLat::new_deg(-74.0060, 40.7128),
        ];

        // Test trait methods
        let hint_size = coords.hint_size();
        assert_eq!(hint_size, Some(2));

        let format_hint = coords.hint_format();
        assert!(matches!(format_hint, FormatHint::LngLat));

        // Test iterator
        let collected: Vec<LngLat> = coords.get_coords().collect();
        assert_eq!(collected.len(), 2);
        assert_eq!(collected[0], LngLat::new_deg(-122.4194, 37.7749));
        assert_eq!(collected[1], LngLat::new_deg(-74.0060, 40.7128));
    }

    #[test]
    fn test_coord_source_vec_tuples() {
        let tuples = vec![
            (-122.4194, 37.7749),
            (-74.0060, 40.7128),
            (-87.6298, 41.8781),
        ];

        assert_eq!(tuples.hint_size(), Some(3));
        assert!(matches!(tuples.hint_format(), FormatHint::Unknown));

        let collected: Vec<LngLat> = tuples.get_coords().collect();
        assert_eq!(collected.len(), 3);
        assert_eq!(collected[0], LngLat::new_deg(-122.4194, 37.7749));
        assert_eq!(collected[1], LngLat::new_deg(-74.0060, 40.7128));
        assert_eq!(collected[2], LngLat::new_deg(-87.6298, 41.8781));
    }

    #[test]
    fn test_coord_source_flat_array() {
        let flat_array = vec![-122.4194, 37.7749, -74.0060, 40.7128];

        assert_eq!(flat_array.hint_size(), Some(2));
        assert!(matches!(flat_array.hint_format(), FormatHint::Unknown));

        let collected: Vec<LngLat> = flat_array.get_coords().collect();
        assert_eq!(collected.len(), 2);
        assert_eq!(collected[0], LngLat::new_deg(-122.4194, 37.7749));
        assert_eq!(collected[1], LngLat::new_deg(-74.0060, 40.7128));
    }

    #[test]
    fn test_coord_source_empty_collections() {
        // Empty Vec<LngLat>
        let empty_coords: Vec<LngLat> = vec![];
        assert_eq!(empty_coords.hint_size(), Some(0));
        assert!(matches!(empty_coords.hint_format(), FormatHint::LngLat));
        assert_eq!(empty_coords.get_coords().collect::<Vec<_>>().len(), 0);

        // Empty Vec<(f64, f64)>
        let empty_tuples: Vec<(f64, f64)> = vec![];
        assert_eq!(empty_tuples.hint_size(), Some(0));
        assert!(matches!(empty_tuples.hint_format(), FormatHint::Unknown));
        assert_eq!(empty_tuples.get_coords().collect::<Vec<_>>().len(), 0);

        // Empty flat array
        let empty_flat: Vec<f64> = vec![];
        assert_eq!(empty_flat.hint_size(), Some(0));
        assert!(matches!(empty_flat.hint_format(), FormatHint::Unknown));
        assert_eq!(empty_flat.get_coords().collect::<Vec<_>>().len(), 0);
    }

    #[test]
    fn test_coord_source_single_element() {
        // Single coordinate in Vec<LngLat>
        let single_coord = vec![LngLat::new_deg(0.0, 0.0)];
        assert_eq!(single_coord.hint_size(), Some(1));
        let collected: Vec<LngLat> = single_coord.get_coords().collect();
        assert_eq!(collected.len(), 1);
        assert_eq!(collected[0], LngLat::new_deg(0.0, 0.0));

        // Single tuple
        let single_tuple = vec![(1.0, 2.0)];
        assert_eq!(single_tuple.hint_size(), Some(1));
        let collected: Vec<LngLat> = single_tuple.get_coords().collect();
        assert_eq!(collected.len(), 1);
        assert_eq!(collected[0], LngLat::new_deg(1.0, 2.0));

        // Single coordinate in flat array
        let single_flat = vec![3.0, 4.0];
        assert_eq!(single_flat.hint_size(), Some(1));
        let collected: Vec<LngLat> = single_flat.get_coords().collect();
        assert_eq!(collected.len(), 1);
        assert_eq!(collected[0], LngLat::new_deg(3.0, 4.0));
    }

    #[test]
    fn test_coord_source_flat_array_size_calculation() {
        // Test various flat array sizes
        let sizes_and_expected = vec![
            (0, 0),    // Empty
            (1, 0),    // Single element (incomplete pair)
            (2, 1),    // One coordinate pair
            (3, 1),    // One pair + incomplete
            (4, 2),    // Two coordinate pairs
            (5, 2),    // Two pairs + incomplete
            (100, 50), // Large array
            (101, 50), // Large array + incomplete
        ];

        for (array_len, expected_coords) in sizes_and_expected {
            let flat_array: Vec<f64> = (0..array_len).map(|i| i as f64).collect();
            assert_eq!(
                flat_array.hint_size(),
                Some(expected_coords),
                "Failed for array length {}",
                array_len
            );

            let collected: Vec<LngLat> = flat_array.get_coords().collect();
            assert_eq!(
                collected.len(),
                expected_coords,
                "Iterator returned wrong count for array length {}",
                array_len
            );
        }
    }

    #[test]
    fn test_coord_source_consistency_across_formats() {
        // Same data in different formats should produce identical results
        let test_coords = vec![
            (-122.4194, 37.7749),
            (-74.0060, 40.7128),
            (-87.6298, 41.8781),
        ];

        // Convert to different formats
        let lnglat_vec: Vec<LngLat> = test_coords
            .iter()
            .map(|&(lng, lat)| LngLat::new_deg(lng, lat))
            .collect();

        let tuple_vec = test_coords.clone();

        let flat_array: Vec<f64> = test_coords
            .iter()
            .flat_map(|&(lng, lat)| vec![lng, lat])
            .collect();

        // Collect results from all formats
        let from_lnglat: Vec<LngLat> = lnglat_vec.get_coords().collect();
        let from_tuples: Vec<LngLat> = tuple_vec.get_coords().collect();
        let from_flat: Vec<LngLat> = flat_array.get_coords().collect();

        // All should be identical
        assert_eq!(from_lnglat.len(), 3);
        assert_eq!(from_tuples.len(), 3);
        assert_eq!(from_flat.len(), 3);

        for i in 0..3 {
            assert_eq!(from_lnglat[i], from_tuples[i]);
            assert_eq!(from_lnglat[i], from_flat[i]);
            assert_eq!(from_tuples[i], from_flat[i]);
        }
    }

    #[test]
    fn test_coord_source_extreme_coordinates() {
        // Test with extreme valid coordinates
        let extreme_coords = vec![
            (-180.0, -90.0),          // Southwest corner
            (180.0, 90.0),            // Northeast corner
            (0.0, 0.0),               // Origin
            (-179.999999, 89.999999), // Near limits
        ];

        let collected: Vec<LngLat> = extreme_coords.get_coords().collect();

        assert_eq!(collected.len(), 4);
        assert_eq!(collected[0], LngLat::new_deg(-180.0, -90.0));
        assert_eq!(collected[1], LngLat::new_deg(180.0, 90.0));
        assert_eq!(collected[2], LngLat::new_deg(0.0, 0.0));
        assert_eq!(collected[3], LngLat::new_deg(-179.999999, 89.999999));
    }

    #[test]
    fn test_coord_source_precision_preservation() {
        // Test that high-precision coordinates are preserved exactly
        let high_precision = vec![
            (-122.419416123456, 37.774928987654),
            (-74.006012345679, 40.712776543211),
        ];

        let flat_array = vec![
            -122.419416123456,
            37.774928987654,
            -74.006012345679,
            40.712776543211,
        ];

        let from_tuples: Vec<LngLat> = high_precision.get_coords().collect();
        let from_flat: Vec<LngLat> = flat_array.get_coords().collect();

        assert_eq!(from_tuples.len(), 2);
        assert_eq!(from_flat.len(), 2);

        // Verify exact precision preservation
        assert_eq!(from_tuples[0].lng_deg, -122.419416123456);
        assert_eq!(from_tuples[0].lat_deg, 37.774928987654);
        assert_eq!(from_flat[0].lng_deg, -122.419416123456);
        assert_eq!(from_flat[0].lat_deg, 37.774928987654);

        assert_eq!(from_tuples[1].lng_deg, -74.006012345679);
        assert_eq!(from_tuples[1].lat_deg, 40.712776543211);
        assert_eq!(from_flat[1].lng_deg, -74.006012345679);
        assert_eq!(from_flat[1].lat_deg, 40.712776543211);
    }

    #[test]
    fn test_coord_source_large_dataset_consistency() {
        // Test with a larger dataset to ensure iterator doesn't have off-by-one errors
        let large_dataset: Vec<(f64, f64)> = (0..1000)
            .map(|i| (i as f64 * 0.001, (i + 1) as f64 * 0.001))
            .collect();

        let collected: Vec<LngLat> = large_dataset.get_coords().collect();

        assert_eq!(collected.len(), 1000);

        // Spot check first, middle, and last elements
        assert_eq!(collected[0], LngLat::new_deg(0.0, 0.001));
        assert_eq!(collected[499], LngLat::new_deg(0.499, 0.5));
        assert_eq!(collected[999], LngLat::new_deg(0.999, 1.0));

        // Verify all elements are correct
        for (i, coord) in collected.iter().enumerate() {
            let expected_lng = i as f64 * 0.001;
            let expected_lat = (i + 1) as f64 * 0.001;
            assert_eq!(
                coord.lng_deg, expected_lng,
                "Longitude mismatch at index {}",
                i
            );
            assert_eq!(
                coord.lat_deg, expected_lat,
                "Latitude mismatch at index {}",
                i
            );
        }
    }

    #[test]
    fn test_coords_to_lnglat_vec_tuples_lnglat() {
        let tuples = vec![
            (-122.4194, 37.7749), // San Francisco (lng, lat)
            (-74.0060, 40.7128),  // New York
        ];

        let input = CoordinateInput::Tuples(tuples);
        let result = coords_to_lnglat_vec(&input);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0], LngLat::new_deg(-122.4194, 37.7749));
        assert_eq!(result[1], LngLat::new_deg(-74.0060, 40.7128));
    }

    #[test]
    fn test_coords_to_lnglat_vec_tuples_latlng() {
        let tuples = vec![
            (37.7749, -122.4194), // San Francisco (lat, lng)
            (40.7128, -74.0060),  // New York
        ];

        let input = CoordinateInput::Tuples(tuples);
        let result = coords_to_lnglat_vec(&input);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0], LngLat::new_deg(-122.4194, 37.7749));
        assert_eq!(result[1], LngLat::new_deg(-74.0060, 40.7128));
    }

    #[test]
    fn test_coords_to_lnglat_vec_flat_array() {
        let flat = vec![
            -122.4194, 37.7749, // San Francisco
            -74.0060, 40.7128, // New York
        ];

        let input = CoordinateInput::FlatArray(flat);
        let result = coords_to_lnglat_vec(&input);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0], LngLat::new_deg(-122.4194, 37.7749));
        assert_eq!(result[1], LngLat::new_deg(-74.0060, 40.7128));
    }

    #[test]
    fn test_coords_to_lnglat_vec_geojson() {
        let geojson = vec![
            GeoPoint {
                coordinates: [-122.4194, 37.7749],
            },
            GeoPoint {
                coordinates: [-74.0060, 40.7128],
            },
        ];

        let input = CoordinateInput::GeoJson(geojson);
        let result = coords_to_lnglat_vec(&input);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0], LngLat::new_deg(-122.4194, 37.7749));
        assert_eq!(result[1], LngLat::new_deg(-74.0060, 40.7128));
    }

    #[test]
    fn test_coords_to_lnglat_vec_already_lnglat() {
        let coords = vec![
            LngLat::new_deg(-122.4194, 37.7749),
            LngLat::new_deg(-74.0060, 40.7128),
        ];

        let input = CoordinateInput::Already(coords.clone());
        let result = coords_to_lnglat_vec(&input);

        assert_eq!(result, coords);
    }

    #[test]
    fn test_tuples_to_lnglat_vec_auto_empty() {
        let empty: Vec<(f64, f64)> = vec![];
        let result = tuples_to_lnglat_vec_auto(&empty);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_flat_array_to_lnglat_vec_auto_odd_length() {
        let odd_array = vec![-122.4194, 37.7749, -74.0060];
        let result = flat_array_to_lnglat_vec_auto(&odd_array);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0], LngLat::new_deg(-122.4194, 37.7749));
    }

    #[test]
    fn test_coordinate_input_from_conversions() {
        // Test From implementations
        let tuples = vec![(-122.4194, 37.7749)];
        let _input: CoordinateInput = tuples.into();

        let flat = vec![-122.4194, 37.7749];
        let _input: CoordinateInput = flat.into();

        let geojson = vec![GeoPoint {
            coordinates: [-122.4194, 37.7749],
        }];
        let _input: CoordinateInput = geojson.into();

        let coords = vec![LngLat::new_deg(-122.4194, 37.7749)];
        let _input: CoordinateInput = coords.into();
    }

    #[test]
    fn test_geopoint_coord_source() {
        let points = vec![
            GeoPoint {
                coordinates: [-122.4194, 37.7749],
            },
            GeoPoint {
                coordinates: [-74.0060, 40.7128],
            },
        ];

        assert_eq!(points.hint_size(), Some(2));
        assert_eq!(points.hint_format(), FormatHint::LngLat);

        let collected: Vec<LngLat> = points.get_coords().collect();
        assert_eq!(collected.len(), 2);
        assert_eq!(collected[0], LngLat::new_deg(-122.4194, 37.7749));
        assert_eq!(collected[1], LngLat::new_deg(-74.0060, 40.7128));
    }

    #[test]
    fn test_large_dataset_conversion_performance() {
        // Test with large dataset to ensure performance
        let large_tuples: Vec<(f64, f64)> = (0..10000)
            .map(|i| (i as f64 * 0.001, (i + 1) as f64 * 0.001))
            .collect();

        let result = tuples_to_lnglat_vec_auto(&large_tuples);

        assert_eq!(result.len(), 10000);
        assert_eq!(result[0], LngLat::new_deg(0.0, 0.001));
        assert_eq!(result[9999], LngLat::new_deg(9.999, 10.0));
    }

    #[test]
    fn test_coord_source_f64_slice() {
        let flat_array = [
            -122.4194, 37.7749, // San Francisco
            -74.0060, 40.7128, // New York City
        ];

        let coords: Vec<_> = flat_array.get_coords().collect();
        assert_eq!(coords.len(), 2);
        assert_eq!(coords[0], LngLat::new_deg(-122.4194, 37.7749));
        assert_eq!(coords[1], LngLat::new_deg(-74.0060, 40.7128));

        assert_eq!(flat_array.hint_size(), Some(2));
        assert!(matches!(flat_array.hint_format(), FormatHint::Unknown));
    }

    #[test]
    fn test_coord_source_empty_f64_slice() {
        let empty_slice: &[f64] = &[];

        let coords: Vec<_> = empty_slice.get_coords().collect();
        assert_eq!(coords.len(), 0);

        assert_eq!(empty_slice.hint_size(), Some(0));
        assert!(matches!(empty_slice.hint_format(), FormatHint::Unknown));
    }

    #[test]
    fn test_coord_source_odd_length_f64_slice() {
        let odd_array = [-122.4194, 37.7749, -74.0060];

        let coords: Vec<_> = odd_array.get_coords().collect();
        assert_eq!(coords.len(), 1);
        assert_eq!(coords[0], LngLat::new_deg(-122.4194, 37.7749));

        assert_eq!(odd_array.hint_size(), Some(1)); // 3 / 2 = 1
        assert!(matches!(odd_array.hint_format(), FormatHint::Unknown));
    }
}
