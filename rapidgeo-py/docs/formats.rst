Coordinate Format Detection
===========================

rapidgeo automatically detects and converts coordinate data from various formats into its standard longitude, latitude (lng, lat) representation. This system handles the common problem of coordinate data coming in different structures and orderings.

How Format Detection Works
--------------------------

The system uses a two-stage approach:

1. **Structure Detection**: Examines the data type and structure to identify the format
2. **Coordinate Detection**: For ambiguous formats, analyzes coordinate values to determine lng,lat vs lat,lng ordering

The detection process follows this hierarchy:

1. NumPy arrays (when available) - fastest path with zero-copy when possible
2. Python lists with format-specific parsing
3. Automatic coordinate ordering detection for ambiguous cases

Supported Input Formats
-----------------------

Tuple/List Format
~~~~~~~~~~~~~~~~~

Coordinate pairs as tuples or lists:

.. code-block:: python

    from rapidgeo.formats import coords_to_lnglat
    
    # Tuple format - detected automatically
    coords = [
        (-122.4194, 37.7749),  # San Francisco
        (-74.0060, 40.7128),   # New York
        (-87.6298, 41.8781),   # Chicago
    ]
    result = coords_to_lnglat(coords)

    # Also works with lists
    coords = [
        [-122.4194, 37.7749],
        [-74.0060, 40.7128]
    ]
    result = coords_to_lnglat(coords)

Flat Array Format
~~~~~~~~~~~~~~~~~

Coordinates as a flat array of alternating longitude and latitude values:

.. code-block:: python

    # Flat array: [lng1, lat1, lng2, lat2, ...]
    coords = [
        -122.4194, 37.7749,  # San Francisco
        -74.0060, 40.7128,   # New York
        -87.6298, 41.8781,   # Chicago
    ]
    result = coords_to_lnglat(coords)

This format is common in graphics APIs, database storage, and NumPy arrays.

GeoJSON-like Format
~~~~~~~~~~~~~~~~~~~

Dictionary objects with coordinate arrays following GeoJSON Point structure:

.. code-block:: python

    coords = [
        {"coordinates": [-122.4194, 37.7749]},  # San Francisco
        {"coordinates": [-74.0060, 40.7128]},   # New York
        {"coordinates": [-87.6298, 41.8781]},   # Chicago
    ]
    result = coords_to_lnglat(coords)

The coordinates array must contain exactly two elements: [longitude, latitude].

NumPy Array Format
~~~~~~~~~~~~~~~~~~

When the numpy feature is enabled, various NumPy array formats are supported:

.. code-block:: python

    import numpy as np
    from rapidgeo.formats import coords_to_lnglat

    # 2D array (N, 2) - fastest path
    coords = np.array([
        [-122.4194, 37.7749],
        [-74.0060, 40.7128]
    ])
    result = coords_to_lnglat(coords)

    # 1D flat array
    coords = np.array([-122.4194, 37.7749, -74.0060, 40.7128])
    result = coords_to_lnglat(coords)

    # Dynamic arrays also supported
    coords = np.array([[-122.4194, 37.7749], [-74.0060, 40.7128]], dtype=object)
    result = coords_to_lnglat(coords)

Automatic Coordinate Ordering Detection
---------------------------------------

For tuple and flat array formats, the system automatically determines whether coordinates are in lng,lat or lat,lng order by analyzing the coordinate values.

Detection Algorithm
~~~~~~~~~~~~~~~~~~~

The algorithm uses statistical analysis of coordinate ranges:

1. **Validation**: Checks if coordinates fit within valid ranges:
   - Longitude: -180° to +180°
   - Latitude: -90° to +90°

2. **Sampling**: Examines up to 100 coordinate pairs for performance

3. **Scoring**: Counts valid coordinates for each interpretation (lng,lat vs lat,lng)

4. **Confidence**: Uses 95% confidence threshold with early termination

5. **Decision**: Returns the format with more valid coordinates

Examples of Automatic Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clear lng,lat format (negative longitudes in Western Hemisphere):

.. code-block:: python

    # These are clearly lng,lat due to longitude values < -90
    coords = [
        (-122.4194, 37.7749),  # San Francisco: lng=-122° (clearly longitude)
        (-74.0060, 40.7128),   # New York: lng=-74° (clearly longitude)
    ]
    result = coords_to_lnglat(coords)
    # Result: coordinates used as-is

Clear lat,lng format (detected and corrected):

.. code-block:: python

    # These appear to be lat,lng and will be automatically swapped
    coords = [
        (37.7749, -122.4194),  # San Francisco: 37° lat, -122° lng  
        (40.7128, -74.0060),   # New York: 40° lat, -74° lng
    ]
    result = coords_to_lnglat(coords)
    # Result: automatically corrected to lng,lat order

Ambiguous coordinates (fallback to lng,lat):

.. code-block:: python

    # These could be valid in either order
    coords = [
        (45.0, 60.0),    # Both values within ±90°
        (30.0, -80.0),   # Could be interpreted either way
    ]
    result = coords_to_lnglat(coords)
    # Result: treats as lng,lat (default assumption)

Performance Characteristics
---------------------------

Format Detection Speed
~~~~~~~~~~~~~~~~~~~~~~

- **NumPy 2D arrays**: Zero-copy for contiguous arrays (~1μs)
- **Flat arrays**: Direct memory copy (~10μs for 1000 points)
- **Tuple lists**: Python iteration required (~100μs for 1000 points)
- **GeoJSON objects**: Dictionary access overhead (~500μs for 1000 points)

Detection Optimizations
~~~~~~~~~~~~~~~~~~~~~~~

- **Early termination**: Stops when 95% confidence reached (typically after 10-20 samples)
- **Sampling limit**: Maximum 100 coordinates analyzed regardless of input size
- **Zero-copy paths**: Direct memory access for compatible NumPy arrays
- **Format caching**: Structure detection happens once per input

Memory Usage
~~~~~~~~~~~~

- **Zero additional memory**: For already-correct lng,lat format
- **Single copy**: For format conversion (input size × 2 × 8 bytes)
- **Minimal overhead**: Detection uses <1KB regardless of input size

Error Handling and Edge Cases
-----------------------------

Format Errors
~~~~~~~~~~~~~~

.. code-block:: python

    from rapidgeo.formats import coords_to_lnglat

    # Empty input - returns empty list
    coords = []
    result = coords_to_lnglat(coords)  # []

    # Malformed GeoJSON - raises KeyError
    coords = [{"not_coordinates": [1.0, 2.0]}]
    try:
        result = coords_to_lnglat(coords)
    except KeyError as e:
        print(f"Missing coordinates key: {e}")

    # Wrong coordinate count - raises ValueError  
    coords = [{"coordinates": [1.0]}]  # Only one coordinate
    try:
        result = coords_to_lnglat(coords)
    except ValueError as e:
        print(f"Invalid coordinate array: {e}")

Invalid Coordinates
~~~~~~~~~~~~~~~~~~~

The system preserves invalid coordinates but they don't affect format detection:

.. code-block:: python

    # Out-of-range coordinates are preserved
    coords = [
        (-122.4194, 37.7749),   # Valid
        (200.0, 95.0),          # Invalid (out of range)
        (-74.0060, 40.7128),    # Valid
    ]
    result = coords_to_lnglat(coords)
    # Detection based only on valid coordinates
    # Invalid coordinates passed through unchanged

Handling Mixed Data
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Mix of valid and invalid affects confidence but not correctness
    coords = [
        (-122.4194, 37.7749),   # Clearly lng,lat
        (0.0, 0.0),             # Ambiguous but valid both ways
        (-74.0060, 40.7128),    # Clearly lng,lat
        (500.0, 600.0),         # Invalid coordinates
    ]
    result = coords_to_lnglat(coords)
    # Algorithm detects lng,lat from the clear examples

Practical Usage Examples
------------------------

Converting GPS Track Data
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from rapidgeo.formats import coords_to_lnglat

    # GPS data might come in various formats
    def standardize_gps_track(track_data):
        """Convert any GPS track format to standard LngLat."""
        return coords_to_lnglat(track_data)

    # Works with different input formats
    gps_track_tuples = [(-122.41, 37.77), (-122.42, 37.78)]
    gps_track_flat = [-122.41, 37.77, -122.42, 37.78]
    gps_track_geojson = [
        {"coordinates": [-122.41, 37.77]},
        {"coordinates": [-122.42, 37.78]}
    ]
    
    # All produce identical results
    track1 = standardize_gps_track(gps_track_tuples)
    track2 = standardize_gps_track(gps_track_flat)  
    track3 = standardize_gps_track(gps_track_geojson)

Working with DataFrames
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import pandas as pd
    from rapidgeo.formats import coords_to_lnglat

    # DataFrame with separate lat/lng columns
    df = pd.DataFrame({
        'latitude': [37.7749, 40.7128, 41.8781],
        'longitude': [-122.4194, -74.0060, -87.6298]
    })
    
    # Convert to coordinate pairs (note: lat,lng order from DataFrame)
    coord_pairs = list(zip(df['latitude'], df['longitude']))
    
    # System will detect this is lat,lng and correct it
    standardized = coords_to_lnglat(coord_pairs)

API Integration
~~~~~~~~~~~~~~~

.. code-block:: python

    from rapidgeo.formats import coords_to_lnglat

    def process_api_coordinates(api_response):
        """Handle coordinates from external API."""
        
        # API might return various formats
        if 'coordinates' in api_response:
            # GeoJSON-style
            coords = [{"coordinates": coord} for coord in api_response['coordinates']]
        elif 'points' in api_response:
            # Flat array style  
            coords = api_response['points']
        else:
            # Assume tuple/list format
            coords = api_response['data']
        
        # Automatic detection and conversion
        return coords_to_lnglat(coords)

Integration with Other rapidgeo Functions
-----------------------------------------

The format detection system is automatically used by other rapidgeo functions:

.. code-block:: python

    from rapidgeo import polyline, distance, simplify

    # These functions automatically detect coordinate formats
    coords = [(37.7749, -122.4194), (40.7128, -74.0060)]  # lat,lng format

    # Automatically detected and corrected to lng,lat internally
    encoded = polyline.encode(coords)
    
    # Distance calculation also handles format detection
    dist = distance.geo.haversine(*coords_to_lnglat(coords[:2]))
    
    # Simplification with automatic format handling
    simplified = simplify.douglas_peucker(coords, tolerance=0.001)

Best Practices
--------------

Input Validation
~~~~~~~~~~~~~~~~

While the system is robust, validating your input helps catch issues early:

.. code-block:: python

    def validate_and_convert_coordinates(coords):
        """Safely convert coordinates with validation."""
        if not coords:
            return []
        
        if not isinstance(coords, (list, tuple)):
            raise TypeError("Coordinates must be a list or tuple")
            
        # Convert and let system handle format detection
        try:
            return coords_to_lnglat(coords)
        except (ValueError, KeyError, TypeError) as e:
            raise ValueError(f"Invalid coordinate format: {e}")

Performance Tips
~~~~~~~~~~~~~~~~

For maximum performance with large datasets:

.. code-block:: python

    import numpy as np
    from rapidgeo.formats import coords_to_lnglat

    # Use NumPy arrays when possible (fastest)
    coords = np.array([[-122.4194, 37.7749], [-74.0060, 40.7128]])
    result = coords_to_lnglat(coords)  # Zero-copy path

    # Pre-convert to correct format if you know the ordering
    # (skips detection overhead for very large datasets)
    if coordinates_are_lng_lat:
        # Direct conversion without detection
        result = [LngLat.new_deg(lng, lat) for lng, lat in coord_pairs]

Format Consistency
~~~~~~~~~~~~~~~~~~

Within the same application, try to standardize on one coordinate format:

.. code-block:: python

    # Good: Consistent format throughout application
    COORDINATE_FORMAT = "lng_lat_tuples"  # or "flat_array", "geojson", etc.

    def load_coordinates(source):
        """Load coordinates in standardized format."""
        raw_data = fetch_from_source(source)
        return coords_to_lnglat(raw_data)  # Always returns LngLat format

