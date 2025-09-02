Polyline Encoding
=================

The ``rapidgeo.polyline`` module provides Google Polyline Algorithm encoding and decoding with optional simplification support.

The Google Polyline Algorithm is a lossy compression format for storing sequences of coordinates as ASCII strings. It's commonly used in mapping applications like Google Maps.

Core Functions
--------------

.. automodule:: rapidgeo.polyline
   :members:
   :no-index:

Basic Usage
-----------

**Encoding:**

.. code-block:: python

    from rapidgeo.distance import LngLat
    from rapidgeo.polyline import encode, decode
    
    # Define a simple path
    path = [
        LngLat.new_deg(-122.4194, 37.7749),  # San Francisco
        LngLat.new_deg(-87.6298, 41.8781),   # Chicago  
        LngLat.new_deg(-74.0060, 40.7128),   # New York City
    ]
    
    # Encode to polyline string
    polyline_str = encode(path, precision=5)
    print(f"Encoded: {polyline_str}")
    
    # Decode back to coordinates
    decoded_path = decode(polyline_str, precision=5)
    print(f"Decoded {len(decoded_path)} points")

**Precision Levels:**

* **precision=5**: Standard precision (~1 meter accuracy)
* **precision=6**: High precision (~0.1 meter accuracy)
* **precision=7**: Very high precision (~0.01 meter accuracy)

Higher precision results in longer encoded strings but better coordinate accuracy.

Simplification
--------------

Polylines can be simplified during encoding to reduce size while maintaining shape:

.. code-block:: python

    from rapidgeo.polyline import encode_simplified
    
    # Create a detailed path with many points
    detailed_path = [
        LngLat.new_deg(-122.4194, 37.7749),
        LngLat.new_deg(-122.4180, 37.7755),  # Close intermediate point
        LngLat.new_deg(-122.4160, 37.7765),  # Another close point
        LngLat.new_deg(-87.6298, 41.8781),   # Far point - will be kept
    ]
    
    # Encode with simplification (1km tolerance)
    simplified = encode_simplified(detailed_path, tolerance_m=1000.0, precision=5)
    print(f"Simplified polyline: {simplified}")
    
    # You can also simplify an existing polyline
    from rapidgeo.polyline import simplify_polyline
    
    original = encode(detailed_path, precision=5)
    simplified = simplify_polyline(original, tolerance_m=1000.0, precision=5)

Batch Operations
----------------

Process multiple polylines efficiently:

.. code-block:: python

    from rapidgeo.polyline import encode_batch, decode_batch, encode_simplified_batch
    
    # Multiple paths to process
    paths = [
        [LngLat.new_deg(-122.4, 37.7), LngLat.new_deg(-122.3, 37.8)],
        [LngLat.new_deg(-74.0, 40.7), LngLat.new_deg(-74.1, 40.8)],
    ]
    
    # Batch encode
    encoded_polylines = encode_batch(paths, precision=5)
    print(f"Encoded {len(encoded_polylines)} polylines")
    
    # Batch decode
    decoded_paths = decode_batch(encoded_polylines, precision=5)
    print(f"Decoded {len(decoded_paths)} paths")
    
    # Batch encode with simplification
    simplified_polylines = encode_simplified_batch(
        paths, tolerance_m=100.0, precision=5
    )

Real-World Example
------------------

Here's a complete example showing how you might use polylines in a web mapping application:

.. code-block:: python

    from rapidgeo.distance import LngLat
    from rapidgeo.polyline import encode_simplified, decode
    from rapidgeo.distance.batch import path_length_haversine
    
    # GPS track data (e.g., from a mobile app)
    gps_track = [
        LngLat.new_deg(-122.4194, 37.7749),
        LngLat.new_deg(-122.4180, 37.7755),
        LngLat.new_deg(-122.4160, 37.7765),
        LngLat.new_deg(-122.4140, 37.7775),
        # ... many more points
        LngLat.new_deg(-122.4000, 37.7900),
    ]
    
    # Calculate original track length
    original_length = path_length_haversine(gps_track)
    print(f"Original track: {len(gps_track)} points, {original_length/1000:.2f} km")
    
    # Encode with simplification for web transmission
    # 10m tolerance reduces data size while preserving route shape
    polyline_str = encode_simplified(gps_track, tolerance_m=10.0, precision=5)
    print(f"Encoded polyline: {len(polyline_str)} characters")
    
    # On the client side, decode the polyline
    simplified_track = decode(polyline_str, precision=5)
    simplified_length = path_length_haversine(simplified_track)
    
    print(f"Simplified track: {len(simplified_track)} points, {simplified_length/1000:.2f} km")
    print(f"Size reduction: {(1 - len(simplified_track)/len(gps_track))*100:.1f}%")

Algorithm Details
-----------------

**Encoding Process:**

1. Convert coordinates to integers by multiplying by 10^precision
2. Calculate deltas between consecutive points
3. Apply zigzag encoding to handle negative numbers
4. Convert to base32 representation using specific character set

**Precision vs. Size:**

* Higher precision = more accurate coordinates but longer strings
* Lower precision = smaller strings but coordinate quantization
* Choose precision based on your accuracy requirements

**Simplification:**

* Uses Douglas-Peucker algorithm during encoding
* Tolerance specified in meters (real-world distance)  
* Points are retained if they deviate more than tolerance from the simplified line

Implementation Notes
--------------------

**Processing:**

* Batch operations handle multiple polylines efficiently
* Simplification uses Douglas-Peucker algorithm
* Memory usage scales with input size

**Precision Trade-offs:**

* Higher precision = more accurate coordinates, longer strings
* Lower precision = shorter strings, some coordinate rounding
* Choose based on your accuracy requirements