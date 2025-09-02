Troubleshooting
===============

Common issues and their solutions.

Installation Issues
-------------------

Module Not Found
~~~~~~~~~~~~~~~~~

**Problem:** ``ImportError: No module named 'rapidgeo'``

**Solution:**

.. code-block:: bash

    pip install rapidgeo

If you're using a virtual environment, make sure it's activated:

.. code-block:: bash

    source venv/bin/activate  # Linux/Mac
    # or
    venv\Scripts\activate     # Windows
    pip install rapidgeo

NumPy Features Not Available
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** ``ImportError`` when trying to import numpy submodule

**Solutions:**

1. Install with NumPy support:

.. code-block:: bash

    pip install rapidgeo[numpy]

2. Check if NumPy features are available:

.. code-block:: python

    try:
        from rapidgeo.distance import numpy as rgeo_numpy
        print("NumPy support available")
    except ImportError:
        print("NumPy support not available - install with: pip install rapidgeo[numpy]")

Coordinate Problems
-------------------

Wrong Coordinate Order
~~~~~~~~~~~~~~~~~~~~~~

**Problem:** Results don't match expected distances or locations

**Cause:** Using (lat, lng) instead of (lng, lat)

**Solution:**

.. code-block:: python

    # Wrong - latitude first
    point = LngLat(37.7749, -122.4194)  
    
    # Correct - longitude first  
    point = LngLat(-122.4194, 37.7749)

**Check:** San Francisco should be approximately LngLat(-122.42, 37.77), not LngLat(37.77, -122.42).

Invalid Coordinates
~~~~~~~~~~~~~~~~~~~

**Problem:** ``ValueError`` or unexpected results

**Common issues:**

* Coordinates outside valid ranges (lng: -180 to 180, lat: -90 to 90)
* Using None or NaN values
* Mixing coordinate systems

**Solution:**

.. code-block:: python

    def validate_coordinate(lng, lat):
        """Validate coordinate values."""
        if not (-180 <= lng <= 180):
            raise ValueError(f"Longitude {lng} out of range [-180, 180]")
        if not (-90 <= lat <= 90):
            raise ValueError(f"Latitude {lat} out of range [-90, 90]")
        return LngLat(lng, lat)
    
    # Use validation for untrusted input
    point = validate_coordinate(-122.4194, 37.7749)

Distance Calculation Issues
---------------------------

Unexpected Zero Distances
~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** Distance calculations return 0 when points are different

**Causes:**

1. **Identical coordinates:** Points are actually the same
2. **Precision issues:** Coordinates differ by less than floating-point precision

**Solution:**

.. code-block:: python

    # Check if points are truly identical
    if point1.lng == point2.lng and point1.lat == point2.lat:
        print("Points are identical")
    else:
        distance = haversine(point1, point2)
        if distance < 0.001:  # Less than 1mm
            print("Points are effectively identical")

Very Large Distances
~~~~~~~~~~~~~~~~~~~~

**Problem:** Distance calculations return unexpectedly large values

**Causes:**

1. **Wrong coordinate order:** Using (lat, lng) instead of (lng, lat)  
2. **Decimal degree confusion:** Using degrees-minutes-seconds instead of decimal degrees
3. **Wrong coordinate system:** Mixing projected coordinates with geographic

**Solution:**

.. code-block:: python

    # Sanity check distances
    distance = haversine(point1, point2)
    
    if distance > 20_000_000:  # More than half Earth's circumference
        print("Warning: Distance is very large - check coordinate order")
    elif distance > 1_000_000:  # More than 1000km  
        print("Large distance - verify coordinates are correct")

Vincenty Algorithm Failures
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** ``ValueError`` from Vincenty distance calculation

**Cause:** Algorithm fails to converge (rare, usually with antipodal points)

**Solution:**

.. code-block:: python

    from rapidgeo.distance.geo import haversine, vincenty_distance
    
    def safe_vincenty_distance(point1, point2):
        """Calculate Vincenty distance with fallback to Haversine."""
        try:
            return vincenty_distance(point1, point2)
        except ValueError:
            # Fallback to Haversine for problematic cases
            return haversine(point1, point2)
    
    distance = safe_vincenty_distance(point1, point2)

Polyline Issues
---------------

Encoding/Decoding Mismatches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** Decoded coordinates don't match original coordinates

**Cause:** Different precision levels used for encoding and decoding

**Solution:**

.. code-block:: python

    # Always use same precision for encode/decode
    precision = 5
    encoded = encode(points, precision=precision)
    decoded = decode(encoded, precision=precision)

**Note:** Some rounding is expected due to polyline compression. Higher precision reduces but doesn't eliminate rounding.

Empty Polyline Strings
~~~~~~~~~~~~~~~~~~~~~~

**Problem:** Encoding returns empty string or very short string

**Causes:**

1. **Empty input:** No coordinates provided
2. **Single point:** Only one coordinate (polylines need at least 2 points for meaningful encoding)
3. **Identical points:** All coordinates are the same

**Solution:**

.. code-block:: python

    def safe_encode(points, precision=5):
        """Safely encode points to polyline."""
        if len(points) < 2:
            raise ValueError(f"Need at least 2 points, got {len(points)}")
        
        # Remove consecutive duplicates
        cleaned = [points[0]]
        for point in points[1:]:
            if point != cleaned[-1]:
                cleaned.append(point)
        
        if len(cleaned) < 2:
            raise ValueError("All points are identical")
        
        return encode(cleaned, precision=precision)

Memory and Performance Issues
-----------------------------

Out of Memory Errors
~~~~~~~~~~~~~~~~~~~~~

**Problem:** ``MemoryError`` when processing large datasets

**Solutions:**

1. **Process in chunks:**

.. code-block:: python

    def process_large_dataset(all_tracks, chunk_size=1000):
        """Process tracks in chunks to manage memory."""
        results = []
        
        for i in range(0, len(all_tracks), chunk_size):
            chunk = all_tracks[i:i + chunk_size]
            chunk_results = encode_batch(chunk, precision=5)
            results.extend(chunk_results)
            
        return results

2. **Use generators for streaming:**

.. code-block:: python

    def process_tracks_streaming(track_iterator):
        """Stream process tracks without loading all into memory."""
        for track in track_iterator:
            if len(track) >= 2:  # Skip invalid tracks
                yield encode(track, precision=5)

Similarity Algorithm Size Limits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** ``ValueError: curve size limited`` when comparing large tracks

**Cause:** Built-in limits to prevent memory exhaustion

**Solutions:**

1. **Simplify tracks before comparison:**

.. code-block:: python

    from rapidgeo.simplify import douglas_peucker
    
    # Simplify large tracks before similarity calculation
    simplified_track1 = douglas_peucker(large_track1, tolerance_m=10.0)
    simplified_track2 = douglas_peucker(large_track2, tolerance_m=10.0)
    
    similarity = discrete_frechet(simplified_track1, simplified_track2)

2. **Sample tracks:**

.. code-block:: python

    def sample_track(track, max_points=5000):
        """Sample track to reduce point count."""
        if len(track) <= max_points:
            return track
        
        step = len(track) // max_points
        return track[::step]
    
    sampled1 = sample_track(track1)
    sampled2 = sample_track(track2)

Data Type Issues
----------------

Wrong Input Types
~~~~~~~~~~~~~~~~~

**Problem:** ``TypeError`` when calling functions

**Causes:**

* Passing tuples instead of LngLat objects
* Using wrong data structures

**Solution:**

.. code-block:: python

    # Convert various input types to LngLat
    def to_lnglat(coord):
        """Convert various coordinate formats to LngLat."""
        if isinstance(coord, LngLat):
            return coord
        elif isinstance(coord, (list, tuple)) and len(coord) == 2:
            return LngLat(coord[0], coord[1])
        else:
            raise ValueError(f"Cannot convert {type(coord)} to LngLat")
    
    # Convert list of tuples
    coords_as_tuples = [(-122.4, 37.7), (-122.3, 37.8)]
    coords_as_lnglat = [to_lnglat(coord) for coord in coords_as_tuples]

Getting Help
------------

When reporting issues:

1. **Include your code:** Show the specific function calls causing problems
2. **Provide sample data:** Include coordinates that reproduce the issue  
3. **Include error messages:** Copy the full traceback
4. **Specify versions:** Include rapidgeo version and Python version

**Check versions:**

.. code-block:: python

    import rapidgeo
    import sys
    
    print(f"rapidgeo version: {rapidgeo.__version__}")
    print(f"Python version: {sys.version}")

**Minimal example:**

.. code-block:: python

    from rapidgeo import LngLat
    from rapidgeo.distance.geo import haversine
    
    # This causes the problem
    point1 = LngLat(-122.4194, 37.7749)
    point2 = LngLat(-74.0060, 40.7128)
    
    distance = haversine(point1, point2)
    print(f"Distance: {distance}")  # What went wrong here?