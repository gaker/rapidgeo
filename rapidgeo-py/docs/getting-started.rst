Getting Started
===============

This guide walks through common tasks with rapidgeo using real examples.

Installation
------------

Install rapidgeo from PyPI:

.. code-block:: bash

    pip install rapidgeo

For NumPy integration (if available):

.. code-block:: bash

    pip install rapidgeo[numpy]

Coordinate Basics
-----------------

All coordinates use longitude, latitude ordering (lng, lat):

.. code-block:: python

    from rapidgeo import LngLat
    
    # Create coordinates
    san_francisco = LngLat(-122.4194, 37.7749)  # lng first, lat second
    new_york = LngLat(-74.0060, 40.7128)
    
    print(f"SF: {san_francisco}")
    print(f"NYC: {new_york}")

Common Tasks
------------

Calculate Distance Between Points
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from rapidgeo import LngLat
    from rapidgeo.distance.geo import haversine
    
    sf = LngLat(-122.4194, 37.7749)
    nyc = LngLat(-74.0060, 40.7128)
    
    distance_meters = haversine(sf, nyc)
    distance_km = distance_meters / 1000
    
    print(f"Distance: {distance_km:.1f} km")

Process GPS Track Data
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from rapidgeo import LngLat
    from rapidgeo.distance.batch import path_length_haversine, pairwise_haversine
    
    # Your GPS track points
    gps_track = [
        LngLat(-122.4194, 37.7749),
        LngLat(-122.4180, 37.7755),
        LngLat(-122.4160, 37.7765),
        LngLat(-122.4140, 37.7775),
    ]
    
    # Calculate total distance
    total_distance = path_length_haversine(gps_track)
    print(f"Total track length: {total_distance/1000:.2f} km")
    
    # Get distances between consecutive points
    segment_distances = list(pairwise_haversine(gps_track))
    print(f"Segments: {[f'{d:.1f}m' for d in segment_distances]}")

Encode GPS Data for Storage
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from rapidgeo.polyline import encode, decode
    
    # Encode your GPS track to a compact string
    polyline_string = encode(gps_track, precision=5)
    print(f"Encoded: {polyline_string}")
    print(f"Original: {len(gps_track)} points")
    print(f"Encoded: {len(polyline_string)} characters")
    
    # Decode it back
    decoded_points = decode(polyline_string, precision=5)
    print(f"Decoded: {len(decoded_points)} points")

Simplify GPS Tracks
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from rapidgeo.simplify import douglas_peucker
    
    # Remove unnecessary points while preserving track shape
    simplified_track = douglas_peucker(gps_track, tolerance_m=10.0)
    
    print(f"Original: {len(gps_track)} points")
    print(f"Simplified: {len(simplified_track)} points")
    
    # Calculate how much distance accuracy we lost
    original_length = path_length_haversine(gps_track)
    simplified_length = path_length_haversine(simplified_track)
    error_percent = abs(simplified_length - original_length) / original_length * 100
    
    print(f"Length error: {error_percent:.2f}%")

Compare Similar Routes
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from rapidgeo.similarity.frechet import discrete_frechet
    
    # Two similar but different routes
    route_a = [
        LngLat(-122.4194, 37.7749),
        LngLat(-122.4100, 37.7800),
        LngLat(-122.4000, 37.7850),
    ]
    
    route_b = [
        LngLat(-122.4194, 37.7749),  # Same start
        LngLat(-122.4120, 37.7790),  # Slightly different path
        LngLat(-122.4000, 37.7850),  # Same end
    ]
    
    similarity = discrete_frechet(route_a, route_b)
    print(f"Routes differ by up to {similarity:.1f} meters")
    
    # Check if routes are similar within tolerance
    if similarity < 50:  # 50 meter tolerance
        print("Routes are very similar")
    else:
        print("Routes are quite different")

Working with Different Data Sources
-----------------------------------

From Lists of Tuples
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # If you have coordinate data as (lng, lat) tuples
    coordinate_tuples = [(-122.4194, 37.7749), (-122.4100, 37.7800)]
    
    # Convert to LngLat objects
    track = [LngLat(lng, lat) for lng, lat in coordinate_tuples]
    
    # Now you can use with rapidgeo functions
    distance = path_length_haversine(track)

From Pandas DataFrames
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import pandas as pd
    
    # Example DataFrame with GPS data
    df = pd.DataFrame({
        'longitude': [-122.4194, -122.4100, -122.4000],
        'latitude': [37.7749, 37.7800, 37.7850],
        'timestamp': ['2024-01-01 10:00:00', '2024-01-01 10:05:00', '2024-01-01 10:10:00']
    })
    
    # Convert to LngLat objects
    track = [LngLat(row['longitude'], row['latitude']) for _, row in df.iterrows()]
    
    # Process the track
    total_distance = path_length_haversine(track)
    simplified = douglas_peucker(track, tolerance_m=5.0)

Choosing the Right Algorithm
----------------------------

Distance Calculations
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from rapidgeo.distance.geo import haversine, vincenty_distance
    from rapidgeo.distance.euclid import euclid
    
    sf = LngLat(-122.4194, 37.7749)
    la = LngLat(-118.2437, 34.0522)
    
    # For most geographic calculations
    distance_haversine = haversine(sf, la)
    print(f"Haversine: {distance_haversine/1000:.1f} km")
    
    # For highest precision (takes more computation)
    distance_vincenty = vincenty_distance(sf, la)
    print(f"Vincenty: {distance_vincenty/1000:.3f} km")
    
    # For fast approximation or projected coordinates
    distance_euclidean = euclid(sf, la) * 111320  # rough conversion to meters
    print(f"Euclidean: {distance_euclidean/1000:.1f} km (approximate)")

Polyline Precision
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Standard precision (about 1 meter accuracy)
    encoded_p5 = encode(track, precision=5)
    
    # Higher precision (about 0.1 meter accuracy, longer strings)
    encoded_p6 = encode(track, precision=6)
    
    print(f"Precision 5: {len(encoded_p5)} chars")
    print(f"Precision 6: {len(encoded_p6)} chars")
    
    # Choose based on your accuracy needs vs storage/bandwidth

Error Handling
--------------

.. code-block:: python

    try:
        # Your rapidgeo operations
        distance = haversine(sf, nyc)
        simplified = douglas_peucker(track, tolerance_m=10.0)
        
    except ValueError as e:
        print(f"Input error: {e}")
        # Handle invalid coordinates, empty tracks, etc.
        
    except Exception as e:
        print(f"Unexpected error: {e}")

Tips for Better Performance
---------------------------

Use Batch Operations
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # More efficient for multiple polylines
    from rapidgeo.polyline import encode_batch
    
    multiple_tracks = [track1, track2, track3, track4]
    
    # Process all at once
    encoded_tracks = encode_batch(multiple_tracks, precision=5)

Preprocess Your Data
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Remove duplicate consecutive points
    def remove_duplicates(track):
        if not track:
            return track
        result = [track[0]]
        for point in track[1:]:
            if point != result[-1]:
                result.append(point)
        return result
    
    cleaned_track = remove_duplicates(noisy_gps_track)

Choose Appropriate Tolerances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # For different simplification needs:
    
    # Remove GPS noise but keep detail
    cleaned = douglas_peucker(track, tolerance_m=2.0)
    
    # Significant simplification for overview maps
    overview = douglas_peucker(track, tolerance_m=50.0)
    
    # Aggressive simplification for thumbnails
    thumbnail = douglas_peucker(track, tolerance_m=200.0)

Next Steps
----------

* Read the :doc:`distance` guide for detailed distance calculation options
* See :doc:`polyline` for advanced polyline encoding features  
* Check :doc:`simplify` for line simplification techniques
* Explore :doc:`similarity` for comparing GPS tracks and routes
* Review :doc:`performance` for optimization strategies
* 