Distance Calculations
=====================

The ``rapidgeo.distance`` module provides fast geographic and planar distance calculations with multiple algorithms optimized for different use cases.

Core Types
----------

.. autoclass:: rapidgeo.distance.LngLat
   :members:
   :no-index:

The ``LngLat`` type represents a coordinate pair in longitude, latitude order. All rapidgeo functions use this consistent ordering.

Example:

.. code-block:: python

    from rapidgeo.distance import LngLat
    
    # Create coordinates (longitude first, latitude second)
    sf = LngLat.new_deg(-122.4194, 37.7749)   # San Francisco
    nyc = LngLat.new_deg(-74.0060, 40.7128)   # New York City
    
    print(f"San Francisco: {sf}")
    print(f"New York City: {nyc}")

Geographic Distance
-------------------

Geographic distances calculate the actual distance between points on Earth's surface.

.. automodule:: rapidgeo.distance.geo
   :members:
   :no-index:

**Algorithm Selection:**

* **Haversine**: Good for most applications. Assumes spherical Earth.
* **Vincenty**: Higher precision using ellipsoidal Earth model. More computation required.

Example:

.. code-block:: python

    from rapidgeo.distance import LngLat
    from rapidgeo.distance.geo import haversine, vincenty_distance
    
    sf = LngLat.new_deg(-122.4194, 37.7749)
    nyc = LngLat.new_deg(-74.0060, 40.7128)
    
    # Fast, good accuracy for most use cases
    distance = haversine(sf, nyc)
    print(f"Haversine: {distance / 1000:.1f} km")
    
    # High precision, slower
    precise = vincenty_distance(sf, nyc)
    print(f"Vincenty: {precise / 1000:.3f} km")

Planar Distance
---------------

Planar distances treat coordinates as points on a flat plane, ignoring Earth's curvature.

.. automodule:: rapidgeo.distance.euclid
   :members:
   :no-index:

**Use Cases:**

* Comparing distances when you only need relative ordering
* Small geographic areas where Earth's curvature doesn't matter
* Point-to-line segment calculations

Example:

.. code-block:: python

    from rapidgeo.distance import LngLat
    from rapidgeo.distance.euclid import euclid, squared, point_to_segment
    
    p1 = LngLat.new_deg(-122.0, 37.0)
    p2 = LngLat.new_deg(-121.0, 37.0)
    
    # Euclidean distance in degrees
    distance = euclid(p1, p2)
    print(f"Euclidean: {distance:.6f} degrees")
    
    # Squared distance (faster, avoid sqrt)
    distance_sq = squared(p1, p2)
    print(f"Squared: {distance_sq:.6f} degrees²")
    
    # Distance from point to line segment
    point = LngLat.new_deg(-121.5, 37.1)
    seg_distance = point_to_segment(point, p1, p2)
    print(f"Point to segment: {seg_distance:.6f} degrees")

Batch Operations
----------------

Batch operations process multiple coordinates efficiently.

.. automodule:: rapidgeo.distance.batch
   :members:
   :no-index:

Example:

.. code-block:: python

    from rapidgeo.distance import LngLat
    from rapidgeo.distance.batch import pairwise_haversine, path_length_haversine
    
    # Define a path
    path = [
        LngLat.new_deg(-122.4194, 37.7749),  # San Francisco
        LngLat.new_deg(-87.6298, 41.8781),   # Chicago
        LngLat.new_deg(-74.0060, 40.7128),   # New York City
    ]
    
    # Calculate distances between consecutive points
    distances = list(pairwise_haversine(path))
    print(f"Segment distances: {[d/1000 for d in distances]} km")
    
    # Calculate total path length
    total_length = path_length_haversine(path)
    print(f"Total path length: {total_length / 1000:.1f} km")

NumPy Integration
-----------------

When rapidgeo is compiled with NumPy support, additional functions are available for processing NumPy arrays efficiently.

.. code-block:: python

    # Check if NumPy support is available
    try:
        from rapidgeo.distance import numpy as rgeo_numpy
        print("NumPy support available")
    except ImportError:
        print("NumPy support not available")

Performance Notes
-----------------

**Algorithm Speed:**

* Euclidean is fastest
* Haversine is moderate speed
* Vincenty takes the most computation

**Batch Operations:**

* Process multiple items at once for better efficiency
* Iterator-based results to manage memory

**Memory:**

* Each ``LngLat`` uses 16 bytes (two 64-bit floats)
* Batch operations don't duplicate input data unnecessarily