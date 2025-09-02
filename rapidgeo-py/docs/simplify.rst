Line Simplification
===================

The ``rapidgeo.simplify`` module provides Douglas-Peucker line simplification with multiple distance calculation methods.

Line simplification reduces the number of points in a polyline while preserving its essential shape. This is useful for reducing data size, improving rendering performance, and eliminating noise from GPS tracks.

Core Functions
--------------

.. automodule:: rapidgeo.simplify
   :members:
   :no-index:

Basic Usage
-----------

.. code-block:: python

    from rapidgeo.distance import LngLat
    from rapidgeo.simplify import douglas_peucker
    
    # Create a polyline with redundant points
    original_path = [
        LngLat.new_deg(-122.4194, 37.7749),  # San Francisco
        LngLat.new_deg(-122.4180, 37.7755),  # Close intermediate point
        LngLat.new_deg(-122.4160, 37.7760),  # Another close point  
        LngLat.new_deg(-122.4140, 37.7765),  # Yet another close point
        LngLat.new_deg(-87.6298, 41.8781),   # Chicago (far point - will be kept)
        LngLat.new_deg(-74.0060, 40.7128),   # New York City
    ]
    
    print(f"Original: {len(original_path)} points")
    
    # Simplify with 1km tolerance
    simplified = douglas_peucker(original_path, tolerance_m=1000.0)
    print(f"Simplified: {len(simplified)} points")
    
    # Show which points were kept
    mask = douglas_peucker(original_path, tolerance_m=1000.0, return_mask=True)
    print(f"Points kept: {mask}")

Distance Methods
----------------

The Douglas-Peucker algorithm can use different distance calculation methods:

**great_circle** (default):
  Uses geodesic distance (Haversine). Best for geographic data covering large areas.

**planar**:
  Uses planar distance calculations. Faster but less accurate for large geographic areas.

**euclidean**:
  Uses simple Euclidean distance. Fastest but only suitable for small areas or projected coordinates.

.. code-block:: python

    # Test different methods
    for method in ['great_circle', 'planar', 'euclidean']:
        simplified = douglas_peucker(
            original_path, 
            tolerance_m=1000.0, 
            method=method
        )
        print(f"{method}: {len(simplified)} points")

Batch Operations
----------------

Process multiple polylines efficiently:

.. automodule:: rapidgeo.simplify.batch
   :members:
   :no-index:

.. code-block:: python

    from rapidgeo.simplify.batch import simplify_multiple
    
    # Multiple paths to simplify
    paths = [
        [LngLat.new_deg(-122.4, 37.7), LngLat.new_deg(-122.3, 37.8), LngLat.new_deg(-122.2, 37.9)],
        [LngLat.new_deg(-74.0, 40.7), LngLat.new_deg(-74.1, 40.8), LngLat.new_deg(-74.2, 40.9)],
    ]
    
    # Batch simplification
    simplified_paths = simplify_multiple(paths, tolerance_m=100.0)
    print(f"Simplified {len(simplified_paths)} paths")
    
    # Get masks showing which points were kept
    masks = simplify_multiple(paths, tolerance_m=100.0, return_masks=True)
    for i, mask in enumerate(masks):
        print(f"Path {i} mask: {mask}")

Practical Examples
------------------

**GPS Track Cleaning:**

.. code-block:: python

    # Raw GPS track with noise
    gps_track = [
        LngLat.new_deg(-122.4194, 37.7749),
        LngLat.new_deg(-122.4195, 37.7750),  # GPS noise (1m away)
        LngLat.new_deg(-122.4193, 37.7748),  # GPS noise
        LngLat.new_deg(-122.4200, 37.7760),  # Significant movement
        LngLat.new_deg(-122.4210, 37.7770),  # Continue movement
    ]
    
    # Remove GPS noise with 5m tolerance
    cleaned_track = douglas_peucker(gps_track, tolerance_m=5.0, method='great_circle')
    print(f"Cleaned track: {len(gps_track)} -> {len(cleaned_track)} points")

**Map Rendering Optimization:**

.. code-block:: python

    # Detailed coastline data
    coastline = [/* many points */]
    
    # Simplify based on zoom level
    zoom_tolerances = {
        1: 10000.0,   # World view - 10km tolerance
        5: 1000.0,    # Country view - 1km tolerance  
        10: 100.0,    # City view - 100m tolerance
        15: 10.0,     # Street view - 10m tolerance
    }
    
    simplified_coastlines = {}
    for zoom, tolerance in zoom_tolerances.items():
        simplified = douglas_peucker(coastline, tolerance_m=tolerance)
        simplified_coastlines[zoom] = simplified
        print(f"Zoom {zoom}: {len(simplified)} points ({tolerance}m tolerance)")

**Data Compression:**

.. code-block:: python

    from rapidgeo.distance.batch import path_length_haversine
    
    # Original detailed path
    detailed_path = [/* many GPS points */]
    original_length = path_length_haversine(detailed_path)
    
    # Try different tolerance levels
    tolerances = [1.0, 5.0, 10.0, 50.0, 100.0]
    
    for tolerance in tolerances:
        simplified = douglas_peucker(detailed_path, tolerance_m=tolerance)
        simplified_length = path_length_haversine(simplified)
        
        point_reduction = (1 - len(simplified) / len(detailed_path)) * 100
        length_error = abs(simplified_length - original_length) / original_length * 100
        
        print(f"Tolerance {tolerance}m:")
        print(f"  Points: {len(detailed_path)} -> {len(simplified)} ({point_reduction:.1f}% reduction)")
        print(f"  Length error: {length_error:.2f}%")

Algorithm Details
-----------------

**Douglas-Peucker Algorithm:**

1. Draw a line between the first and last point
2. Find the point with the maximum distance from this line
3. If the maximum distance is greater than tolerance, split the line at that point
4. Recursively apply the algorithm to each segment
5. Return all points that were splitting points

**Distance Calculation:**

* **great_circle**: Point-to-line distance using great circle arcs
* **planar**: Point-to-line distance on flat plane (faster)
* **euclidean**: Simple Euclidean distance (fastest)

**Tolerance Units:**

* All tolerances are specified in meters (real-world distance)
* The algorithm converts between coordinate degrees and meters internally
* For projected coordinates, consider using 'euclidean' method with tolerance in coordinate units

Implementation Notes
--------------------

**Distance Methods:**

* **great_circle**: Most accurate for geographic data
* **planar**: Good balance of accuracy and computation
* **euclidean**: Fastest option

**Memory Usage:**

* Memory scales with input size
* Batch operations process one polyline at a time
* Optional masks require additional storage per polyline

**Typical Results:**

Results depend heavily on your data and tolerance:
* GPS tracks with noise: Often significant point reduction
* Smooth curves: Less reduction
* Detailed coastlines: Varies widely with tolerance

Choosing Tolerance
------------------

**GPS Tracks:**
* 1-5m: Remove GPS noise while preserving path detail
* 10-20m: Significant simplification for storage/transmission
* 50-100m: Aggressive simplification for overview displays

**Geographic Features:**
* Coastlines: 10-1000m depending on map scale
* Borders: 100-5000m depending on detail level
* Rivers: 5-100m depending on scale and importance

**Rule of thumb**: Start with tolerance = map_scale / 1000