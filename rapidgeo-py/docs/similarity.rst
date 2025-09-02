Curve Similarity
================

The ``rapidgeo.similarity`` module provides algorithms for measuring the similarity between two polygonal curves. These algorithms are essential for comparing trajectories, GPS tracks, and other geographic paths.

Overview
--------

Two primary similarity measures are available:

**Fréchet Distance:**
  Considers the ordering of points along each curve. Often described as the "dog walking" distance - imagine a person walking their dog, each along their own path. The Fréchet distance is the shortest leash length that allows both to complete their walks.

**Hausdorff Distance:**
  Measures the maximum distance from any point in one curve to the closest point in the other curve. Order-independent and symmetric.

All functions work with sequences of ``LngLat`` coordinates and return distances in meters.

Core Modules
------------

.. automodule:: rapidgeo.similarity.frechet
   :members:
   :no-index:

.. automodule:: rapidgeo.similarity.hausdorff
   :members:
   :no-index:

Fréchet Distance
----------------

The discrete Fréchet distance is ideal for comparing trajectories where point order matters, such as GPS tracks or movement paths.

Basic Usage:

.. code-block:: python

    from rapidgeo.distance import LngLat
    from rapidgeo.similarity.frechet import discrete_frechet
    
    # Two similar GPS tracks with slight variations
    track_a = [
        LngLat.new_deg(-122.4194, 37.7749),  # San Francisco
        LngLat.new_deg(-122.4180, 37.7755),  # Slight detour
        LngLat.new_deg(-122.4000, 37.7800),  # Continuing north
        LngLat.new_deg(-122.3900, 37.7850),  # End point
    ]
    
    track_b = [
        LngLat.new_deg(-122.4194, 37.7749),  # Same start
        LngLat.new_deg(-122.4170, 37.7760),  # Different route
        LngLat.new_deg(-122.4010, 37.7790),  # Close to track_a
        LngLat.new_deg(-122.3900, 37.7850),  # Same end
    ]
    
    # Calculate similarity
    distance = discrete_frechet(track_a, track_b)
    print(f"Fréchet distance: {distance:.1f} meters")

**With Early Termination:**

For performance when you only need to know if curves are similar within a threshold:

.. code-block:: python

    from rapidgeo.similarity.frechet import discrete_frechet_with_threshold
    
    threshold = 50.0  # 50 meters
    distance = discrete_frechet_with_threshold(track_a, track_b, threshold)
    
    if distance <= threshold:
        print("Tracks are similar within 50 meters")
    else:
        print(f"Tracks differ by at least {distance:.1f} meters")

Hausdorff Distance
------------------

The Hausdorff distance is useful when point order doesn't matter, such as comparing building outlines or coastline segments.

Basic Usage:

.. code-block:: python

    from rapidgeo.similarity.hausdorff import hausdorff
    
    # Two representations of the same geographic feature
    coastline_detailed = [
        LngLat.new_deg(-123.0, 37.0),
        LngLat.new_deg(-122.9, 37.1),
        LngLat.new_deg(-122.8, 37.0),
        LngLat.new_deg(-122.7, 37.1),
        LngLat.new_deg(-122.6, 37.0),
    ]
    
    coastline_simplified = [
        LngLat.new_deg(-123.0, 37.0),
        LngLat.new_deg(-122.8, 37.05),  # Simplified curve
        LngLat.new_deg(-122.6, 37.0),
    ]
    
    # Calculate maximum deviation
    max_distance = hausdorff(coastline_detailed, coastline_simplified)
    print(f"Maximum deviation: {max_distance:.1f} meters")

**With Early Termination:**

.. code-block:: python

    from rapidgeo.similarity.hausdorff import hausdorff_with_threshold
    
    threshold = 100.0  # 100 meters
    distance = hausdorff_with_threshold(coastline_detailed, coastline_simplified, threshold)
    
    if distance <= threshold:
        print("Simplified coastline is within 100 meters of original")
    else:
        print(f"Simplification error exceeds {threshold} meters")

Practical Applications
----------------------

**GPS Track Comparison:**

Compare recorded GPS tracks to detect similar routes:

.. code-block:: python

    def are_routes_similar(route1, route2, tolerance_m=25.0):
        """Check if two GPS routes are similar within tolerance."""
        distance = discrete_frechet_with_threshold(route1, route2, tolerance_m)
        return distance <= tolerance_m
    
    # Example usage
    morning_commute = [LngLat.new_deg(-122.4, 37.7), LngLat.new_deg(-122.3, 37.8)]
    evening_commute = [LngLat.new_deg(-122.4, 37.7), LngLat.new_deg(-122.3, 37.8)]
    
    if are_routes_similar(morning_commute, evening_commute):
        print("Similar commute pattern detected")

**Map Data Quality Assessment:**

Check how well simplified data represents original features:

.. code-block:: python

    def assess_simplification_quality(original, simplified, max_error_m=10.0):
        """Assess quality of line simplification."""
        error = hausdorff(original, simplified)
        
        if error <= max_error_m:
            return "Excellent quality"
        elif error <= max_error_m * 2:
            return "Good quality"
        elif error <= max_error_m * 5:
            return "Acceptable quality"
        else:
            return "Poor quality"
    
    # Example usage
    detailed_border = [/* many points */]
    simplified_border = [/* fewer points */]
    
    quality = assess_simplification_quality(detailed_border, simplified_border)
    print(f"Simplification quality: {quality}")

**Trajectory Clustering:**

Group similar trajectories using Fréchet distance:

.. code-block:: python

    def find_similar_trajectories(trajectories, reference, threshold_m=50.0):
        """Find all trajectories similar to a reference trajectory."""
        similar = []
        
        for i, trajectory in enumerate(trajectories):
            distance = discrete_frechet_with_threshold(reference, trajectory, threshold_m)
            if distance <= threshold_m:
                similar.append((i, trajectory, distance))
        
        # Sort by similarity (ascending distance)
        similar.sort(key=lambda x: x[2])
        return similar
    
    # Example usage
    all_routes = [route1, route2, route3, route4]
    reference_route = route1
    
    similar_routes = find_similar_trajectories(all_routes, reference_route)
    print(f"Found {len(similar_routes)} similar routes")

Algorithm Comparison
--------------------

**When to Use Fréchet Distance:**

* Comparing GPS tracks or trajectories
* Order of points matters (start → end sequence)
* Want to find similar paths or routes
* Temporal data where sequence is important

**When to Use Hausdorff Distance:**

* Comparing shapes or outlines 
* Order of points doesn't matter
* Assessing simplification quality
* Measuring maximum deviation between curves

**Algorithm Complexity:**

* **Fréchet**: Requires storing intermediate results, memory usage grows with input size
* **Hausdorff**: Uses constant memory regardless of input size
* Both algorithms compare each point in one curve to points in the other curve

Security Limits
---------------

To prevent memory exhaustion, the following limits are enforced:

**Fréchet Distance:**
* Maximum 10,000 points per curve
* Memory usage proportional to n × m

**Hausdorff Distance:**
* Maximum 50,000 points per curve  
* Constant memory usage regardless of input size

Attempting to process curves exceeding these limits will raise a ``ValueError``.

**Example:**

.. code-block:: python

    # This will raise ValueError: curve size limited
    huge_curve = [LngLat.new_deg(i*0.001, 0) for i in range(15000)]
    small_curve = [LngLat.new_deg(0, 0)]
    
    try:
        distance = discrete_frechet(huge_curve, small_curve)
    except ValueError as e:
        print(f"Error: {e}")

Error Conditions
----------------

Common errors and their causes:

**ValueError: "Empty input curves"**
  One or both input curves contain no points.

**ValueError: "curve size limited"**  
  Input curves exceed security limits (see above).

**TypeError**
  Input is not a sequence of LngLat objects.

**Example Error Handling:**

.. code-block:: python

    def safe_frechet_distance(curve1, curve2):
        """Calculate Fréchet distance with error handling."""
        try:
            return discrete_frechet(curve1, curve2)
        except ValueError as e:
            if "Empty input" in str(e):
                print("Error: Cannot compare empty curves")
            elif "size limited" in str(e):
                print("Error: Input curves too large")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

Tips and Best Practices
-----------------------

**Preprocessing:**

.. code-block:: python

    # Remove duplicate consecutive points to improve performance
    def remove_duplicates(curve):
        if not curve:
            return curve
        
        result = [curve[0]]
        for point in curve[1:]:
            if point != result[-1]:  # LngLat implements __eq__
                result.append(point)
        return result
    
    cleaned_curve = remove_duplicates(noisy_gps_track)

**Performance Optimization:**

.. code-block:: python

    # Use threshold variants when possible
    threshold = 100.0  # meters
    
    # Faster - early termination
    distance = discrete_frechet_with_threshold(curve1, curve2, threshold)
    
    # Slower - always computes full distance  
    distance = discrete_frechet(curve1, curve2)

**Coordinate Precision:**

.. code-block:: python

    # High precision for small-scale comparisons
    building_corner_a = LngLat.new_deg(-122.419400, 37.774900)
    building_corner_b = LngLat.new_deg(-122.419401, 37.774901)
    
    # Lower precision acceptable for large-scale comparisons
    city_a = LngLat.new_deg(-122.42, 37.77)
    city_b = LngLat.new_deg(-122.41, 37.78)