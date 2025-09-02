Examples
========

Real-world examples showing how to use rapidgeo for common tasks.

GPS Track Analysis
------------------

Analyze a Day's Walking Route
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from rapidgeo import LngLat
    from rapidgeo.distance.batch import path_length_haversine, pairwise_haversine
    from rapidgeo.simplify import douglas_peucker
    from rapidgeo.polyline import encode_simplified
    
    # GPS track from a morning walk (recorded every 5 seconds)
    morning_walk = [
        LngLat(-122.4194, 37.7749),  # Start: home
        LngLat(-122.4190, 37.7752),  # Down the street
        LngLat(-122.4180, 37.7760),  # Turn corner
        LngLat(-122.4170, 37.7765),  # Keep walking
        LngLat(-122.4160, 37.7770),  # Approach park
        LngLat(-122.4150, 37.7780),  # Into park
        LngLat(-122.4140, 37.7785),  # Walk through park
        LngLat(-122.4130, 37.7790),  # Loop back
        LngLat(-122.4140, 37.7785),  # Return path
        LngLat(-122.4150, 37.7780),  # Exit park
        LngLat(-122.4160, 37.7770),  # Head home
        LngLat(-122.4180, 37.7760),  # Back corner
        LngLat(-122.4194, 37.7749),  # End: home
    ]
    
    # Basic statistics
    total_distance = path_length_haversine(morning_walk)
    print(f"Total distance walked: {total_distance/1000:.2f} km")
    print(f"GPS points recorded: {len(morning_walk)}")
    
    # Find longest single segment
    segments = list(pairwise_haversine(morning_walk))
    max_segment = max(segments)
    max_index = segments.index(max_segment)
    print(f"Longest segment: {max_segment:.0f}m (between points {max_index} and {max_index+1})")
    
    # Simplify for storage (remove GPS noise)
    simplified = douglas_peucker(morning_walk, tolerance_m=5.0)
    print(f"After simplification: {len(simplified)} points ({len(simplified)/len(morning_walk)*100:.1f}% of original)")
    
    # Encode for database storage
    encoded = encode_simplified(morning_walk, tolerance_m=5.0, precision=5)
    print(f"Encoded polyline: {len(encoded)} characters")

Delivery Route Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from rapidgeo.distance.geo import haversine
    
    # Delivery depot and customer locations
    depot = LngLat(-122.4000, 37.7800)
    customers = [
        ("Alice", LngLat(-122.3950, 37.7820)),
        ("Bob", LngLat(-122.4050, 37.7780)),
        ("Carol", LngLat(-122.3980, 37.7850)),
        ("Dave", LngLat(-122.4020, 37.7760)),
    ]
    
    # Calculate distances from depot to each customer
    distances = []
    for name, location in customers:
        distance = haversine(depot, location)
        distances.append((name, distance, location))
    
    # Sort by distance (simple nearest-first routing)
    distances.sort(key=lambda x: x[1])
    
    print("Delivery order (nearest first):")
    for i, (name, distance, location) in enumerate(distances, 1):
        print(f"{i}. {name}: {distance:.0f}m from depot")
    
    # Calculate total route distance
    route = [depot] + [location for _, _, location in distances] + [depot]
    total_route_distance = path_length_haversine(route)
    print(f"\nTotal route distance: {total_route_distance/1000:.1f} km")

Map Data Processing
-------------------

Simplify Coastline Data by Zoom Level
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Detailed coastline data (many points)
    detailed_coastline = [
        LngLat(-123.0000, 37.0000),
        LngLat(-122.9950, 37.0100),
        LngLat(-122.9900, 37.0080),
        LngLat(-122.9850, 37.0120),
        # ... imagine hundreds more points
        LngLat(-122.5000, 37.3000),
    ]
    
    # Different simplification levels for different map zoom levels
    zoom_configs = {
        'world': {'tolerance': 5000.0, 'description': 'World view'},
        'country': {'tolerance': 1000.0, 'description': 'Country view'},  
        'state': {'tolerance': 200.0, 'description': 'State view'},
        'city': {'tolerance': 50.0, 'description': 'City view'},
        'street': {'tolerance': 10.0, 'description': 'Street view'},
    }
    
    print(f"Original coastline: {len(detailed_coastline)} points")
    
    simplified_coastlines = {}
    for zoom_level, config in zoom_configs.items():
        simplified = douglas_peucker(detailed_coastline, tolerance_m=config['tolerance'])
        simplified_coastlines[zoom_level] = simplified
        
        reduction = (1 - len(simplified) / len(detailed_coastline)) * 100
        print(f"{config['description']}: {len(simplified)} points ({reduction:.1f}% reduction)")
    
    # Encode each level for storage
    for zoom_level, coastline in simplified_coastlines.items():
        encoded = encode(coastline, precision=5)
        print(f"{zoom_level} encoded: {len(encoded)} characters")

Building Outline Simplification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Detailed building outline (from high-resolution satellite data)
    building_outline = [
        LngLat(-122.4100, 37.7800),  # Northwest corner
        LngLat(-122.4100, 37.7801),  # Small detail
        LngLat(-122.4099, 37.7801),  # More detail
        LngLat(-122.4090, 37.7801),  # North wall
        LngLat(-122.4090, 37.7790),  # Northeast corner
        LngLat(-122.4100, 37.7790),  # East wall
        LngLat(-122.4100, 37.7800),  # Back to start
    ]
    
    print(f"Detailed outline: {len(building_outline)} points")
    
    # Simplify for different uses
    uses = [
        ('property_records', 1.0),      # High accuracy for legal documents
        ('navigation', 5.0),            # Navigation apps
        ('overview_map', 10.0),         # Small-scale overview maps
    ]
    
    for use_case, tolerance in uses:
        simplified = douglas_peucker(building_outline, tolerance_m=tolerance)
        print(f"{use_case} ({tolerance}m tolerance): {len(simplified)} points")

Route Similarity Analysis
-------------------------

Compare Commute Routes
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from rapidgeo.similarity.frechet import discrete_frechet_with_threshold
    
    # Two different routes to work
    route_highway = [
        LngLat(-122.4194, 37.7749),  # Home
        LngLat(-122.4000, 37.7600),  # Get on highway
        LngLat(-122.3500, 37.7400),  # Highway section
        LngLat(-122.3000, 37.7200),  # Exit highway
        LngLat(-122.2800, 37.7100),  # Work
    ]
    
    route_surface_streets = [
        LngLat(-122.4194, 37.7749),  # Home (same start)
        LngLat(-122.4100, 37.7700),  # Surface streets
        LngLat(-122.3900, 37.7600),  # Continue on streets
        LngLat(-122.3200, 37.7300),  # More streets
        LngLat(-122.2800, 37.7100),  # Work (same end)
    ]
    
    # Calculate how different the routes are
    max_deviation = discrete_frechet(route_highway, route_surface_streets)
    print(f"Maximum deviation between routes: {max_deviation:.0f} meters")
    
    # Check if routes are similar within tolerance
    tolerance = 200.0  # 200 meters
    similarity = discrete_frechet_with_threshold(route_highway, route_surface_streets, tolerance)
    
    if similarity <= tolerance:
        print(f"Routes are similar (within {tolerance}m)")
    else:
        print(f"Routes are quite different (exceed {tolerance}m threshold)")

Detect Route Variations
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Same route taken on different days with variations
    monday_route = [
        LngLat(-122.4194, 37.7749),
        LngLat(-122.4100, 37.7800),
        LngLat(-122.4000, 37.7850),
        LngLat(-122.3900, 37.7900),
    ]
    
    tuesday_route = [
        LngLat(-122.4194, 37.7749),  # Same start
        LngLat(-122.4090, 37.7810),  # Slight detour
        LngLat(-122.4010, 37.7840),  # Different path
        LngLat(-122.3900, 37.7900),  # Same end
    ]
    
    wednesday_route = [
        LngLat(-122.4194, 37.7749),  # Same start
        LngLat(-122.4200, 37.7820),  # Major detour
        LngLat(-122.4150, 37.7870),  # Different route
        LngLat(-122.3900, 37.7900),  # Same end
    ]
    
    routes = [("Monday", monday_route), ("Tuesday", tuesday_route), ("Wednesday", wednesday_route)]
    
    # Compare all routes to Monday baseline
    baseline = monday_route
    threshold = 50.0  # 50 meter threshold for "similar" routes
    
    print("Route similarity analysis:")
    for day, route in routes[1:]:  # Skip Monday (baseline)
        similarity = discrete_frechet_with_threshold(baseline, route, threshold)
        
        if similarity <= threshold:
            status = "similar"
        else:
            status = "different"
            
        print(f"{day} vs Monday: {similarity:.0f}m ({status})")

Data Format Conversion
----------------------

Handle Mixed Coordinate Formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from rapidgeo.formats import coords_to_lnglat
    from rapidgeo import LngLat
    
    def handle_mixed_coordinate_data():
        """Example of handling coordinates from different sources."""
        
        # GPS data from mobile app (lat, lng order)
        mobile_data = [
            (37.7749, -122.4194),  # San Francisco
            (40.7128, -74.0060),   # New York  
            (41.8781, -87.6298),   # Chicago
        ]
        
        # API response with GeoJSON format
        api_response = [
            {"coordinates": [-122.4194, 37.7749]},  # San Francisco (lng, lat)
            {"coordinates": [-74.0060, 40.7128]},   # New York
            {"coordinates": [-87.6298, 41.8781]},   # Chicago
        ]
        
        # Database export as flat array
        database_coords = [
            -122.4194, 37.7749,  # San Francisco  
            -74.0060, 40.7128,   # New York
            -87.6298, 41.8781,   # Chicago
        ]
        
        # All formats automatically converted to standardized LngLat
        mobile_coords = coords_to_lnglat(mobile_data)      # Detects lat,lng and swaps
        api_coords = coords_to_lnglat(api_response)        # Uses GeoJSON format
        db_coords = coords_to_lnglat(database_coords)      # Treats as flat array
        
        print("Mobile app coordinates:")
        for coord in mobile_coords:
            print(f"  {coord.lng:.4f}, {coord.lat:.4f}")
            
        print("API coordinates:")  
        for coord in api_coords:
            print(f"  {coord.lng:.4f}, {coord.lat:.4f}")
            
        print("Database coordinates:")
        for coord in db_coords:
            print(f"  {coord.lng:.4f}, {coord.lat:.4f}")
        
        # All should produce identical results
        assert mobile_coords[0].lng == api_coords[0].lng == db_coords[0].lng
        assert mobile_coords[0].lat == api_coords[0].lat == db_coords[0].lat

Convert Between Different Coordinate Formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # From CSV data
    import csv
    
    def load_track_from_csv(filename):
        """Load GPS track from CSV file."""
        track = []
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                lng = float(row['longitude'])
                lat = float(row['latitude'])
                track.append(LngLat(lng, lat))
        return track
    
    # From GeoJSON
    import json
    
    def load_track_from_geojson(filename):
        """Load GPS track from GeoJSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        coordinates = data['features'][0]['geometry']['coordinates']
        track = [LngLat(lng, lat) for lng, lat in coordinates]
        return track
    
    # To polyline format
    def save_track_as_polyline(track, filename, precision=5):
        """Save track as polyline string."""
        encoded = encode(track, precision=precision)
        with open(filename, 'w') as f:
            f.write(encoded)
        print(f"Saved {len(track)} points as {len(encoded)} character polyline")

Batch Processing Multiple GPS Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import os
    from rapidgeo.polyline import encode_batch
    from rapidgeo.simplify.batch import simplify_multiple
    
    def process_gps_directory(input_dir, output_dir):
        """Process all GPS track files in a directory."""
        
        # Load all tracks
        tracks = []
        filenames = []
        
        for filename in os.listdir(input_dir):
            if filename.endswith('.csv'):
                filepath = os.path.join(input_dir, filename)
                track = load_track_from_csv(filepath)
                if len(track) >= 2:  # Valid track
                    tracks.append(track)
                    filenames.append(filename.replace('.csv', ''))
        
        print(f"Loaded {len(tracks)} tracks")
        
        # Batch simplify all tracks
        simplified_tracks = simplify_multiple(tracks, tolerance_m=10.0)
        print(f"Simplified tracks (10m tolerance)")
        
        # Batch encode all tracks
        encoded_tracks = encode_batch(simplified_tracks, precision=5)
        print(f"Encoded all tracks")
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        for filename, encoded in zip(filenames, encoded_tracks):
            output_path = os.path.join(output_dir, f"{filename}_simplified.polyline")
            with open(output_path, 'w') as f:
                f.write(encoded)
        
        print(f"Saved {len(encoded_tracks)} processed tracks to {output_dir}")
    
    # Usage
    # process_gps_directory("raw_gps_data/", "processed_polylines/")

Quality Assessment
------------------

Measure GPS Track Quality
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def assess_gps_track_quality(track):
        """Assess the quality of a GPS track."""
        if len(track) < 2:
            return {"status": "invalid", "reason": "too few points"}
        
        # Check for obvious errors
        segments = list(pairwise_haversine(track))
        
        # Look for impossibly fast movement (assuming walking/driving)
        max_reasonable_speed = 50.0  # 50 m/s = 180 km/h
        time_between_points = 5.0    # Assume 5 seconds between GPS readings
        max_reasonable_distance = max_reasonable_speed * time_between_points
        
        speed_violations = sum(1 for d in segments if d > max_reasonable_distance)
        
        # Look for stationary periods (GPS noise)
        noise_threshold = 2.0  # 2 meters
        stationary_points = sum(1 for d in segments if d < noise_threshold)
        
        total_distance = sum(segments)
        
        return {
            "total_points": len(track),
            "total_distance_km": total_distance / 1000,
            "average_segment_m": total_distance / len(segments),
            "speed_violations": speed_violations,
            "stationary_points": stationary_points,
            "noise_ratio": stationary_points / len(segments),
            "status": "good" if speed_violations == 0 else "questionable"
        }
    
    # Example usage
    sample_track = [
        LngLat(-122.4194, 37.7749),
        LngLat(-122.4194, 37.7750),  # 1m movement (might be noise)
        LngLat(-122.4180, 37.7760),  # Normal movement
        LngLat(-122.4000, 37.8000),  # Large jump (might be error)
    ]
    
    quality = assess_gps_track_quality(sample_track)
    print(f"Track quality assessment: {quality}")

Compare Original vs Simplified Tracks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from rapidgeo.similarity.hausdorff import hausdorff
    
    def evaluate_simplification(original_track, tolerance_m):
        """Evaluate the effect of different simplification tolerances."""
        
        simplified = douglas_peucker(original_track, tolerance_m=tolerance_m)
        
        # Calculate metrics
        original_length = path_length_haversine(original_track)
        simplified_length = path_length_haversine(simplified)
        
        # Measure maximum deviation
        max_deviation = hausdorff(original_track, simplified)
        
        # Calculate reductions
        point_reduction = (1 - len(simplified) / len(original_track)) * 100
        length_error = abs(simplified_length - original_length) / original_length * 100
        
        return {
            "tolerance_m": tolerance_m,
            "original_points": len(original_track),
            "simplified_points": len(simplified),
            "point_reduction_pct": point_reduction,
            "length_error_pct": length_error,
            "max_deviation_m": max_deviation,
            "original_length_km": original_length / 1000,
            "simplified_length_km": simplified_length / 1000,
        }
    
    # Test different tolerance levels
    sample_track = [LngLat(-122.4 + i*0.001, 37.7 + i*0.001) for i in range(50)]
    
    print("Simplification analysis:")
    for tolerance in [1, 5, 10, 25, 50, 100]:
        results = evaluate_simplification(sample_track, tolerance)
        print(f"Tolerance {tolerance}m: {results['point_reduction_pct']:.1f}% fewer points, "
              f"{results['length_error_pct']:.2f}% length error, "
              f"max deviation {results['max_deviation_m']:.1f}m")
