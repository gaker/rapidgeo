rapidgeo documentation
======================

Geographic calculations for Python.

rapidgeo provides distance calculations, polyline encoding/decoding, simplification, and similarity measures for coordinate data. Built with Rust bindings for reliable performance.

Installation
------------

.. code-block:: bash

    pip install rapidgeo          # Base package
    pip install rapidgeo[numpy]   # With NumPy support

Quick Start
-----------

.. code-block:: python

    from rapidgeo.distance import LngLat
    from rapidgeo.distance.geo import haversine, vincenty_distance
    
    # Create coordinates (longitude, latitude)
    sf = LngLat.new_deg(-122.4194, 37.7749)   # San Francisco
    nyc = LngLat.new_deg(-74.0060, 40.7128)   # New York City
    
    # Calculate distance using Haversine formula
    distance = haversine(sf, nyc)
    print(f"Distance: {distance / 1000:.1f} km")  # ~4,130 km

What it does
------------

* Distance calculations with multiple algorithms (Haversine, Vincenty, Euclidean)
* Google Polyline encoding and decoding
* Line simplification using Douglas-Peucker algorithm  
* Curve similarity measures (Fréchet and Hausdorff distances)
* Batch operations for processing multiple datasets
* Optional NumPy integration

Coordinate System
-----------------

All coordinates in rapidgeo use **longitude, latitude** ordering (lng, lat):

.. code-block:: python

    # Correct
    point = LngLat.new_deg(-122.4194, 37.7749)  # lng first, lat second
    
    # Common mistake
    # point = LngLat.new_deg(37.7749, -122.4194)  # lat, lng - WRONG

Contents
--------

.. toctree::
   :maxdepth: 2
   
   getting-started
   formats
   distance
   polyline
   simplify
   similarity
   examples
   performance
   troubleshooting
   api

License
-------

Licensed under either of Apache License, Version 2.0 or MIT License at your option.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`