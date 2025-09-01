rapidgeo documentation
======================

Fast geographic and planar distance calculations for Python.

rapidgeo is a Python library built with Rust that provides high-performance geographic and planar distance calculations. It supports various distance algorithms including Haversine, Vincenty, and Euclidean distance calculations, along with polyline encoding/decoding, Douglas-Peucker simplification, and curve similarity measures.

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

Key Features
------------

* **Fast**: Rust-based implementation for maximum performance
* **Accurate**: Multiple distance algorithms for different precision needs
* **Flexible**: Support for batch operations and NumPy arrays
* **Complete**: Distance calculation, polyline encoding, simplification, and similarity measures

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
   
   distance
   polyline
   simplify
   similarity
   performance
   api

License
-------

Licensed under either of Apache License, Version 2.0 or MIT License at your option.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`