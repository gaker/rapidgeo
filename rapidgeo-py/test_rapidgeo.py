#!/usr/bin/env python3

import rapidgeo
from rapidgeo.distance import LngLat
from rapidgeo.distance.geo import haversine, vincenty_distance
from rapidgeo.distance.euclid import euclid, squared, point_to_segment, point_to_segment_squared
from rapidgeo.distance.batch import pairwise_haversine, path_length_haversine

def test_basic_functionality():
    print(f"rapidgeo version: {rapidgeo.__version__}")
    
    sf = LngLat(-122.4194, 37.7749)
    nyc = LngLat(-74.0060, 40.7128)
    
    print(f"San Francisco: {sf}")
    print(f"New York City: {nyc}")
    
    # Test haversine distance
    distance_haversine = haversine(sf, nyc)
    print(f"Haversine distance SF->NYC: {distance_haversine:.2f} meters")
    
    # Test vincenty distance
    try:
        distance_vincenty = vincenty_distance(sf, nyc)
        print(f"Vincenty distance SF->NYC: {distance_vincenty:.2f} meters")
    except ValueError as e:
        print(f"Vincenty failed: {e}")
    
    # Test euclidean distance
    distance_euclidean = euclid(sf, nyc)
    print(f"Euclidean distance SF->NYC: {distance_euclidean:.6f} degrees")
    
    # Test squared distance
    distance_squared = squared(sf, nyc)
    print(f"Squared distance SF->NYC: {distance_squared:.6f} degrees²")
    
    # Test point to segment
    p1 = LngLat(-122.0, 37.0)
    p2 = LngLat(-121.0, 37.0)
    point = LngLat(-121.5, 37.01)
    
    segment_distance = point_to_segment(point, p1, p2)
    print(f"Point to segment distance: {segment_distance:.6f} degrees")
    
    segment_distance_sq = point_to_segment_squared(point, p1, p2)
    print(f"Point to segment squared distance: {segment_distance_sq:.6f} degrees²")
    
    # Test batch operations
    points = [
        LngLat(0.0, 0.0),
        LngLat(1.0, 0.0),
        LngLat(1.0, 1.0),
        LngLat(0.0, 1.0)
    ]
    
    # Test pairwise distances
    pairwise_distances = pairwise_haversine(points)
    print(f"Pairwise distances: {[f'{d:.2f}' for d in pairwise_distances]}")
    
    # Test path length
    total_length = path_length_haversine(points)
    print(f"Total path length: {total_length:.2f} meters")

if __name__ == "__main__":
    test_basic_functionality()