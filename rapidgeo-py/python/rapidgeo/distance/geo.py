"""
Geographic distance functions
"""

from .._rapidgeo import distance

# Re-export functions from the compiled module
haversine = distance.geo.haversine
haversine_km = distance.geo.haversine_km
haversine_miles = distance.geo.haversine_miles
haversine_nautical = distance.geo.haversine_nautical
bearing = distance.geo.bearing
destination = distance.geo.destination
vincenty_distance = distance.geo.vincenty_distance

__all__ = [
    "haversine",
    "haversine_km", 
    "haversine_miles",
    "haversine_nautical",
    "bearing",
    "destination",
    "vincenty_distance",
]
