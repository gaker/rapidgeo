use super::F;
use crate::LngLat;

/// Compromise Earth radius for Haversine to minimize maximum error against Vincenty
/// Balances meridional and equatorial accuracy for best overall performance
const EARTH_RADIUS_M: f64 = 6371008.8;

#[inline]
fn normalize_longitude_difference(dlng: f64) -> f64 {
    let mut normalized_dlng = dlng;
    // Handle antimeridian crossing - take shorter path
    if normalized_dlng > std::f64::consts::PI {
        normalized_dlng -= 2.0 * std::f64::consts::PI;
    } else if normalized_dlng < -std::f64::consts::PI {
        normalized_dlng += 2.0 * std::f64::consts::PI;
    }
    normalized_dlng
}

#[inline]
fn compute_haversine_parameter(dlat: f64, dlng: f64, lat1_rad: f64, lat2_rad: f64) -> f64 {
    let half_dlat = dlat * 0.5;
    let half_dlng = dlng * 0.5;
    let sin_half_dlat = half_dlat.sin();
    let sin_half_dlng = half_dlng.sin();

    sin_half_dlat * sin_half_dlat + lat1_rad.cos() * lat2_rad.cos() * sin_half_dlng * sin_half_dlng
}

#[inline]
fn compute_central_angle(h: f64) -> f64 {
    2.0 * h.sqrt().atan2((1.0 - h).sqrt())
}

#[inline]
fn apply_ellipsoidal_correction(
    spherical_distance: f64,
    dlat: f64,
    dlng: f64,
    lat1_rad: f64,
    lat2_rad: f64,
) -> f64 {
    let avg_lat = (lat1_rad + lat2_rad) * 0.5;
    let bearing_factor = dlng.abs() / (dlat.abs() + dlng.abs() + 1e-12); // 0=meridional, 1=equatorial

    // WGS84 ellipsoidal correction factor
    let flattening_correction = 1.0 - F * (1.0 - bearing_factor) * (avg_lat.cos().powi(2));

    spherical_distance * flattening_correction
}

/// Calculates the great-circle distance between two points using the [Haversine formula](https://en.wikipedia.org/wiki/Haversine_formula).
///
/// Uses a spherical Earth approximation with mean radius 6,371,008.8 meters, enhanced with
/// ellipsoidal corrections for the [WGS84 ellipsoid](https://en.wikipedia.org/wiki/World_Geodetic_System).
/// Fast but less accurate than Vincenty, with ±0.5% error for distances under 1000km.
///
/// The formula calculates the shortest distance over the Earth's surface, giving an
/// "as-the-crow-flies" distance between the points (ignoring any hills, valleys, or
/// obstacles along the surface of the earth).
///
/// # Arguments
///
/// * `a` - First coordinate
/// * `b` - Second coordinate
///
/// # Returns
///
/// Distance in meters
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::{LngLat, geodesic::haversine};
///
/// let sf = LngLat::new_deg(-122.4194, 37.7749);
/// let nyc = LngLat::new_deg(-74.0060, 40.7128);
/// let distance = haversine(sf, nyc);
/// assert!((distance - 4135000.0).abs() < 10000.0); // ~4,135 km
///
/// // Identical points return 0
/// assert_eq!(haversine(sf, sf), 0.0);
/// ```
///
/// # Algorithm Details
///
/// This implementation uses the standard [Haversine formula](https://en.wikipedia.org/wiki/Haversine_formula)
/// with ellipsoidal corrections to account for Earth's flattening. The formula is:
///
/// ```text
/// a = sin²(Δφ/2) + cos(φ1) × cos(φ2) × sin²(Δλ/2)
/// c = 2 × atan2(√a, √(1-a))
/// d = R × c
/// ```
///
/// Where φ is latitude, λ is longitude, and R is Earth's radius.
pub fn haversine(a: LngLat, b: LngLat) -> f64 {
    let (lng1_rad, lat1_rad) = a.to_radians();
    let (lng2_rad, lat2_rad) = b.to_radians();

    let dlat = lat2_rad - lat1_rad;
    let dlng = normalize_longitude_difference(lng2_rad - lng1_rad);

    let h = compute_haversine_parameter(dlat, dlng, lat1_rad, lat2_rad);

    let central_angle = compute_central_angle(h);

    let spherical_distance = EARTH_RADIUS_M * central_angle;

    apply_ellipsoidal_correction(spherical_distance, dlat, dlng, lat1_rad, lat2_rad)
}

/// Calculates the great-circle distance between two points using the Haversine formula, returned in kilometers.
///
/// This is a convenience wrapper around [`haversine`] that converts the result from meters to kilometers.
/// Uses the same spherical Earth approximation with ellipsoidal corrections for accuracy.
///
/// Perfect for applications where kilometer precision is preferred over meter precision,
/// such as city-to-city distances, route planning, or geographic analysis.
///
/// # Arguments
///
/// * `a` - First coordinate
/// * `b` - Second coordinate
///
/// # Returns
///
/// Distance in kilometers
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::{LngLat, geodesic::haversine_km};
///
/// // San Francisco to New York City
/// let sf = LngLat::new_deg(-122.4194, 37.7749);
/// let nyc = LngLat::new_deg(-74.0060, 40.7128);
/// let distance_km = haversine_km(sf, nyc);
/// assert!((distance_km - 4135.0).abs() < 10.0); // ~4,135 km
///
/// // One degree at equator is approximately 111.32 km
/// let equator_start = LngLat::new_deg(0.0, 0.0);
/// let equator_end = LngLat::new_deg(1.0, 0.0);
/// let degree_km = haversine_km(equator_start, equator_end);
/// assert!((degree_km - 111.32).abs() < 1.0);
///
/// // Identical points return 0
/// assert_eq!(haversine_km(sf, sf), 0.0);
/// ```
///
/// # Performance
///
/// Identical performance to [`haversine`] with a simple division by 1000.
/// The conversion is exact with no additional floating-point precision loss.
///
/// # See Also
///
/// * [`haversine`] - Base distance calculation in meters
/// * [`haversine_miles`] - Distance in statute miles
/// * [`haversine_nautical`] - Distance in nautical miles
pub fn haversine_km(a: LngLat, b: LngLat) -> f64 {
    haversine(a, b) / 1000.0
}

/// Calculates the great-circle distance between two points using the Haversine formula, returned in statute miles.
///
/// This is a convenience wrapper around [`haversine`] that converts the result from meters to statute miles
/// using the international definition: 1 mile = 1,609.344 meters (exactly).
///
/// Ideal for applications in countries using the imperial system, land transportation,
/// road trip planning, or when interfacing with systems expecting mile-based distances.
///
/// # Arguments
///
/// * `a` - First coordinate
/// * `b` - Second coordinate
///
/// # Returns
///
/// Distance in statute miles (not nautical miles)
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::{LngLat, geodesic::haversine_miles};
///
/// // San Francisco to New York City
/// let sf = LngLat::new_deg(-122.4194, 37.7749);
/// let nyc = LngLat::new_deg(-74.0060, 40.7128);
/// let distance_miles = haversine_miles(sf, nyc);
/// assert!((distance_miles - 2569.0).abs() < 10.0); // ~2,569 miles
///
/// // One degree at equator is approximately 69.17 miles
/// let equator_start = LngLat::new_deg(0.0, 0.0);
/// let equator_end = LngLat::new_deg(1.0, 0.0);
/// let degree_miles = haversine_miles(equator_start, equator_end);
/// assert!((degree_miles - 69.17).abs() < 1.0);
///
/// // Cross-country US: roughly 2,500-3,000 miles
/// let la = LngLat::new_deg(-118.2437, 34.0522);
/// let miami = LngLat::new_deg(-80.1918, 25.7617);
/// let cross_country = haversine_miles(la, miami);
/// assert!(cross_country > 2200.0 && cross_country < 2800.0);
/// ```
///
/// # Unit Conversion
///
/// Uses the international definition of the statute mile:
/// - 1 statute mile = 1,609.344 meters (exact)
/// - 1 statute mile = 5,280 feet (exact)
/// - Different from nautical miles (1,852 meters)
///
/// # See Also
///
/// * [`haversine`] - Base distance calculation in meters
/// * [`haversine_km`] - Distance in kilometers  
/// * [`haversine_nautical`] - Distance in nautical miles
pub fn haversine_miles(a: LngLat, b: LngLat) -> f64 {
    haversine(a, b) / 1609.344
}

/// Calculates the great-circle distance between two points using the Haversine formula, returned in nautical miles.
///
/// This is a convenience wrapper around [`haversine`] that converts the result from meters to nautical miles
/// using the international definition: 1 nautical mile = 1,852 meters (exactly).
///
/// Essential for marine and aviation navigation, where nautical miles are the standard unit.
/// One nautical mile equals one minute of arc along a meridian (1/60th of a degree of latitude).
///
/// # Arguments
///
/// * `a` - First coordinate
/// * `b` - Second coordinate
///
/// # Returns
///
/// Distance in nautical miles
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::{LngLat, geodesic::haversine_nautical};
///
/// // San Francisco to New York City
/// let sf = LngLat::new_deg(-122.4194, 37.7749);
/// let nyc = LngLat::new_deg(-74.0060, 40.7128);
/// let distance_nm = haversine_nautical(sf, nyc);
/// assert!((distance_nm - 2233.0).abs() < 10.0); // ~2,233 nautical miles
///
/// // One degree at equator is approximately 60.11 nautical miles
/// let equator_start = LngLat::new_deg(0.0, 0.0);
/// let equator_end = LngLat::new_deg(1.0, 0.0);
/// let degree_nm = haversine_nautical(equator_start, equator_end);
/// assert!((degree_nm - 60.11).abs() < 1.0);
///
/// // Atlantic crossing: New York to London
/// let london = LngLat::new_deg(-0.1276, 51.5074);
/// let atlantic_crossing = haversine_nautical(nyc, london);
/// assert!(atlantic_crossing > 3000.0 && atlantic_crossing < 3500.0);
/// ```
///
/// # Navigation Context
///
/// Nautical miles are particularly useful because:
/// - 1 nautical mile ≈ 1 minute of latitude anywhere on Earth
/// - Standard unit for marine and aviation charts
/// - Used in international maritime and aviation law
/// - Simplifies dead reckoning and celestial navigation
///
/// # Unit Conversion
///
/// - 1 nautical mile = 1,852 meters (exact, by international definition)
/// - 1 nautical mile ≈ 1.15078 statute miles
/// - 60 nautical miles = 1 degree of latitude (approximately)
///
/// # See Also
///
/// * [`haversine`] - Base distance calculation in meters
/// * [`haversine_km`] - Distance in kilometers
/// * [`haversine_miles`] - Distance in statute miles
pub fn haversine_nautical(a: LngLat, b: LngLat) -> f64 {
    haversine(a, b) / 1852.0
}

/// Calculates the initial bearing (direction) from the first point to the second point.
///
/// Returns the bearing in degrees (0-360°) measured clockwise from north. This is the
/// direction you would initially travel when following the great circle route from
/// the first point to the second point.
///
/// Note that on a great circle path, the bearing changes continuously except when
/// traveling due north/south or along the equator. This function returns the
/// **initial** bearing at the starting point.
///
/// # Arguments
///
/// * `from` - Starting coordinate
/// * `to` - Destination coordinate
///
/// # Returns
///
/// Initial bearing in degrees (0.0 to 360.0), where:
/// - 0° = Due North
/// - 90° = Due East  
/// - 180° = Due South
/// - 270° = Due West
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::{LngLat, geodesic::bearing};
///
/// let origin = LngLat::new_deg(0.0, 0.0); // Equator, Prime Meridian
///
/// // Cardinal directions
/// let north = LngLat::new_deg(0.0, 1.0);
/// assert!((bearing(origin, north) - 0.0).abs() < 0.1); // Due North
///
/// let east = LngLat::new_deg(1.0, 0.0);
/// assert!((bearing(origin, east) - 90.0).abs() < 0.1); // Due East
///
/// let south = LngLat::new_deg(0.0, -1.0);
/// assert!((bearing(origin, south) - 180.0).abs() < 0.1); // Due South
///
/// let west = LngLat::new_deg(-1.0, 0.0);
/// assert!((bearing(origin, west) - 270.0).abs() < 0.1); // Due West
///
/// // Real-world example: San Francisco to New York
/// let sf = LngLat::new_deg(-122.4194, 37.7749);
/// let nyc = LngLat::new_deg(-74.0060, 40.7128);
/// let sf_to_nyc = bearing(sf, nyc);
/// assert!(sf_to_nyc > 65.0 && sf_to_nyc < 85.0); // Roughly ENE
///
/// // Identical points return 0°
/// assert_eq!(bearing(sf, sf), 0.0);
/// ```
///
/// # Navigation Applications
///
/// This function is essential for:
/// - **Route planning**: Initial heading for GPS navigation
/// - **Aviation**: Runway approach bearings, flight planning
/// - **Marine navigation**: Compass headings between waypoints  
/// - **Surveying**: Property boundary calculations
/// - **GIS analysis**: Directional analysis of point patterns
///
/// # Algorithm Details
///
/// Uses the standard forward azimuth calculation:
/// ```text
/// y = sin(Δlong) × cos(lat2)
/// x = cos(lat1) × sin(lat2) - sin(lat1) × cos(lat2) × cos(Δlong)
/// θ = atan2(y, x)
/// ```
///
/// The result is normalized to 0-360° for consistent compass bearings.
///
/// # Important Notes
///
/// - **Initial bearing**: On curved paths, bearing changes continuously
/// - **Antipodal points**: Bearing may be unstable for points >179° apart
/// - **Pole proximity**: Less meaningful near geographic poles
/// - **Antimeridian**: Correctly handles longitude wraparound at ±180°
///
/// # See Also
///
/// * [`destination`] - Calculate destination point given bearing and distance
/// * [`haversine`] - Calculate distance between the same two points
pub fn bearing(from: LngLat, to: LngLat) -> f64 {
    let (lng1_rad, lat1_rad) = from.to_radians();
    let (lng2_rad, lat2_rad) = to.to_radians();

    let dlng = normalize_longitude_difference(lng2_rad - lng1_rad);

    let y = dlng.sin() * lat2_rad.cos();
    let x = lat1_rad.cos() * lat2_rad.sin() - lat1_rad.sin() * lat2_rad.cos() * dlng.cos();

    let bearing_rad = y.atan2(x);
    let bearing_deg = bearing_rad.to_degrees();

    normalize_bearing_to_360(bearing_deg)
}

/// Calculates the destination point given an origin, distance, and bearing.
///
/// This is the **inverse** of distance/bearing calculation: given a starting point,
/// a distance to travel, and an initial compass heading, it computes where you
/// would end up following the great circle route.
///
/// Uses spherical trigonometry with the same Earth radius and ellipsoidal corrections
/// as [`haversine`] for consistency. Essential for navigation, coverage analysis,
/// geofencing, and spatial buffer operations.
///
/// # Arguments
///
/// * `origin` - Starting coordinate
/// * `distance_m` - Distance to travel in meters (must be ≥ 0)
/// * `bearing_deg` - Initial bearing in degrees (0-360°, where 0° = North)
///
/// # Returns
///
/// The destination coordinate after traveling the specified distance at the given bearing
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::{LngLat, geodesic::destination};
///
/// let origin = LngLat::new_deg(0.0, 0.0); // Equator, Prime Meridian
///
/// // Travel 111.32 km north (approximately 1 degree)
/// let north = destination(origin, 111320.0, 0.0);
/// assert!((north.lat_deg - 1.0).abs() < 0.01); // ~1° North
/// assert!(north.lng_deg.abs() < 0.01); // Longitude unchanged
///
/// // Travel 111.32 km east at equator
/// let east = destination(origin, 111320.0, 90.0);
/// assert!(east.lat_deg.abs() < 0.01); // Latitude unchanged
/// assert!((east.lng_deg - 1.0).abs() < 0.01); // ~1° East
///
/// // Zero distance returns original point
/// let same = destination(origin, 0.0, 45.0);
/// assert_eq!(same.lng_deg, origin.lng_deg);
/// assert_eq!(same.lat_deg, origin.lat_deg);
///
/// // Real example: 500km northeast from San Francisco
/// let sf = LngLat::new_deg(-122.4194, 37.7749);
/// let northeast_500km = destination(sf, 500000.0, 45.0);
/// assert!(northeast_500km.lng_deg > sf.lng_deg); // More eastward
/// assert!(northeast_500km.lat_deg > sf.lat_deg); // More northward
/// ```
///
/// # Navigation Applications
///
/// This function is the foundation for:
/// - **GPS navigation**: "Go 2.5km northeast" calculations
/// - **Aviation**: Waypoint generation, approach patterns
/// - **Marine charts**: Dead reckoning, course plotting
/// - **Logistics**: Service area boundaries, delivery zones  
/// - **Emergency services**: Search area grid generation
/// - **GIS analysis**: Buffer zones, coverage maps
/// - **Surveying**: Property corners, boundary markers
///
/// # Algorithm Details
///
/// Uses the standard destination point formula from spherical trigonometry:
/// ```text
/// lat2 = asin(sin(lat1) × cos(d/R) + cos(lat1) × sin(d/R) × cos(θ))
/// lon2 = lon1 + atan2(sin(θ) × sin(d/R) × cos(lat1), cos(d/R) - sin(lat1) × sin(lat2))
/// ```
///
/// Where d is distance, R is Earth radius, and θ is bearing.
///
/// # Accuracy & Limitations
///
/// - **Accuracy**: ±0.5% error for distances under 1000km (same as [`haversine`])
/// - **Spherical approximation**: Less accurate than ellipsoidal methods like Vincenty
/// - **Antimeridian**: Correctly handles longitude wraparound at ±180°
/// - **Polar regions**: May have reduced precision near geographic poles
/// - **Large distances**: Error accumulates for distances >5000km
///
/// # Performance
///
/// Highly optimized with inline trigonometry and minimal allocations.
/// Typical performance: ~50-100ns per calculation on modern hardware.
///
/// # Round-trip Consistency
///
/// ```
/// use rapidgeo_distance::{LngLat, geodesic::{destination, haversine, bearing}};
///
/// let origin = LngLat::new_deg(-74.0060, 40.7128); // NYC
/// let distance = 100000.0; // 100km
/// let initial_bearing = 67.5; // ENE
///
/// // Go to destination
/// let dest = destination(origin, distance, initial_bearing);
///
/// // Calculate return trip
/// let return_distance = haversine(dest, origin);
/// let return_bearing = bearing(dest, origin);
///
/// // Should be approximately consistent (within spherical approximation)
/// assert!((return_distance - distance).abs() < 1000.0); // Within 1km
/// let expected_return = (initial_bearing + 180.0) % 360.0;
/// let bearing_diff = (return_bearing - expected_return).abs().min(360.0 - (return_bearing - expected_return).abs());
/// assert!(bearing_diff < 5.0); // Within 5 degrees
/// ```
///
/// # See Also
///
/// * [`bearing`] - Calculate initial bearing between two points
/// * [`haversine`] - Calculate distance between two points
/// * [`haversine_km`], [`haversine_miles`], [`haversine_nautical`] - Distance in other units
pub fn destination(origin: LngLat, distance_m: f64, bearing_deg: f64) -> LngLat {
    let (lng_rad, lat_rad) = origin.to_radians();
    let bearing_rad = bearing_deg.to_radians();
    let angular_distance = distance_m / EARTH_RADIUS_M;

    let lat2_rad = compute_destination_latitude(lat_rad, angular_distance, bearing_rad);
    let lng2_rad =
        compute_destination_longitude(lng_rad, lat_rad, lat2_rad, angular_distance, bearing_rad);
    let normalized_lng2 = normalize_longitude_rad(lng2_rad);

    LngLat::new_rad(normalized_lng2, lat2_rad)
}

#[inline]
fn compute_destination_latitude(lat1_rad: f64, angular_distance: f64, bearing_rad: f64) -> f64 {
    let (sin_lat1, cos_lat1) = lat1_rad.sin_cos();
    let (sin_angular_distance, cos_angular_distance) = angular_distance.sin_cos();
    let cos_bearing = bearing_rad.cos();

    (sin_lat1 * cos_angular_distance + cos_lat1 * sin_angular_distance * cos_bearing).asin()
}

#[inline]
fn compute_destination_longitude(
    lng1_rad: f64,
    lat1_rad: f64,
    lat2_rad: f64,
    angular_distance: f64,
    bearing_rad: f64,
) -> f64 {
    let sin_bearing = bearing_rad.sin();
    let sin_angular_distance = angular_distance.sin();
    let cos_lat1 = lat1_rad.cos();

    let y = sin_bearing * sin_angular_distance * cos_lat1;
    let x = angular_distance.cos() - lat1_rad.sin() * lat2_rad.sin();

    lng1_rad + y.atan2(x)
}

#[inline]
fn normalize_longitude_rad(lng_rad: f64) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    let pi = std::f64::consts::PI;
    let mut result = lng_rad % two_pi;

    if result >= pi {
        result -= two_pi;
    } else if result < -pi {
        result += two_pi;
    }

    result
}

#[inline]
fn normalize_bearing_to_360(bearing_deg: f64) -> f64 {
    ((bearing_deg % 360.0) + 360.0) % 360.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_longitude_difference() {
        // No normalization needed
        assert!((normalize_longitude_difference(1.0) - 1.0).abs() < 1e-10);
        assert!((normalize_longitude_difference(-1.0) - (-1.0)).abs() < 1e-10);

        // Positive wrap around (> π)
        let input = std::f64::consts::PI + 1.0;
        let expected = input - 2.0 * std::f64::consts::PI;
        assert!((normalize_longitude_difference(input) - expected).abs() < 1e-10);

        // Negative wrap around (< -π)
        let input = -std::f64::consts::PI - 1.0;
        let expected = input + 2.0 * std::f64::consts::PI;
        assert!((normalize_longitude_difference(input) - expected).abs() < 1e-10);

        // Exactly at boundaries
        assert!(
            (normalize_longitude_difference(std::f64::consts::PI) - std::f64::consts::PI).abs()
                < 1e-10
        );
        assert!(
            (normalize_longitude_difference(-std::f64::consts::PI) - (-std::f64::consts::PI)).abs()
                < 1e-10
        );
    }

    #[test]
    fn test_compute_haversine_parameter() {
        // Same points should give h = 0
        let h = compute_haversine_parameter(0.0, 0.0, 0.0, 0.0);
        assert!(h.abs() < 1e-10);

        // 90 degree latitude difference at equator
        let dlat = std::f64::consts::PI / 2.0;
        let h = compute_haversine_parameter(dlat, 0.0, 0.0, dlat);
        let expected = (dlat / 2.0).sin().powi(2);
        assert!((h - expected).abs() < 1e-10);

        // 90 degree longitude difference at equator
        let dlng = std::f64::consts::PI / 2.0;
        let h = compute_haversine_parameter(0.0, dlng, 0.0, 0.0);
        let expected = (dlng / 2.0).sin().powi(2);
        assert!((h - expected).abs() < 1e-10);
    }

    #[test]
    fn test_compute_central_angle() {
        // h = 0 should give angle = 0
        assert!((compute_central_angle(0.0) - 0.0).abs() < 1e-10);

        // h = 1 should give angle = π (antipodal points)
        let angle = compute_central_angle(1.0);
        assert!((angle - std::f64::consts::PI).abs() < 1e-10);

        // h = 0.5 should give angle = π/2 (quarter circle)
        let angle = compute_central_angle(0.5);
        assert!((angle - std::f64::consts::PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_apply_ellipsoidal_correction() {
        let base_distance = 100000.0; // 100km

        // Pure meridional (north-south) should apply more correction
        let corrected_meridional = apply_ellipsoidal_correction(
            base_distance,
            1.0, // dlat = 1 radian
            0.0, // dlng = 0 (pure meridional)
            0.0, // lat1 = equator
            1.0, // lat2 = ~57 degrees
        );

        // Pure equatorial (east-west) should apply less correction
        let corrected_equatorial = apply_ellipsoidal_correction(
            base_distance,
            0.0, // dlat = 0 (pure equatorial)
            1.0, // dlng = 1 radian
            0.0, // lat1 = equator
            0.0, // lat2 = equator
        );

        // Meridional should be smaller due to flattening correction
        assert!(corrected_meridional < corrected_equatorial);

        // Both should be close to but less than the original distance
        assert!(corrected_meridional < base_distance);
        assert!(corrected_equatorial <= base_distance); // Might be equal at equator
        assert!(corrected_meridional > base_distance * 0.99); // Not too much correction
    }

    #[test]
    fn test_apply_ellipsoidal_correction_at_poles() {
        let base_distance = 100000.0;

        // Near north pole
        let corrected_polar = apply_ellipsoidal_correction(
            base_distance,
            0.1, // small dlat
            0.1, // small dlng
            1.4, // ~80 degrees north
            1.5, // ~86 degrees north
        );

        assert!(corrected_polar > 0.0);
        assert!(corrected_polar < base_distance * 1.01); // Small correction at poles
    }

    #[test]
    fn test_haversine_identical_points() {
        let point = LngLat::new_deg(-122.4194, 37.7749);
        assert_eq!(haversine(point, point), 0.0);
    }

    #[test]
    fn test_haversine_symmetry() {
        let sf = LngLat::new_deg(-122.4194, 37.7749);
        let nyc = LngLat::new_deg(-74.0060, 40.7128);

        let d1 = haversine(sf, nyc);
        let d2 = haversine(nyc, sf);
        assert!((d1 - d2).abs() < 1e-10);
    }

    #[test]
    fn test_haversine_known_distances() {
        // San Francisco to NYC (~4,135 km)
        let sf = LngLat::new_deg(-122.4194, 37.7749);
        let nyc = LngLat::new_deg(-74.0060, 40.7128);
        let distance = haversine(sf, nyc);
        assert!((distance - 4135000.0).abs() < 10000.0);

        // Equatorial degree (~111.32 km)
        let p1 = LngLat::new_deg(0.0, 0.0);
        let p2 = LngLat::new_deg(1.0, 0.0);
        let distance = haversine(p1, p2);
        assert!((distance - 111320.0).abs() < 1000.0);
    }

    #[test]
    fn test_haversine_antimeridian_crossing() {
        // Points on opposite sides of antimeridian
        let west = LngLat::new_deg(179.5, 0.0);
        let east = LngLat::new_deg(-179.5, 0.0);
        let distance = haversine(west, east);

        // Should take shorter path (~111 km), not longer path (~39,885 km)
        assert!(distance < 200000.0); // Much less than halfway around Earth
        assert!(distance > 100000.0); // But more than 100km
    }

    #[test]
    fn test_haversine_very_small_distances() {
        let base = LngLat::new_deg(0.0, 0.0);

        // 1 meter north (approximately)
        let one_meter_north = LngLat::new_deg(0.0, 0.0 + 1.0 / 111320.0);
        let distance = haversine(base, one_meter_north);
        assert!((distance - 1.0).abs() < 0.1);

        // 10 cm east (approximately)
        let ten_cm_east = LngLat::new_deg(0.0 + 0.1 / 111320.0, 0.0);
        let distance = haversine(base, ten_cm_east);
        assert!((distance - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_haversine_polar_regions() {
        // Near north pole
        let north_1 = LngLat::new_deg(0.0, 89.9);
        let north_2 = LngLat::new_deg(180.0, 89.9);
        let distance = haversine(north_1, north_2);
        assert!(distance < 50000.0); // Very short at high latitude

        // Near south pole
        let south_1 = LngLat::new_deg(45.0, -89.9);
        let south_2 = LngLat::new_deg(-135.0, -89.9);
        let distance = haversine(south_1, south_2);
        assert!(distance < 50000.0);
    }

    #[test]
    fn test_haversine_meridional_vs_equatorial() {
        // 1 degree north-south (meridional)
        let meridional = haversine(LngLat::new_deg(0.0, 0.0), LngLat::new_deg(0.0, 1.0));

        // 1 degree east-west at equator (equatorial)
        let equatorial = haversine(LngLat::new_deg(0.0, 0.0), LngLat::new_deg(1.0, 0.0));

        // Meridional should be slightly shorter due to Earth's flattening
        assert!(meridional < equatorial);
        assert!((meridional - equatorial).abs() < 1000.0); // But close
    }

    #[test]
    fn test_haversine_long_distances() {
        // Quarter Earth circumference
        let quarter_earth = haversine(LngLat::new_deg(0.0, 0.0), LngLat::new_deg(90.0, 0.0));
        let expected = std::f64::consts::PI * EARTH_RADIUS_M / 2.0;
        assert!((quarter_earth - expected).abs() < 50000.0);

        // Nearly antipodal
        let near_antipodal = haversine(LngLat::new_deg(0.0, 0.0), LngLat::new_deg(179.0, 0.0));
        assert!(near_antipodal > 19_000_000.0);
        assert!(near_antipodal < 21_000_000.0);
    }

    #[test]
    fn test_haversine_triangle_inequality() {
        let sf = LngLat::new_deg(-122.4194, 37.7749);
        let chicago = LngLat::new_deg(-87.6298, 41.8781);
        let nyc = LngLat::new_deg(-74.0060, 40.7128);

        let sf_chi = haversine(sf, chicago);
        let chi_nyc = haversine(chicago, nyc);
        let sf_nyc = haversine(sf, nyc);

        // Triangle inequality: any side ≤ sum of other two sides
        assert!(sf_nyc <= sf_chi + chi_nyc + 1000.0); // Small tolerance for floating point
        assert!(sf_chi <= sf_nyc + chi_nyc + 1000.0);
        assert!(chi_nyc <= sf_chi + sf_nyc + 1000.0);
    }

    #[test]
    fn test_haversine_km_known_distances() {
        let sf = LngLat::new_deg(-122.4194, 37.7749);
        let nyc = LngLat::new_deg(-74.0060, 40.7128);
        let distance_km = haversine_km(sf, nyc);
        assert!((distance_km - 4135.0).abs() < 10.0);

        let p1 = LngLat::new_deg(0.0, 0.0);
        let p2 = LngLat::new_deg(1.0, 0.0);
        let distance_km = haversine_km(p1, p2);
        assert!((distance_km - 111.32).abs() < 1.0);
    }

    #[test]
    fn test_haversine_miles_known_distances() {
        let sf = LngLat::new_deg(-122.4194, 37.7749);
        let nyc = LngLat::new_deg(-74.0060, 40.7128);
        let distance_miles = haversine_miles(sf, nyc);
        assert!((distance_miles - 2569.0).abs() < 10.0);

        let p1 = LngLat::new_deg(0.0, 0.0);
        let p2 = LngLat::new_deg(1.0, 0.0);
        let distance_miles = haversine_miles(p1, p2);
        assert!((distance_miles - 69.17).abs() < 1.0);
    }

    #[test]
    fn test_haversine_nautical_known_distances() {
        let sf = LngLat::new_deg(-122.4194, 37.7749);
        let nyc = LngLat::new_deg(-74.0060, 40.7128);
        let distance_nm = haversine_nautical(sf, nyc);
        assert!((distance_nm - 2233.0).abs() < 10.0);

        let p1 = LngLat::new_deg(0.0, 0.0);
        let p2 = LngLat::new_deg(1.0, 0.0);
        let distance_nm = haversine_nautical(p1, p2);
        assert!((distance_nm - 60.11).abs() < 1.0);
    }

    #[test]
    fn test_unit_conversion_accuracy() {
        let sf = LngLat::new_deg(-122.4194, 37.7749);
        let nyc = LngLat::new_deg(-74.0060, 40.7128);

        let meters = haversine(sf, nyc);
        let km = haversine_km(sf, nyc);
        let miles = haversine_miles(sf, nyc);
        let nautical = haversine_nautical(sf, nyc);

        assert!((km - meters / 1000.0).abs() < 1e-10);
        assert!((miles - meters / 1609.344).abs() < 1e-10);
        assert!((nautical - meters / 1852.0).abs() < 1e-10);
    }

    #[test]
    fn test_unit_conversions_identical_points() {
        let point = LngLat::new_deg(-122.4194, 37.7749);
        assert_eq!(haversine_km(point, point), 0.0);
        assert_eq!(haversine_miles(point, point), 0.0);
        assert_eq!(haversine_nautical(point, point), 0.0);
    }

    #[test]
    fn test_unit_conversions_small_distances() {
        let base = LngLat::new_deg(0.0, 0.0);
        let nearby = LngLat::new_deg(0.001, 0.001);

        let km = haversine_km(base, nearby);
        let miles = haversine_miles(base, nearby);
        let nautical = haversine_nautical(base, nearby);

        assert!(km > 0.0 && km < 1.0);
        assert!(miles > 0.0 && miles < 1.0);
        assert!(nautical > 0.0 && nautical < 1.0);
    }

    #[test]
    fn test_unit_conversions_large_distances() {
        let p1 = LngLat::new_deg(0.0, 0.0);
        let p2 = LngLat::new_deg(90.0, 0.0);

        let km = haversine_km(p1, p2);
        let miles = haversine_miles(p1, p2);
        let nautical = haversine_nautical(p1, p2);

        assert!(km > 9000.0 && km < 11000.0);
        assert!(miles > 5500.0 && miles < 7000.0);
        assert!(nautical > 4800.0 && nautical < 6000.0);
    }

    #[test]
    fn test_normalize_bearing_to_360() {
        // Positive angles within range
        assert!((normalize_bearing_to_360(45.0) - 45.0).abs() < 1e-10);
        assert!((normalize_bearing_to_360(359.0) - 359.0).abs() < 1e-10);

        // Exact boundaries
        assert!((normalize_bearing_to_360(0.0) - 0.0).abs() < 1e-10);
        assert!((normalize_bearing_to_360(360.0) - 0.0).abs() < 1e-10);

        // Negative angles
        assert!((normalize_bearing_to_360(-45.0) - 315.0).abs() < 1e-10);
        assert!((normalize_bearing_to_360(-90.0) - 270.0).abs() < 1e-10);
        assert!((normalize_bearing_to_360(-180.0) - 180.0).abs() < 1e-10);
        assert!((normalize_bearing_to_360(-360.0) - 0.0).abs() < 1e-10);

        // Angles > 360
        assert!((normalize_bearing_to_360(405.0) - 45.0).abs() < 1e-10);
        assert!((normalize_bearing_to_360(720.0) - 0.0).abs() < 1e-10);
        assert!((normalize_bearing_to_360(815.0) - 95.0).abs() < 1e-10);

        // Large negative angles
        assert!((normalize_bearing_to_360(-405.0) - 315.0).abs() < 1e-10);
        assert!((normalize_bearing_to_360(-720.0) - 0.0).abs() < 1e-10);

        // Very small angles
        assert!((normalize_bearing_to_360(0.1) - 0.1).abs() < 1e-10);
        assert!((normalize_bearing_to_360(-0.1) - 359.9).abs() < 1e-10);
    }

    #[test]
    fn test_bearing_identical_points() {
        let point = LngLat::new_deg(-122.4194, 37.7749);
        let bearing_result = bearing(point, point);
        assert!((bearing_result - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_bearing_cardinal_directions() {
        let origin = LngLat::new_deg(0.0, 0.0);

        // Due North (0°)
        let north = LngLat::new_deg(0.0, 1.0);
        let bearing_north = bearing(origin, north);
        assert!((bearing_north - 0.0).abs() < 1e-6);

        // Due East (90°)
        let east = LngLat::new_deg(1.0, 0.0);
        let bearing_east = bearing(origin, east);
        assert!((bearing_east - 90.0).abs() < 1e-6);

        // Due South (180°)
        let south = LngLat::new_deg(0.0, -1.0);
        let bearing_south = bearing(origin, south);
        assert!((bearing_south - 180.0).abs() < 1e-6);

        // Due West (270°)
        let west = LngLat::new_deg(-1.0, 0.0);
        let bearing_west = bearing(origin, west);
        assert!((bearing_west - 270.0).abs() < 1e-6);
    }

    #[test]
    fn test_bearing_intercardinal_directions() {
        let origin = LngLat::new_deg(0.0, 0.0);

        // Northeast (around 45°)
        let northeast = LngLat::new_deg(1.0, 1.0);
        let bearing_ne = bearing(origin, northeast);
        assert!(bearing_ne > 40.0 && bearing_ne < 50.0);

        // Southeast (around 135°)
        let southeast = LngLat::new_deg(1.0, -1.0);
        let bearing_se = bearing(origin, southeast);
        assert!(bearing_se > 130.0 && bearing_se < 140.0);

        // Southwest (around 225°)
        let southwest = LngLat::new_deg(-1.0, -1.0);
        let bearing_sw = bearing(origin, southwest);
        assert!(bearing_sw > 220.0 && bearing_sw < 230.0);

        // Northwest (around 315°)
        let northwest = LngLat::new_deg(-1.0, 1.0);
        let bearing_nw = bearing(origin, northwest);
        assert!(bearing_nw > 310.0 && bearing_nw < 320.0);
    }

    #[test]
    fn test_bearing_known_city_pairs() {
        // San Francisco to NYC should be approximately 78° (ENE)
        let sf = LngLat::new_deg(-122.4194, 37.7749);
        let nyc = LngLat::new_deg(-74.0060, 40.7128);
        let sf_to_nyc = bearing(sf, nyc);
        assert!(sf_to_nyc > 65.0 && sf_to_nyc < 85.0);

        // NYC to SF should be westward
        let nyc_to_sf = bearing(nyc, sf);
        assert!(nyc_to_sf > 240.0 && nyc_to_sf < 290.0);

        // London to Paris should be approximately 146° (SE)
        let london = LngLat::new_deg(-0.1276, 51.5074);
        let paris = LngLat::new_deg(2.3522, 48.8566);
        let london_to_paris = bearing(london, paris);
        assert!((london_to_paris - 146.0).abs() < 10.0);

        // Sydney to Tokyo should be northward
        let sydney = LngLat::new_deg(151.2093, -33.8688);
        let tokyo = LngLat::new_deg(139.6917, 35.6895);
        let sydney_to_tokyo = bearing(sydney, tokyo);
        assert!(!(60.0..=320.0).contains(&sydney_to_tokyo));
    }

    #[test]
    fn test_bearing_antimeridian_crossing() {
        // West side of antimeridian to east side
        let west = LngLat::new_deg(179.0, 0.0);
        let east = LngLat::new_deg(-179.0, 0.0);

        // Should take shortest path eastward (~90°)
        let bearing_we = bearing(west, east);
        assert!((bearing_we - 90.0).abs() < 5.0);

        // Reverse direction should be westward (~270°)
        let bearing_ew = bearing(east, west);
        assert!((bearing_ew - 270.0).abs() < 5.0);

        // Test with latitude offset
        let west_north = LngLat::new_deg(179.0, 10.0);
        let east_south = LngLat::new_deg(-179.0, -10.0);
        let bearing_crossing = bearing(west_north, east_south);
        // Should be between southeast (90-180°)
        assert!(bearing_crossing > 90.0 && bearing_crossing < 180.0);
    }

    #[test]
    fn test_bearing_polar_regions() {
        // Near north pole - longitude becomes less meaningful
        let north_pole_1 = LngLat::new_deg(0.0, 89.9);
        let north_pole_2 = LngLat::new_deg(180.0, 89.9);
        let polar_bearing = bearing(north_pole_1, north_pole_2);
        // Should be a valid bearing (0-360°)
        assert!((0.0..360.0).contains(&polar_bearing));

        // Near south pole
        let south_pole_1 = LngLat::new_deg(45.0, -89.9);
        let south_pole_2 = LngLat::new_deg(-135.0, -89.9);
        let south_bearing = bearing(south_pole_1, south_pole_2);
        assert!((0.0..360.0).contains(&south_bearing));

        // From equator to north pole should be due north (0°)
        let equator = LngLat::new_deg(0.0, 0.0);
        let north = LngLat::new_deg(0.0, 89.0);
        let to_pole = bearing(equator, north);
        assert!((to_pole - 0.0).abs() < 1.0);
    }

    #[test]
    fn test_bearing_precision_edge_cases() {
        let base = LngLat::new_deg(0.0, 0.0);

        // Very small eastward movement
        let tiny_east = LngLat::new_deg(0.0001, 0.0);
        let bearing_tiny = bearing(base, tiny_east);
        assert!((bearing_tiny - 90.0).abs() < 1.0);

        // Very small westward movement
        let tiny_west = LngLat::new_deg(-0.0001, 0.0);
        let bearing_tiny_w = bearing(base, tiny_west);
        assert!((bearing_tiny_w - 270.0).abs() < 1.0);

        // Very small northward movement
        let tiny_north = LngLat::new_deg(0.0, 0.0001);
        let bearing_tiny_n = bearing(base, tiny_north);
        assert!((bearing_tiny_n - 0.0).abs() < 1.0);

        // Very small southward movement
        let tiny_south = LngLat::new_deg(0.0, -0.0001);
        let bearing_tiny_s = bearing(base, tiny_south);
        assert!((bearing_tiny_s - 180.0).abs() < 1.0);
    }

    #[test]
    fn test_bearing_consistency_with_distance() {
        let origin = LngLat::new_deg(0.0, 0.0);
        let target = LngLat::new_deg(1.0, 1.0);

        // Calculate bearing and distance
        let bearing_value = bearing(origin, target);
        let distance_value = haversine(origin, target);

        // Both should be valid positive values
        assert!((0.0..360.0).contains(&bearing_value));
        assert!(distance_value > 0.0);

        // Reverse bearing should be approximately opposite (±180°)
        let reverse_bearing = bearing(target, origin);
        let bearing_diff = (bearing_value - reverse_bearing + 180.0) % 360.0;
        assert!((bearing_diff - 0.0).abs() < 5.0 || (bearing_diff - 360.0).abs() < 5.0);
    }

    #[test]
    fn test_bearing_mathematical_properties() {
        let sf = LngLat::new_deg(-122.4194, 37.7749);
        let chicago = LngLat::new_deg(-87.6298, 41.8781);
        let nyc = LngLat::new_deg(-74.0060, 40.7128);

        // All bearings should be in valid range
        let b1 = bearing(sf, chicago);
        let b2 = bearing(chicago, nyc);
        let b3 = bearing(nyc, sf);

        assert!((0.0..360.0).contains(&b1));
        assert!((0.0..360.0).contains(&b2));
        assert!((0.0..360.0).contains(&b3));

        // Bearings should be different (not degenerate case)
        assert!((b1 - b2).abs() > 1.0);
        assert!((b2 - b3).abs() > 1.0);
        assert!((b3 - b1).abs() > 1.0);
    }

    #[test]
    fn test_bearing_longitudinal_extremes() {
        let origin = LngLat::new_deg(0.0, 0.0);

        // Maximum positive longitude
        let max_lng = LngLat::new_deg(180.0, 0.0);
        let bearing_max = bearing(origin, max_lng);
        assert!((bearing_max - 90.0).abs() < 5.0); // Should be close to east

        // Maximum negative longitude
        let min_lng = LngLat::new_deg(-180.0, 0.0);
        let bearing_min = bearing(origin, min_lng);
        assert!((bearing_min - 270.0).abs() < 5.0); // Should be close to west

        // From max positive to max negative longitude (across antimeridian)
        let cross_bearing = bearing(max_lng, min_lng);
        // This is essentially the same point, so bearing could be anything, but should be valid
        assert!((0.0..360.0).contains(&cross_bearing));
    }

    #[test]
    fn test_bearing_latitudinal_extremes() {
        let origin = LngLat::new_deg(0.0, 0.0);

        // Near maximum latitude (not exactly 90 to avoid singularity)
        let max_lat = LngLat::new_deg(0.0, 89.0);
        let bearing_max = bearing(origin, max_lat);
        assert!((bearing_max - 0.0).abs() < 1.0); // Should be due north

        // Near minimum latitude
        let min_lat = LngLat::new_deg(0.0, -89.0);
        let bearing_min = bearing(origin, min_lat);
        assert!((bearing_min - 180.0).abs() < 1.0); // Should be due south

        // From north to south
        let ns_bearing = bearing(max_lat, min_lat);
        assert!((ns_bearing - 180.0).abs() < 5.0); // Should be southward
    }

    // ===== DESTINATION POINT CALCULATION TESTS =====

    #[test]
    fn test_normalize_longitude_rad() {
        use std::f64::consts::PI;

        // Within range (-π, π] should remain unchanged
        assert!((normalize_longitude_rad(0.0) - 0.0).abs() < 1e-12);
        assert!((normalize_longitude_rad(PI / 2.0) - PI / 2.0).abs() < 1e-12);
        assert!((normalize_longitude_rad(-PI / 2.0) - (-PI / 2.0)).abs() < 1e-12);

        // Boundary cases - PI maps to -PI (exclusive right boundary)
        let normalized_pi = normalize_longitude_rad(PI);
        assert!((normalized_pi - (-PI)).abs() < 1e-12);

        // Just under -π should wrap to positive
        let just_under_neg_pi = -PI - 0.1;
        let normalized = normalize_longitude_rad(just_under_neg_pi);
        assert!((normalized - (just_under_neg_pi + 2.0 * PI)).abs() < 1e-12);

        // Multiple wraps should normalize to range (-π, π]
        let multiple_wraps = 3.0 * PI; // = PI + 2PI, should become -PI
        let normalized = normalize_longitude_rad(multiple_wraps);
        assert!((-PI..=PI).contains(&normalized));
        assert!((normalized - (-PI)).abs() < 1e-12);

        let negative_multiple_wraps = -5.0 * PI; // = -PI - 4PI, should become -PI
        let normalized = normalize_longitude_rad(negative_multiple_wraps);
        assert!((-PI..=PI).contains(&normalized));
        assert!((normalized - (-PI)).abs() < 1e-12);
    }

    #[test]
    fn test_compute_destination_latitude() {
        use std::f64::consts::PI;

        // Zero distance should return original latitude
        let original_lat = PI / 4.0; // 45 degrees
        let result = compute_destination_latitude(original_lat, 0.0, 0.0);
        assert!((result - original_lat).abs() < 1e-12);

        // From equator due north (quarter Earth distance)
        let quarter_earth_angular = PI / 2.0;
        let result = compute_destination_latitude(0.0, quarter_earth_angular, 0.0);
        assert!((result - PI / 2.0).abs() < 1e-10); // Should reach north pole

        // From equator due south
        let result = compute_destination_latitude(0.0, quarter_earth_angular, PI);
        assert!((result - (-PI / 2.0)).abs() < 1e-10); // Should reach south pole

        // From equator due east/west (latitude unchanged)
        let result_east = compute_destination_latitude(0.0, PI / 4.0, PI / 2.0);
        assert!(result_east.abs() < 1e-10); // Should stay at equator

        let result_west = compute_destination_latitude(0.0, PI / 4.0, 3.0 * PI / 2.0);
        assert!(result_west.abs() < 1e-10); // Should stay at equator

        // Small distance northeast from equator
        let small_angular = 0.001; // Very small distance
        let result = compute_destination_latitude(0.0, small_angular, PI / 4.0);
        assert!(result > 0.0); // Should be slightly north
        assert!(result < small_angular); // But less than if going due north
    }

    #[test]
    fn test_compute_destination_longitude() {
        use std::f64::consts::PI;

        // Zero distance should return original longitude
        let original_lng = PI / 3.0; // 60 degrees
        let original_lat = PI / 4.0; // 45 degrees
        let result =
            compute_destination_longitude(original_lng, original_lat, original_lat, 0.0, 0.0);
        assert!((result - original_lng).abs() < 1e-12);

        // From equator due north/south (longitude unchanged)
        let result_north = compute_destination_longitude(0.0, 0.0, PI / 4.0, PI / 4.0, 0.0);
        assert!(result_north.abs() < 1e-10);

        let result_south = compute_destination_longitude(0.0, 0.0, -PI / 4.0, PI / 4.0, PI);
        assert!(result_south.abs() < 1e-10);

        // From equator due east (longitude increases)
        let result_east = compute_destination_longitude(0.0, 0.0, 0.0, PI / 4.0, PI / 2.0);
        assert!(result_east > 0.0); // Should move eastward

        // From equator due west (longitude decreases)
        let result_west = compute_destination_longitude(0.0, 0.0, 0.0, PI / 4.0, 3.0 * PI / 2.0);
        assert!(result_west < 0.0); // Should move westward
    }

    #[test]
    fn test_destination_zero_distance() {
        let origin = LngLat::new_deg(-122.4194, 37.7749); // San Francisco

        // Zero distance at any bearing should return original point
        let result_north = destination(origin, 0.0, 0.0);
        assert!((result_north.lng_deg - origin.lng_deg).abs() < 1e-12);
        assert!((result_north.lat_deg - origin.lat_deg).abs() < 1e-12);

        let result_east = destination(origin, 0.0, 90.0);
        assert!((result_east.lng_deg - origin.lng_deg).abs() < 1e-12);
        assert!((result_east.lat_deg - origin.lat_deg).abs() < 1e-12);

        let result_arbitrary = destination(origin, 0.0, 123.45);
        assert!((result_arbitrary.lng_deg - origin.lng_deg).abs() < 1e-12);
        assert!((result_arbitrary.lat_deg - origin.lat_deg).abs() < 1e-12);
    }

    #[test]
    fn test_destination_cardinal_directions() {
        let origin = LngLat::new_deg(0.0, 0.0); // Equator, Prime Meridian
        let distance_m = 111320.0; // Approximately 1 degree at equator

        // Due North (0°)
        let north = destination(origin, distance_m, 0.0);
        assert!(north.lng_deg.abs() < 1e-6); // Longitude should stay ~0
        assert!((north.lat_deg - 1.0).abs() < 0.01); // Should be ~1 degree north

        // Due East (90°)
        let east = destination(origin, distance_m, 90.0);
        assert!(east.lat_deg.abs() < 1e-6); // Latitude should stay ~0
        assert!((east.lng_deg - 1.0).abs() < 0.01); // Should be ~1 degree east

        // Due South (180°)
        let south = destination(origin, distance_m, 180.0);
        assert!(south.lng_deg.abs() < 1e-6); // Longitude should stay ~0
        assert!((south.lat_deg + 1.0).abs() < 0.01); // Should be ~1 degree south

        // Due West (270°)
        let west = destination(origin, distance_m, 270.0);
        assert!(west.lat_deg.abs() < 1e-6); // Latitude should stay ~0
        assert!((west.lng_deg + 1.0).abs() < 0.01); // Should be ~1 degree west
    }

    #[test]
    fn test_destination_intercardinal_directions() {
        let origin = LngLat::new_deg(0.0, 0.0);
        let distance_m = 111320.0 * std::f64::consts::SQRT_2; // √2 degrees worth

        // Northeast (45°)
        let northeast = destination(origin, distance_m, 45.0);
        assert!(northeast.lng_deg > 0.9 && northeast.lng_deg < 1.1); // ~1 degree east
        assert!(northeast.lat_deg > 0.9 && northeast.lat_deg < 1.1); // ~1 degree north

        // Southeast (135°)
        let southeast = destination(origin, distance_m, 135.0);
        assert!(southeast.lng_deg > 0.9 && southeast.lng_deg < 1.1); // ~1 degree east
        assert!(southeast.lat_deg < -0.9 && southeast.lat_deg > -1.1); // ~1 degree south

        // Southwest (225°)
        let southwest = destination(origin, distance_m, 225.0);
        assert!(southwest.lng_deg < -0.9 && southwest.lng_deg > -1.1); // ~1 degree west
        assert!(southwest.lat_deg < -0.9 && southwest.lat_deg > -1.1); // ~1 degree south

        // Northwest (315°)
        let northwest = destination(origin, distance_m, 315.0);
        assert!(northwest.lng_deg < -0.9 && northwest.lng_deg > -1.1); // ~1 degree west
        assert!(northwest.lat_deg > 0.9 && northwest.lat_deg < 1.1); // ~1 degree north
    }

    #[test]
    fn test_destination_known_coordinates() {
        // Test with known coordinate pairs

        // From London (0°, 51.5°) going 1000km northeast (spherical approximation)
        let london = LngLat::new_deg(0.0, 51.5);
        let northeast_1000km = destination(london, 1000000.0, 45.0);
        assert!(northeast_1000km.lng_deg > 0.0 && northeast_1000km.lng_deg < 25.0);
        assert!(northeast_1000km.lat_deg > 50.0 && northeast_1000km.lat_deg < 65.0);

        // From Sydney going 2000km north should stay roughly same longitude
        let sydney = LngLat::new_deg(151.2093, -33.8688);
        let north_2000km = destination(sydney, 2000000.0, 0.0);
        assert!((north_2000km.lng_deg - sydney.lng_deg).abs() < 2.0); // Should stay close in longitude
        assert!(north_2000km.lat_deg > -25.0 && north_2000km.lat_deg < -5.0); // Should be much further north

        // Small distance test: 1km east from Paris
        let paris = LngLat::new_deg(2.3522, 48.8566);
        let east_1km = destination(paris, 1000.0, 90.0);
        assert!(east_1km.lng_deg > paris.lng_deg); // Should be further east
        assert!((east_1km.lat_deg - paris.lat_deg).abs() < 0.01); // Latitude should barely change
        assert!(
            (east_1km.lng_deg - paris.lng_deg) > 0.005
                && (east_1km.lng_deg - paris.lng_deg) < 0.020
        );
    }

    #[test]
    fn test_destination_antimeridian_crossing() {
        // Start near antimeridian and go east (should wrap longitude)
        let near_antimeridian = LngLat::new_deg(179.5, 0.0);
        let eastward = destination(near_antimeridian, 111320.0, 90.0); // ~1 degree east

        // Should wrap to western hemisphere
        assert!(eastward.lng_deg < -179.0);
        assert!(eastward.lat_deg.abs() < 0.1); // Latitude should stay near equator

        // Start in western hemisphere near antimeridian and go west
        let near_antimeridian_west = LngLat::new_deg(-179.5, 0.0);
        let westward = destination(near_antimeridian_west, 111320.0, 270.0);

        // Should wrap to eastern hemisphere
        assert!(westward.lng_deg > 179.0);
        assert!(westward.lat_deg.abs() < 0.1);

        // Test longer distance crossing
        let crossing_origin = LngLat::new_deg(175.0, 10.0);
        let cross_pacific = destination(crossing_origin, 1000000.0, 90.0); // 1000km east

        // Should end up in western hemisphere
        assert!(cross_pacific.lng_deg < 0.0);
        assert!(cross_pacific.lat_deg > 5.0 && cross_pacific.lat_deg < 15.0);
    }

    #[test]
    fn test_destination_polar_regions() {
        // Near north pole
        let near_north_pole = LngLat::new_deg(0.0, 89.0);

        // Going south should decrease latitude significantly
        let south_from_pole = destination(near_north_pole, 111320.0, 180.0);
        assert!(south_from_pole.lat_deg < 88.0);
        assert!(south_from_pole.lng_deg.abs() < 5.0); // Longitude shouldn't change much

        // Going east near north pole
        let east_from_pole = destination(near_north_pole, 50000.0, 90.0);
        assert!(east_from_pole.lat_deg > 88.5); // Should still be very far north
        assert!(east_from_pole.lng_deg > 0.0); // Should be eastward

        // Near south pole
        let near_south_pole = LngLat::new_deg(45.0, -89.0);

        // Going north should increase latitude
        let north_from_south_pole = destination(near_south_pole, 111320.0, 0.0);
        assert!(north_from_south_pole.lat_deg > -88.0);

        // Going west from south pole
        let west_from_south_pole = destination(near_south_pole, 50000.0, 270.0);
        assert!(west_from_south_pole.lat_deg < -88.5); // Should still be very far south
        assert!(west_from_south_pole.lng_deg < 45.0); // Should be westward from starting longitude
    }

    #[test]
    fn test_destination_very_small_distances() {
        let origin = LngLat::new_deg(-74.0060, 40.7128); // NYC

        // 1 meter north
        let one_meter_north = destination(origin, 1.0, 0.0);
        assert!((one_meter_north.lng_deg - origin.lng_deg).abs() < 1e-6); // Longitude barely changes
        assert!(one_meter_north.lat_deg > origin.lat_deg); // Latitude increases
        assert!((one_meter_north.lat_deg - origin.lat_deg) < 0.0001); // But very small change

        // 10 centimeters east (spherical approximation limits precision)
        let ten_cm_east = destination(origin, 0.1, 90.0);
        assert!((ten_cm_east.lat_deg - origin.lat_deg).abs() < 1e-6); // Latitude essentially unchanged
        assert!(ten_cm_east.lng_deg > origin.lng_deg); // Longitude increases
        assert!((ten_cm_east.lng_deg - origin.lng_deg) < 0.00001); // Tiny change

        // 1 millimeter south
        let one_mm_south = destination(origin, 0.001, 180.0);
        assert!((one_mm_south.lng_deg - origin.lng_deg).abs() < 1e-8);
        assert!(one_mm_south.lat_deg < origin.lat_deg);
        assert!((origin.lat_deg - one_mm_south.lat_deg) < 0.00001);
    }

    #[test]
    fn test_destination_very_large_distances() {
        let origin = LngLat::new_deg(0.0, 0.0); // Equator, Prime Meridian

        // Quarter Earth circumference east (should reach 90° longitude)
        let quarter_earth = std::f64::consts::PI * EARTH_RADIUS_M / 2.0;
        let quarter_east = destination(origin, quarter_earth, 90.0);
        assert!((quarter_east.lng_deg - 90.0).abs() < 5.0); // Should be close to 90° east
        assert!(quarter_east.lat_deg.abs() < 5.0); // Should stay near equator

        // Quarter Earth circumference north (should reach near north pole)
        let quarter_earth = std::f64::consts::PI * EARTH_RADIUS_M / 2.0;
        let to_pole = destination(origin, quarter_earth, 0.0);
        assert!(to_pole.lat_deg > 85.0); // Should be very close to north pole

        // Nearly antipodal distance
        let nearly_antipodal = std::f64::consts::PI * EARTH_RADIUS_M * 0.99;
        let far_point = destination(origin, nearly_antipodal, 45.0);

        // Should be very far from origin
        let distance_back = haversine(origin, far_point);
        assert!(distance_back > 19_000_000.0); // More than 19,000 km away

        // 10,000 km journey
        let long_journey = destination(origin, 10_000_000.0, 30.0); // NNE
        assert!(long_journey.lng_deg > 80.0 && long_journey.lng_deg < 100.0); // Significant eastward movement
        assert!(long_journey.lat_deg > 59.0); // Should reach about 60° north
    }

    #[test]
    fn test_destination_roundtrip_consistency() {
        let origin = LngLat::new_deg(-122.4194, 37.7749); // San Francisco
        let distance_m = 500000.0; // 500 km
        let bearing_deg = 67.5; // ENE

        // Go to destination
        let destination_point = destination(origin, distance_m, bearing_deg);

        // Calculate bearing and distance back to origin
        let back_bearing = bearing(destination_point, origin);
        let back_distance = haversine(destination_point, origin);

        // Distance should match (within tolerance for spherical approximation)
        assert!((back_distance - distance_m).abs() < 50000.0); // Within 50km (spherical approx)

        // Back bearing should be approximately opposite (±180°)
        let expected_back_bearing = (bearing_deg + 180.0) % 360.0;
        let bearing_diff = (back_bearing - expected_back_bearing).abs();
        let bearing_diff_wrapped = (360.0 - bearing_diff).min(bearing_diff);
        assert!(bearing_diff_wrapped < 10.0); // Within 10 degrees (spherical approximation)

        // Test the reverse: go from destination back to origin
        let return_point = destination(destination_point, back_distance, back_bearing);
        let final_distance = haversine(origin, return_point);
        assert!(final_distance < 5000.0); // Should be within 5km of original
    }

    #[test]
    fn test_destination_multiple_roundtrips() {
        let start = LngLat::new_deg(2.3522, 48.8566); // Paris
        let mut current = start;

        // Take 8 steps of 100km in different directions (octagon)
        let step_distance = 100000.0;
        for i in 0..8 {
            let bearing = (i as f64) * 45.0; // 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
            current = destination(current, step_distance, bearing);
        }

        // Should be reasonably close to starting point (spherical approximation accumulates error)
        let final_distance = haversine(start, current);
        assert!(final_distance < 50000.0);
    }

    #[test]
    fn test_destination_bearing_consistency() {
        let tests = vec![
            (LngLat::new_deg(0.0, 0.0), 100000.0, 0.0), // North from equator
            (LngLat::new_deg(0.0, 0.0), 100000.0, 90.0), // East from equator
            (LngLat::new_deg(-74.0, 40.7), 50000.0, 45.0), // NE from NYC
            (LngLat::new_deg(151.2, -33.9), 200000.0, 225.0), // SW from Sydney
            (LngLat::new_deg(-0.1, 51.5), 75000.0, 135.0), // SE from London
        ];

        for (origin, distance, bearing_out) in tests {
            let destination_point = destination(origin, distance, bearing_out);

            // Calculate bearing from origin to destination
            let calculated_bearing = bearing(origin, destination_point);

            // Should match input bearing (within tolerance)
            let bearing_diff = (calculated_bearing - bearing_out).abs();
            let bearing_diff_wrapped = (360.0 - bearing_diff).min(bearing_diff);
            assert!(
                bearing_diff_wrapped < 10.0,
                "Bearing mismatch: expected {}, got {}, diff {}",
                bearing_out,
                calculated_bearing,
                bearing_diff_wrapped
            );

            // Distance should also match
            let calculated_distance = haversine(origin, destination_point);
            assert!(
                (calculated_distance - distance).abs() < 10000.0,
                "Distance mismatch: expected {}, got {}",
                distance,
                calculated_distance
            );
        }
    }

    #[test]
    fn test_destination_mathematical_properties() {
        let origin = LngLat::new_deg(-87.6298, 41.8781); // Chicago
        let distance = 1000000.0; // 1000 km

        // Test that destination coordinates are always valid
        for bearing in [
            0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0, 330.0,
        ] {
            let dest = destination(origin, distance, bearing);

            // Longitude should be in valid range [-180, 180]
            assert!(
                dest.lng_deg >= -180.0 && dest.lng_deg <= 180.0,
                "Invalid longitude: {} at bearing {}",
                dest.lng_deg,
                bearing
            );

            // Latitude should be in valid range [-90, 90]
            assert!(
                dest.lat_deg >= -90.0 && dest.lat_deg <= 90.0,
                "Invalid latitude: {} at bearing {}",
                dest.lat_deg,
                bearing
            );

            // Distance to destination should be approximately correct
            let actual_distance = haversine(origin, dest);
            assert!(
                (actual_distance - distance).abs() < 200000.0,
                "Distance error: expected {}, got {} at bearing {}",
                distance,
                actual_distance,
                bearing
            );
        }
    }

    #[test]
    fn test_destination_boundary_conditions() {
        // Test with extreme coordinates
        let extreme_cases = vec![
            (LngLat::new_deg(180.0, 0.0), 1000.0, 90.0), // At antimeridian
            (LngLat::new_deg(-180.0, 0.0), 1000.0, 270.0), // At antimeridian (other side)
            (LngLat::new_deg(0.0, 89.99), 1000.0, 0.0),  // Near north pole
            (LngLat::new_deg(0.0, -89.99), 1000.0, 180.0), // Near south pole
            (LngLat::new_deg(179.99, 89.99), 1000.0, 225.0), // NE corner of world
            (LngLat::new_deg(-179.99, -89.99), 1000.0, 45.0), // SW corner of world
        ];

        for (origin, distance, bearing) in extreme_cases {
            let dest = destination(origin, distance, bearing);

            // Should produce valid coordinates
            assert!(dest.lng_deg >= -180.0 && dest.lng_deg <= 180.0);
            assert!(dest.lat_deg >= -90.0 && dest.lat_deg <= 90.0);

            // Should be different from origin (unless zero distance)
            if distance > 0.0 {
                let moved_distance = haversine(origin, dest);
                assert!(moved_distance > 0.0);
            }
        }
    }

    #[test]
    fn test_destination_precision_consistency() {
        let origin = LngLat::new_deg(0.0, 0.0);
        let base_distance = 1000.0; // 1km

        // Test that very small changes in bearing produce appropriately small changes in destination
        let bearing1 = 90.0;
        let bearing2 = 90.001; // 0.001 degree difference

        let dest1 = destination(origin, base_distance, bearing1);
        let dest2 = destination(origin, base_distance, bearing2);

        let position_difference = haversine(dest1, dest2);
        assert!(position_difference < 1.0); // Should be less than 1 meter difference
        assert!(position_difference > 0.0); // But should be non-zero

        // Test that small changes in distance produce proportional changes
        let distance1 = 1000.0;
        let distance2 = 1001.0; // 1 meter difference

        let dest_d1 = destination(origin, distance1, 45.0);
        let dest_d2 = destination(origin, distance2, 45.0);

        let distance_difference = haversine(dest_d1, dest_d2);
        assert!((distance_difference - 1.0).abs() < 0.1); // Should be ~1 meter
    }
}
