//! Cross-track distance calculation implementations.
//!
//! This module provides different methods for calculating the perpendicular
//! distance from a point to a line segment, which is the core operation
//! in the Douglas-Peucker algorithm.
//!
//! The choice of distance calculation method affects both accuracy and performance:
//!
//! - [`XtGreatCircle`]: Most accurate for geographic data, works globally
//! - [`XtEnu`]: Good balance of accuracy and performance for regional data
//! - [`XtEuclid`]: Fastest, suitable for projected or non-geographic data

use rapidgeo_distance::{geodesic, LngLat};

/// Trait for calculating perpendicular distance from a point to a line segment.
///
/// This abstraction allows the Douglas-Peucker algorithm to work with different
/// distance calculation methods without changing the core algorithm.
pub trait PerpDistance {
    /// Calculate the perpendicular distance from point `p` to line segment `a`-`b`.
    ///
    /// # Arguments
    ///
    /// * `a` - Start point of the line segment
    /// * `b` - End point of the line segment  
    /// * `p` - Point to measure distance from
    ///
    /// # Returns
    ///
    /// Perpendicular distance in the implementation's units (usually meters)
    fn d_perp_m(&self, a: LngLat, b: LngLat, p: LngLat) -> f64;
}

/// Great circle distance calculation using spherical geometry.
///
/// Uses the [great circle method](https://en.wikipedia.org/wiki/Great_circle)
/// to calculate the shortest distance between a point and a line segment on
/// the Earth's surface. This is the most accurate method for geographic data
/// but requires more computation.
///
/// # Examples
///
/// ```rust
/// use rapidgeo_distance::LngLat;
/// use rapidgeo_simplify::xt::{PerpDistance, XtGreatCircle};
///
/// let backend = XtGreatCircle;
/// let distance = backend.d_perp_m(
///     LngLat::new_deg(-122.0, 37.0), // San Francisco area
///     LngLat::new_deg(-121.0, 37.0), // Point east
///     LngLat::new_deg(-121.5, 37.1), // Point slightly north of line
/// );
///
/// // Distance should be roughly 11km (0.1 degree latitude difference)
/// assert!((distance - 11100.0).abs() < 1000.0);
/// ```
pub struct XtGreatCircle;

impl PerpDistance for XtGreatCircle {
    fn d_perp_m(&self, a: LngLat, b: LngLat, p: LngLat) -> f64 {
        geodesic::great_circle_point_to_seg(p, (a, b))
    }
}

/// East-North-Up (ENU) planar projection distance calculation.
///
/// Projects coordinates to a local [East-North-Up coordinate system](https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates)
/// around the specified origin point, then calculates Euclidean distance.
/// This provides better performance than great circle calculations while
/// maintaining reasonable accuracy for regional datasets.
///
/// # Examples
///
/// ```rust
/// use rapidgeo_distance::LngLat;
/// use rapidgeo_simplify::xt::{PerpDistance, XtEnu};
///
/// let origin = LngLat::new_deg(-121.5, 37.0); // Midpoint
/// let backend = XtEnu { origin };
///
/// let distance = backend.d_perp_m(
///     LngLat::new_deg(-122.0, 37.0),
///     LngLat::new_deg(-121.0, 37.0),
///     LngLat::new_deg(-121.5, 37.1), // 0.1 degree north
/// );
///
/// // Should be close to great circle result for this regional example
/// assert!(distance > 10000.0 && distance < 12000.0);
/// ```
pub struct XtEnu {
    /// Origin point for the ENU projection. Usually set to the midpoint
    /// of the polyline being simplified for optimal accuracy.
    pub origin: LngLat,
}

impl PerpDistance for XtEnu {
    fn d_perp_m(&self, a: LngLat, b: LngLat, p: LngLat) -> f64 {
        geodesic::point_to_segment_enu_m(p, (a, b))
    }
}

/// Raw Euclidean distance calculation between coordinates.
///
/// Treats longitude and latitude values as Cartesian coordinates and
/// calculates standard Euclidean distance. This is the fastest method
/// but should only be used for:
///
/// - Non-geographic coordinate systems (screen coordinates, etc.)
/// - Already-projected data where coordinates represent planar distances
/// - Data where geographic accuracy is not important
///
/// # Examples
///
/// ```rust
/// use rapidgeo_distance::LngLat;
/// use rapidgeo_simplify::xt::{PerpDistance, XtEuclid};
///
/// let backend = XtEuclid;
///
/// // Using as screen coordinates (not geographic)
/// let distance = backend.d_perp_m(
///     LngLat::new_deg(0.0, 0.0),   // Point A
///     LngLat::new_deg(10.0, 0.0),  // Point B  
///     LngLat::new_deg(5.0, 3.0),   // Point P (3 units above midpoint)
/// );
///
/// assert!((distance - 3.0).abs() < 0.001); // Should be exactly 3.0
/// ```
pub struct XtEuclid;

impl PerpDistance for XtEuclid {
    /// Calculate Euclidean distance from point to line segment.
    ///
    /// Uses the standard [point-to-line distance formula](https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points)
    /// with projection clamping to handle line segments (not infinite lines).
    ///
    /// # Algorithm
    ///
    /// 1. If start and end points are identical, return direct distance
    /// 2. Project point P onto the line AB to find the closest point
    /// 3. Clamp projection parameter to \[0,1\] to stay within segment bounds
    /// 4. Calculate distance from P to the clamped projection point
    fn d_perp_m(&self, a: LngLat, b: LngLat, p: LngLat) -> f64 {
        let (ax, ay) = (a.lng_deg, a.lat_deg);
        let (bx, by) = (b.lng_deg, b.lat_deg);
        let (px, py) = (p.lng_deg, p.lat_deg);

        // Handle degenerate case where A and B are the same point
        if ax == bx && ay == by {
            let dx = px - ax;
            let dy = py - ay;
            return (dx * dx + dy * dy).sqrt();
        }

        // Vector from A to B
        let dx = bx - ax;
        let dy = by - ay;

        // Project P onto line AB: t = (AP · AB) / |AB|²
        let t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy);
        // Clamp to segment: t ∈ [0,1]
        let t = t.clamp(0.0, 1.0);

        // Find closest point on segment
        let proj_x = ax + t * dx;
        let proj_y = ay + t * dy;

        // Distance from P to closest point
        let dpx = px - proj_x;
        let dpy = py - proj_y;

        (dpx * dpx + dpy * dpy).sqrt()
    }
}
