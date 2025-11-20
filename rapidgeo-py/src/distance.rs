#![allow(non_local_definitions)]

use pyo3::prelude::*;
use pyo3::types::PyList;
use rapidgeo_distance::{geodesic, LngLat as CoreLngLat};
use std::sync::OnceLock;

/// Cached numpy availability check to avoid repeated import overhead
static NUMPY_AVAILABLE: OnceLock<bool> = OnceLock::new();

/// Check if numpy is available, caching the result for performance
fn is_numpy_available(py: Python) -> bool {
    *NUMPY_AVAILABLE.get_or_init(|| py.import("numpy").is_ok())
}

/// Geographic coordinate representing longitude and latitude in decimal degrees.
///
/// Coordinates use longitude, latitude ordering (x, y convention).
/// All functions expect coordinates in decimal degrees.
///
/// Examples:
///     >>> from rapidgeo import LngLat
///     >>> sf = LngLat(-122.4194, 37.7749)  # San Francisco
///     >>> print(sf.lng, sf.lat)
///     -122.4194 37.7749
#[pyclass]
#[derive(Clone, Copy)]
pub struct LngLat {
    inner: CoreLngLat,
}

#[pymethods]
impl LngLat {
    /// Create a new coordinate from longitude and latitude in decimal degrees.
    ///
    /// Args:
    ///     lng (float): Longitude in decimal degrees (-180 to +180)
    ///     lat (float): Latitude in decimal degrees (-90 to +90)
    ///
    /// Returns:
    ///     LngLat: A new coordinate object
    ///
    /// Examples:
    ///     >>> coord = LngLat(-122.4194, 37.7749)
    ///     >>> print(coord)
    ///     LngLat(-122.4194, 37.7749)
    #[new]
    pub fn new(lng: f64, lat: f64) -> Self {
        Self {
            inner: CoreLngLat::new_deg(lng, lat),
        }
    }

    /// Longitude in decimal degrees.
    ///
    /// Returns:
    ///     float: Longitude coordinate (-180 to +180)
    #[getter]
    pub fn lng(&self) -> f64 {
        self.inner.lng_deg
    }

    /// Latitude in decimal degrees.
    ///
    /// Returns:
    ///     float: Latitude coordinate (-90 to +90)
    #[getter]
    pub fn lat(&self) -> f64 {
        self.inner.lat_deg
    }

    fn __repr__(&self) -> String {
        format!("LngLat({}, {})", self.lng(), self.lat())
    }
}

impl From<LngLat> for CoreLngLat {
    fn from(val: LngLat) -> Self {
        val.inner
    }
}

impl From<CoreLngLat> for LngLat {
    fn from(val: CoreLngLat) -> Self {
        Self { inner: val }
    }
}

pub mod geo {
    use super::*;

    /// Calculate the great-circle distance between two points using the Haversine formula.
    ///
    /// Uses spherical Earth approximation for fast distance calculations.
    /// Accurate to within 0.5% for distances under 1000km.
    ///
    /// Args:
    ///     a (LngLat): First coordinate
    ///     b (LngLat): Second coordinate
    ///
    /// Returns:
    ///     float: Distance in meters
    ///
    /// Examples:
    ///     >>> from rapidgeo import LngLat
    ///     >>> from rapidgeo.distance.geo import haversine
    ///     >>> sf = LngLat(-122.4194, 37.7749)
    ///     >>> nyc = LngLat(-74.0060, 40.7128)
    ///     >>> distance = haversine(sf, nyc)
    ///     >>> print(f"Distance: {distance/1000:.0f} km")
    ///     Distance: 4135 km
    #[pyfunction]
    pub fn haversine(a: LngLat, b: LngLat) -> f64 {
        geodesic::haversine(a.into(), b.into())
    }

    /// Calculate distance in kilometers using the Haversine formula.
    ///
    /// Convenient wrapper that returns distance in kilometers instead of meters.
    /// Fast spherical approximation accurate to within 0.5% for distances under 1000km.
    ///
    /// Args:
    ///     a (LngLat): First coordinate
    ///     b (LngLat): Second coordinate
    ///
    /// Returns:
    ///     float: Distance in kilometers
    ///
    /// Examples:
    ///     >>> from rapidgeo import LngLat
    ///     >>> from rapidgeo.distance.geo import haversine_km
    ///     >>> sf = LngLat(-122.4194, 37.7749)
    ///     >>> nyc = LngLat(-74.0060, 40.7128)
    ///     >>> distance = haversine_km(sf, nyc)
    ///     >>> print(f"Distance: {distance:.0f} km")
    ///     Distance: 4135 km
    #[pyfunction]
    pub fn haversine_km(a: LngLat, b: LngLat) -> f64 {
        geodesic::haversine_km(a.into(), b.into())
    }

    /// Calculate distance in statute miles using the Haversine formula.
    ///
    /// Convenient wrapper that returns distance in statute miles.
    /// Fast spherical approximation accurate to within 0.5% for distances under 1000km.
    ///
    /// Args:
    ///     a (LngLat): First coordinate
    ///     b (LngLat): Second coordinate
    ///
    /// Returns:
    ///     float: Distance in statute miles
    ///
    /// Examples:
    ///     >>> from rapidgeo import LngLat
    ///     >>> from rapidgeo.distance.geo import haversine_miles
    ///     >>> sf = LngLat(-122.4194, 37.7749)
    ///     >>> nyc = LngLat(-74.0060, 40.7128)
    ///     >>> distance = haversine_miles(sf, nyc)
    ///     >>> print(f"Distance: {distance:.0f} miles")
    ///     Distance: 2570 miles
    #[pyfunction]
    pub fn haversine_miles(a: LngLat, b: LngLat) -> f64 {
        geodesic::haversine_miles(a.into(), b.into())
    }

    /// Calculate distance in nautical miles using the Haversine formula.
    ///
    /// Convenient wrapper that returns distance in nautical miles.
    /// Fast spherical approximation accurate to within 0.5% for distances under 1000km.
    ///
    /// Args:
    ///     a (LngLat): First coordinate
    ///     b (LngLat): Second coordinate
    ///
    /// Returns:
    ///     float: Distance in nautical miles
    ///
    /// Examples:
    ///     >>> from rapidgeo import LngLat
    ///     >>> from rapidgeo.distance.geo import haversine_nautical
    ///     >>> sf = LngLat(-122.4194, 37.7749)
    ///     >>> nyc = LngLat(-74.0060, 40.7128)
    ///     >>> distance = haversine_nautical(sf, nyc)
    ///     >>> print(f"Distance: {distance:.0f} nm")
    ///     Distance: 2232 nm
    #[pyfunction]
    pub fn haversine_nautical(a: LngLat, b: LngLat) -> f64 {
        geodesic::haversine_nautical(a.into(), b.into())
    }

    /// Calculate the initial bearing from one point to another.
    ///
    /// Returns the compass bearing (azimuth) in degrees from the first point
    /// to the second point along the great circle path.
    ///
    /// Args:
    ///     from_point (LngLat): Starting coordinate
    ///     to_point (LngLat): Destination coordinate
    ///
    /// Returns:
    ///     float: Initial bearing in degrees (0-360°, where 0° is North)
    ///
    /// Examples:
    ///     >>> from rapidgeo import LngLat
    ///     >>> from rapidgeo.distance.geo import bearing
    ///     >>> sf = LngLat(-122.4194, 37.7749)
    ///     >>> nyc = LngLat(-74.0060, 40.7128)
    ///     >>> bearing_deg = bearing(sf, nyc)
    ///     >>> print(f"Bearing: {bearing_deg:.1f}°")
    ///     Bearing: 65.4°
    #[pyfunction]
    pub fn bearing(from_point: LngLat, to_point: LngLat) -> f64 {
        geodesic::bearing(from_point.into(), to_point.into())
    }

    /// Calculate the destination point given origin, distance, and bearing.
    ///
    /// Uses spherical trigonometry to find the point that is at the specified
    /// distance and bearing from the origin point.
    ///
    /// Args:
    ///     origin (LngLat): Starting coordinate
    ///     distance_m (float): Distance to travel in meters
    ///     bearing_deg (float): Compass bearing in degrees (0-360°, where 0° is North)
    ///
    /// Returns:
    ///     LngLat: Destination coordinate
    ///
    /// Examples:
    ///     >>> from rapidgeo import LngLat
    ///     >>> from rapidgeo.distance.geo import destination
    ///     >>> london = LngLat(-0.1278, 51.5074)
    ///     >>> dest = destination(london, 100000, 90)  # 100km due east
    ///     >>> print(f"Destination: {dest.lng:.4f}, {dest.lat:.4f}")
    ///     Destination: 1.2644, 51.5074
    #[pyfunction]
    pub fn destination(origin: LngLat, distance_m: f64, bearing_deg: f64) -> LngLat {
        geodesic::destination(origin.into(), distance_m, bearing_deg).into()
    }

    /// Calculate high-precision distance using Vincenty's formulae for the WGS84 ellipsoid.
    ///
    /// Provides millimeter accuracy for geodesic distances but slower than Haversine.
    /// May fail for nearly antipodal points (opposite sides of Earth).
    ///
    /// Args:
    ///     a (LngLat): First coordinate
    ///     b (LngLat): Second coordinate
    ///
    /// Returns:
    ///     float: Distance in meters with millimeter precision
    ///
    /// Raises:
    ///     ValueError: If the algorithm fails to converge for antipodal points
    ///
    /// Examples:
    ///     >>> from rapidgeo import LngLat
    ///     >>> from rapidgeo.distance.geo import vincenty_distance
    ///     >>> sf = LngLat(-122.4194, 37.7749)
    ///     >>> nyc = LngLat(-74.0060, 40.7128)
    ///     >>> distance = vincenty_distance(sf, nyc)
    ///     >>> print(f"Precise distance: {distance:.1f} m")
    ///     Precise distance: 4134785.2 m
    #[pyfunction]
    pub fn vincenty_distance(a: LngLat, b: LngLat) -> PyResult<f64> {
        match geodesic::vincenty_distance_m(a.into(), b.into()) {
            Ok(distance) => Ok(distance),
            Err(_) => Err(pyo3::exceptions::PyValueError::new_err(
                "Vincenty algorithm failed to converge",
            )),
        }
    }

    pub fn create_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
        let m = PyModule::new(py, "geo")?;
        m.add_function(wrap_pyfunction!(haversine, &m)?)?;
        m.add_function(wrap_pyfunction!(haversine_km, &m)?)?;
        m.add_function(wrap_pyfunction!(haversine_miles, &m)?)?;
        m.add_function(wrap_pyfunction!(haversine_nautical, &m)?)?;
        m.add_function(wrap_pyfunction!(bearing, &m)?)?;
        m.add_function(wrap_pyfunction!(destination, &m)?)?;
        m.add_function(wrap_pyfunction!(vincenty_distance, &m)?)?;
        Ok(m)
    }
}

pub mod euclid_mod {
    use super::*;

    /// Calculate Euclidean distance between coordinates treating them as points on a flat plane.
    ///
    /// Uses the Pythagorean theorem: d = √[(x₂-x₁)² + (y₂-y₁)²]
    /// Fast but only accurate for small geographic areas or projected coordinates.
    ///
    /// Args:
    ///     a (LngLat): First coordinate
    ///     b (LngLat): Second coordinate
    ///
    /// Returns:
    ///     float: Euclidean distance in decimal degrees
    ///
    /// Examples:
    ///     >>> from rapidgeo import LngLat
    ///     >>> from rapidgeo.distance.euclid import euclid
    ///     >>> p1 = LngLat(0.0, 0.0)
    ///     >>> p2 = LngLat(1.0, 1.0)
    ///     >>> distance = euclid(p1, p2)
    ///     >>> print(f"Distance: {distance:.4f} degrees")
    ///     Distance: 1.4142 degrees
    #[pyfunction]
    pub fn euclid(a: LngLat, b: LngLat) -> f64 {
        rapidgeo_distance::euclid::distance_euclid(a.into(), b.into())
    }

    /// Calculate squared Euclidean distance (avoids expensive square root).
    ///
    /// Useful for distance comparisons where you don't need the actual distance value.
    /// Faster than euclid() when you only need to compare relative distances.
    ///
    /// Args:
    ///     a (LngLat): First coordinate
    ///     b (LngLat): Second coordinate
    ///
    /// Returns:
    ///     float: Squared distance in decimal degrees²
    ///
    /// Examples:
    ///     >>> from rapidgeo.distance.euclid import squared
    ///     >>> from rapidgeo import LngLat
    ///     >>> p1 = LngLat(0.0, 0.0)
    ///     >>> p2 = LngLat(3.0, 4.0)
    ///     >>> dist_sq = squared(p1, p2)
    ///     >>> print(f"Squared distance: {dist_sq}")
    ///     Squared distance: 25.0
    #[pyfunction]
    pub fn squared(a: LngLat, b: LngLat) -> f64 {
        rapidgeo_distance::euclid::distance_squared(a.into(), b.into())
    }

    /// Calculate the minimum Euclidean distance from a point to a line segment.
    ///
    /// Projects the point onto the line segment and returns the shortest distance.
    /// Uses flat-plane geometry - not suitable for long geographic distances.
    ///
    /// Args:
    ///     point (LngLat): Point to measure from
    ///     seg_start (LngLat): Start of line segment
    ///     seg_end (LngLat): End of line segment
    ///
    /// Returns:
    ///     float: Minimum distance in decimal degrees
    ///
    /// Examples:
    ///     >>> from rapidgeo.distance.euclid import point_to_segment
    ///     >>> from rapidgeo import LngLat
    ///     >>> point = LngLat(1.0, 1.0)
    ///     >>> seg_start = LngLat(0.0, 0.0)
    ///     >>> seg_end = LngLat(2.0, 0.0)
    ///     >>> dist = point_to_segment(point, seg_start, seg_end)
    ///     >>> print(f"Distance to segment: {dist:.1f}")
    ///     Distance to segment: 1.0
    #[pyfunction]
    pub fn point_to_segment(point: LngLat, seg_start: LngLat, seg_end: LngLat) -> f64 {
        rapidgeo_distance::euclid::point_to_segment(
            point.into(),
            (seg_start.into(), seg_end.into()),
        )
    }

    /// Calculate squared distance from point to line segment (avoids square root).
    ///
    /// Faster version of point_to_segment() when you only need relative distances.
    /// Useful for finding the closest segment among many options.
    ///
    /// Args:
    ///     point (LngLat): Point to measure from
    ///     seg_start (LngLat): Start of line segment
    ///     seg_end (LngLat): End of line segment
    ///
    /// Returns:
    ///     float: Squared minimum distance in decimal degrees²
    ///
    /// Examples:
    ///     >>> from rapidgeo.distance.euclid import point_to_segment_squared
    ///     >>> from rapidgeo import LngLat
    ///     >>> point = LngLat(0.0, 1.0)
    ///     >>> seg_start = LngLat(0.0, 0.0)
    ///     >>> seg_end = LngLat(1.0, 0.0)
    ///     >>> dist_sq = point_to_segment_squared(point, seg_start, seg_end)
    ///     >>> print(f"Squared distance: {dist_sq}")
    ///     Squared distance: 1.0
    #[pyfunction]
    pub fn point_to_segment_squared(point: LngLat, seg_start: LngLat, seg_end: LngLat) -> f64 {
        rapidgeo_distance::euclid::point_to_segment_squared(
            point.into(),
            (seg_start.into(), seg_end.into()),
        )
    }

    pub fn create_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
        let m = PyModule::new(py, "euclid")?;
        m.add_function(wrap_pyfunction!(euclid, &m)?)?;
        m.add_function(wrap_pyfunction!(squared, &m)?)?;
        m.add_function(wrap_pyfunction!(point_to_segment, &m)?)?;
        m.add_function(wrap_pyfunction!(point_to_segment_squared, &m)?)?;
        Ok(m)
    }
}

pub mod batch_mod {
    use super::*;

    /// Calculate haversine distances between consecutive points in a path.
    ///
    /// Computes the great-circle distance between each pair of consecutive points
    /// using the Haversine formula. Returns a list of distances with length ``len(points) - 1``.
    ///
    /// Parameters
    /// ----------
    /// points : list[LngLat]
    ///     List of coordinates representing a path
    ///
    /// Returns
    /// -------
    /// list[float]
    ///     Distances in meters between consecutive points. Length is ``len(points) - 1``.
    ///
    /// Examples
    /// --------
    /// >>> from rapidgeo import LngLat
    /// >>> from rapidgeo.distance.batch import pairwise_haversine
    /// >>> path = [
    /// ...     LngLat(-122.4194, 37.7749),  # San Francisco
    /// ...     LngLat(-87.6298, 41.8781),   # Chicago
    /// ...     LngLat(-74.0060, 40.7128),   # New York
    /// ... ]
    /// >>> distances = pairwise_haversine(path)
    /// >>> [f"{d/1000:.0f} km" for d in distances]
    /// ['2984 km', '1145 km']
    ///
    /// Notes
    /// -----
    /// - Uses spherical Earth approximation (accurate to ±0.5% for distances <1000km)
    /// - Releases GIL during computation
    /// - For high precision, use Vincenty-based functions
    ///
    /// See Also
    /// --------
    /// path_length_haversine : Sum of all consecutive distances
    /// pairwise_bearings : Bearings between consecutive points
    #[pyfunction]
    pub fn pairwise_haversine(py: Python, points: &Bound<'_, PyList>) -> PyResult<Vec<f64>> {
        let core_pts: Vec<CoreLngLat> = points
            .iter()
            .map(|item| {
                let pt: LngLat = item.extract()?;
                Ok(pt.into())
            })
            .collect::<PyResult<Vec<_>>>()?;

        Ok(py.detach(move || {
            core_pts
                .windows(2)
                .map(|pair| rapidgeo_distance::geodesic::haversine(pair[0], pair[1]))
                .collect()
        }))
    }

    /// Calculate the total haversine distance along a path.
    ///
    /// Computes the sum of great-circle distances between all consecutive points
    /// using the Haversine formula.
    ///
    /// Parameters
    /// ----------
    /// points : list[LngLat]
    ///     List of coordinates representing a path (minimum 2 points)
    ///
    /// Returns
    /// -------
    /// float
    ///     Total path length in meters
    ///
    /// Examples
    /// --------
    /// >>> from rapidgeo import LngLat
    /// >>> from rapidgeo.distance.batch import path_length_haversine
    /// >>> route = [
    /// ...     LngLat(-122.4194, 37.7749),  # San Francisco
    /// ...     LngLat(-87.6298, 41.8781),   # Chicago
    /// ...     LngLat(-74.0060, 40.7128),   # New York
    /// ... ]
    /// >>> total_km = path_length_haversine(route) / 1000
    /// >>> print(f"Total route: {total_km:.0f} km")
    /// Total route: 4129 km
    ///
    /// Notes
    /// -----
    /// - Uses spherical Earth approximation (accurate to ±0.5% for distances <1000km)
    /// - Releases Python GIL during computation
    /// - Returns 0.0 for paths with fewer than 2 points
    /// - For millimeter precision, use ``path_length_vincenty()``
    ///
    /// See Also
    /// --------
    /// pairwise_haversine : Get individual segment distances
    /// pairwise_bearings : Get bearings between consecutive points
    #[pyfunction]
    pub fn path_length_haversine(py: Python, points: &Bound<'_, PyList>) -> PyResult<f64> {
        let core_pts: Vec<CoreLngLat> = points
            .iter()
            .map(|item| {
                let pt: LngLat = item.extract()?;
                Ok(pt.into())
            })
            .collect::<PyResult<Vec<_>>>()?;

        Ok(py.detach(move || {
            core_pts
                .windows(2)
                .map(|pair| rapidgeo_distance::geodesic::haversine(pair[0], pair[1]))
                .sum()
        }))
    }

    #[pyfunction]
    #[pyo3(signature = (paths))]
    pub fn path_length_haversine_batch(
        py: Python,
        paths: &Bound<'_, PyList>,
    ) -> PyResult<Vec<f64>> {
        use crate::formats::python_to_coordinate_input;
        use pyo3::types::PyList;

        let all_paths: Vec<Vec<CoreLngLat>> = paths
            .iter()
            .map(|path_item| {
                let path_list = path_item.cast::<PyList>()?;

                // Try to extract as LngLat objects first, fall back to coordinate input conversion
                if path_list.len() > 0 {
                    if let Ok(_first_lnglat) = path_list.get_item(0)?.extract::<LngLat>() {
                        // Path contains LngLat objects
                        return path_list
                            .iter()
                            .map(|item| {
                                let pt: LngLat = item.extract()?;
                                Ok(pt.into())
                            })
                            .collect::<PyResult<Vec<_>>>();
                    }
                }

                // Not LngLat objects, use format detection
                let input = python_to_coordinate_input(&path_item)?;
                let core_coords = rapidgeo_distance::formats::coords_to_lnglat_vec(&input);
                Ok(core_coords)
            })
            .collect::<PyResult<Vec<_>>>()?;

        Ok(py.detach(move || {
            all_paths
                .into_iter()
                .map(|core_pts| {
                    core_pts
                        .windows(2)
                        .map(|pair| rapidgeo_distance::geodesic::haversine(pair[0], pair[1]))
                        .sum()
                })
                .collect()
        }))
    }

    /// Calculate initial bearings between consecutive points in a path.
    ///
    /// Computes the compass bearing (azimuth) from each point to the next point
    /// along the great circle path. Returns a list of bearings with length ``len(points) - 1``.
    ///
    /// Bearings are measured in degrees (0-360°) clockwise from North:
    /// 0° = North, 90° = East, 180° = South, 270° = West
    ///
    /// Parameters
    /// ----------
    /// points : list[LngLat]
    ///     List of coordinates representing a path
    ///
    /// Returns
    /// -------
    /// list[float]
    ///     Initial bearings in degrees (0-360°). Length is ``len(points) - 1``.
    ///
    /// Examples
    /// --------
    /// >>> from rapidgeo import LngLat
    /// >>> from rapidgeo.distance.batch import pairwise_bearings
    /// >>> path = [
    /// ...     LngLat(0.0, 0.0),    # Origin
    /// ...     LngLat(1.0, 0.0),    # East
    /// ...     LngLat(1.0, 1.0),    # North
    /// ... ]
    /// >>> bearings = pairwise_bearings(path)
    /// >>> [f"{b:.1f}°" for b in bearings]
    /// ['90.0°', '0.0°']
    ///
    /// Notes
    /// -----
    /// - Returns initial bearing at each point (bearing changes along great circles)
    /// - Releases Python GIL during computation
    /// - Returns empty list for paths with fewer than 2 points
    /// - Handles antimeridian crossing correctly
    ///
    /// See Also
    /// --------
    /// pairwise_haversine : Distances between consecutive points
    /// bearing : Single bearing calculation
    #[pyfunction]
    pub fn pairwise_bearings(py: Python, points: &Bound<'_, PyList>) -> PyResult<Vec<f64>> {
        let core_pts: Vec<CoreLngLat> = points
            .iter()
            .map(|item| {
                let pt: LngLat = item.extract()?;
                Ok(pt.into())
            })
            .collect::<PyResult<Vec<_>>>()?;

        Ok(py.detach(move || {
            core_pts
                .windows(2)
                .map(|pair| rapidgeo_distance::geodesic::bearing(pair[0], pair[1]))
                .collect()
        }))
    }

    #[cfg(feature = "vincenty")]
    #[pyfunction]
    pub fn path_length_vincenty(py: Python, points: &Bound<'_, PyList>) -> PyResult<f64> {
        let core_pts: Vec<CoreLngLat> = points
            .iter()
            .map(|item| {
                let pt: LngLat = item.extract()?;
                Ok(pt.into())
            })
            .collect::<PyResult<Vec<_>>>()?;

        py.detach(move || -> PyResult<f64> {
            let mut total = 0.0;
            for pair in core_pts.windows(2) {
                match rapidgeo_distance::geodesic::vincenty_distance_m(pair[0], pair[1]) {
                    Ok(distance) => total += distance,
                    Err(_) => {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "Vincenty algorithm failed to converge",
                        ))
                    }
                }
            }
            Ok(total)
        })
    }

    pub fn create_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
        let m = PyModule::new(py, "batch")?;
        m.add_function(wrap_pyfunction!(pairwise_haversine, &m)?)?;
        m.add_function(wrap_pyfunction!(path_length_haversine, &m)?)?;
        m.add_function(wrap_pyfunction!(path_length_haversine_batch, &m)?)?;
        m.add_function(wrap_pyfunction!(pairwise_bearings, &m)?)?;

        #[cfg(feature = "vincenty")]
        m.add_function(wrap_pyfunction!(path_length_vincenty, &m)?)?;

        Ok(m)
    }
}

pub fn create_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "distance")?;
    m.add_class::<LngLat>()?;
    m.add_submodule(&geo::create_module(py)?)?;
    m.add_submodule(&euclid_mod::create_module(py)?)?;
    m.add_submodule(&batch_mod::create_module(py)?)?;

    // Only add numpy submodule if numpy is available
    if is_numpy_available(py) {
        use crate::numpy_batch;
        m.add_submodule(&numpy_batch::create_module(py)?)?;
    }

    Ok(m)
}
