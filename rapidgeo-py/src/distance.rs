#![allow(non_local_definitions)]

use pyo3::prelude::*;
use pyo3::types::PyList;
use rapidgeo_distance::{geodesic, LngLat as CoreLngLat};

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

    #[cfg(feature = "numpy")]
    {
        use crate::numpy_batch;
        m.add_submodule(&numpy_batch::create_module(py)?)?;
    }

    Ok(m)
}
