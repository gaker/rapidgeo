#![allow(non_local_definitions)]

use pyo3::prelude::*;
use pyo3::types::PyList;
use rapidgeo_distance::{geodesic, LngLat as CoreLngLat};

#[pyclass]
#[derive(Clone, Copy)]
pub struct LngLat {
    inner: CoreLngLat,
}

#[pymethods]
impl LngLat {
    #[new]
    pub fn new(lng: f64, lat: f64) -> Self {
        Self {
            inner: CoreLngLat::new_deg(lng, lat),
        }
    }

    #[getter]
    pub fn lng(&self) -> f64 {
        self.inner.lng_deg
    }

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

    #[pyfunction]
    pub fn haversine(a: LngLat, b: LngLat) -> f64 {
        geodesic::haversine(a.into(), b.into())
    }

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

    #[pyfunction]
    pub fn euclid(a: LngLat, b: LngLat) -> f64 {
        rapidgeo_distance::euclid::distance_euclid(a.into(), b.into())
    }

    #[pyfunction]
    pub fn squared(a: LngLat, b: LngLat) -> f64 {
        rapidgeo_distance::euclid::distance_squared(a.into(), b.into())
    }

    #[pyfunction]
    pub fn point_to_segment(point: LngLat, seg_start: LngLat, seg_end: LngLat) -> f64 {
        rapidgeo_distance::euclid::point_to_segment(
            point.into(),
            (seg_start.into(), seg_end.into()),
        )
    }

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
