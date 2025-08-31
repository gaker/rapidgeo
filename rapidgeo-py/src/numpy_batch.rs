use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rapidgeo_distance::{geodesic, LngLat as CoreLngLat};
use rayon::prelude::*;

#[pyfunction]
pub fn pairwise_haversine_numpy(
    py: Python,
    points_lng: PyReadonlyArray1<f64>,
    points_lat: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let lng = points_lng.as_array();
    let lat = points_lat.as_array();

    if lng.len() != lat.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Longitude and latitude arrays must have same length",
        ));
    }

    if lng.len() < 2 {
        return Ok(PyArray1::from_vec(py, vec![]).into());
    }

    let core_pts: Vec<CoreLngLat> = lng
        .iter()
        .zip(lat.iter())
        .map(|(&lng, &lat)| CoreLngLat::new_deg(lng, lat))
        .collect();

    let result = py.detach(move || {
        core_pts
            .par_windows(2)
            .map(|pair| geodesic::haversine(pair[0], pair[1]))
            .collect::<Vec<f64>>()
    });

    Ok(PyArray1::from_vec(py, result).into())
}

#[cfg(feature = "numpy")]
#[pyfunction]
pub fn distances_to_point_numpy(
    py: Python,
    points_lng: PyReadonlyArray1<f64>,
    points_lat: PyReadonlyArray1<f64>,
    target_lng: f64,
    target_lat: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let lng = points_lng.as_array();
    let lat = points_lat.as_array();

    if lng.len() != lat.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Longitude and latitude arrays must have same length",
        ));
    }

    let core_pts: Vec<CoreLngLat> = lng
        .iter()
        .zip(lat.iter())
        .map(|(&lng, &lat)| CoreLngLat::new_deg(lng, lat))
        .collect();

    let target_core = CoreLngLat::new_deg(target_lng, target_lat);

    let result = py.detach(move || {
        core_pts
            .par_iter()
            .map(|&point| geodesic::haversine(point, target_core))
            .collect::<Vec<f64>>()
    });

    Ok(PyArray1::from_vec(py, result).into())
}

#[cfg(feature = "numpy")]
#[pyfunction]
pub fn path_length_haversine_numpy(
    py: Python,
    points_lng: PyReadonlyArray1<f64>,
    points_lat: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let lng = points_lng.as_array();
    let lat = points_lat.as_array();

    if lng.len() != lat.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Longitude and latitude arrays must have same length",
        ));
    }

    if lng.len() < 2 {
        return Ok(0.0);
    }

    let core_pts: Vec<CoreLngLat> = lng
        .iter()
        .zip(lat.iter())
        .map(|(&lng, &lat)| CoreLngLat::new_deg(lng, lat))
        .collect();

    Ok(py.detach(move || {
        core_pts
            .par_windows(2)
            .map(|pair| geodesic::haversine(pair[0], pair[1]))
            .sum()
    }))
}

#[cfg(feature = "numpy")]
pub fn create_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "numpy")?;
    m.add_function(wrap_pyfunction!(pairwise_haversine_numpy, &m)?)?;
    m.add_function(wrap_pyfunction!(distances_to_point_numpy, &m)?)?;
    m.add_function(wrap_pyfunction!(path_length_haversine_numpy, &m)?)?;
    Ok(m)
}
