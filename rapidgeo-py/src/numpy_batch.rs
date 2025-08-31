#[cfg(feature = "numpy")]
use pyo3::prelude::*;
#[cfg(feature = "numpy")]
use numpy::{PyArray1, PyReadonlyArray1};
#[cfg(feature = "numpy")]
use map_distance::{LngLat as CoreLngLat, geodesic};

#[cfg(feature = "numpy")]
#[pyfunction]
pub fn pairwise_haversine_numpy(py: Python, points_lng: PyReadonlyArray1<f64>, points_lat: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let lng = points_lng.as_array();
    let lat = points_lat.as_array();
    
    if lng.len() != lat.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Longitude and latitude arrays must have same length"
        ));
    }
    
    if lng.len() < 2 {
        return Ok(PyArray1::from_vec(py, vec![]).to_owned());
    }
    
    let core_pts: Vec<CoreLngLat> = lng.iter()
        .zip(lat.iter())
        .map(|(&lng, &lat)| CoreLngLat::new_deg(lng, lat))
        .collect();
    
    let result = py.allow_threads(move || {
        core_pts.windows(2)
            .map(|pair| geodesic::haversine(pair[0], pair[1]))
            .collect::<Vec<f64>>()
    });
    
    Ok(PyArray1::from_vec(py, result).to_owned())
}

#[cfg(feature = "numpy")]
#[pyfunction]
pub fn distances_to_point_numpy(py: Python, points_lng: PyReadonlyArray1<f64>, points_lat: PyReadonlyArray1<f64>, target_lng: f64, target_lat: f64) -> PyResult<Py<PyArray1<f64>>> {
    let lng = points_lng.as_array();
    let lat = points_lat.as_array();
    
    if lng.len() != lat.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Longitude and latitude arrays must have same length"
        ));
    }
    
    let core_pts: Vec<CoreLngLat> = lng.iter()
        .zip(lat.iter())
        .map(|(&lng, &lat)| CoreLngLat::new_deg(lng, lat))
        .collect();
    
    let target_core = CoreLngLat::new_deg(target_lng, target_lat);
    
    let result = py.allow_threads(move || {
        core_pts.iter()
            .map(|&point| geodesic::haversine(point, target_core))
            .collect::<Vec<f64>>()
    });
    
    Ok(PyArray1::from_vec(py, result).to_owned())
}

#[cfg(feature = "numpy")]
#[pyfunction]
pub fn path_length_haversine_numpy(py: Python, points_lng: PyReadonlyArray1<f64>, points_lat: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let lng = points_lng.as_array();
    let lat = points_lat.as_array();
    
    if lng.len() != lat.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Longitude and latitude arrays must have same length"
        ));
    }
    
    if lng.len() < 2 {
        return Ok(0.0);
    }
    
    let core_pts: Vec<CoreLngLat> = lng.iter()
        .zip(lat.iter())
        .map(|(&lng, &lat)| CoreLngLat::new_deg(lng, lat))
        .collect();
    
    Ok(py.allow_threads(move || {
        core_pts.windows(2)
            .map(|pair| geodesic::haversine(pair[0], pair[1]))
            .sum()
    }))
}

#[cfg(feature = "numpy")]
pub fn create_module(py: Python) -> PyResult<&PyModule> {
    let m = PyModule::new(py, "numpy")?;
    m.add_function(wrap_pyfunction!(pairwise_haversine_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(distances_to_point_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(path_length_haversine_numpy, m)?)?;
    Ok(m)
}