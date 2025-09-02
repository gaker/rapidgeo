//! Coordinate format detection and conversion utilities.
//!
//! This module provides functions to convert Python coordinate data into standardized
//! longitude/latitude format for processing by the RapidGeo system. It automatically
//! detects the input format and converts to the internal representation efficiently.
//!
//! The system follows the lng,lat coordinate ordering convention (longitude first,
//! latitude second) throughout the API.
//!
//! # Supported Formats
//!
//! - **Tuple format**: ``[(lng, lat), (lng, lat), ...]``
//! - **Flat array**: ``[lng1, lat1, lng2, lat2, ...]``  
//! - **GeoJSON-like**: ``[{"coordinates": [lng, lat]}, ...]``
//!
//! # Performance Notes
//!
//! Format detection uses minimal inspection of the input data structure.
//! The conversion process delegates to optimized Rust core functions for
//! bulk processing operations.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use rapidgeo_distance::formats::{coords_to_lnglat_vec, CoordinateInput, GeoPoint};

#[cfg(feature = "numpy")]
use numpy::{PyArray1, PyArray2, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};

use crate::distance::LngLat;

/// Converts Python coordinate data to standardized LngLat format.
///
/// Automatically detects the input format and converts coordinate data to the
/// internal LngLat representation used throughout the RapidGeo system.
/// All coordinates are normalized to longitude/latitude ordering.
///
/// # Arguments
///
/// * ``coords`` - Python coordinate data in one of the supported formats
///
/// # Returns
///
/// A vector of ``LngLat`` objects representing the coordinates in longitude/latitude order.
///
/// # Errors
///
/// * ``PyTypeError`` - If the input is not a list or cannot be converted
/// * ``PyValueError`` - If coordinate data is malformed (e.g., wrong number of elements)
/// * ``PyKeyError`` - If GeoJSON objects are missing required keys
///
/// # Format Detection
///
/// The function automatically detects the input format by inspecting the first element:
/// - If the first element is a number, treats input as flat array format
/// - If the first element is a dictionary with "coordinates" key, treats as GeoJSON
/// - Otherwise, treats as tuple/list format
///
/// # Examples
///
/// Tuple Format::
///
///     coords = [(1.0, 2.0), (3.0, 4.0)]
///     result = coords_to_lnglat(coords)  # Returns [LngLat(1.0, 2.0), LngLat(3.0, 4.0)]
///
/// Flat Array Format::
///
///     coords = [1.0, 2.0, 3.0, 4.0]  # lng1, lat1, lng2, lat2
///     result = coords_to_lnglat(coords)  # Returns [LngLat(1.0, 2.0), LngLat(3.0, 4.0)]
///
/// GeoJSON-like Format::
///
///     coords = [
///         {"coordinates": [1.0, 2.0]},
///         {"coordinates": [3.0, 4.0]}
///     ]
///     result = coords_to_lnglat(coords)  # Returns [LngLat(1.0, 2.0), LngLat(3.0, 4.0)]
///     
/// # Performance
///
/// The function performs minimal format detection overhead and delegates bulk
/// processing to optimized Rust core functions. Conversion is done in-place
/// where possible to minimize memory allocations.
#[pyfunction]
pub fn coords_to_lnglat(_py: Python, coords: &Bound<'_, PyAny>) -> PyResult<Vec<LngLat>> {
    let input = python_to_coordinate_input(coords)?;
    let core_coords = coords_to_lnglat_vec(&input);
    Ok(core_coords.into_iter().map(LngLat::from).collect())
}

/// Converts Python coordinate data to the internal CoordinateInput enum.
///
/// Performs format detection by examining the first element of the input list
/// and dispatches to the appropriate parsing function. This function handles
/// the core logic of determining which coordinate format is being used.
///
/// # Arguments
///
/// * ``coords`` - Python object expected to be a list of coordinates
///
/// # Returns
///
/// A ``CoordinateInput`` enum variant containing the parsed coordinate data.
///
/// # Errors
///
/// * ``PyTypeError`` - If the input cannot be converted to a Python list
///
/// # Format Detection Logic
///
/// 1. Empty list → Returns empty tuple format
/// 2. First element is numeric → Flat array format
/// 3. First element is dict with "coordinates" → GeoJSON format  
/// 4. Otherwise → Tuple/list format
pub fn python_to_coordinate_input(coords: &Bound<'_, PyAny>) -> PyResult<CoordinateInput> {
    // Try NumPy array first (zero-copy when possible)
    #[cfg(feature = "numpy")]
    {
        // PRIORITY 1: Contiguous 2D arrays (FASTEST PATH)
        if let Ok(array) = coords.downcast::<PyArray2<f64>>() {
            return parse_numpy_2d_array(array);
        }

        // PRIORITY 2: Dynamic f64 arrays
        if let Ok(array) = coords.downcast::<PyArrayDyn<f64>>() {
            return parse_numpy_array(array);
        }

        // PRIORITY 3: 1D f64 arrays
        if let Ok(array) = coords.downcast::<PyArray1<f64>>() {
            return parse_numpy_array_1d(array);
        }

        // PRIORITY 4: Object arrays (SLOWEST - but still better than Python loops)
        if let Ok(array) = coords.downcast::<PyArrayDyn<Py<PyAny>>>() {
            return parse_numpy_object_array(array);
        }
    }

    // Fall back to Python list processing
    let py_list = coords.downcast::<PyList>()?;

    if py_list.len() == 0 {
        return Ok(CoordinateInput::Tuples(vec![]));
    }

    let first_item = py_list.get_item(0)?;

    if first_item.extract::<f64>().is_ok() {
        return parse_flat_array(py_list);
    }

    if let Ok(dict) = first_item.downcast::<PyDict>() {
        if dict.contains("coordinates")? {
            return parse_geojson_list(py_list);
        }
    }

    parse_tuple_list(py_list)
}

/// Parses a flat array of coordinates in the format ``[lng1, lat1, lng2, lat2, ...]``.
///
/// Extracts floating-point coordinate values from a Python list where coordinates
/// are stored as alternating longitude and latitude values.
///
/// # Arguments
///
/// * ``py_list`` - Python list containing numeric coordinate values
///
/// # Returns
///
/// A ``CoordinateInput::FlatArray`` containing the extracted coordinate values.
///
/// # Errors
///
/// * ``PyTypeError`` - If any list element cannot be converted to f64
///
/// # Expected Format
///
/// ::
///
///     coords = [lng1, lat1, lng2, lat2, lng3, lat3, ...]
///
/// The list length should be even (pairs of lng/lat values).
#[inline]
fn parse_flat_array(py_list: &Bound<'_, PyList>) -> PyResult<CoordinateInput> {
    let len = py_list.len();
    let mut flat = Vec::with_capacity(len);

    for item in py_list.iter() {
        flat.push(item.extract::<f64>()?);
    }

    Ok(CoordinateInput::FlatArray(flat))
}

/// Parses a list of GeoJSON-like coordinate objects.
///
/// Processes a list where each element is a dictionary containing coordinate
/// data in GeoJSON Point format. Each object must have a "coordinates" key
/// with a two-element array of ``[longitude, latitude]`` values.
///
/// # Arguments
///
/// * ``py_list`` - Python list of dictionary objects with GeoJSON structure
///
/// # Returns
///
/// A ``CoordinateInput::GeoJson`` containing the parsed GeoPoint objects.
///
/// # Errors
///
/// * ``PyTypeError`` - If list elements are not dictionaries
/// * ``PyKeyError`` - If any dictionary lacks a "coordinates" key
/// * ``PyValueError`` - If coordinate arrays don't have exactly 2 elements
///
/// # Expected Format
///
/// ```python
/// coords = [
///     {"coordinates": [lng1, lat1]},
///     {"coordinates": [lng2, lat2]},
///     # ... additional GeoJSON-like objects
/// ]
/// ```
#[inline]
fn parse_geojson_list(py_list: &Bound<'_, PyList>) -> PyResult<CoordinateInput> {
    let len = py_list.len();
    let mut geo_points = Vec::with_capacity(len);

    for item in py_list.iter() {
        geo_points.push(parse_geojson_point(item)?);
    }

    Ok(CoordinateInput::GeoJson(geo_points))
}

#[inline]
fn parse_geojson_point(item: Bound<'_, PyAny>) -> PyResult<GeoPoint> {
    let dict: &Bound<'_, PyDict> = item.downcast()?;
    let coords_item = dict
        .get_item("coordinates")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Missing 'coordinates' key"))?;
    let coords_list: &Bound<'_, PyList> = coords_item.downcast()?;
    if coords_list.len() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "GeoJSON coordinates must have exactly 2 elements",
        ));
    }
    let lng: f64 = coords_list.get_item(0)?.extract()?;
    let lat: f64 = coords_list.get_item(1)?.extract()?;
    Ok(GeoPoint {
        coordinates: [lng, lat],
    })
}

#[inline]
fn parse_tuple_list(py_list: &Bound<'_, PyList>) -> PyResult<CoordinateInput> {
    let len = py_list.len();
    let mut tuples = Vec::with_capacity(len);

    for item in py_list.iter() {
        // Handle both tuples and lists
        let (x, y) = if let Ok(tuple) = item.downcast::<PyTuple>() {
            // Tuple format: (lng, lat)
            let x: f64 = tuple.get_item(0)?.extract()?;
            let y: f64 = tuple.get_item(1)?.extract()?;
            (x, y)
        } else if let Ok(list) = item.downcast::<PyList>() {
            // List format: [lng, lat]
            let x: f64 = list.get_item(0)?.extract()?;
            let y: f64 = list.get_item(1)?.extract()?;
            (x, y)
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Expected tuple or list for coordinate pair",
            ));
        };

        tuples.push((x, y));
    }

    Ok(CoordinateInput::Tuples(tuples))
}

#[cfg(feature = "numpy")]
#[inline]
fn parse_numpy_2d_array(array: &Bound<'_, PyArray2<f64>>) -> PyResult<CoordinateInput> {
    // ULTRA-FAST PATH: Handle contiguous 2D arrays like [[lng, lat], [lng, lat], ...]
    let readonly = array.readonly();
    let shape = array.shape();

    // Validate shape: must be (N, 2) for coordinate pairs
    if shape[1] != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "2D NumPy array must have shape (N, 2) for coordinate pairs",
        ));
    }

    // Zero-copy access for contiguous arrays
    if array.is_c_contiguous() {
        let slice = readonly.as_slice()?;
        // Slice is [lng1, lat1, lng2, lat2, ...] - exactly what we want!
        return Ok(CoordinateInput::FlatArray(slice.to_vec()));
    }

    // Handle non-contiguous arrays (rare but possible)
    let slice = readonly.as_slice()?;
    Ok(CoordinateInput::FlatArray(slice.to_vec()))
}

#[cfg(feature = "numpy")]
#[inline]
fn parse_numpy_array(array: &Bound<'_, PyArrayDyn<f64>>) -> PyResult<CoordinateInput> {
    // Get read-only access to NumPy array
    let readonly = array.readonly();
    let slice = readonly.as_slice()?;

    // Convert to our format (still faster than Python iteration)
    Ok(CoordinateInput::FlatArray(slice.to_vec()))
}

#[cfg(feature = "numpy")]
#[inline]
fn parse_numpy_array_1d(array: &Bound<'_, PyArray1<f64>>) -> PyResult<CoordinateInput> {
    // Get read-only access to NumPy array
    let readonly = array.readonly();
    let slice = readonly.as_slice()?;

    // Convert to our format (still faster than Python iteration)
    Ok(CoordinateInput::FlatArray(slice.to_vec()))
}

#[cfg(feature = "numpy")]
#[inline]
fn parse_numpy_object_array(array: &Bound<'_, PyArrayDyn<Py<PyAny>>>) -> PyResult<CoordinateInput> {
    // Handle array-of-arrays like [array([lng1, lat1]), array([lng2, lat2]), ...]
    let readonly = array.readonly();
    let objects = readonly.as_slice()?;

    let mut tuples = Vec::with_capacity(objects.len());

    for obj in objects {
        // Each object should be a NumPy array with 2 elements
        if let Ok(coord_array) = obj.downcast_bound::<PyArrayDyn<f64>>(array.py()) {
            let coord_readonly = coord_array.readonly();
            let coord_slice = coord_readonly.as_slice()?;

            if coord_slice.len() >= 2 {
                tuples.push((coord_slice[0], coord_slice[1]));
            }
        }
    }

    Ok(CoordinateInput::Tuples(tuples))
}

/// Create module for coordinate format utilities.
pub fn create_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "formats")?;

    m.add_function(wrap_pyfunction!(coords_to_lnglat, &m)?)?;

    Ok(m)
}
