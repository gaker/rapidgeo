use crate::distance::LngLat;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rapidgeo_polyline::{decode, encode, encode_simplified, simplify_polyline};
use rapidgeo_simplify::SimplifyMethod;

#[cfg(feature = "batch")]
use rapidgeo_polyline::batch::{decode_batch, encode_batch, encode_simplified_batch};

/// Encode a sequence of coordinates into a Google Polyline Algorithm string.
///
/// Compresses coordinate sequences using variable-length encoding and delta compression.
/// Commonly used for GPS tracks, routes, and mapping applications.
///
/// Args:
///     coordinates (List[LngLat]): Sequence of coordinates to encode
///     precision (int, optional): Decimal places of precision (1-11). Defaults to 5.
///         - 5: ~1 meter accuracy (standard)
///         - 6: ~10 centimeter accuracy (high precision)
///
/// Returns:
///     str: Encoded polyline string
///
/// Raises:
///     ValueError: If precision is invalid or coordinates cause overflow
///
/// Examples:
///     >>> from rapidgeo import LngLat
///     >>> from rapidgeo.polyline import encode
///     >>> coords = [LngLat(-120.2, 38.5), LngLat(-120.95, 40.7)]
///     >>> polyline = encode(coords, 5)
///     >>> print(polyline)
///     '_p~iF~ps|U_ulLnnqC_mqNvxq`@'
#[pyfunction]
#[pyo3(signature = (coordinates, precision = 5))]
fn py_encode(coordinates: Vec<LngLat>, precision: u8) -> PyResult<String> {
    let coords: Vec<rapidgeo_distance::LngLat> = coordinates
        .iter()
        .map(|c| rapidgeo_distance::LngLat::new_deg(c.lng(), c.lat()))
        .collect();

    encode(&coords, precision).map_err(|e| PyValueError::new_err(format!("Encoding error: {}", e)))
}

/// Decode a Google Polyline Algorithm string back into coordinates.
///
/// Reverses the polyline encoding process to reconstruct the original coordinate sequence.
/// The precision must match what was used during encoding.
///
/// Args:
///     polyline (str): Encoded polyline string
///     precision (int, optional): Decimal places used during encoding. Defaults to 5.
///
/// Returns:
///     List[LngLat]: Decoded coordinate sequence
///
/// Raises:
///     ValueError: If polyline is malformed or precision is invalid
///
/// Examples:
///     >>> from rapidgeo.polyline import decode
///     >>> polyline = '_p~iF~ps|U_ulLnnqC_mqNvxq`@'
///     >>> coords = decode(polyline, 5)
///     >>> print(f"First coord: {coords[0].lng}, {coords[0].lat}")
///     First coord: -120.2, 38.5
#[pyfunction]
#[pyo3(signature = (polyline, precision = 5))]
fn py_decode(polyline: &str, precision: u8) -> PyResult<Vec<LngLat>> {
    let coords = decode(polyline, precision)
        .map_err(|e| PyValueError::new_err(format!("Decoding error: {}", e)))?;

    Ok(coords
        .iter()
        .map(|c| LngLat::new(c.lng_deg, c.lat_deg))
        .collect())
}

#[pyfunction]
#[pyo3(signature = (coordinates, tolerance_m, method = "great_circle", precision = 5))]
fn py_encode_simplified(
    coordinates: Vec<LngLat>,
    tolerance_m: f64,
    method: &str,
    precision: u8,
) -> PyResult<String> {
    let coords: Vec<rapidgeo_distance::LngLat> = coordinates
        .iter()
        .map(|c| rapidgeo_distance::LngLat::new_deg(c.lng(), c.lat()))
        .collect();

    let simplify_method = match method {
        "great_circle" => SimplifyMethod::GreatCircleMeters,
        "planar" => SimplifyMethod::PlanarMeters,
        "euclidean" => SimplifyMethod::EuclidRaw,
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unknown simplification method: {}",
                method
            )))
        }
    };

    encode_simplified(&coords, tolerance_m, simplify_method, precision)
        .map_err(|e| PyValueError::new_err(format!("Encoding error: {}", e)))
}

#[pyfunction]
#[pyo3(signature = (polyline, tolerance_m, method = "great_circle", precision = 5))]
fn py_simplify_polyline(
    polyline: &str,
    tolerance_m: f64,
    method: &str,
    precision: u8,
) -> PyResult<String> {
    let simplify_method = match method {
        "great_circle" => SimplifyMethod::GreatCircleMeters,
        "planar" => SimplifyMethod::PlanarMeters,
        "euclidean" => SimplifyMethod::EuclidRaw,
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unknown simplification method: {}",
                method
            )))
        }
    };

    simplify_polyline(polyline, tolerance_m, simplify_method, precision)
        .map_err(|e| PyValueError::new_err(format!("Simplification error: {}", e)))
}

#[cfg(feature = "batch")]
#[pyfunction]
#[pyo3(signature = (coordinates_list, precision = 5))]
fn py_encode_batch(coordinates_list: Vec<Vec<LngLat>>, precision: u8) -> PyResult<Vec<String>> {
    let coord_batches: Vec<Vec<rapidgeo_distance::LngLat>> = coordinates_list
        .iter()
        .map(|coords| {
            coords
                .iter()
                .map(|c| rapidgeo_distance::LngLat::new_deg(c.lng(), c.lat()))
                .collect()
        })
        .collect();

    encode_batch(&coord_batches, precision)
        .map_err(|e| PyValueError::new_err(format!("Batch encoding error: {}", e)))
}

#[cfg(feature = "batch")]
#[pyfunction]
#[pyo3(signature = (polylines, precision = 5))]
fn py_decode_batch(polylines: Vec<String>, precision: u8) -> PyResult<Vec<Vec<LngLat>>> {
    let coord_batches = decode_batch(&polylines, precision)
        .map_err(|e| PyValueError::new_err(format!("Batch decoding error: {}", e)))?;

    Ok(coord_batches
        .iter()
        .map(|coords| {
            coords
                .iter()
                .map(|c| LngLat::new(c.lng_deg, c.lat_deg))
                .collect()
        })
        .collect())
}

#[cfg(feature = "batch")]
#[pyfunction]
#[pyo3(signature = (coordinates_list, tolerance_m, method = "great_circle", precision = 5))]
fn py_encode_simplified_batch(
    coordinates_list: Vec<Vec<LngLat>>,
    tolerance_m: f64,
    method: &str,
    precision: u8,
) -> PyResult<Vec<String>> {
    let coord_batches: Vec<Vec<rapidgeo_distance::LngLat>> = coordinates_list
        .iter()
        .map(|coords| {
            coords
                .iter()
                .map(|c| rapidgeo_distance::LngLat::new_deg(c.lng(), c.lat()))
                .collect()
        })
        .collect();

    let simplify_method = match method {
        "great_circle" => SimplifyMethod::GreatCircleMeters,
        "planar" => SimplifyMethod::PlanarMeters,
        "euclidean" => SimplifyMethod::EuclidRaw,
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unknown simplification method: {}",
                method
            )))
        }
    };

    encode_simplified_batch(&coord_batches, tolerance_m, simplify_method, precision)
        .map_err(|e| PyValueError::new_err(format!("Batch simplified encoding error: {}", e)))
}

pub fn create_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "polyline")?;

    m.add_function(wrap_pyfunction!(py_encode, &m)?)?;
    m.add_function(wrap_pyfunction!(py_decode, &m)?)?;
    m.add_function(wrap_pyfunction!(py_encode_simplified, &m)?)?;
    m.add_function(wrap_pyfunction!(py_simplify_polyline, &m)?)?;

    #[cfg(feature = "batch")]
    {
        m.add_function(wrap_pyfunction!(py_encode_batch, &m)?)?;
        m.add_function(wrap_pyfunction!(py_decode_batch, &m)?)?;
        m.add_function(wrap_pyfunction!(py_encode_simplified_batch, &m)?)?;
    }

    Ok(m)
}
