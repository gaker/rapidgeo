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

/// Encode coordinates with line simplification into a Google Polyline string.
///
/// Combines Douglas-Peucker line simplification with polyline encoding in one step.
/// This reduces coordinate count while maintaining essential shape, then compresses
/// the result using Google's polyline algorithm.
///
/// Args:
///     coordinates (List[LngLat]): Sequence of coordinates to simplify and encode
///     tolerance_m (float): Simplification tolerance in meters. Higher values = more simplification.
///     method (str, optional): Distance calculation method. Defaults to "great_circle".
///         - "great_circle": Accurate geographic distance (recommended)
///         - "planar": Fast approximation for small areas  
///         - "euclidean": Fastest, treat coordinates as flat plane
///     precision (int, optional): Decimal places of precision (1-11). Defaults to 5.
///
/// Returns:
///     str: Simplified and encoded polyline string
///
/// Raises:
///     ValueError: If tolerance is negative, method is unknown, or precision is invalid
///
/// Examples:
///     >>> from rapidgeo import LngLat
///     >>> from rapidgeo.polyline import encode_simplified
///     >>> # GPS track with noise - simplify to 10 meter tolerance
///     >>> track = [LngLat(-120.2, 38.5), LngLat(-120.201, 38.501), LngLat(-120.95, 40.7)]
///     >>> simplified = encode_simplified(track, tolerance_m=10.0)
///     >>> # Result has fewer points than original track
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

/// Simplify an already-encoded Google Polyline string.
///
/// Decodes a polyline string, applies Douglas-Peucker simplification, then re-encodes it.
/// Useful when you have polyline data from external sources that needs simplification
/// without converting back to coordinate arrays.
///
/// Args:
///     polyline (str): Encoded polyline string to simplify
///     tolerance_m (float): Simplification tolerance in meters. Higher values = more simplification.
///     method (str, optional): Distance calculation method. Defaults to "great_circle".
///         - "great_circle": Accurate geographic distance (recommended)
///         - "planar": Fast approximation for small areas
///         - "euclidean": Fastest, treat coordinates as flat plane
///     precision (int, optional): Decimal places of precision (1-11). Defaults to 5.
///
/// Returns:
///     str: Simplified polyline string with same precision as input
///
/// Raises:
///     ValueError: If polyline is malformed, tolerance is negative, method is unknown, or precision is invalid
///
/// Examples:
///     >>> from rapidgeo.polyline import simplify_polyline
///     >>> # Simplify detailed polyline from mapping API to reduce size
///     >>> detailed = '_p~iF~ps|U_ulLnnqC_c~vLvxq`@'
///     >>> simplified = simplify_polyline(detailed, tolerance_m=50.0)
///     >>> # Result string is shorter with fewer encoded points
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

/// Encode multiple sequences of coordinates into Google Polyline Algorithm strings.
///
/// Batch processing version of encode() that efficiently handles multiple coordinate sequences
/// in parallel. Useful for encoding many routes, tracks, or boundaries at once.
///
/// Args:
///     coordinates_list (List[List[LngLat]]): List of coordinate sequences to encode
///     precision (int, optional): Decimal places of precision (1-11). Defaults to 5.
///         - 5: ~1 meter accuracy (standard)
///         - 6: ~10 centimeter accuracy (high precision)
///
/// Returns:
///     List[str]: List of encoded polyline strings, one for each input sequence
///
/// Raises:
///     ValueError: If precision is invalid or any coordinates cause overflow
///
/// Examples:
///     >>> from rapidgeo import LngLat
///     >>> from rapidgeo.polyline import encode_batch
///     >>> routes = [
///     ...     [LngLat(-120.2, 38.5), LngLat(-120.95, 40.7)],
///     ...     [LngLat(-121.0, 39.0), LngLat(-122.0, 40.0)]
///     ... ]
///     >>> polylines = encode_batch(routes, 5)
///     >>> len(polylines)
///     2
#[cfg(feature = "batch")]
#[pyfunction]
#[pyo3(signature = (coordinates_list, precision = 5))]
fn py_encode_batch(coordinates_list: &Bound<'_, PyAny>, precision: u8) -> PyResult<Vec<String>> {
    use crate::formats::python_to_coordinate_input;
    use pyo3::types::PyList;
    use rapidgeo_distance::formats::coords_to_lnglat_vec;

    // Try to handle as Vec<Vec<LngLat>> first (existing API)
    if let Ok(lnglat_list) = coordinates_list.extract::<Vec<Vec<LngLat>>>() {
        let coord_batches: Vec<Vec<rapidgeo_distance::LngLat>> = lnglat_list
            .iter()
            .map(|coords| {
                coords
                    .iter()
                    .map(|c| rapidgeo_distance::LngLat::new_deg(c.lng(), c.lat()))
                    .collect()
            })
            .collect();

        return encode_batch(&coord_batches, precision)
            .map_err(|e| PyValueError::new_err(format!("Batch encoding error: {}", e)));
    }

    // Fallback: Handle as raw coordinate data (new functionality)
    // HYBRID APPROACH: Use tolist() for pandas Series (faster bulk processing)
    // but keep zero-copy for other types where it helps
    let py_list = if let Ok(list) = coordinates_list.downcast::<PyList>() {
        // Already a PyList - use directly (zero-copy)
        list.clone()
    } else {
        // For pandas Series - use tolist() for faster bulk processing
        // The memory allocation is worth it for the algorithmic efficiency
        let list_obj = coordinates_list.call_method0("tolist")?;
        list_obj.downcast::<PyList>()?.clone()
    };

    let mut coord_batches: Vec<Vec<rapidgeo_distance::LngLat>> = Vec::with_capacity(py_list.len());

    for item in py_list.iter() {
        // Use our optimized format conversion for each coordinate array
        let input = python_to_coordinate_input(&item)?;
        let coords = coords_to_lnglat_vec(&input);
        coord_batches.push(coords);
    }

    encode_batch(&coord_batches, precision)
        .map_err(|e| PyValueError::new_err(format!("Batch encoding error: {}", e)))
}

/// Decode multiple Google Polyline Algorithm strings back into coordinate sequences.
///
/// Batch processing version of decode() that efficiently handles multiple polyline strings
/// in parallel. Useful for decoding many encoded routes, tracks, or boundaries at once.
///
/// Args:
///     polylines (List[str]): List of encoded polyline strings to decode
///     precision (int, optional): Decimal places used during encoding. Defaults to 5.
///         Must match the precision used when the polylines were originally encoded.
///
/// Returns:
///     List[List[LngLat]]: List of decoded coordinate sequences, one for each input polyline
///
/// Raises:
///     ValueError: If any polyline is malformed or precision is invalid
///
/// Examples:
///     >>> from rapidgeo.polyline import decode_batch
///     >>> polylines = [
///     ...     '_p~iF~ps|U_ulLnnqC_mqNvxq`@',
///     ...     'u{~vFvyys@fS]'
///     ... ]
///     >>> routes = decode_batch(polylines, 5)
///     >>> len(routes)
///     2
///     >>> len(routes[0])  # Number of points in first route
///     3
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

/// Encode multiple coordinate sequences with line simplification into Google Polyline strings.
///
/// Batch processing version of encode_simplified() that efficiently handles multiple coordinate
/// sequences in parallel. Combines Douglas-Peucker line simplification with polyline encoding
/// for each sequence. Useful for processing many GPS tracks, routes, or boundaries with
/// noise reduction.
///
/// Args:
///     coordinates_list (List[List[LngLat]]): List of coordinate sequences to simplify and encode
///     tolerance_m (float): Simplification tolerance in meters. Higher values = more simplification.
///         Applied uniformly to all sequences in the batch.
///     method (str, optional): Distance calculation method. Defaults to "great_circle".
///         - "great_circle": Accurate geographic distance (recommended)
///         - "planar": Fast approximation for small areas  
///         - "euclidean": Fastest, treat coordinates as flat plane
///     precision (int, optional): Decimal places of precision (1-11). Defaults to 5.
///
/// Returns:
///     List[str]: List of simplified and encoded polyline strings, one for each input sequence
///
/// Raises:
///     ValueError: If tolerance is negative, method is unknown, or precision is invalid
///
/// Examples:
///     >>> from rapidgeo import LngLat
///     >>> from rapidgeo.polyline import encode_simplified_batch
///     >>> # Multiple GPS tracks with noise - simplify all to 10 meter tolerance
///     >>> tracks = [
///     ...     [LngLat(-120.2, 38.5), LngLat(-120.201, 38.501), LngLat(-120.95, 40.7)],
///     ...     [LngLat(-121.0, 39.0), LngLat(-121.001, 39.001), LngLat(-122.0, 40.0)]
///     ... ]
///     >>> simplified = encode_simplified_batch(tracks, tolerance_m=10.0)
///     >>> len(simplified)
///     2
///     >>> # Each result has fewer points than original tracks
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

/// Encode an entire pandas column/Series of coordinates to polylines
///
/// This is the fastest way to convert coordinate data to polylines. Takes a pandas Series
/// or any iterable of coordinate arrays and returns all polylines in one shot.
/// Uses Rayon parallelization internally for maximum performance.
///
/// Args:
///     coordinates_column: pandas Series, list, or iterable of coordinate arrays
///     precision (int, optional): Decimal places of precision (1-11). Defaults to 5.
///
/// Returns:
///     List[str]: List of encoded polyline strings, one per input coordinate array
///
/// Examples:
///     >>> df['polylines'] = rapidgeo.polyline.encode_column(df['coordinates'])
///     >>> # processes entire column at once with parallel encoding
#[pyfunction]
#[pyo3(signature = (coordinates_column, precision = 5))]
fn py_encode_column(coordinates_column: &Bound<'_, PyAny>, precision: u8) -> PyResult<Vec<String>> {
    // This is exactly the same as encode_batch but with a more descriptive name
    // and documentation focused on pandas/column usage
    py_encode_batch(coordinates_column, precision)
}

pub fn create_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "polyline")?;

    m.add_function(wrap_pyfunction!(py_encode, &m)?)?;
    m.add_function(wrap_pyfunction!(py_decode, &m)?)?;
    m.add_function(wrap_pyfunction!(py_encode_simplified, &m)?)?;
    m.add_function(wrap_pyfunction!(py_simplify_polyline, &m)?)?;
    m.add_function(wrap_pyfunction!(py_encode_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(py_decode_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(py_encode_simplified_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(py_encode_column, &m)?)?;

    Ok(m)
}
