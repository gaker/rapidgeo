use crate::distance::LngLat;
use pyo3::prelude::*;
use pyo3::types::PyList;
use rapidgeo_similarity::{
    frechet::{discrete_frechet_distance, discrete_frechet_distance_with_threshold},
    hausdorff::{hausdorff_distance, hausdorff_distance_with_threshold},
    LngLat as CoreLngLat, SimilarityError,
};

fn convert_similarity_error(err: SimilarityError) -> PyErr {
    match err {
        SimilarityError::EmptyInput => {
            pyo3::exceptions::PyValueError::new_err("Input polylines cannot be empty")
        }
        SimilarityError::InvalidInput(msg) => {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid input: {}", msg))
        }
    }
}

fn py_list_to_core_points(points: &Bound<'_, PyList>) -> PyResult<Vec<CoreLngLat>> {
    let len = points.len();
    let mut result = Vec::with_capacity(len); // Pre-allocate to avoid reallocations

    // Batch extraction to reduce Python C API overhead
    for i in 0..len {
        let item = points.get_item(i)?;
        let pt: LngLat = item.extract()?;
        result.push(pt.into());
    }

    Ok(result)
}

pub mod frechet_mod {
    use super::*;

    /// Calculate discrete Fréchet distance between two polylines.
    ///
    /// The Fréchet distance measures the similarity between two polygonal curves.
    /// It considers the ordering of points along each curve, making it suitable for
    /// comparing trajectories, paths, or time-series data.
    ///
    /// This implementation uses dynamic programming for optimal alignment between curves.
    /// Uses Haversine distance for point-to-point calculations.
    ///
    /// Args:
    ///     polyline_a (List[LngLat]): First polyline as list of coordinates
    ///     polyline_b (List[LngLat]): Second polyline as list of coordinates
    ///
    /// Returns:
    ///     float: Fréchet distance in meters
    ///
    /// Raises:
    ///     ValueError: If either polyline is empty
    ///
    /// Examples:
    ///     >>> from rapidgeo import LngLat
    ///     >>> from rapidgeo.similarity.frechet import discrete_frechet
    ///     >>> path1 = [LngLat(0.0, 0.0), LngLat(1.0, 0.0)]
    ///     >>> path2 = [LngLat(0.0, 1.0), LngLat(1.0, 1.0)]
    ///     >>> distance = discrete_frechet(path1, path2)
    #[pyfunction]
    pub fn discrete_frechet(
        polyline_a: &Bound<'_, PyList>,
        polyline_b: &Bound<'_, PyList>,
    ) -> PyResult<f64> {
        // Input validation
        if polyline_a.len() == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "First polyline cannot be empty",
            ));
        }

        if polyline_b.len() == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Second polyline cannot be empty",
            ));
        }

        // Size limits to prevent memory exhaustion attacks
        const MAX_POLYLINE_SIZE: usize = 10_000;
        if polyline_a.len() > MAX_POLYLINE_SIZE || polyline_b.len() > MAX_POLYLINE_SIZE {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Polyline size limited to {} points for security",
                MAX_POLYLINE_SIZE
            )));
        }

        // Memory usage check to prevent allocation bombs
        let estimated_memory = polyline_a.len() * polyline_b.len() * 8; // 8 bytes per f64
        const MAX_MEMORY_MB: usize = 100; // 100 MB limit
        if estimated_memory > MAX_MEMORY_MB * 1_000_000 {
            return Err(pyo3::exceptions::PyMemoryError::new_err(format!(
                "Computation would use {} MB, limit is {} MB",
                estimated_memory / 1_000_000,
                MAX_MEMORY_MB
            )));
        }

        let a = py_list_to_core_points(polyline_a)?;
        let b = py_list_to_core_points(polyline_b)?;

        discrete_frechet_distance(&a, &b).map_err(convert_similarity_error)
    }

    /// Calculate discrete Fréchet distance with early termination threshold.
    ///
    /// Optimized version that can terminate early if the distance exceeds a threshold.
    /// Useful when you only need to know if curves are within a certain similarity bound.
    ///
    /// Args:
    ///     polyline_a (List[LngLat]): First polyline as list of coordinates
    ///     polyline_b (List[LngLat]): Second polyline as list of coordinates
    ///     threshold (float): Maximum distance threshold in meters
    ///
    /// Returns:
    ///     float: Fréchet distance in meters, or value > threshold if exceeded
    ///
    /// Raises:
    ///     ValueError: If either polyline is empty
    ///
    /// Examples:
    ///     >>> from rapidgeo.similarity.frechet import discrete_frechet_with_threshold
    ///     >>> distance = discrete_frechet_with_threshold(path1, path2, 1000.0)
    #[pyfunction]
    pub fn discrete_frechet_with_threshold(
        polyline_a: &Bound<'_, PyList>,
        polyline_b: &Bound<'_, PyList>,
        threshold: f64,
    ) -> PyResult<f64> {
        // Input validation
        if polyline_a.len() == 0 || polyline_b.len() == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Both polylines must be non-empty",
            ));
        }

        // Size limits to prevent memory exhaustion attacks
        const MAX_POLYLINE_SIZE: usize = 10_000;
        if polyline_a.len() > MAX_POLYLINE_SIZE || polyline_b.len() > MAX_POLYLINE_SIZE {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Polyline size limited to {} points for security",
                MAX_POLYLINE_SIZE
            )));
        }

        // Memory usage check
        let estimated_memory = polyline_a.len() * polyline_b.len() * 8;
        const MAX_MEMORY_MB: usize = 100;
        if estimated_memory > MAX_MEMORY_MB * 1_000_000 {
            return Err(pyo3::exceptions::PyMemoryError::new_err(format!(
                "Computation would use {} MB, limit is {} MB",
                estimated_memory / 1_000_000,
                MAX_MEMORY_MB
            )));
        }

        let a = py_list_to_core_points(polyline_a)?;
        let b = py_list_to_core_points(polyline_b)?;

        discrete_frechet_distance_with_threshold(&a, &b, threshold)
            .map_err(convert_similarity_error)
    }

    pub fn create_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
        let m = PyModule::new(py, "frechet")?;
        m.add_function(wrap_pyfunction!(discrete_frechet, &m)?)?;
        m.add_function(wrap_pyfunction!(discrete_frechet_with_threshold, &m)?)?;
        Ok(m)
    }
}

pub mod hausdorff_mod {
    use super::*;

    /// Calculate Hausdorff distance between two polylines.
    ///
    /// The Hausdorff distance measures the maximum distance between any point in one
    /// set and the closest point in another set. Unlike Fréchet distance, it doesn't
    /// consider the ordering of points along curves.
    ///
    /// This is the symmetric version: max(directed_hausdorff(A,B), directed_hausdorff(B,A)).
    /// Uses Haversine distance for point-to-point calculations.
    ///
    /// Args:
    ///     polyline_a (List[LngLat]): First polyline as list of coordinates
    ///     polyline_b (List[LngLat]): Second polyline as list of coordinates
    ///
    /// Returns:
    ///     float: Hausdorff distance in meters
    ///
    /// Raises:
    ///     ValueError: If either polyline is empty
    ///
    /// Examples:
    ///     >>> from rapidgeo import LngLat
    ///     >>> from rapidgeo.similarity.hausdorff import hausdorff
    ///     >>> points1 = [LngLat(0.0, 0.0), LngLat(1.0, 0.0)]
    ///     >>> points2 = [LngLat(0.0, 1.0), LngLat(1.0, 1.0)]
    ///     >>> distance = hausdorff(points1, points2)
    #[pyfunction]
    pub fn hausdorff(
        polyline_a: &Bound<'_, PyList>,
        polyline_b: &Bound<'_, PyList>,
    ) -> PyResult<f64> {
        // Input validation
        if polyline_a.len() == 0 || polyline_b.len() == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Both polylines must be non-empty",
            ));
        }

        // Size limits for security (Hausdorff is O(n²) in distance calculations)
        const MAX_POLYLINE_SIZE: usize = 50_000; // Higher limit since no matrix allocation
        if polyline_a.len() > MAX_POLYLINE_SIZE || polyline_b.len() > MAX_POLYLINE_SIZE {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Polyline size limited to {} points for security",
                MAX_POLYLINE_SIZE
            )));
        }

        let a = py_list_to_core_points(polyline_a)?;
        let b = py_list_to_core_points(polyline_b)?;

        hausdorff_distance(&a, &b).map_err(convert_similarity_error)
    }

    /// Calculate Hausdorff distance with early termination threshold.
    ///
    /// Optimized version that can terminate early if the distance exceeds a threshold.
    /// Useful for filtering point sets based on similarity bounds.
    ///
    /// Args:
    ///     polyline_a (List[LngLat]): First polyline as list of coordinates
    ///     polyline_b (List[LngLat]): Second polyline as list of coordinates
    ///     threshold (float): Maximum distance threshold in meters
    ///
    /// Returns:
    ///     float: Hausdorff distance in meters, or value > threshold if exceeded
    ///
    /// Raises:
    ///     ValueError: If either polyline is empty
    ///
    /// Examples:
    ///     >>> from rapidgeo.similarity.hausdorff import hausdorff_with_threshold
    ///     >>> distance = hausdorff_with_threshold(points1, points2, 500.0)
    #[pyfunction]
    pub fn hausdorff_with_threshold(
        polyline_a: &Bound<'_, PyList>,
        polyline_b: &Bound<'_, PyList>,
        threshold: f64,
    ) -> PyResult<f64> {
        let a = py_list_to_core_points(polyline_a)?;
        let b = py_list_to_core_points(polyline_b)?;

        hausdorff_distance_with_threshold(&a, &b, threshold).map_err(convert_similarity_error)
    }

    pub fn create_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
        let m = PyModule::new(py, "hausdorff")?;
        m.add_function(wrap_pyfunction!(hausdorff, &m)?)?;
        m.add_function(wrap_pyfunction!(hausdorff_with_threshold, &m)?)?;
        Ok(m)
    }
}

pub fn create_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "similarity")?;
    m.add_submodule(&frechet_mod::create_module(py)?)?;
    m.add_submodule(&hausdorff_mod::create_module(py)?)?;
    Ok(m)
}
