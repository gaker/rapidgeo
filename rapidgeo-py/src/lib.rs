use pyo3::prelude::*;

mod distance;
#[cfg(feature = "numpy")]
mod numpy_batch;
mod polyline;
mod similarity;
mod simplify;

use distance::{create_module as create_distance_module, LngLat};
use polyline::create_module as create_polyline_module;
use similarity::create_module as create_similarity_module;
use simplify::create_module as create_simplify_module;

#[pymodule]
fn _rapidgeo(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;

    // Add LngLat class directly to main module
    m.add_class::<LngLat>()?;

    // Add submodules
    m.add_submodule(&create_distance_module(py)?)?;
    m.add_submodule(&create_simplify_module(py)?)?;
    m.add_submodule(&create_polyline_module(py)?)?;
    m.add_submodule(&create_similarity_module(py)?)?;

    Ok(())
}
