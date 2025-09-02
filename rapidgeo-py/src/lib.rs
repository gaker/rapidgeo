use pyo3::prelude::*;

mod distance;
mod formats;
#[cfg(feature = "numpy")]
mod numpy_batch;
mod polyline;
mod similarity;
mod simplify;

use distance::{create_module as create_distance_module, LngLat};
use formats::create_module as create_formats_module;
use polyline::create_module as create_polyline_module;
use similarity::create_module as create_similarity_module;
use simplify::create_module as create_simplify_module;

#[pymodule]
fn _rapidgeo(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Add LngLat class directly to main module
    m.add_class::<LngLat>()?;

    // Add submodules
    m.add_submodule(&create_distance_module(py)?)?;
    m.add_submodule(&create_formats_module(py)?)?;
    m.add_submodule(&create_simplify_module(py)?)?;
    m.add_submodule(&create_polyline_module(py)?)?;
    m.add_submodule(&create_similarity_module(py)?)?;

    Ok(())
}
