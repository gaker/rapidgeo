use pyo3::prelude::*;

mod distance;
#[cfg(feature = "numpy")]
mod numpy_batch;

use distance::{create_module as create_distance_module, LngLat};

#[pymodule]
fn _rapidgeo(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;

    // Add LngLat class directly to main module
    m.add_class::<LngLat>()?;

    // Add submodules
    m.add_submodule(&create_distance_module(py)?)?;

    Ok(())
}
