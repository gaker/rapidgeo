use pyo3::prelude::*;

mod distance;
#[cfg(feature = "numpy")]
mod numpy_batch;

use distance::{LngLat, geo, euclid_mod, batch_mod};

#[pymodule]
fn _rapidgeo(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;
    
    // Add LngLat class directly to main module
    m.add_class::<LngLat>()?;
    
    // Add submodules
    let distance_m = PyModule::new(py, "distance")?;
    distance_m.add_class::<LngLat>()?;
    distance_m.add_submodule(geo::create_module(py)?)?;
    distance_m.add_submodule(euclid_mod::create_module(py)?)?;
    distance_m.add_submodule(batch_mod::create_module(py)?)?;
    
    #[cfg(feature = "numpy")]
    distance_m.add_submodule(numpy_batch::create_module(py)?)?;
    
    m.add_submodule(distance_m)?;
    
    Ok(())
}