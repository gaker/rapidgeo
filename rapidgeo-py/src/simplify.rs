#![allow(non_local_definitions)]

use pyo3::prelude::*;
use pyo3::types::PyList;
use rapidgeo_simplify::{simplify_dp_into, simplify_dp_mask, SimplifyMethod};
use rayon::prelude::*;

use crate::distance::LngLat;

fn parse_method(method: &str) -> PyResult<SimplifyMethod> {
    match method {
        "great_circle" => Ok(SimplifyMethod::GreatCircleMeters),
        "planar" => Ok(SimplifyMethod::PlanarMeters),
        "euclidean" => Ok(SimplifyMethod::EuclidRaw),
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            "Invalid method. Use 'great_circle', 'planar', or 'euclidean'",
        )),
    }
}

#[pyfunction]
#[pyo3(signature = (points, tolerance_m, method="great_circle", return_mask=false))]
pub fn douglas_peucker(
    py: Python,
    points: &Bound<'_, PyList>,
    tolerance_m: f64,
    method: &str,
    return_mask: bool,
) -> PyResult<Py<PyAny>> {
    let pts_len = points.len();
    let mut core_pts = Vec::with_capacity(pts_len);
    for item in points.iter() {
        let pt: LngLat = item.extract()?;
        core_pts.push(pt.into());
    }

    let simplify_method = parse_method(method)?;

    if return_mask {
        let result = py.detach(move || {
            let mut mask = Vec::with_capacity(pts_len);
            simplify_dp_mask(&core_pts, tolerance_m, simplify_method, &mut mask);
            mask
        });
        Ok(result.into_pyobject(py)?.into_any().unbind())
    } else {
        let result = py.detach(move || {
            let mut out = Vec::with_capacity(pts_len);
            simplify_dp_into(&core_pts, tolerance_m, simplify_method, &mut out);
            out.into_iter().map(LngLat::from).collect::<Vec<_>>()
        });
        Ok(result.into_pyobject(py)?.into_any().unbind())
    }
}

pub mod batch_mod {
    use super::*;

    #[pyfunction]
    #[pyo3(signature = (polylines, tolerance_m, method="great_circle", return_masks=false))]
    pub fn simplify_multiple(
        py: Python,
        polylines: &Bound<'_, PyList>,
        tolerance_m: f64,
        method: &str,
        return_masks: bool,
    ) -> PyResult<Py<PyAny>> {
        let simplify_method = parse_method(method)?;

        let polylines_len = polylines.len();
        let mut core_polylines = Vec::with_capacity(polylines_len);
        for polyline_item in polylines.iter() {
            let polyline: &Bound<'_, PyList> = polyline_item.downcast()?;
            let pts_len = polyline.len();
            let mut pts = Vec::with_capacity(pts_len);
            for pt_item in polyline.iter() {
                let pt: LngLat = pt_item.extract()?;
                pts.push(pt.into());
            }
            core_polylines.push(pts);
        }

        if return_masks {
            let result = py.detach(move || {
                core_polylines
                    .into_par_iter()
                    .map(|pts| {
                        let mut mask = Vec::with_capacity(pts.len());
                        simplify_dp_mask(&pts, tolerance_m, simplify_method, &mut mask);
                        mask
                    })
                    .collect::<Vec<_>>()
            });
            Ok(result.into_pyobject(py)?.into_any().unbind())
        } else {
            let result = py.detach(move || {
                core_polylines
                    .into_par_iter()
                    .map(|pts| {
                        let mut out = Vec::with_capacity(pts.len());
                        simplify_dp_into(&pts, tolerance_m, simplify_method, &mut out);
                        out.into_iter().map(LngLat::from).collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            });
            Ok(result.into_pyobject(py)?.into_any().unbind())
        }
    }

    pub fn create_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
        let m = PyModule::new(py, "batch")?;
        m.add_function(wrap_pyfunction!(simplify_multiple, &m)?)?;
        Ok(m)
    }
}

pub fn create_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "simplify")?;
    m.add_function(wrap_pyfunction!(douglas_peucker, &m)?)?;
    m.add_submodule(&batch_mod::create_module(py)?)?;
    Ok(m)
}
