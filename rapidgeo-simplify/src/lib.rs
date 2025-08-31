pub mod dp;
pub mod xt;

#[cfg(feature = "batch")]
pub mod batch;

use rapidgeo_distance::LngLat;

#[derive(Clone, Copy, Debug)]
pub enum SimplifyMethod {
    PlanarMeters,
    GreatCircleMeters,
    EuclidRaw,
}

pub fn simplify_dp_into(
    pts: &[LngLat],
    tolerance_m: f64,
    method: SimplifyMethod,
    out: &mut Vec<LngLat>,
) -> usize {
    out.clear();

    let mut mask = vec![false; pts.len()];
    simplify_dp_mask(pts, tolerance_m, method, &mut mask);

    for (i, &keep) in mask.iter().enumerate() {
        if keep {
            out.push(pts[i]);
        }
    }

    out.len()
}

pub fn simplify_dp_mask(
    pts: &[LngLat],
    tolerance_m: f64,
    method: SimplifyMethod,
    mask: &mut Vec<bool>,
) {
    use xt::*;

    match method {
        SimplifyMethod::GreatCircleMeters => {
            let backend = XtGreatCircle;
            dp::simplify_mask(pts, tolerance_m, &backend, mask);
        }
        SimplifyMethod::PlanarMeters => {
            let midpoint = compute_midpoint(pts);
            let backend = XtEnu { origin: midpoint };
            dp::simplify_mask(pts, tolerance_m, &backend, mask);
        }
        SimplifyMethod::EuclidRaw => {
            let backend = XtEuclid;
            dp::simplify_mask(pts, tolerance_m, &backend, mask);
        }
    }
}

pub(crate) fn compute_midpoint(pts: &[LngLat]) -> LngLat {
    if pts.is_empty() {
        return LngLat::new_deg(0.0, 0.0);
    }

    let mut sum_lng = 0.0;
    let mut sum_lat = 0.0;

    for pt in pts {
        sum_lng += pt.lng_deg;
        sum_lat += pt.lat_deg;
    }

    LngLat::new_deg(sum_lng / pts.len() as f64, sum_lat / pts.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_midpoint_with_empty_points() {
        let pts = vec![];
        let res = compute_midpoint(&pts);
        assert_eq!(res, LngLat::new_deg(0.0, 0.0));
    }

    #[test]
    fn test_endpoints_always_preserved() {
        let pts = vec![
            LngLat::new_deg(-122.0, 37.0),
            LngLat::new_deg(-121.5, 37.5),
            LngLat::new_deg(-121.0, 37.0),
        ];

        let mut mask = Vec::new();
        simplify_dp_mask(&pts, 1000.0, SimplifyMethod::GreatCircleMeters, &mut mask);

        assert_eq!(mask.len(), 3);
        assert!(mask[0]); // first endpoint
        assert!(mask[2]); // last endpoint
    }

    #[test]
    fn test_zero_length_segments() {
        let pts = vec![
            LngLat::new_deg(-122.0, 37.0),
            LngLat::new_deg(-122.0, 37.0),
            LngLat::new_deg(-122.0, 37.0),
        ];

        let mut mask = Vec::new();
        simplify_dp_mask(&pts, 10.0, SimplifyMethod::GreatCircleMeters, &mut mask);

        for &keep in &mask {
            assert!(keep);
        }
    }

    #[test]
    fn test_tolerance_zero_returns_original() {
        let pts = vec![
            LngLat::new_deg(-122.0, 37.0),
            LngLat::new_deg(-121.9, 37.1),
            LngLat::new_deg(-121.8, 37.2),
            LngLat::new_deg(-121.7, 37.0),
        ];

        let mut mask = Vec::new();
        simplify_dp_mask(&pts, 0.0, SimplifyMethod::GreatCircleMeters, &mut mask);

        for &keep in &mask {
            assert!(keep);
        }
    }

    #[test]
    fn test_very_large_tolerance_returns_endpoints() {
        let pts = vec![
            LngLat::new_deg(-122.0, 37.0),
            LngLat::new_deg(-121.9, 37.1),
            LngLat::new_deg(-121.8, 37.2),
            LngLat::new_deg(-121.7, 37.0),
        ];

        let mut mask = Vec::new();
        simplify_dp_mask(
            &pts,
            1_000_000.0,
            SimplifyMethod::GreatCircleMeters,
            &mut mask,
        );

        assert!(mask[0]); // first endpoint
        assert!(mask[3]); // last endpoint
        assert!(!mask[1] || !mask[2]); // at least one middle point should be removed
    }

    #[test]
    fn test_antimeridian_crossing() {
        let pts = vec![
            LngLat::new_deg(179.0, 0.0),
            LngLat::new_deg(179.5, 0.1),
            LngLat::new_deg(-179.5, 0.2),
            LngLat::new_deg(-179.0, 0.0),
        ];

        let mut mask = Vec::new();
        simplify_dp_mask(&pts, 1000.0, SimplifyMethod::GreatCircleMeters, &mut mask);

        assert!(mask[0]); // first endpoint
        assert!(mask[3]); // last endpoint
    }

    #[test]
    fn test_high_latitude_longitude_squeeze() {
        let pts = vec![
            LngLat::new_deg(-1.0, 89.0),
            LngLat::new_deg(0.0, 89.1),
            LngLat::new_deg(1.0, 89.0),
        ];

        let mut mask = Vec::new();
        simplify_dp_mask(&pts, 1000.0, SimplifyMethod::GreatCircleMeters, &mut mask);

        assert!(mask[0]); // first endpoint
        assert!(mask[2]); // last endpoint
    }

    #[test]
    fn test_simplify_dp_into() {
        let pts = vec![
            LngLat::new_deg(-122.0, 37.0),
            LngLat::new_deg(-121.5, 37.5),
            LngLat::new_deg(-121.0, 37.0),
        ];

        let mut out = Vec::new();
        let count = simplify_dp_into(&pts, 1000.0, SimplifyMethod::GreatCircleMeters, &mut out);

        assert_eq!(count, out.len());
        assert!(count >= 2); // at least endpoints
        assert_eq!(out[0], pts[0]); // first endpoint preserved
        assert_eq!(out[out.len() - 1], pts[pts.len() - 1]); // last endpoint preserved
    }

    #[test]
    fn test_different_methods() {
        let pts = vec![
            LngLat::new_deg(-122.0, 37.0),
            LngLat::new_deg(-121.5, 37.5),
            LngLat::new_deg(-121.0, 37.0),
        ];

        for method in [
            SimplifyMethod::GreatCircleMeters,
            SimplifyMethod::PlanarMeters,
            SimplifyMethod::EuclidRaw,
        ] {
            let mut mask = Vec::new();
            simplify_dp_mask(&pts, 1000.0, method, &mut mask);

            assert!(mask[0]);
            assert!(mask[2]);
        }
    }

    #[test]
    fn test_single_point() {
        let pts = vec![LngLat::new_deg(-122.0, 37.0)];

        let mut mask = Vec::new();
        simplify_dp_mask(&pts, 10.0, SimplifyMethod::GreatCircleMeters, &mut mask);

        assert_eq!(mask.len(), 1);
        assert!(mask[0]);
    }

    #[test]
    fn test_two_points() {
        let pts = vec![LngLat::new_deg(-122.0, 37.0), LngLat::new_deg(-121.0, 37.0)];

        let mut mask = Vec::new();
        simplify_dp_mask(&pts, 10.0, SimplifyMethod::GreatCircleMeters, &mut mask);

        assert_eq!(mask.len(), 2);
        assert!(mask[0]);
        assert!(mask[1]);
    }

    #[test]
    fn test_empty_points() {
        let pts = vec![];

        let mut mask = Vec::new();
        simplify_dp_mask(&pts, 10.0, SimplifyMethod::GreatCircleMeters, &mut mask);

        assert_eq!(mask.len(), 0);
    }
}
