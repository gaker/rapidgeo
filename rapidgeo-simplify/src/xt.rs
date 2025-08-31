use rapidgeo_distance::{geodesic, LngLat};

pub trait PerpDistance {
    fn d_perp_m(&self, a: LngLat, b: LngLat, p: LngLat) -> f64;
}

pub struct XtGreatCircle;

impl PerpDistance for XtGreatCircle {
    fn d_perp_m(&self, a: LngLat, b: LngLat, p: LngLat) -> f64 {
        geodesic::great_circle_point_to_seg(p, (a, b))
    }
}

pub struct XtEnu {
    pub origin: LngLat,
}

impl PerpDistance for XtEnu {
    fn d_perp_m(&self, a: LngLat, b: LngLat, p: LngLat) -> f64 {
        geodesic::point_to_segment_enu_m(p, (a, b))
    }
}

pub struct XtEuclid;

impl PerpDistance for XtEuclid {
    fn d_perp_m(&self, a: LngLat, b: LngLat, p: LngLat) -> f64 {
        let (ax, ay) = (a.lng_deg, a.lat_deg);
        let (bx, by) = (b.lng_deg, b.lat_deg);
        let (px, py) = (p.lng_deg, p.lat_deg);

        if ax == bx && ay == by {
            let dx = px - ax;
            let dy = py - ay;
            return (dx * dx + dy * dy).sqrt();
        }

        let dx = bx - ax;
        let dy = by - ay;

        let t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy);
        let t = t.clamp(0.0, 1.0);

        let proj_x = ax + t * dx;
        let proj_y = ay + t * dy;

        let dpx = px - proj_x;
        let dpy = py - proj_y;

        (dpx * dpx + dpy * dpy).sqrt()
    }
}
