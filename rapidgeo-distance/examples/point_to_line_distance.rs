use rapidgeo_distance::{
    euclid::{point_to_segment, point_to_segment_squared},
    geodesic::{great_circle_point_to_seg, point_to_segment_enu_m},
    LngLat,
};

fn main() {
    // sf to la route
    let sf = LngLat::new_deg(-122.4194, 37.7749);
    let la = LngLat::new_deg(-118.2437, 34.0522);
    let segment = (sf, la);

    // some point off the route
    let fresno = LngLat::new_deg(-119.7871, 36.7378);

    println!("distance from fresno to sf-la route:");
    println!("  enu: {:.0} m", point_to_segment_enu_m(fresno, segment));
    println!(
        "  great circle: {:.0} m",
        great_circle_point_to_seg(fresno, segment)
    );
    println!("  euclidean: {:.6} deg", point_to_segment(fresno, segment));
    println!(
        "  euclidean²: {:.6} deg²",
        point_to_segment_squared(fresno, segment)
    );

    // simple test case
    let start = LngLat::new_deg(-122.0, 37.0);
    let end = LngLat::new_deg(-120.0, 37.0); // 2 deg east
    let horizontal_segment = (start, end);

    // point directly above midpoint
    let above = LngLat::new_deg(-121.0, 38.0);

    println!("\npoint above horizontal segment:");
    println!(
        "  enu: {:.0} m",
        point_to_segment_enu_m(above, horizontal_segment)
    );
    println!(
        "  euclidean: {:.3} deg",
        point_to_segment(above, horizontal_segment)
    );

    // degenerate case - zero length segment
    let zero_seg = (sf, sf);
    println!("\nzero-length segment (should equal point-to-point):");
    println!("  distance: {:.0} m", point_to_segment_enu_m(la, zero_seg));
}
