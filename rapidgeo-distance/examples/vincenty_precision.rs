use rapidgeo_distance::{
    geodesic::{haversine, vincenty_distance_m, VincentyError},
    LngLat,
};

fn main() {
    println!("testing vincenty vs haversine");

    let sf = LngLat::new_deg(-122.4194, 37.7749);
    let nyc = LngLat::new_deg(-74.0060, 40.7128);

    let v_dist = vincenty_distance_m(sf, nyc).unwrap();
    let h_dist = haversine(sf, nyc);

    println!("vincenty: {:.3} km", v_dist / 1000.0);
    println!("haversine: {:.3} km", h_dist / 1000.0);
    println!("diff: {:.0} m", (v_dist - h_dist).abs());

    // try some edge cases
    let pole = LngLat::new_deg(0.0, 89.9999);
    let equator = LngLat::new_deg(0.0, 0.0);

    match vincenty_distance_m(pole, equator) {
        Ok(d) => println!("pole to equator: {:.0} km", d / 1000.0),
        Err(VincentyError::DidNotConverge) => {
            println!("vincenty failed, falling back to haversine");
            println!("fallback: {:.0} km", haversine(pole, equator) / 1000.0);
        }
        Err(VincentyError::Domain) => println!("bad coordinates"),
    }

    // test with garbage input
    let bad_coord = LngLat::new_deg(f64::NAN, 0.0);
    match vincenty_distance_m(sf, bad_coord) {
        Ok(_) => println!("unexpectedly worked"),
        Err(e) => println!("rejected bad input: {:?}", e),
    }
}
