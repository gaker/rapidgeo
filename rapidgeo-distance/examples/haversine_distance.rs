use rapidgeo_distance::{geodesic::haversine, LngLat};

fn main() {
    // testing some city distances
    let sf = LngLat::new_deg(-122.4194, 37.7749);
    let nyc = LngLat::new_deg(-74.0060, 40.7128);
    let london = LngLat::new_deg(-0.1276, 51.5074);
    let tokyo = LngLat::new_deg(139.6503, 35.6762);

    println!("sf to nyc: {:.0} km", haversine(sf, nyc) / 1000.0);
    println!("sf to london: {:.0} km", haversine(sf, london) / 1000.0);
    println!("nyc to london: {:.0} km", haversine(nyc, london) / 1000.0);
    println!("tokyo to sf: {:.0} km", haversine(tokyo, sf) / 1000.0);

    // check small distances work
    let here = LngLat::new_deg(0.0, 0.0);
    let nearby = LngLat::new_deg(0.01, 0.01); // about 1.5km northeast

    println!("tiny distance: {:.0} m", haversine(here, nearby));

    // lng,lat order check
    let p1 = LngLat::new_deg(-122.0, 37.0);
    let p2 = LngLat::new_deg(-121.0, 37.0); // 1 degree east

    println!("1 deg longitude: {:.0} m", haversine(p1, p2));
    println!("should be ~111km at this lat");
}
