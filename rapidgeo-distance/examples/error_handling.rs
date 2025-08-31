use rapidgeo_distance::{
    geodesic::{haversine, vincenty_distance_m, VincentyError},
    LngLat,
};

fn main() {
    println!("testing error cases");

    let valid_point = LngLat::new_deg(-122.0, 37.0);

    // test invalid coordinates
    let invalid_coords = vec![
        ("NaN lng", LngLat::new_deg(f64::NAN, 37.0)),
        ("NaN lat", LngLat::new_deg(-122.0, f64::NAN)),
        ("inf lng", LngLat::new_deg(f64::INFINITY, 37.0)),
        ("inf lat", LngLat::new_deg(-122.0, f64::INFINITY)),
        ("-inf lng", LngLat::new_deg(f64::NEG_INFINITY, 37.0)),
    ];

    for (name, bad_coord) in invalid_coords {
        match vincenty_distance_m(valid_point, bad_coord) {
            Ok(d) => println!("{}: unexpectedly got {:.0} km", name, d / 1000.0),
            Err(VincentyError::Domain) => println!("{}: correctly rejected", name),
            Err(e) => println!("{}: unexpected error {:?}", name, e),
        }
    }

    // test near-antipodal points (convergence problems)
    println!("\ntesting near-antipodal convergence:");

    let base = LngLat::new_deg(0.0, 0.0);
    let near_antipodal_cases = [
        LngLat::new_deg(179.999999, 0.000001),
        LngLat::new_deg(179.9999999, 0.0000001),
        LngLat::new_deg(180.0, 0.0),
    ];

    for (i, antipodal) in near_antipodal_cases.iter().enumerate() {
        match vincenty_distance_m(base, *antipodal) {
            Ok(d) => {
                println!("case {}: converged to {:.0} km", i, d / 1000.0);
                // should be close to half earth circumference (~20003 km)
                if (d / 1000.0 - 20003.0).abs() > 1000.0 {
                    println!("  warning: distance seems wrong");
                }
            }
            Err(VincentyError::DidNotConverge) => {
                println!("case {}: failed to converge", i);
                let fallback = haversine(base, *antipodal);
                println!("  fallback haversine: {:.0} km", fallback / 1000.0);
            }
            Err(e) => println!("case {}: error {:?}", i, e),
        }
    }

    // practical error handling pattern
    println!("\npractical fallback pattern:");

    let p1 = LngLat::new_deg(0.0, 0.0);
    let p2 = LngLat::new_deg(179.999999, 0.0); // might fail

    let distance = match vincenty_distance_m(p1, p2) {
        Ok(d) => {
            println!("vincenty succeeded: {:.0} km", d / 1000.0);
            d
        }
        Err(VincentyError::DidNotConverge) => {
            println!("vincenty failed, using haversine fallback");
            let d = haversine(p1, p2);
            println!("fallback result: {:.0} km", d / 1000.0);
            d
        }
        Err(VincentyError::Domain) => {
            println!("bad coordinates, can't calculate");
            return;
        }
    };

    println!("final distance: {:.0} km", distance / 1000.0);
}
