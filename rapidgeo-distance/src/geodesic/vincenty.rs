use super::{compute_reduced_latitudes_and_trig, F};
use crate::LngLat;

/// WGS84 semi-major axis in meters
const A: f64 = 6378137.0;
/// WGS84 semi-minor axis in meters
const B: f64 = 6356752.314245;

/// Errors that can occur during [Vincenty distance calculations](https://en.wikipedia.org/wiki/Vincenty%27s_formulae).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VincentyError {
    /// The iterative algorithm failed to converge within the iteration limit.
    ///
    /// This typically occurs for nearly antipodal points (opposite sides of Earth).
    /// [Vincenty's method](https://en.wikipedia.org/wiki/Vincenty%27s_formulae) uses an iterative approach
    /// that may not converge for points that are nearly 180° apart on the Earth's surface.
    ///
    /// Consider using [`crate::geodesic::haversine()`] as a fallback for such cases.
    DidNotConverge,
    /// Invalid input coordinates (NaN or infinite values).
    ///
    /// Returned when one or more coordinate values are not finite numbers.
    /// Check that all longitude and latitude values are valid before calling.
    Domain,
}

/// Calculates the precise distance between two points using [Vincenty's inverse formula](https://en.wikipedia.org/wiki/Vincenty%27s_formulae).
///
/// Uses the [WGS84 ellipsoid](https://en.wikipedia.org/wiki/World_Geodetic_System) for high-precision calculations with ±1mm accuracy globally.
/// Slower than haversine but much more accurate, especially for long distances.
///
/// This implementation uses the iterative method described in [Vincenty (1975)](https://www.ngs.noaa.gov/PUBS_LIB/inverse.pdf)
/// for solving the inverse geodetic problem on an ellipsoid.
///
/// # Arguments
///
/// * `a` - First coordinate
/// * `b` - Second coordinate
///
/// # Returns
///
/// - `Ok(distance)` - Distance in meters
/// - `Err(VincentyError::DidNotConverge)` - Algorithm failed for nearly antipodal points
/// - `Err(VincentyError::Domain)` - Invalid coordinates (NaN/infinite)
///
/// # Examples
///
/// ```
/// use rapidgeo_distance::{LngLat, geodesic::{vincenty_distance_m, VincentyError}};
///
/// let sf = LngLat::new_deg(-122.4194, 37.7749);
/// let nyc = LngLat::new_deg(-74.0060, 40.7128);
///
/// match vincenty_distance_m(sf, nyc) {
///     Ok(distance) => {
///         assert!(distance > 4_120_000.0 && distance < 4_170_000.0);
///     },
///     Err(VincentyError::DidNotConverge) => {
///         // Use haversine as fallback for antipodal points
///         use rapidgeo_distance::geodesic::haversine;
///         let fallback_distance = haversine(sf, nyc);
///     },
///     Err(VincentyError::Domain) => {
///         // Handle invalid coordinates
///     }
/// }
///
/// // Identical points return 0
/// assert_eq!(vincenty_distance_m(sf, sf).unwrap(), 0.0);
///
/// // Invalid coordinates return Domain error
/// let invalid = LngLat::new_deg(f64::NAN, 0.0);
/// assert_eq!(vincenty_distance_m(sf, invalid), Err(VincentyError::Domain));
/// ```
///
/// # Algorithm Details
///
/// [Vincenty's formulae](https://en.wikipedia.org/wiki/Vincenty%27s_formulae) are used to calculate geodesic
/// distances with high accuracy on the [WGS84 ellipsoid](https://en.wikipedia.org/wiki/World_Geodetic_System).
/// The method is iterative and may fail to converge for nearly antipodal points
/// (points on opposite sides of the Earth).
#[inline]
fn wrap_pi(x: f64) -> f64 {
    let two_pi = std::f64::consts::TAU;
    let mut y = (x + std::f64::consts::PI) % two_pi;
    if y < 0.0 {
        y += two_pi;
    }
    y - std::f64::consts::PI
}

#[inline]
fn validate_vincenty_inputs(a: LngLat, b: LngLat) -> Result<(), VincentyError> {
    if !a.lng_deg.is_finite()
        || !a.lat_deg.is_finite()
        || !b.lng_deg.is_finite()
        || !b.lat_deg.is_finite()
    {
        return Err(VincentyError::Domain);
    }
    Ok(())
}

#[inline]
fn check_coincident_points(lng1_rad: f64, lat1_rad: f64, lng2_rad: f64, lat2_rad: f64) -> bool {
    const EPS: f64 = 1e-12;
    (lng1_rad - lng2_rad).abs() <= EPS && (lat1_rad - lat2_rad).abs() <= EPS
}

struct VincentyIteration {
    cos_sq_alpha: f64,
    sin_sigma: f64,
    cos_sigma: f64,
    sigma: f64,
    cos_2sigma_m: f64,
}

#[inline]
fn compute_sigma_values(
    lambda: f64,
    sin_u1: f64,
    cos_u1: f64,
    sin_u2: f64,
    cos_u2: f64,
) -> (f64, f64, f64) {
    let sin_lambda = lambda.sin();
    let cos_lambda = lambda.cos();

    let x = cos_u2 * sin_lambda;
    let y = cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lambda;
    let sin_sigma_sq = x * x + y * y;
    let sin_sigma = sin_sigma_sq.sqrt();

    let cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lambda;
    let sigma = sin_sigma.atan2(cos_sigma);

    (sin_sigma, cos_sigma, sigma)
}

#[inline]
fn compute_lambda_update(
    l0: f64,
    sigma_values: (f64, f64, f64), // (sin_sigma, cos_sigma, sigma)
    cos_sq_alpha: f64,
    cos_2sigma_m: f64,
    cos_u_values: (f64, f64), // (cos_u1, cos_u2)
    lambda: f64,
) -> f64 {
    let (sin_sigma, cos_sigma, sigma) = sigma_values;
    let (cos_u1, cos_u2) = cos_u_values;
    let sin_alpha = (cos_u1 * cos_u2 * lambda.sin()) / sin_sigma;
    let c = (F / 16.0) * cos_sq_alpha * (4.0 + F * (4.0 - 3.0 * cos_sq_alpha));

    l0 + (1.0 - c)
        * F
        * sin_alpha
        * (sigma
            + c * sin_sigma
                * (cos_2sigma_m + c * cos_sigma * (-1.0 + 2.0 * (cos_2sigma_m * cos_2sigma_m))))
}

#[inline]
fn compute_iteration_parameters(
    l0: f64,
    lambda: f64,
    sigma_values: (f64, f64, f64), // (sin_sigma, cos_sigma, sigma)
    u_trig: (f64, f64, f64, f64),  // (sin_u1, cos_u1, sin_u2, cos_u2)
) -> (f64, f64, f64) {
    let (sin_sigma, cos_sigma, sigma) = sigma_values;
    let (sin_u1, cos_u1, sin_u2, cos_u2) = u_trig;
    const EPS: f64 = 1e-12;

    let sin_alpha = (cos_u1 * cos_u2 * lambda.sin()) / sin_sigma;
    let cos_sq_alpha = 1.0 - sin_alpha * sin_alpha;

    let cos_2sigma_m = if cos_sq_alpha <= EPS {
        0.0
    } else {
        cos_sigma - 2.0 * sin_u1 * sin_u2 / cos_sq_alpha
    };

    let new_lambda = compute_lambda_update(
        l0,
        (sin_sigma, cos_sigma, sigma),
        cos_sq_alpha,
        cos_2sigma_m,
        (cos_u1, cos_u2),
        lambda,
    );

    (cos_sq_alpha, cos_2sigma_m, new_lambda)
}

#[inline]
fn perform_vincenty_iteration_step(
    lambda: f64,
    l0: f64,
    sin_u1: f64,
    cos_u1: f64,
    sin_u2: f64,
    cos_u2: f64,
) -> Result<(f64, VincentyIteration), VincentyIteration> {
    const EPS: f64 = 1e-12;

    let (sin_sigma, cos_sigma, sigma) =
        compute_sigma_values(lambda, sin_u1, cos_u1, sin_u2, cos_u2);

    if sin_sigma <= EPS {
        return Err(VincentyIteration {
            cos_sq_alpha: 0.0,
            sin_sigma: 0.0,
            cos_sigma: 1.0,
            sigma: 0.0,
            cos_2sigma_m: 0.0,
        });
    }

    let (cos_sq_alpha, cos_2sigma_m, new_lambda) = compute_iteration_parameters(
        l0,
        lambda,
        (sin_sigma, cos_sigma, sigma),
        (sin_u1, cos_u1, sin_u2, cos_u2),
    );

    let iteration = VincentyIteration {
        cos_sq_alpha,
        sin_sigma,
        cos_sigma,
        sigma,
        cos_2sigma_m,
    };

    Ok((new_lambda, iteration))
}

#[inline]
fn handle_iteration_result(
    iteration_result: Result<(f64, VincentyIteration), VincentyIteration>,
) -> Result<Option<(f64, VincentyIteration)>, VincentyIteration> {
    match iteration_result {
        Err(result) => Err(result), // Early termination case (sin_sigma <= EPS)
        Ok((new_lambda, iteration)) => Ok(Some((new_lambda, iteration))),
    }
}

#[inline]
fn check_convergence(new_lambda: f64, lambda: f64) -> bool {
    const EPS: f64 = 1e-12;
    (new_lambda - lambda).abs() < EPS
}

#[inline]
fn check_iteration_limit(iter_limit: &mut i32) -> Result<(), VincentyError> {
    *iter_limit -= 1;
    if *iter_limit == 0 {
        return Err(VincentyError::DidNotConverge);
    }
    Ok(())
}

#[inline]
fn vincenty_iterate(
    l0: f64,
    sin_u1: f64,
    cos_u1: f64,
    sin_u2: f64,
    cos_u2: f64,
) -> Result<VincentyIteration, VincentyError> {
    let mut lambda = l0;
    let mut iter_limit = 100;

    loop {
        let iteration_result =
            perform_vincenty_iteration_step(lambda, l0, sin_u1, cos_u1, sin_u2, cos_u2);

        match handle_iteration_result(iteration_result) {
            Err(result) => return Ok(result), // Early termination case
            Ok(Some((new_lambda, iteration))) => {
                if check_convergence(new_lambda, lambda) {
                    return Ok(iteration);
                }
                lambda = new_lambda;
            }
            Ok(None) => unreachable!(), // handle_iteration_result never returns Ok(None)
        }

        check_iteration_limit(&mut iter_limit)?;
    }
}

#[inline]
fn compute_vincenty_correction(iteration: &VincentyIteration) -> f64 {
    let a2 = A * A;
    let b2 = B * B;
    let u_sq = iteration.cos_sq_alpha * (a2 - b2) / b2;

    let big_a = 1.0 + u_sq / 16384.0 * (4096.0 + u_sq * (-768.0 + u_sq * (320.0 - 175.0 * u_sq)));
    let big_b = u_sq / 1024.0 * (256.0 + u_sq * (-128.0 + u_sq * (74.0 - 47.0 * u_sq)));

    let cos_2sigma_m2 = iteration.cos_2sigma_m * iteration.cos_2sigma_m;
    let sin_sigma2 = iteration.sin_sigma * iteration.sin_sigma;

    let delta_sigma = big_b
        * iteration.sin_sigma
        * (iteration.cos_2sigma_m
            + (big_b / 4.0)
                * (iteration.cos_sigma * (-1.0 + 2.0 * cos_2sigma_m2)
                    - (big_b / 6.0)
                        * iteration.cos_2sigma_m
                        * (-3.0 + 4.0 * sin_sigma2)
                        * (-3.0 + 4.0 * cos_2sigma_m2)));

    B * big_a * (iteration.sigma - delta_sigma)
}

type CoordinateParams = (f64, f64, f64, f64, f64, f64);

#[inline]
fn preprocess_coordinates(a: LngLat, b: LngLat) -> Result<Option<CoordinateParams>, VincentyError> {
    validate_vincenty_inputs(a, b)?;

    let (lng1_rad, lat1_rad) = a.to_radians();
    let (lng2_rad, lat2_rad) = b.to_radians();

    if check_coincident_points(lng1_rad, lat1_rad, lng2_rad, lat2_rad) {
        return Ok(None); // Indicates coincident points, distance is 0
    }

    let l0 = wrap_pi(lng2_rad - lng1_rad);
    let (_u1, _u2, sin_u1, cos_u1, sin_u2, cos_u2) =
        compute_reduced_latitudes_and_trig(lat1_rad, lat2_rad);

    Ok(Some((l0, sin_u1, cos_u1, sin_u2, cos_u2, 0.0))) // Last 0.0 is placeholder
}

#[inline]
fn check_final_iteration_state(iteration: &VincentyIteration) -> Option<f64> {
    if iteration.sin_sigma == 0.0 {
        Some(0.0)
    } else {
        None
    }
}

#[inline]
pub fn vincenty_distance_m(a: LngLat, b: LngLat) -> Result<f64, VincentyError> {
    match preprocess_coordinates(a, b)? {
        None => Ok(0.0), // Coincident points
        Some((l0, sin_u1, cos_u1, sin_u2, cos_u2, _)) => {
            let iteration = vincenty_iterate(l0, sin_u1, cos_u1, sin_u2, cos_u2)?;

            if let Some(distance) = check_final_iteration_state(&iteration) {
                return Ok(distance);
            }

            Ok(compute_vincenty_correction(&iteration))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LngLat;

    #[test]
    fn test_wrap_pi() {
        // Case where y < 0 (input causes negative intermediate result)
        let input = -5.0; // This will make y negative in the modulo operation
        let result = wrap_pi(input);
        assert!(result >= -std::f64::consts::PI);
        assert!(result <= std::f64::consts::PI);
        assert!((result - (-5.0 + 2.0 * std::f64::consts::PI)).abs() < 1e-10);

        // Case at 0.0
        let result = wrap_pi(0.0);
        assert!((result - 0.0).abs() < 1e-10);

        // Case where y > 0 (normal positive case)
        let input = 2.0; // Less than PI, so should remain unchanged
        let result = wrap_pi(input);
        assert!((result - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_validate_vincenty_inputs() {
        // Case 1: a.lng_deg is infinite
        let a_inf_lng = LngLat::new_deg(f64::INFINITY, 37.0);
        let b_valid = LngLat::new_deg(-74.0, 40.0);
        assert_eq!(
            validate_vincenty_inputs(a_inf_lng, b_valid),
            Err(VincentyError::Domain)
        );

        // Case 2: a.lat_deg is infinite
        let a_inf_lat = LngLat::new_deg(-122.0, f64::INFINITY);
        let b_valid = LngLat::new_deg(-74.0, 40.0);
        assert_eq!(
            validate_vincenty_inputs(a_inf_lat, b_valid),
            Err(VincentyError::Domain)
        );

        // Case 3: b.lng_deg is infinite
        let a_valid = LngLat::new_deg(-122.0, 37.0);
        let b_inf_lng = LngLat::new_deg(f64::INFINITY, 40.0);
        assert_eq!(
            validate_vincenty_inputs(a_valid, b_inf_lng),
            Err(VincentyError::Domain)
        );

        // Case 4: b.lat_deg is infinite
        let a_valid = LngLat::new_deg(-122.0, 37.0);
        let b_inf_lat = LngLat::new_deg(-74.0, f64::INFINITY);
        assert_eq!(
            validate_vincenty_inputs(a_valid, b_inf_lat),
            Err(VincentyError::Domain)
        );

        // Case 5: Both points are valid (should pass)
        let a_valid = LngLat::new_deg(-122.4194, 37.7749);
        let b_valid = LngLat::new_deg(-74.0060, 40.7128);
        assert_eq!(validate_vincenty_inputs(a_valid, b_valid), Ok(()));
    }

    #[test]
    fn test_check_coincident_points() {
        const EPS: f64 = 1e-12;

        // Exactly identical points
        assert!(check_coincident_points(0.0, 0.0, 0.0, 0.0));
        assert!(check_coincident_points(1.5, 0.7, 1.5, 0.7));

        // Points within epsilon tolerance
        assert!(check_coincident_points(0.0, 0.0, EPS * 0.5, EPS * 0.5));
        assert!(check_coincident_points(
            1.0,
            1.0,
            1.0 + EPS * 0.5,
            1.0 + EPS * 0.5
        ));

        // Points just outside epsilon tolerance (should NOT be coincident)
        assert!(!check_coincident_points(0.0, 0.0, EPS * 2.0, 0.0));
        assert!(!check_coincident_points(0.0, 0.0, 0.0, EPS * 2.0));
        assert!(!check_coincident_points(
            1.0,
            1.0,
            1.0 + EPS * 2.0,
            1.0 + EPS * 2.0
        ));

        // Clearly different points
        assert!(!check_coincident_points(0.0, 0.0, 1.0, 0.0));
        assert!(!check_coincident_points(0.0, 0.0, 0.0, 1.0));
        assert!(!check_coincident_points(-1.0, -1.0, 1.0, 1.0));
    }

    #[test]
    fn test_compute_sigma_values() {
        // Test with lambda = 0 (same meridian)
        let (sin_sigma, cos_sigma, sigma) = compute_sigma_values(
            0.0, // lambda
            0.0, // sin_u1
            1.0, // cos_u1
            0.0, // sin_u2
            1.0, // cos_u2
        );
        assert!((sin_sigma - 0.0).abs() < 1e-10);
        assert!((cos_sigma - 1.0).abs() < 1e-10);
        assert!((sigma - 0.0).abs() < 1e-10);

        // Test with 90 degree longitude difference at equator
        let lambda = std::f64::consts::PI / 2.0;
        let (sin_sigma, cos_sigma, sigma) = compute_sigma_values(
            lambda, // 90 degrees
            0.0,    // sin_u1 (equator)
            1.0,    // cos_u1 (equator)
            0.0,    // sin_u2 (equator)
            1.0,    // cos_u2 (equator)
        );
        // At equator with 90° longitude difference, sin_sigma should be 1.0
        assert!((sin_sigma - 1.0).abs() < 1e-10);
        assert!((cos_sigma - 0.0).abs() < 1e-10);
        assert!((sigma - std::f64::consts::PI / 2.0).abs() < 1e-10);

        // Test with small lambda to check numerical stability
        let small_lambda = 1e-6;
        let (sin_sigma, cos_sigma, sigma) = compute_sigma_values(
            small_lambda,
            0.1,   // sin_u1
            0.995, // cos_u1 (approximately)
            0.2,   // sin_u2
            0.98,  // cos_u2 (approximately)
        );
        assert!(sin_sigma > 0.0);
        assert!(cos_sigma > 0.0);
        assert!(sigma > 0.0);
        assert!(sigma < std::f64::consts::PI);

        // Test with antipodal-like case (lambda close to PI)
        let antipodal_lambda = std::f64::consts::PI * 0.99;
        let (sin_sigma, cos_sigma, sigma) = compute_sigma_values(
            antipodal_lambda,
            0.0, // sin_u1 (equator)
            1.0, // cos_u1 (equator)
            0.0, // sin_u2 (equator)
            1.0, // cos_u2 (equator)
        );

        // Let me debug what's actually happening
        // For equatorial points: sin_u1 = sin_u2 = 0, cos_u1 = cos_u2 = 1
        // x = cos_u2 * sin_lambda = 1 * sin(lambda) = sin(lambda)
        // y = cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lambda = 1*0 - 0*1*cos(lambda) = 0
        // sin_sigma = sqrt(x^2 + y^2) = sqrt(sin^2(lambda) + 0) = |sin(lambda)|
        // cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lambda = 0*0 + 1*1*cos(lambda) = cos(lambda)

        let expected_sin_sigma = antipodal_lambda.sin().abs();
        let expected_cos_sigma = antipodal_lambda.cos();

        assert!((sin_sigma - expected_sin_sigma).abs() < 1e-10);
        assert!((cos_sigma - expected_cos_sigma).abs() < 1e-10);
        assert!(sigma > 3.0); // Should be close to PI
    }

    #[test]
    fn test_perform_vincenty_iteration_step_early_termination() {
        // Create a case where sin_sigma <= EPS to hit the early termination branch
        // This happens when points are numerically coincident during iteration
        const EPS: f64 = 1e-12;

        // Use very small lambda and coincident-like u values to force sin_sigma ≈ 0
        let result = perform_vincenty_iteration_step(
            EPS * 0.1, // Very small lambda
            0.0,       // l0
            0.0,       // sin_u1 (both points at equator)
            1.0,       // cos_u1
            0.0,       // sin_u2 (both points at equator)
            1.0,       // cos_u2
        );

        // Should return the early termination case (Err variant)
        match result {
            Err(iteration) => {
                assert_eq!(iteration.cos_sq_alpha, 0.0);
                assert_eq!(iteration.sin_sigma, 0.0);
                assert_eq!(iteration.cos_sigma, 1.0);
                assert_eq!(iteration.sigma, 0.0);
                assert_eq!(iteration.cos_2sigma_m, 0.0);
            }
            Ok(_) => panic!("Expected early termination case, but got normal iteration"),
        }
    }

    #[test]
    fn test_vincenty_distance_m_sin_sigma_zero_safety_net() {
        const EPS: f64 = 1e-12;

        let a = LngLat::new_deg(0.0, 0.0);
        let b = LngLat::new_deg((EPS * 1.1).to_degrees(), 0.0);

        let (lng1_rad, lat1_rad) = a.to_radians();
        let (lng2_rad, lat2_rad) = b.to_radians();
        assert!(!check_coincident_points(
            lng1_rad, lat1_rad, lng2_rad, lat2_rad
        ));

        let result = vincenty_distance_m(a, b);
        assert!(result.is_ok());

        let distance = result.unwrap();
        assert!(distance > 0.0 && distance < 1e-4);
    }

    #[test]
    fn test_handle_iteration_result() {
        // Test early termination case (Err input)
        let early_termination = VincentyIteration {
            cos_sq_alpha: 0.0,
            sin_sigma: 0.0,
            cos_sigma: 1.0,
            sigma: 0.0,
            cos_2sigma_m: 0.0,
        };
        let input = Err(early_termination);
        let result = handle_iteration_result(input);
        match result {
            Err(iteration) => {
                assert_eq!(iteration.sin_sigma, 0.0);
                assert_eq!(iteration.cos_sigma, 1.0);
            }
            Ok(_) => panic!("Expected Err, got Ok"),
        }

        // Test normal iteration case (Ok input)
        let normal_iteration = VincentyIteration {
            cos_sq_alpha: 0.5,
            sin_sigma: 0.8,
            cos_sigma: 0.6,
            sigma: 0.927,
            cos_2sigma_m: 0.3,
        };
        let input = Ok((1.5, normal_iteration));
        let result = handle_iteration_result(input);
        match result {
            Ok(Some((lambda, iteration))) => {
                assert_eq!(lambda, 1.5);
                assert_eq!(iteration.sin_sigma, 0.8);
                assert_eq!(iteration.cos_sigma, 0.6);
            }
            _ => panic!("Expected Ok(Some(_)), got something else"),
        }
    }

    #[test]
    fn test_check_convergence() {
        const EPS: f64 = 1e-12;

        // Test convergence (difference less than EPS)
        assert!(check_convergence(1.0, 1.0 + EPS * 0.5));
        assert!(check_convergence(0.0, EPS * 0.9));

        // Test non-convergence (difference greater than EPS)
        assert!(!check_convergence(1.0, 1.0 + EPS * 2.0));
        assert!(!check_convergence(0.0, EPS * 2.0));
        assert!(!check_convergence(1.0, 2.0));
    }

    #[test]
    fn test_check_iteration_limit() {
        // Test normal case (limit not reached)
        let mut limit = 5;
        assert!(check_iteration_limit(&mut limit).is_ok());
        assert_eq!(limit, 4);

        // Test limit approaching zero (should still be ok)
        let mut limit = 2;
        assert!(check_iteration_limit(&mut limit).is_ok());
        assert_eq!(limit, 1);

        // Test limit reached (should return error) - when limit is 1, it becomes 0 after decrement, triggering error
        let mut limit = 1;
        match check_iteration_limit(&mut limit) {
            Err(VincentyError::DidNotConverge) => (),
            _ => panic!("Expected DidNotConverge error"),
        }
        assert_eq!(limit, 0);

        // Test limit already zero - this becomes -1 after decrement, so it does NOT trigger the error condition
        let mut limit = 0;
        assert!(check_iteration_limit(&mut limit).is_ok());
        assert_eq!(limit, -1);
    }

    #[test]
    fn test_preprocess_coordinates() {
        // Test coincident points (should return None)
        let a = LngLat::new_deg(0.0, 0.0);
        let b = LngLat::new_deg(0.0, 0.0);
        match preprocess_coordinates(a, b).unwrap() {
            None => (), // Expected for coincident points
            Some(_) => panic!("Expected None for coincident points"),
        }

        // Test distinct points (should return Some)
        let a = LngLat::new_deg(-122.4194, 37.7749);
        let b = LngLat::new_deg(-74.0060, 40.7128);
        match preprocess_coordinates(a, b).unwrap() {
            Some((l0, sin_u1, cos_u1, sin_u2, cos_u2, _)) => {
                // Verify l0 is wrapped properly
                assert!(l0 >= -std::f64::consts::PI);
                assert!(l0 <= std::f64::consts::PI);

                // Verify trig values are in valid ranges
                assert!(sin_u1.abs() <= 1.0);
                assert!(cos_u1.abs() <= 1.0);
                assert!(sin_u2.abs() <= 1.0);
                assert!(cos_u2.abs() <= 1.0);

                // Verify trig identity: sin^2 + cos^2 = 1
                assert!((sin_u1 * sin_u1 + cos_u1 * cos_u1 - 1.0).abs() < 1e-10);
                assert!((sin_u2 * sin_u2 + cos_u2 * cos_u2 - 1.0).abs() < 1e-10);
            }
            None => panic!("Expected Some for distinct points"),
        }

        // Test invalid coordinates (should return error)
        let a_invalid = LngLat::new_deg(f64::NAN, 37.0);
        let b_valid = LngLat::new_deg(-74.0, 40.0);
        match preprocess_coordinates(a_invalid, b_valid) {
            Err(VincentyError::Domain) => (),
            _ => panic!("Expected Domain error for invalid coordinates"),
        }
    }

    #[test]
    fn test_check_final_iteration_state() {
        // Test case where sin_sigma == 0.0 (should return Some(0.0))
        let zero_iteration = VincentyIteration {
            cos_sq_alpha: 0.5,
            sin_sigma: 0.0, // This triggers the early return
            cos_sigma: 1.0,
            sigma: 0.0,
            cos_2sigma_m: 0.0,
        };
        match check_final_iteration_state(&zero_iteration) {
            Some(distance) => assert_eq!(distance, 0.0),
            None => panic!("Expected Some(0.0) for zero sin_sigma"),
        }

        // Test case where sin_sigma != 0.0 (should return None)
        let normal_iteration = VincentyIteration {
            cos_sq_alpha: 0.5,
            sin_sigma: 0.8, // Non-zero, so should return None
            cos_sigma: 0.6,
            sigma: 0.927,
            cos_2sigma_m: 0.3,
        };
        match check_final_iteration_state(&normal_iteration) {
            None => (), // Expected
            Some(_) => panic!("Expected None for non-zero sin_sigma"),
        }
    }

    #[test]
    fn test_compute_iteration_parameters() {
        const EPS: f64 = 1e-12;

        // Test normal case (cos_sq_alpha > EPS)
        let (cos_sq_alpha, cos_2sigma_m, new_lambda) = compute_iteration_parameters(
            1.0,                     // l0
            0.5,                     // lambda
            (0.8, 0.6, 0.927),       // (sin_sigma, cos_sigma, sigma)
            (0.1, 0.995, 0.2, 0.98), // (sin_u1, cos_u1, sin_u2, cos_u2)
        );

        // Verify cos_sq_alpha calculation: 1 - sin_alpha^2
        let expected_sin_alpha = (0.995 * 0.98 * 0.5_f64.sin()) / 0.8;
        let expected_cos_sq_alpha = 1.0 - expected_sin_alpha * expected_sin_alpha;
        assert!((cos_sq_alpha - expected_cos_sq_alpha).abs() < 1e-10);

        // Verify cos_2sigma_m calculation (normal case)
        let expected_cos_2sigma_m = 0.6 - 2.0 * 0.1 * 0.2 / cos_sq_alpha;
        assert!((cos_2sigma_m - expected_cos_2sigma_m).abs() < 1e-10);

        // new_lambda should be different from input lambda
        assert!((new_lambda - 0.5).abs() > 1e-6);

        // Test equatorial case (cos_sq_alpha <= EPS)
        let (cos_sq_alpha_eq, cos_2sigma_m_eq, _) = compute_iteration_parameters(
            0.0,                                    // l0
            EPS * 0.1, // Very small lambda to make sin_alpha ≈ 0, cos_sq_alpha ≈ 1
            (1.0, 0.0, std::f64::consts::PI / 2.0), // (sin_sigma, cos_sigma, sigma)
            (0.0, 1.0, 0.0, 1.0), // (sin_u1, cos_u1, sin_u2, cos_u2) - equator
        );

        // For this case, sin_alpha should be very small, cos_sq_alpha close to 1
        assert!(cos_sq_alpha_eq > 0.99); // Should be close to 1

        // If cos_sq_alpha > EPS, normal calculation should occur
        if cos_sq_alpha_eq > EPS {
            let expected_cos_2sigma_m_eq = 0.0 - 2.0 * 0.0 * 0.0 / cos_sq_alpha_eq;
            assert!((cos_2sigma_m_eq - expected_cos_2sigma_m_eq).abs() < 1e-10);
        }

        // Test case where cos_sq_alpha <= EPS (force equatorial line condition)
        // This is hard to achieve naturally, so let's create a synthetic test
        // by using parameters that would make sin_alpha ≈ 1
        let (cos_sq_alpha_small, cos_2sigma_m_small, _) = compute_iteration_parameters(
            0.0,                                    // l0
            std::f64::consts::PI / 2.0,             // lambda = 90 degrees
            (1.0, 0.0, std::f64::consts::PI / 2.0), // (sin_sigma, cos_sigma, sigma)
            (0.0, 1.0, 0.0, 1.0),                   // (sin_u1, cos_u1, sin_u2, cos_u2) - equator
        );

        // sin_alpha = (1.0 * 1.0 * sin(PI/2)) / 1.0 = 1.0
        // cos_sq_alpha = 1 - 1^2 = 0, which is <= EPS
        assert!(cos_sq_alpha_small <= EPS + 1e-10);
        assert_eq!(cos_2sigma_m_small, 0.0); // Should be 0 for equatorial line
    }
}
