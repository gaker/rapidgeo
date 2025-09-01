use rapidgeo_polyline::{decode, encode, LngLat, PolylineError};

#[test]
fn test_coordinate_overflow_detection() {
    // Test coordinates that would overflow with high precision
    let extreme_coords = vec![
        LngLat::new_deg(1e8, 1e8), // Extreme but possible values
        LngLat::new_deg(-1e8, -1e8),
    ];

    // With precision 11, this should handle or reject gracefully
    match encode(&extreme_coords, 11) {
        Ok(encoded) => {
            // If encoding succeeds, decoding should produce consistent results
            let decoded = decode(&encoded, 11).unwrap();

            // Results should be consistent (may be clamped but not corrupted)
            for (original, decoded_coord) in extreme_coords.iter().zip(decoded.iter()) {
                // Allow for precision loss but not complete corruption
                let lng_diff = (original.lng_deg - decoded_coord.lng_deg).abs();
                let lat_diff = (original.lat_deg - decoded_coord.lat_deg).abs();

                // If difference is huge, we have silent overflow corruption
                assert!(
                    lng_diff < original.lng_deg.abs() * 0.1,
                    "Longitude corruption detected: {} vs {} (diff: {})",
                    original.lng_deg,
                    decoded_coord.lng_deg,
                    lng_diff
                );
                assert!(
                    lat_diff < original.lat_deg.abs() * 0.1,
                    "Latitude corruption detected: {} vs {} (diff: {})",
                    original.lat_deg,
                    decoded_coord.lat_deg,
                    lat_diff
                );
            }
        }
        Err(PolylineError::CoordinateOverflow) => {
            // This is the expected behavior for values that would overflow
        }
        Err(PolylineError::InvalidCoordinate(_)) => {
            // This is also expected - coordinates outside valid geographic bounds
        }
        Err(other) => {
            panic!("Unexpected error for extreme coordinates: {:?}", other);
        }
    }
}

#[test]
fn test_malicious_precision_values() {
    let coords = vec![LngLat::new_deg(0.0, 0.0)];

    // Test boundary values
    assert!(encode(&coords, 0).is_err()); // Too low
    assert!(encode(&coords, 12).is_err()); // Too high
    assert!(encode(&coords, 255).is_err()); // Way too high

    // Test valid boundaries
    assert!(encode(&coords, 1).is_ok()); // Minimum valid
    assert!(encode(&coords, 11).is_ok()); // Maximum valid
}

#[test]
fn test_nan_infinity_coordinates() {
    let problematic_coords = [
        LngLat::new_deg(f64::NAN, 0.0),
        LngLat::new_deg(0.0, f64::NAN),
        LngLat::new_deg(f64::INFINITY, 0.0),
        LngLat::new_deg(0.0, f64::INFINITY),
        LngLat::new_deg(f64::NEG_INFINITY, 0.0),
        LngLat::new_deg(0.0, f64::NEG_INFINITY),
    ];

    for coords in problematic_coords.iter().map(|c| vec![*c]) {
        // These should either error gracefully or handle consistently
        match encode(&coords, 5) {
            Ok(encoded) => {
                // If encoding succeeds, decoding should not panic or produce invalid results
                match decode(&encoded, 5) {
                    Ok(decoded) => {
                        // Verify no NaN propagation in successful decode
                        assert!(
                            !decoded[0].lng_deg.is_nan(),
                            "NaN propagated through encoding"
                        );
                        assert!(
                            !decoded[0].lat_deg.is_nan(),
                            "NaN propagated through encoding"
                        );
                        assert!(
                            decoded[0].lng_deg.is_finite(),
                            "Infinity propagated through encoding"
                        );
                        assert!(
                            decoded[0].lat_deg.is_finite(),
                            "Infinity propagated through encoding"
                        );
                    }
                    Err(_) => {
                        // Error during decode is acceptable
                    }
                }
            }
            Err(_) => {
                // Error during encode is acceptable and preferred
            }
        }
    }
}

#[test]
fn test_invalid_geographic_bounds() {
    let invalid_coords = vec![
        LngLat::new_deg(200.0, 0.0),  // Invalid longitude > 180
        LngLat::new_deg(-200.0, 0.0), // Invalid longitude < -180
        LngLat::new_deg(0.0, 100.0),  // Invalid latitude > 90
        LngLat::new_deg(0.0, -100.0), // Invalid latitude < -90
    ];

    // Currently the library doesn't validate coordinate bounds
    // This test documents the current behavior and can be updated
    // when proper validation is added
    for coord in invalid_coords {
        let result = encode(&[coord], 5);
        // Currently this succeeds but probably shouldn't for invalid geography
        // When validation is added, this should return an appropriate error
        match result {
            Ok(_) => {
                // Document current behavior: invalid coords are processed
                println!(
                    "Warning: Invalid coordinate processed: lng={}, lat={}",
                    coord.lng_deg, coord.lat_deg
                );
            }
            Err(_) => {
                // Future behavior: validation rejects invalid coordinates
            }
        }
    }
}

#[test]
fn test_malformed_polyline_strings() {
    let malformed_polylines = vec![
        "\x1f\x00\x20",      // Invalid characters (< 63)
        "\x7f\x7f\x20",      // Invalid characters (> 126)
        "_p~iF~ps|U_",       // Truncated polyline
        "",                  // Empty string (should be valid)
        "validstart\x00end", // Null bytes in middle
        "validstart\x7fend", // High bytes > 126
    ];

    for polyline in malformed_polylines {
        match decode(polyline, 5) {
            Ok(coords) => {
                if polyline.is_empty() {
                    assert_eq!(coords.len(), 0);
                } else {
                    // If decode succeeds with malformed input, that's concerning
                    println!(
                        "Warning: Malformed polyline decoded successfully: {:?}",
                        polyline
                    );
                }
            }
            Err(PolylineError::InvalidCharacter {
                character,
                position,
            }) => {
                // Expected behavior for invalid characters
                assert!(
                    !('\x3f'..='\x7e').contains(&character),
                    "Invalid character error for valid character: {:?} at {}",
                    character,
                    position
                );
            }
            Err(PolylineError::TruncatedData) => {
                // Expected behavior for truncated data
            }
            Err(other) => {
                println!("Unexpected error for malformed polyline: {:?}", other);
            }
        }
    }
}

#[test]
fn test_excessive_precision_memory_usage() {
    // Test that high precision values don't cause excessive memory allocation
    let coords = vec![LngLat::new_deg(1.0, 1.0)];

    // Test maximum allowed precision
    let result = encode(&coords, 11);
    match result {
        Ok(encoded) => {
            // Encoded string should not be unreasonably long
            assert!(
                encoded.len() < 100,
                "Excessive string length for single coordinate: {}",
                encoded.len()
            );

            // Should decode correctly
            let decoded = decode(&encoded, 11).unwrap();
            assert_eq!(decoded.len(), 1);
        }
        Err(e) => {
            println!("High precision encoding failed (may be expected): {:?}", e);
        }
    }
}

#[test]
fn test_decode_memory_exhaustion_protection() {
    // Test very long polyline strings that could cause excessive memory allocation
    let long_valid_polyline = "_p~iF~ps|U".repeat(1000); // Repeat valid segment

    match decode(&long_valid_polyline, 5) {
        Ok(coords) => {
            // Should handle long polylines gracefully
            println!("Decoded {} coordinates from long polyline", coords.len());
            assert!(!coords.is_empty());
        }
        Err(e) => {
            println!("Long polyline decode failed: {:?}", e);
            // Failing is acceptable if it's due to reasonable memory limits
        }
    }

    // Test string that suggests many coordinates but is actually invalid
    let misleading_polyline = "A".repeat(10000); // Invalid characters

    match decode(&misleading_polyline, 5) {
        Ok(coords) => {
            // If it somehow decodes, ensure the coordinates are reasonable
            for coord in coords {
                assert!(coord.lng_deg >= -180.1 && coord.lng_deg <= 180.1);
                assert!(coord.lat_deg >= -90.1 && coord.lat_deg <= 90.1);
            }
        }
        Err(PolylineError::InvalidCharacter { .. }) => {
            // Expected behavior - should fail fast on invalid characters
            // rather than allocating excessive memory
        }
        Err(PolylineError::InvalidCoordinate(_)) => {
            // Also acceptable - coordinate validation may reject decoded values
        }
        Err(other) => {
            println!("Unexpected error for invalid long string: {:?}", other);
        }
    }
}

#[test]
fn test_shift_overflow_protection() {
    // Test that bit shift operations in decode don't overflow
    // This tests the shift >= 64 check in decode_signed_number

    // Create a polyline that would cause excessive bit shifting
    // Each byte with high bit set continues the number
    let excessive_shift_polyline: String = (0..20)
        .map(|_| char::from(63 + 0x20)) // Characters that continue the number
        .chain(std::iter::once(char::from(63))) // Final byte
        .collect();

    match decode(&excessive_shift_polyline, 5) {
        Ok(_) => {
            panic!("Should not decode polyline with excessive shift operations");
        }
        Err(PolylineError::CoordinateOverflow) => {
            // Expected behavior - should detect shift overflow
        }
        Err(other) => {
            println!("Unexpected error for excessive shift: {:?}", other);
        }
    }
}
