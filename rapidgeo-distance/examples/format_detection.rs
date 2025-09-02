use rapidgeo_distance::{
    detection::{confidence_score, detect_format},
    formats::{coords_to_lnglat_vec, CoordinateInput, GeoPoint},
};

fn main() {
    println!("RapidGeo Format Detection Examples\n");

    // Example 1: Clear LngLat format (longitude, latitude)
    println!("Example 1: Clear LngLat Format");
    let lng_lat_coords = vec![
        (-122.4194, 37.7749), // San Francisco
        (-74.0060, 40.7128),  // New York
        (-87.6298, 41.8781),  // Chicago
        (2.3522, 48.8566),    // Paris
    ];

    let format = detect_format(&lng_lat_coords);
    let confidence = confidence_score(&lng_lat_coords, format);
    println!("  Input: {:?}", lng_lat_coords);
    println!(
        "  Detected: {:?} (confidence: {:.1}%)",
        format,
        confidence * 100.0
    );
    println!("  Valid: All longitudes in [-180, 180], latitudes in [-90, 90]");
    println!("  This is LngLat format because first values are valid longitudes\n");

    // Example 2: Clear LatLng format (latitude, longitude)
    println!("Example 2: Clear LatLng Format");
    let lat_lng_coords = vec![
        (37.7749, -122.4194), // San Francisco (reversed)
        (40.7128, -74.0060),  // New York (reversed)
        (41.8781, -87.6298),  // Chicago (reversed)
        (48.8566, 2.3522),    // Paris (reversed)
    ];

    let format = detect_format(&lat_lng_coords);
    let confidence = confidence_score(&lat_lng_coords, format);
    println!("  Input: {:?}", lat_lng_coords);
    println!(
        "  Detected: {:?} (confidence: {:.1}%)",
        format,
        confidence * 100.0
    );
    println!("  Valid: All first values are valid latitudes, second values are longitudes");
    println!("  This is LatLng format because first values are in latitude range\n");

    // Example 3: Invalid coordinates (out of bounds)
    println!("Example 3: Invalid Coordinates");
    let invalid_coords = vec![
        (200.0, 37.7749),  // Invalid longitude > 180
        (-74.0060, 100.0), // Invalid latitude > 90
        (-250.0, -95.0),   // Both invalid
    ];

    let format = detect_format(&invalid_coords);
    let confidence = confidence_score(&invalid_coords, format);
    println!("  Input: {:?}", invalid_coords);
    println!(
        "  Detected: {:?} (confidence: {:.1}%)",
        format,
        confidence * 100.0
    );
    println!("  Invalid: Coordinates out of valid ranges");
    println!("  Note: Longitude must be [-180, 180], latitude must be [-90, 90]\n");

    // Example 4: Ambiguous format (equatorial/prime meridian)
    println!("Example 4: Ambiguous Format");
    let ambiguous_coords = vec![
        (0.0, 0.0),     // Origin - could be either format
        (10.0, 20.0),   // Both values are valid lat AND lng
        (-10.0, -20.0), // Both values are valid lat AND lng
    ];

    let format = detect_format(&ambiguous_coords);
    let confidence = confidence_score(&ambiguous_coords, format);
    println!("  Input: {:?}", ambiguous_coords);
    println!(
        "  Detected: {:?} (confidence: {:.1}%)",
        format,
        confidence * 100.0
    );
    println!("  Ambiguous: All values are valid for both latitude and longitude");
    println!("  Note: Algorithm assumes LngLat by default when ambiguous\n");

    // Example 5: Mixed validity (some valid, some invalid)
    println!("Example 5: Mixed Validity");
    let mixed_coords = vec![
        (-122.4194, 37.7749), // Valid LngLat
        (40.7128, -74.0060),  // Valid LatLng
        (200.0, 37.7749),     // Invalid longitude
        (-87.6298, 41.8781),  // Valid LngLat
    ];

    let format = detect_format(&mixed_coords);
    let confidence = confidence_score(&mixed_coords, format);
    println!("  Input: {:?}", mixed_coords);
    println!(
        "  Detected: {:?} (confidence: {:.1}%)",
        format,
        confidence * 100.0
    );
    println!("  Mixed: Some coordinates valid as LngLat, others as LatLng");
    println!("  Note: Algorithm picks format with more valid coordinates\n");

    // Example 6: Demonstrating automatic conversion
    println!("Example 6: Automatic Conversion");

    // Convert different formats to Vec<LngLat>
    let tuple_input = CoordinateInput::Tuples(lat_lng_coords.clone());
    let converted = coords_to_lnglat_vec(&tuple_input);

    println!("  Original LatLng: {:?}", lat_lng_coords[0]);
    println!("  Converted LngLat: {:?}", converted[0]);
    println!("  Automatically detected LatLng format and swapped coordinates\n");

    // Example 7: Flat array format
    println!("Example 7: Flat Array Format");
    let flat_array = vec![
        -122.4194, 37.7749, // San Francisco
        -74.0060, 40.7128, // New York
        2.3522, 48.8566, // Paris
    ];

    let flat_input = CoordinateInput::FlatArray(flat_array.clone());
    let converted = coords_to_lnglat_vec(&flat_input);

    println!("  Flat array: {:?}", flat_array);
    println!("  Converted to: {:?}", converted);
    println!("  Note: Flat arrays are chunked into pairs and format-detected\n");

    // Example 8: GeoJSON-like format
    println!("Example 8: GeoJSON Format");
    let geojson_points = vec![
        GeoPoint {
            coordinates: [-122.4194, 37.7749],
        },
        GeoPoint {
            coordinates: [-74.0060, 40.7128],
        },
    ];

    let geojson_input = CoordinateInput::GeoJson(geojson_points.clone());
    let converted = coords_to_lnglat_vec(&geojson_input);

    println!("  GeoJSON: {:?}", geojson_points[0]);
    println!("  Converted: {:?}", converted[0]);
    println!("  GeoJSON coordinates are always [lng, lat] by specification\n");

    // Example 9: Edge cases
    println!("Example 9: Edge Cases");

    // Empty array
    let empty: Vec<(f64, f64)> = vec![];
    let format = detect_format(&empty);
    println!("  Empty array: {:?} → {:?}", empty, format);

    // Single coordinate
    let single = vec![(0.0, 0.0)];
    let format = detect_format(&single);
    let confidence = confidence_score(&single, format);
    println!(
        "  Single coord: {:?} → {:?} ({:.1}%)",
        single,
        format,
        confidence * 100.0
    );

    // Polar regions (high latitude values)
    let polar = vec![
        (-45.0, 89.0),  // Near North Pole
        (120.0, -88.0), // Near South Pole
    ];
    let format = detect_format(&polar);
    let confidence = confidence_score(&polar, format);
    println!(
        "  Polar regions: {:?} → {:?} ({:.1}%)",
        polar,
        format,
        confidence * 100.0
    );
    println!("  Note: High latitude values help distinguish from longitude\n");

    // Example 10: Performance with large dataset
    println!("Example 10: Large Dataset Performance");
    let large_dataset: Vec<(f64, f64)> = (0..10000)
        .map(|i| {
            let lng = -180.0 + (i as f64 / 10000.0) * 360.0; // -180 to 180
            let lat = -90.0 + ((i * 7) % 10000) as f64 / 10000.0 * 180.0; // -90 to 90
            (lng, lat)
        })
        .collect();

    let start = std::time::Instant::now();
    let format = detect_format(&large_dataset);
    let detection_time = start.elapsed();

    let start = std::time::Instant::now();
    let input = CoordinateInput::Tuples(large_dataset.clone());
    let converted = coords_to_lnglat_vec(&input);
    let conversion_time = start.elapsed();

    println!("  Dataset size: {} coordinates", large_dataset.len());
    println!("  Detection time: {:?}", detection_time);
    println!("  Conversion time: {:?}", conversion_time);
    println!("  Detected format: {:?}", format);
    println!("  Converted count: {}", converted.len());
}
