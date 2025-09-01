#!/usr/bin/env python3

# Simple test script to verify polyline functionality
# Run this after building the Python module

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

try:
    import rapidgeo
    
    # Test basic encode/decode
    print("Testing basic polyline encode/decode...")
    
    # Create some test coordinates
    coords = [
        rapidgeo.LngLat(-120.2, 38.5),
        rapidgeo.LngLat(-120.95, 40.7),
        rapidgeo.LngLat(-126.453, 43.252),
    ]
    
    # Test encoding
    encoded = rapidgeo.polyline.encode(coords, 5)
    print(f"Encoded polyline: {encoded}")
    print(f"Expected: _p~iF~ps|U_ulLnnqC_mqNvxq`@")
    
    # Test decoding
    decoded = rapidgeo.polyline.decode(encoded, 5)
    print(f"Decoded {len(decoded)} coordinates:")
    for i, coord in enumerate(decoded):
        print(f"  {i}: lng={coord.lng:.5f}, lat={coord.lat:.5f}")
        
    # Test simplification
    print("\nTesting polyline simplification...")
    simplified = rapidgeo.polyline.simplify_polyline(encoded, 1000.0, "great_circle", 5)
    print(f"Simplified polyline: {simplified}")
    
    # Test direct simplified encoding
    simplified_direct = rapidgeo.polyline.encode_simplified(coords, 1000.0, "great_circle", 5)
    print(f"Direct simplified encoding: {simplified_direct}")
    
    # Test batch operations if available
    try:
        print("\nTesting batch operations...")
        coord_batches = [coords, coords[:2]]
        
        encoded_batch = rapidgeo.polyline.encode_batch(coord_batches, 5)
        print(f"Batch encoded {len(encoded_batch)} polylines:")
        for i, polyline in enumerate(encoded_batch):
            print(f"  {i}: {polyline}")
            
        decoded_batch = rapidgeo.polyline.decode_batch(encoded_batch, 5)
        print(f"Batch decoded {len(decoded_batch)} coordinate sets:")
        for i, coord_set in enumerate(decoded_batch):
            print(f"  Set {i}: {len(coord_set)} coordinates")
            
        # Test batch simplification
        simplified_batch = rapidgeo.polyline.encode_simplified_batch(coord_batches, 1000.0, "great_circle", 5)
        print(f"Batch simplified {len(simplified_batch)} polylines:")
        for i, polyline in enumerate(simplified_batch):
            print(f"  {i}: {polyline}")
            
    except AttributeError:
        print("Batch operations not available (batch feature not enabled)")
        
    print("\n✅ All tests completed successfully!")
    
except ImportError as e:
    print(f"❌ Failed to import rapidgeo: {e}")
    print("Make sure you've built the Python module with:")
    print("  cargo build --features batch,numpy")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ Test failed with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)