#!/usr/bin/env python3
"""
Test the claimed good parameters: 30x52 pitch, position (110,56), threshold 45
Compare with scale-aware results for validation.

Created by: Cursor Agent
Purpose: Test claimed good parameters vs scale-aware method
Date: 2025-07-17
"""

import cv2
import numpy as np
import pandas as pd
import json

def extract_with_claimed_params():
    """Extract using the claimed good parameters."""
    
    print("=== TESTING CLAIMED GOOD PARAMETERS ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    print(f"Image dimensions: {img.shape}")
    
    # Claimed good parameters
    row0, col0 = 110, 56
    row_pitch, col_pitch = 30, 52
    threshold = 45
    
    print(f"Parameters: pos=({row0},{col0}), pitch={row_pitch}x{col_pitch}, threshold={threshold}")
    
    # Extract bits
    bits = []
    positions = []
    values = []
    
    for i in range(1000):  # Extract 1000 bits
        bit_row = i // 8
        bit_col = i % 8
        
        y = row0 + bit_row * row_pitch
        x = col0 + bit_col * col_pitch
        
        if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
            pixel_val = img[y, x]
            bit = 1 if pixel_val > threshold else 0
            
            bits.append(bit)
            positions.append((y, x))
            values.append(pixel_val)
    
    print(f"Extracted {len(bits)} bits")
    
    # Basic statistics
    ones_count = sum(bits)
    ones_ratio = ones_count / len(bits)
    
    print(f"Ones ratio: {ones_count}/{len(bits)} ({ones_ratio:.1%})")
    
    # Calculate entropy
    if ones_count > 0 and ones_count < len(bits):
        p1 = ones_ratio
        p0 = 1 - p1
        entropy = -p1 * np.log2(p1) - p0 * np.log2(p0)
    else:
        entropy = 0
    
    print(f"Bit entropy: {entropy:.3f}")
    
    # Create DataFrame
    dump_data = []
    for i, ((y, x), pixel_val, bit) in enumerate(zip(positions, values, bits)):
        dump_data.append({
            'bit_index': i,
            'pixel_y': y,
            'pixel_x': x,
            'sample_value': pixel_val,
            'threshold': threshold,
            'bit': bit
        })
    
    df = pd.DataFrame(dump_data)
    
    # Save to CSV
    csv_filename = 'tmp/dump_30x52.csv'
    df.to_csv(csv_filename, index=False)
    
    print(f"Saved to {csv_filename}")
    
    return df, bits, ones_ratio, entropy

def compare_extractions():
    """Compare the two extraction methods."""
    
    print("\n=== COMPARISON OF EXTRACTION METHODS ===")
    
    # Read both dumps
    try:
        scale_aware_df = pd.read_csv('output/canonical_raw_bit_dump.csv')
        claimed_df = pd.read_csv('tmp/dump_30x52.csv')
        
        print(f"Scale-aware: {len(scale_aware_df)} bits, {sum(scale_aware_df['bit'])/len(scale_aware_df):.1%} ones")
        print(f"Claimed good: {len(claimed_df)} bits, {sum(claimed_df['bit'])/len(claimed_df):.1%} ones")
        
        # Compare first 128 bits
        scale_bits = ''.join(scale_aware_df['bit'].astype(str).tolist()[:128])
        claimed_bits = ''.join(claimed_df['bit'].astype(str).tolist()[:128])
        
        print(f"\nFirst 128 bits comparison:")
        print(f"Scale-aware: {scale_bits[:64]}...")
        print(f"Claimed good: {claimed_bits[:64]}...")
        
        # ASCII preview
        scale_bytes = bytes(int(scale_bits[i:i+8], 2) for i in range(0, 128, 8))
        claimed_bytes = bytes(int(claimed_bits[i:i+8], 2) for i in range(0, 128, 8))
        
        print(f"\nASCII preview (first 16 bytes):")
        print(f"Scale-aware: {scale_bytes}")
        print(f"Claimed good: {claimed_bytes}")
        
        # Check for "On the..." pattern
        expected_start = "010011110110111000100000011101000110100001100101"  # "On the"
        print(f"\nExpected 'On the...': {expected_start}")
        print(f"Scale-aware match: {scale_bits.startswith(expected_start)}")
        print(f"Claimed good match: {claimed_bits.startswith(expected_start)}")
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")

if __name__ == "__main__":
    # Create tmp directory
    import os
    os.makedirs('tmp', exist_ok=True)
    
    # Test claimed parameters
    df, bits, ones_ratio, entropy = extract_with_claimed_params()
    
    # Compare with scale-aware results
    compare_extractions()
    
    print(f"\n=== RESULTS SUMMARY ===")
    print(f"Claimed parameters extraction:")
    print(f"  Ones ratio: {ones_ratio:.1%}")
    print(f"  Entropy: {entropy:.3f}")
    print(f"  Bits extracted: {len(bits)}")
    
    # Save results
    results = {
        "timestamp": "2025-07-17",
        "analysis_type": "claimed_params_test",
        "parameters": {
            "position": [110, 56],
            "pitch": [30, 52],
            "threshold": 45
        },
        "results": {
            "bits_extracted": len(bits),
            "ones_ratio": ones_ratio,
            "entropy": entropy
        },
        "assessment": "Direct test of claimed good parameters"
    }
    
    with open('tmp/claimed_params_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to tmp/claimed_params_results.json") 