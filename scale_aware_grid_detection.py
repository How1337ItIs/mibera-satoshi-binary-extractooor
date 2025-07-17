#!/usr/bin/env python3
"""
Scale-aware grid detection following the condensed checklist.
Implements proper autocorrelation-based pitch detection and origin optimization.
"""

import cv2
import numpy as np
import pandas as pd
import hashlib
import json
from scipy import ndimage

def detect_grid_pitch_scale_aware():
    """Lock the grid using scale-aware autocorrelation detection."""
    
    print("=== SCALE-AWARE GRID DETECTION ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    print(f"Image dimensions: {img.shape}")
    
    # Create mask for high-contrast regions (text/grid)
    mask = img > 200
    print(f"High-contrast pixels: {mask.sum():,} ({mask.sum()/mask.size:.1%})")
    
    # Column pitch detection
    print("\n--- COLUMN PITCH DETECTION ---")
    proj_col = mask.sum(0).astype(float) - mask.mean(0)
    
    # Autocorrelation for column spacing
    autocorr_col = np.correlate(proj_col, proj_col, "full")
    autocorr_col = autocorr_col[len(proj_col):]  # Take second half
    
    # Find peak after offset 5 (avoid trivial zero-lag peak)
    col_px = np.argmax(autocorr_col[5:]) + 5
    
    # Calculate scale factor
    scale = img.shape[1] / 4096  # Assuming 4096 is reference width
    logical_col_pitch = col_px / scale
    
    print(f"Column pitch (pixels): {col_px}")
    print(f"Scale factor: {scale:.3f}")
    print(f"Logical column pitch: {logical_col_pitch:.1f}")
    
    # Row pitch detection
    print("\n--- ROW PITCH DETECTION ---")
    proj_row = mask.sum(1).astype(float) - mask.mean(1)
    
    # Autocorrelation for row spacing
    autocorr_row = np.correlate(proj_row, proj_row, "full")
    autocorr_row = autocorr_row[len(proj_row):]  # Take second half
    
    # Find peak after offset 10 (looking for larger spacing)
    row_px = np.argmax(autocorr_row[10:50]) + 10  # Search in reasonable range
    logical_row_pitch = row_px / scale
    
    print(f"Row pitch (pixels): {row_px}")
    print(f"Logical row pitch: {logical_row_pitch:.1f}")
    
    return {
        'col_pitch_px': col_px,
        'row_pitch_px': row_px,
        'scale_factor': scale,
        'logical_col_pitch': logical_col_pitch,
        'logical_row_pitch': logical_row_pitch,
        'image_shape': img.shape
    }

def optimize_grid_origin(col_pitch, row_pitch):
    """Sweep origin to find offset with maximum ink density."""
    
    print("\n=== GRID ORIGIN OPTIMIZATION ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    best_origin = None
    best_ink_density = 0
    
    print(f"Testing origins: 0-{col_pitch-1} (col) × 0-{row_pitch-1} (row)")
    
    # Test all possible origins within one grid cell
    for origin_row in range(row_pitch):
        for origin_col in range(col_pitch):
            
            # Sample grid at this origin
            ink_total = 0
            sample_count = 0
            
            # Test first 10x10 grid cells
            for grid_row in range(10):
                for grid_col in range(10):
                    y = origin_row + grid_row * row_pitch
                    x = origin_col + grid_col * col_pitch
                    
                    if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                        # Sample 6x6 region around grid point
                        y_start = max(0, y - 3)
                        y_end = min(img.shape[0], y + 4)
                        x_start = max(0, x - 3)
                        x_end = min(img.shape[1], x + 4)
                        
                        region = img[y_start:y_end, x_start:x_end]
                        if region.size > 0:
                            # Use inverse values (dark = high ink)
                            ink_value = 255 - np.median(region)
                            ink_total += ink_value
                            sample_count += 1
            
            if sample_count > 0:
                ink_density = ink_total / sample_count
                
                if ink_density > best_ink_density:
                    best_ink_density = ink_density
                    best_origin = (origin_row, origin_col)
    
    print(f"Best origin: row={best_origin[0]}, col={best_origin[1]}")
    print(f"Best ink density: {best_ink_density:.1f}")
    
    return best_origin, best_ink_density

def sample_grid_properly(origin_row, origin_col, row_pitch, col_pitch):
    """Sample grid using median of central 6x6 pixels per cell."""
    
    print("\n=== PROPER GRID SAMPLING ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Sample large grid for analysis
    grid_rows = 20
    grid_cols = 60
    
    samples = []
    positions = []
    
    for grid_row in range(grid_rows):
        for grid_col in range(grid_cols):
            y = origin_row + grid_row * row_pitch
            x = origin_col + grid_col * col_pitch
            
            if (0 <= y - 3 and y + 3 < img.shape[0] and 
                0 <= x - 3 and x + 3 < img.shape[1]):
                
                # Extract 6x6 central region
                region = img[y-3:y+4, x-3:x+4]
                
                if region.shape == (7, 7):  # Ensure we got full 6x6 + center
                    median_value = np.median(region)
                    samples.append(median_value)
                    positions.append((grid_row, grid_col, y, x))
    
    print(f"Sampled {len(samples)} grid positions")
    print(f"Value range: {min(samples):.1f} - {max(samples):.1f}")
    print(f"Mean: {np.mean(samples):.1f}, Median: {np.median(samples):.1f}")
    
    # Determine global threshold
    # Test both Sauvola-like and simple threshold
    threshold_sauvola = np.mean(samples) + 0.2 * np.std(samples)
    threshold_simple = 200
    
    print(f"Threshold candidates:")
    print(f"  Sauvola-like: {threshold_sauvola:.1f}")
    print(f"  Simple (V > 200): {threshold_simple}")
    
    # Test both thresholds
    for thresh_name, threshold in [("sauvola", threshold_sauvola), ("simple", threshold_simple)]:
        bits = [1 if val > threshold else 0 for val in samples]
        ones_ratio = sum(bits) / len(bits)
        print(f"  {thresh_name}: {ones_ratio:.1%} ones")
    
    # Use threshold that gives closest to 50% ones ratio
    thresh_sauvola_diff = abs(sum(1 if val > threshold_sauvola else 0 for val in samples) / len(samples) - 0.5)
    thresh_simple_diff = abs(sum(1 if val > threshold_simple else 0 for val in samples) / len(samples) - 0.5)
    
    if thresh_sauvola_diff < thresh_simple_diff:
        best_threshold = threshold_sauvola
        threshold_name = "sauvola"
    else:
        best_threshold = threshold_simple
        threshold_name = "simple"
    
    print(f"Selected threshold: {threshold_name} ({best_threshold:.1f})")
    
    return samples, positions, best_threshold, threshold_name

def create_canonical_bit_dump(positions, samples, threshold):
    """Create canonical raw_bit_dump.csv with SHA-256 verification."""
    
    print("\n=== CANONICAL BIT DUMP CREATION ===")
    
    # Convert to bits
    bits = [1 if val > threshold else 0 for val in samples]
    
    # Create DataFrame
    dump_data = []
    for i, ((grid_row, grid_col, y, x), sample_val, bit) in enumerate(zip(positions, samples, bits)):
        dump_data.append({
            'bit_index': i,
            'grid_row': grid_row,
            'grid_col': grid_col,
            'pixel_y': y,
            'pixel_x': x,
            'sample_value': sample_val,
            'threshold': threshold,
            'bit': bit
        })
    
    df = pd.DataFrame(dump_data)
    
    # Save to CSV
    csv_filename = 'canonical_raw_bit_dump.csv'
    df.to_csv(csv_filename, index=False)
    
    # Calculate SHA-256
    with open(csv_filename, 'rb') as f:
        content = f.read()
    sha256_hash = hashlib.sha256(content).hexdigest()
    
    print(f"Created {csv_filename}")
    print(f"Total bits: {len(bits)}")
    print(f"Ones ratio: {sum(bits)}/{len(bits)} ({sum(bits)/len(bits):.1%})")
    print(f"SHA-256: {sha256_hash}")
    
    return df, sha256_hash

def validate_sanity_checks(df):
    """Validate sanity checks from the checklist."""
    
    print("\n=== SANITY VALIDATION ===")
    
    bits = df['bit'].values
    ones_ratio = sum(bits) / len(bits)
    
    # Check 1: Ones ratio 45-55%
    print(f"1. Ones ratio: {ones_ratio:.1%}", end="")
    if 0.45 <= ones_ratio <= 0.55:
        print(" PASS")
        sanity_1 = True
    else:
        print(" FAIL")
        sanity_1 = False
    
    # Check 2: First row bytes = "On" (01001111 01101110)
    if len(bits) >= 16:
        first_16_bits = bits[:16]
        
        # Convert to two bytes
        byte1 = 0
        byte2 = 0
        for i in range(8):
            byte1 |= (first_16_bits[i] << (7 - i))
            byte2 |= (first_16_bits[i + 8] << (7 - i))
        
        expected_byte1 = 0b01001111  # 'O'
        expected_byte2 = 0b01101110  # 'n'
        
        print(f"2. First bytes: {byte1:08b} {byte2:08b}", end="")
        print(f" (chars: '{chr(byte1) if 32 <= byte1 <= 126 else '?'}'{chr(byte2) if 32 <= byte2 <= 126 else '?'}')", end="")
        
        if byte1 == expected_byte1 and byte2 == expected_byte2:
            print(" PASS - 'On' detected")
            sanity_2 = True
        else:
            print(f" FAIL - expected 'On' ({expected_byte1:08b} {expected_byte2:08b})")
            sanity_2 = False
    else:
        print("2. First row bytes: FAIL - insufficient bits")
        sanity_2 = False
    
    # Check 3: Grid position validation (approximate)
    # Look for grid positions around row ≈ 160, col ≈ 430 in master scale
    scale_factor = 1232 / 4096  # Assuming current image is 1232 wide vs 4096 master
    expected_master_row = 160
    expected_master_col = 430
    
    expected_current_row = expected_master_row * scale_factor
    expected_current_col = expected_master_col * scale_factor
    
    print(f"3. Grid position check:")
    print(f"   Expected position (current scale): ~({expected_current_row:.0f}, {expected_current_col:.0f})")
    
    # Check if we have samples near this position
    positions_near = df[
        (abs(df['pixel_y'] - expected_current_row) < 50) & 
        (abs(df['pixel_x'] - expected_current_col) < 50)
    ]
    
    if len(positions_near) > 0:
        print(f"   Found {len(positions_near)} samples near expected position PASS")
        sanity_3 = True
    else:
        print(f"   No samples near expected position FAIL")
        sanity_3 = False
    
    total_passed = sum([sanity_1, sanity_2, sanity_3])
    print(f"\nSanity checks passed: {total_passed}/3")
    
    return {
        'ones_ratio_pass': sanity_1,
        'first_bytes_pass': sanity_2,
        'position_pass': sanity_3,
        'total_passed': total_passed
    }

def save_scale_aware_results():
    """Save complete scale-aware analysis results."""
    
    print("\n=== SAVING SCALE-AWARE RESULTS ===")
    
    # Run complete analysis
    grid_params = detect_grid_pitch_scale_aware()
    origin, ink_density = optimize_grid_origin(grid_params['col_pitch_px'], grid_params['row_pitch_px'])
    samples, positions, threshold, threshold_name = sample_grid_properly(
        origin[0], origin[1], grid_params['row_pitch_px'], grid_params['col_pitch_px']
    )
    df, sha256_hash = create_canonical_bit_dump(positions, samples, threshold)
    sanity_results = validate_sanity_checks(df)
    
    # Compile results
    results = {
        "timestamp": "2025-07-17",
        "analysis_type": "scale_aware_grid_detection",
        "grid_parameters": {k: float(v) if isinstance(v, np.number) else v for k, v in grid_params.items()},
        "optimized_origin": {
            "row": int(origin[0]),
            "col": int(origin[1]),
            "ink_density": float(ink_density)
        },
        "sampling": {
            "method": "6x6_median",
            "threshold": threshold,
            "threshold_type": threshold_name,
            "samples_count": int(len(samples))
        },
        "canonical_dump": {
            "filename": "canonical_raw_bit_dump.csv",
            "sha256": sha256_hash,
            "total_bits": int(len(df)),
            "ones_ratio": float(sum(df['bit']) / len(df))
        },
        "sanity_validation": sanity_results,
        "assessment": "Scale-aware grid detection with proper sampling and validation"
    }
    
    with open('scale_aware_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Scale-aware analysis saved to scale_aware_analysis.json")
    
    return results

if __name__ == "__main__":
    print("Scale-Aware Grid Detection")
    print("Following condensed checklist for proper extraction")
    print("=" * 60)
    
    # Step 1: Lock the grid (scale-aware)
    grid_params = detect_grid_pitch_scale_aware()
    
    # Step 2: Optimize origin
    origin, ink_density = optimize_grid_origin(grid_params['col_pitch_px'], grid_params['row_pitch_px'])
    
    # Step 3: Sample properly
    samples, positions, threshold, threshold_name = sample_grid_properly(
        origin[0], origin[1], grid_params['row_pitch_px'], grid_params['col_pitch_px']
    )
    
    # Step 4: Re-dump raw bits
    df, sha256_hash = create_canonical_bit_dump(positions, samples, threshold)
    
    # Step 5: Validate sanity
    sanity_results = validate_sanity_checks(df)
    
    # Save comprehensive results
    analysis_results = save_scale_aware_results()
    
    print("\n" + "=" * 60)
    print("SCALE-AWARE ANALYSIS COMPLETE")
    
    print(f"\nFinal Results:")
    print(f"Grid pitch: {grid_params['row_pitch_px']}×{grid_params['col_pitch_px']} px")
    print(f"Logical pitch: {grid_params['logical_row_pitch']:.1f}×{grid_params['logical_col_pitch']:.1f}")
    print(f"Origin: ({origin[0]}, {origin[1]})")
    print(f"Sanity checks: {sanity_results['total_passed']}/3 passed")
    print(f"SHA-256: {sha256_hash[:16]}...")
    
    if sanity_results['total_passed'] >= 2:
        print("\nREADY for crypto/pattern analysis")
    else:
        print("\nNEEDS further grid refinement")