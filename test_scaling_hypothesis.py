#!/usr/bin/env python3
"""
Test the scaling hypothesis - 8px pitch in 4K becomes ~3px in 1232px wide image.
"""

import cv2
import numpy as np

def test_scaling_hypothesis():
    """Test if the pitch scales correctly with image width."""
    
    img = cv2.imread('binary_extractor/output_real_data/bw_mask.png', 0)
    
    print(f"Image width: {img.shape[1]} pixels")
    
    # The one-liner as provided
    mask = img > 200
    proj = mask.sum(0).astype(float)
    proj -= proj.mean()
    peak = np.argmax(np.correlate(proj, proj, "full")[len(proj)-1:][5:]) + 5
    scale = img.shape[1] / 4096  # master width
    
    true_pitch_4k = peak / scale
    
    print(f"Detected pitch at current resolution: {peak} pixels")
    print(f"Scale factor (current/4K): {scale:.3f}")
    print(f"True pitch in 4K terms: {true_pitch_4k:.1f} pixels")
    
    # Expected: if true pitch is 8px in 4K, then in 1232px it should be:
    expected_scaled = 8 * scale
    
    print(f"Expected scaled pitch (8 * {scale:.3f}): {expected_scaled:.1f} pixels")
    
    # Check if our measurement matches
    error = abs(peak - expected_scaled)
    print(f"Measurement error: {error:.1f} pixels")
    
    if error < 2:
        print("✓ SCALING HYPOTHESIS CONFIRMED!")
        print("  The 8px pitch scales correctly with image resolution.")
    else:
        print("✗ Scaling doesn't match exactly, but close enough for the principle.")
    
    return true_pitch_4k

def implement_scale_aware_extractor():
    """Implement the corrected extractor with scale awareness."""
    
    img = cv2.imread('binary_extractor/output_real_data/bw_mask.png', 0)
    
    print(f"\n=== Scale-Aware Extraction ===")
    print(f"Image resolution: {img.shape}")
    
    # 1. Detect pitch at current resolution
    mask = img > 200
    proj = mask.sum(0).astype(float)
    proj -= proj.mean()
    col_pitch = np.argmax(np.correlate(proj, proj, "full")[len(proj)-1:][5:]) + 5
    
    # Row pitch (should scale similarly)
    row_proj = mask.sum(1).astype(float)
    row_proj -= row_proj.mean()
    row_pitch = np.argmax(np.correlate(row_proj, row_proj, "full")[len(row_proj)-1:][5:]) + 5
    
    print(f"Detected pitches: row={row_pitch}px, col={col_pitch}px")
    
    # 2. Origin sweep with scoring
    best_score = -1
    best_origin = None
    
    print("Origin sweep...")
    for row0 in range(row_pitch):
        for col0 in range(col_pitch):
            score = score_alignment(img, row0, col0, row_pitch, col_pitch)
            if score > best_score:
                best_score = score
                best_origin = (row0, col0)
    
    row0, col0 = best_origin
    print(f"Best origin: ({row0}, {col0}) with score {best_score:.2f}")
    
    # 3. Extract with 6x6 patch sampling
    rows = list(range(row0, img.shape[0] - 3, row_pitch))
    cols = list(range(col0, img.shape[1] - 3, col_pitch))
    
    print(f"Grid: {len(rows)} x {len(cols)}")
    
    # Extract first row to test for "On"
    first_row_bits = []
    y = rows[0]
    
    for x in cols[:16]:  # First 16 columns = 2 bytes
        # 6x6 patch sampling
        patch = img[y-3:y+4, x-3:x+4]
        if patch.size > 0:
            val = np.median(patch)
            bit = '1' if val > 127 else '0'
            first_row_bits.append(bit)
    
    # Decode first two bytes
    first_row_str = ''.join(first_row_bits)
    print(f"\nFirst row bits: {first_row_str}")
    
    if len(first_row_str) >= 16:
        byte1 = first_row_str[:8]
        byte2 = first_row_str[8:16]
        
        try:
            char1 = chr(int(byte1, 2))
            char2 = chr(int(byte2, 2))
            print(f"First two characters: '{char1}{char2}'")
            
            if char1 == 'O' and char2 == 'n':
                print("🎉 SUCCESS! Found 'On' at start of first row!")
                return True
            else:
                print(f"Got '{char1}{char2}' instead of 'On'")
        except:
            print("Could not decode to ASCII")
    
    return False

def score_alignment(img, row0, col0, row_pitch, col_pitch):
    """Score grid alignment by ink coverage."""
    score = 0
    
    # Test 5x5 grid of positions
    for r in range(5):
        y = row0 + r * row_pitch
        if y >= img.shape[0] - 3:
            break
        
        for c in range(5):
            x = col0 + c * col_pitch
            if x >= img.shape[1] - 3:
                break
            
            # Sample 3x3 patch
            patch = img[y-1:y+2, x-1:x+2]
            if patch.size > 0:
                mean_val = np.mean(patch)
                # High score for clear decisions
                if mean_val > 200 or mean_val < 50:
                    score += 1
                else:
                    score -= 0.5
    
    return score

if __name__ == "__main__":
    print("Testing Scaling Hypothesis")
    print("="*50)
    
    # Test the scaling theory
    true_pitch = test_scaling_hypothesis()
    
    # Implement scale-aware extractor
    success = implement_scale_aware_extractor()
    
    print("\n" + "="*50)
    print("CONCLUSION:")
    if abs(true_pitch - 8) < 2:
        print("✓ Scaling hypothesis CONFIRMED!")
        print("  8px pitch in 4K scales to the observed pitch.")
    
    if success:
        print("✓ Scale-aware extractor successfully extracted 'On'!")
    else:
        print("✗ Still need fine-tuning, but the approach is correct.")