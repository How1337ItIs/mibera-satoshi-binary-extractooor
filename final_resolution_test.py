#!/usr/bin/env python3
"""
Final test to resolve the pitch debate by examining the actual data more carefully.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def examine_actual_structure():
    """Look at the actual structure in detail."""
    
    img = cv2.imread('binary_extractor/output_real_data/bw_mask.png', 0)
    
    print(f"Binary mask analysis:")
    print(f"  Shape: {img.shape}")
    print(f"  Unique values: {np.unique(img)}")
    
    # Take a representative crop
    crop = img[400:500, 300:800]
    
    # Show the actual pattern
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
    
    ax1.imshow(crop, cmap='gray', interpolation='nearest')
    ax1.set_title('Actual Binary Pattern')
    
    # Draw grid lines at different pitches to see which aligns
    width = crop.shape[1]
    
    # Test pitches
    pitches_to_test = [3, 5, 8, 12, 18, 25, 53]
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink']
    
    for pitch, color in zip(pitches_to_test, colors):
        for x in range(0, width, pitch):
            ax1.axvline(x, color=color, alpha=0.3, linewidth=0.5)
    
    ax1.set_xlabel('Pitch lines: ' + ', '.join([f'{p}px ({c})' for p, c in zip(pitches_to_test, colors)]))
    
    # Column projection
    proj = crop.sum(0)
    ax2.plot(proj)
    ax2.set_title('Column Projection of Crop')
    ax2.set_xlabel('Position')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('structure_analysis.png', dpi=150)
    plt.close()
    
    # Measure actual digit widths and spacings manually
    print("\nManual structure analysis:")
    
    # Find peaks and valleys in the projection
    from scipy.signal import find_peaks
    
    peaks, _ = find_peaks(proj, height=np.max(proj)*0.3, distance=3)
    valleys, _ = find_peaks(-proj, height=-np.max(proj)*0.7, distance=3)
    
    print(f"  Found {len(peaks)} peaks (digit centers)")
    print(f"  Found {len(valleys)} valleys (gaps)")
    
    if len(peaks) > 1:
        peak_spacings = np.diff(peaks)
        print(f"  Peak spacings: {peak_spacings}")
        print(f"  Mean peak spacing: {np.mean(peak_spacings):.1f} pixels")
        
        # Check if this is consistent with any expected pitch
        mean_spacing = np.mean(peak_spacings)
        
        print(f"\nHypothesis check:")
        print(f"  If true pitch is 8px in 4K, then in {img.shape[1]}px width:")
        expected = 8 * (img.shape[1] / 4096)
        print(f"  Expected: {expected:.1f} pixels")
        print(f"  Measured: {mean_spacing:.1f} pixels")
        print(f"  Ratio: {mean_spacing / expected:.1f}x")
        
        # Maybe the binary mask is further processed?
        print(f"\nAlternative: if measured {mean_spacing:.1f}px corresponds to 8px logical:")
        actual_scale = mean_spacing / 8
        implied_width = 4096 * actual_scale
        print(f"  Implied processed width: {implied_width:.0f}px")
        print(f"  Actual width: {img.shape[1]}px")
        print(f"  Processing factor: {img.shape[1] / implied_width:.2f}x")

def test_with_correct_pitch():
    """Test extraction using the measured pitch."""
    
    img = cv2.imread('binary_extractor/output_real_data/bw_mask.png', 0)
    
    # Use the measured pitch from our analysis
    measured_col_pitch = 53  # From previous measurements
    measured_row_pitch = 31  # This was consistent
    
    print(f"\nTesting with measured pitches: {measured_row_pitch}x{measured_col_pitch}")
    
    # Simple origin sweep
    best_score = -1
    best_origin = None
    
    for row0 in range(0, measured_row_pitch, 2):
        for col0 in range(0, measured_col_pitch, 2):
            score = test_origin(img, row0, col0, measured_row_pitch, measured_col_pitch)
            if score > best_score:
                best_score = score
                best_origin = (row0, col0)
    
    row0, col0 = best_origin
    print(f"Best origin: ({row0}, {col0}) score: {best_score:.2f}")
    
    # Extract first row with 6x6 sampling
    y = row0
    bits = []
    
    for i in range(20):  # First 20 bits
        x = col0 + i * measured_col_pitch
        if x < img.shape[1] - 3:
            # 6x6 patch
            patch = img[y-3:y+4, x-3:x+4]
            if patch.size > 0:
                val = np.median(patch)
                bit = '1' if val > 127 else '0'
                bits.append(bit)
    
    bits_str = ''.join(bits)
    print(f"First 20 bits: {bits_str}")
    
    # Try to decode
    if len(bits_str) >= 16:
        try:
            char1 = chr(int(bits_str[:8], 2))
            char2 = chr(int(bits_str[8:16], 2))
            print(f"First two chars: '{char1}{char2}'")
            
            if char1 == 'O' and char2 == 'n':
                print("SUCCESS: Found 'On'!")
                return True
        except:
            print("Could not decode to ASCII")
    
    return False

def test_origin(img, row0, col0, row_pitch, col_pitch):
    """Score an origin by sampling a few positions."""
    score = 0
    
    for r in range(3):
        y = row0 + r * row_pitch
        if y >= img.shape[0] - 3:
            break
        
        for c in range(5):
            x = col0 + c * col_pitch
            if x >= img.shape[1] - 3:
                break
            
            # Sample center pixel
            val = img[y, x]
            # Score based on clarity
            if val > 200:  # Clear white
                score += 2
            elif val < 50:  # Clear black
                score += 1
            else:  # Ambiguous
                score -= 0.5
    
    return score

if __name__ == "__main__":
    print("Final Resolution of Pitch Debate")
    print("="*50)
    
    # Examine the actual structure
    examine_actual_structure()
    
    # Test with measured pitch
    success = test_with_correct_pitch()
    
    print("\n" + "="*50)
    print("FINAL CONCLUSION:")
    print("The key insight is that regardless of whether the 'true' pitch")
    print("is 8px in some reference frame, what matters is:")
    print("1. Measure the actual pitch in YOUR image")
    print("2. Use robust 6x6 patch sampling")
    print("3. Do proper origin sweep")
    print("4. Focus on getting 'On' in the first row")
    
    if success:
        print("\n✓ This approach works with the measured pitch!")
    else:
        print("\n→ Still needs fine-tuning, but methodology is sound.")