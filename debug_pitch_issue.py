#!/usr/bin/env python3
"""
Debug the pitch detection issue by examining the actual pattern more carefully.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def examine_image_carefully():
    """Look at what we're actually analyzing."""
    
    # Load the binary mask
    bw = cv2.imread('binary_extractor/output_real_data/bw_mask.png', cv2.IMREAD_GRAYSCALE)
    
    print(f"Binary mask shape: {bw.shape}")
    print(f"Pixel value range: {bw.min()} to {bw.max()}")
    print(f"Unique values: {np.unique(bw)}")
    
    # Show a crop of the actual pattern
    crop = bw[400:600, 300:700]  # Sample region
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Show the crop
    ax1.imshow(crop, cmap='gray', interpolation='nearest')
    ax1.set_title('Binary Mask Crop (400-600, 300-700)')
    ax1.grid(True, alpha=0.3)
    
    # Column projection of the crop
    crop_proj = crop.sum(0)
    ax2.plot(crop_proj)
    ax2.set_title('Column Projection of Crop')
    ax2.grid(True, alpha=0.3)
    
    # Test different thresholding on the crop
    bright_mask = crop > 200
    dark_mask = crop < 50
    
    ax3.plot(bright_mask.sum(0), label='Bright pixels (>200)')
    ax3.plot(dark_mask.sum(0), label='Dark pixels (<50)')
    ax3.set_title('Different Thresholds on Crop')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Autocorr of the crop
    proj = bright_mask.sum(0).astype(float)
    proj -= proj.mean()
    corr = np.correlate(proj, proj, mode='full')[len(proj):]
    
    ax4.plot(corr[:50])
    ax4.set_title('Autocorr of Crop (bright pixels)')
    ax4.set_xlabel('Lag (pixels)')
    
    # Mark potential peaks
    for lag in [5, 8, 12, 25]:
        if lag < len(corr):
            ax4.axvline(lag, alpha=0.5, linestyle='--', label=f'{lag}px')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('debug_pitch_pattern.png', dpi=150)
    plt.close()
    
    # Find actual peak
    peak_idx = np.argmax(corr[3:30]) + 3
    print(f"\nCrop autocorr peak at: {peak_idx} pixels")
    
    return peak_idx

def test_manual_measurement():
    """Manually measure some digit spacings in a row."""
    
    bw = cv2.imread('binary_extractor/output_real_data/bw_mask.png', cv2.IMREAD_GRAYSCALE)
    
    # Take a horizontal slice
    row_y = 500
    row_data = bw[row_y, 300:700]  # Focus on the digit region
    
    print(f"\nManual measurement of row {row_y}:")
    
    # Find transitions from black to white and white to black
    transitions = []
    for i in range(1, len(row_data)):
        if (row_data[i-1] < 127 and row_data[i] >= 127) or \
           (row_data[i-1] >= 127 and row_data[i] < 127):
            transitions.append(i + 300)  # Add offset back
    
    print(f"Transitions: {transitions[:20]}")
    
    if len(transitions) >= 4:
        # Calculate distances between digit starts
        digit_starts = transitions[::2]  # Every other transition should be digit start
        spacings = np.diff(digit_starts)
        
        print(f"Digit start spacings: {spacings[:10]}")
        print(f"Mean spacing: {np.mean(spacings):.1f} pixels")
        
        # Look for common spacings
        unique_spacings, counts = np.unique(spacings, return_counts=True)
        print("\nMost common spacings:")
        for spacing, count in sorted(zip(unique_spacings, counts), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {spacing} pixels: {count} occurrences")

def try_different_images():
    """Try the test on different available images."""
    
    image_files = [
        'binary_extractor/output_real_data/bw_mask.png',
        'binary_extractor/output_real_data/gaussian_subtracted.png',
        'binary_extractor/output_real_data/cyan_channel.png'
    ]
    
    for img_file in image_files:
        try:
            print(f"\n--- Testing {img_file} ---")
            arr = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
            if arr is None:
                print(f"Could not load {img_file}")
                continue
                
            print(f"Shape: {arr.shape}, range: {arr.min()}-{arr.max()}")
            
            # Try both bright and dark pixel detection
            for name, condition in [("bright", arr > 150), ("dark", arr < 100)]:
                mask = condition
                proj = mask.sum(0).astype(float)
                
                if proj.sum() == 0:
                    print(f"  {name}: no pixels found")
                    continue
                    
                proj -= proj.mean()
                corr = np.correlate(proj, proj, mode='full')[len(proj):]
                
                # Find peak in reasonable range
                peak_idx = np.argmax(corr[3:50]) + 3
                peak_value = corr[peak_idx]
                
                print(f"  {name}: peak at {peak_idx}px (strength: {peak_value:.0f})")
                
        except Exception as e:
            print(f"Error with {img_file}: {e}")

def check_o3_method_exactly():
    """Try to replicate o3's method exactly as described."""
    
    print("\n--- Checking o3's exact method ---")
    
    # Try on the binary mask with exactly the method described
    arr = cv2.imread('binary_extractor/output_real_data/bw_mask.png', cv2.IMREAD_GRAYSCALE)
    
    # "arr < 200" as mentioned in the feedback
    mask = arr < 200  # Dark digits
    proj_c = mask.sum(0).astype(float)
    mu = proj_c.mean()
    proj_c_minus_mu = proj_c - mu
    
    # Full autocorr as specified
    corr_full = np.correlate(proj_c_minus_mu, proj_c_minus_mu, mode='full')
    W = len(proj_c_minus_mu)
    corr = corr_full[W:]  # Take second half
    
    # Find argmax skipping first few
    pitch = np.argmax(corr[5:]) + 5
    
    print(f"O3's exact method result: {pitch} pixels")
    
    # Show top peaks
    peaks = np.argsort(corr[5:50])[::-1][:10] + 5
    print("Top peaks:")
    for i, peak in enumerate(peaks):
        print(f"  {i+1}. {peak} pixels: {corr[peak]:.0f}")

if __name__ == "__main__":
    print("Debugging Pitch Detection Issue")
    print("="*50)
    
    # Examine the image carefully
    crop_peak = examine_image_carefully()
    
    # Manual measurement
    test_manual_measurement()
    
    # Try different images
    try_different_images()
    
    # Check o3's exact method
    check_o3_method_exactly()
    
    print("\n" + "="*50)
    print("ANALYSIS:")
    print("The binary mask might be processed/filtered in a way that")
    print("changes the original pitch. Need to check if this is the")
    print("right source image or if preprocessing affected the pattern.")