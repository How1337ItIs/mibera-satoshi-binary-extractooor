#!/usr/bin/env python3
"""
Test the 8px claim with the exact one-liner provided.
No parameters, no resizing, just raw autocorrelation.
"""

import cv2
import numpy as np

def test_pitch_oneliner():
    """Test the exact one-liner from the feedback."""
    
    # Use the binary mask we have
    arr = cv2.imread('binary_extractor/output_real_data/bw_mask.png', cv2.IMREAD_GRAYSCALE)
    
    print("Testing pitch detection one-liner...")
    print(f"Image shape: {arr.shape}")
    
    # The exact test from feedback
    mask = arr > 200  # very bright digits only
    proj = mask.sum(0).astype(float)
    proj -= proj.mean()
    corr = np.correlate(proj, proj, mode='full')[len(proj)-1:]
    pitch = np.argmax(corr[5:]) + 5  # +5 because we skip zero-lag
    
    print(f"Column pitch = {pitch} pixels")
    
    # Show the autocorr values for verification
    print("\nTop autocorr peaks:")
    peak_indices = np.argsort(corr[5:30])[::-1][:10] + 5
    for i, peak_idx in enumerate(peak_indices):
        print(f"  {i+1}. {peak_idx} pixels: {corr[peak_idx]:.0f}")
    
    return pitch

def test_different_thresholds():
    """Test with different thresholding approaches."""
    
    arr = cv2.imread('binary_extractor/output_real_data/bw_mask.png', cv2.IMREAD_GRAYSCALE)
    
    thresholds = [
        ("arr > 200", lambda x: x > 200),
        ("arr > 150", lambda x: x > 150),
        ("arr > 127", lambda x: x > 127),
        ("arr < 200", lambda x: x < 200),  # Dark digits
        ("arr < 100", lambda x: x < 100),
    ]
    
    print("\nTesting different thresholds:")
    print("-" * 40)
    
    for name, thresh_func in thresholds:
        mask = thresh_func(arr)
        proj = mask.sum(0).astype(float)
        proj -= proj.mean()
        corr = np.correlate(proj, proj, mode='full')[len(proj)-1:]
        pitch = np.argmax(corr[5:]) + 5
        
        print(f"{name:12s}: pitch = {pitch:2d} pixels")

def test_on_gaussian_subtracted():
    """Test on the gaussian_subtracted image which might be cleaner."""
    
    try:
        arr = cv2.imread('binary_extractor/output_real_data/gaussian_subtracted.png', cv2.IMREAD_GRAYSCALE)
        
        print(f"\nTesting on gaussian_subtracted.png:")
        print(f"Image shape: {arr.shape}")
        
        # Test both bright and dark
        for name, condition in [("bright digits", arr > 150), ("dark digits", arr < 100)]:
            mask = condition
            proj = mask.sum(0).astype(float)
            proj -= proj.mean()
            corr = np.correlate(proj, proj, mode='full')[len(proj)-1:]
            pitch = np.argmax(corr[5:]) + 5
            
            print(f"  {name}: pitch = {pitch} pixels")
        
    except Exception as e:
        print(f"\nCould not test gaussian_subtracted.png: {e}")

def verify_with_visual():
    """Create a visual verification of the pitch."""
    import matplotlib.pyplot as plt
    
    arr = cv2.imread('binary_extractor/output_real_data/bw_mask.png', cv2.IMREAD_GRAYSCALE)
    
    # Test the claim
    mask = arr > 200
    proj = mask.sum(0).astype(float)
    proj -= proj.mean()
    corr = np.correlate(proj, proj, mode='full')[len(proj)-1:]
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Column projection
    ax1.plot(proj + proj.mean())  # Add mean back for display
    ax1.set_title('Column Projection (bright pixels)')
    ax1.set_xlabel('X position')
    ax1.grid(True, alpha=0.3)
    
    # Autocorrelation
    ax2.plot(corr[:50])
    ax2.set_title('Autocorrelation')
    ax2.set_xlabel('Lag (pixels)')
    
    # Mark key lags
    ax2.axvline(8, color='green', linestyle='--', label='8px (claimed true)')
    ax2.axvline(25, color='red', linestyle='--', label='25px (observed before)')
    
    # Mark the actual maximum
    max_idx = np.argmax(corr[5:30]) + 5
    ax2.axvline(max_idx, color='blue', linestyle='-', linewidth=2, label=f'Actual max: {max_idx}px')
    
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pitch_verification.png', dpi=150)
    plt.close()
    
    print(f"\nVisual verification saved to pitch_verification.png")
    print(f"Autocorrelation maximum at lag {max_idx} pixels")
    
    return max_idx

if __name__ == "__main__":
    print("Testing the 8px Column Pitch Claim")
    print("Using the exact one-liner from feedback")
    print("="*50)
    
    # Main test
    result = test_pitch_oneliner()
    
    # Additional tests
    test_different_thresholds()
    test_on_gaussian_subtracted()
    
    # Visual verification
    visual_result = verify_with_visual()
    
    print("\n" + "="*50)
    print("CONCLUSION:")
    if result == 8:
        print("✓ The one-liner confirms 8px column pitch!")
        print("  The 25px measurement was indeed an alias.")
    else:
        print(f"✗ One-liner returned {result}px, not 8px")
        print("  Need to investigate further...")
    
    print(f"\nNext step: Implement the 4-point fix for >95% accuracy")