#!/usr/bin/env python3
"""
Adaptive threshold search to find the actual hidden message.
Now that we understand the dark region issue, search for brighter areas or adapt thresholds.
"""

import cv2
import numpy as np
from scipy import ndimage

def find_bright_regions():
    """Find the brightest regions in the image for bit extraction."""
    
    print("=== FINDING BRIGHT REGIONS ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    print(f"Image stats: min={img.min()}, max={img.max()}, mean={img.mean():.1f}")
    
    # Find regions with different brightness levels
    percentiles = [50, 60, 70, 80, 90, 95, 99]
    
    for p in percentiles:
        threshold = np.percentile(img, p)
        bright_mask = img > threshold
        bright_count = np.sum(bright_mask)
        print(f"{p:2d}th percentile: {threshold:3.0f} ({bright_count:6d} pixels, {bright_count/img.size:.1%})")
    
    # Find the brightest rectangular regions
    print(f"\n=== BRIGHT REGION ANALYSIS ===")
    
    # Use a high percentile for truly bright areas
    bright_threshold = np.percentile(img, 90)
    bright_mask = img > bright_threshold
    
    # Find connected components of bright areas
    labeled, num_features = ndimage.label(bright_mask)
    
    if num_features > 0:
        print(f"Found {num_features} bright regions")
        
        # Analyze each region
        bright_regions = []
        for i in range(1, min(num_features + 1, 20)):  # Check top 20 regions
            region_mask = labeled == i
            region_coords = np.where(region_mask)
            
            if len(region_coords[0]) > 100:  # Minimum size
                y_min, y_max = region_coords[0].min(), region_coords[0].max()
                x_min, x_max = region_coords[1].min(), region_coords[1].max()
                
                region_height = y_max - y_min
                region_width = x_max - x_min
                region_size = len(region_coords[0])
                
                bright_regions.append({
                    'id': i,
                    'bbox': (y_min, x_min, y_max, x_max),
                    'size': region_size,
                    'dimensions': (region_height, region_width)
                })
        
        # Sort by size
        bright_regions.sort(key=lambda x: x['size'], reverse=True)
        
        print(f"Top bright regions:")
        for i, region in enumerate(bright_regions[:10]):
            bbox = region['bbox']
            dims = region['dimensions']
            print(f"  {i+1:2d}. Size {region['size']:4d}, bbox=({bbox[0]:3d},{bbox[1]:3d})-({bbox[2]:3d},{bbox[3]:3d}), {dims[0]}x{dims[1]}")
        
        return bright_regions
    
    return []

def adaptive_threshold_extraction():
    """Extract with adaptive thresholds based on local image statistics."""
    
    print(f"\n=== ADAPTIVE THRESHOLD EXTRACTION ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Test grid parameters
    row_pitch = 31
    col_pitch = 53
    patch_size = 5
    
    # Target patterns to test
    targets = [
        ("On", "0100111101101110"),
        ("At", "01000001011101000010000000100000"),
        ("The", "01010100011010000110010100100000"),
        ("Bitcoin", "0100001001101001011101000110001101101111011010010110111000100000"),
        ("Satoshi", "0101001101100001011101000110111101110011011010000110100100100000")
    ]
    
    best_results = []
    
    # Test a broader range of positions
    test_positions = [
        (58, 28),   # Previous best
        (30, 30),   # Upper region
        (100, 50),  # Lower region
        (60, 100),  # Right side
        (80, 30),   # Different row
        (40, 60),   # Middle area
    ]
    
    for pos_name, (row0, col0) in enumerate(test_positions):
        print(f"\nTesting position ({row0}, {col0})...")
        
        # Sample a few positions to get local statistics
        local_values = []
        for test_r in range(2):
            for test_c in range(8):
                y = row0 + test_r * row_pitch
                x = col0 + test_c * col_pitch
                
                if 0 <= y < img.shape[0] - patch_size and 0 <= x < img.shape[1] - patch_size:
                    half = patch_size // 2
                    patch = img[max(0, y-half):min(img.shape[0], y+half+1), 
                               max(0, x-half):min(img.shape[1], x+half+1)]
                    if patch.size > 0:
                        local_values.append(np.median(patch))
        
        if local_values:
            local_min = min(local_values)
            local_max = max(local_values)
            local_mean = np.mean(local_values)
            local_std = np.std(local_values)
            
            print(f"  Local stats: {local_min:.0f}-{local_max:.0f}, mean={local_mean:.0f}, std={local_std:.0f}")
            
            # Try adaptive thresholds
            adaptive_thresholds = [
                local_mean,
                local_mean + local_std/2,
                local_mean - local_std/2,
                (local_min + local_max) / 2,
                local_mean + local_std,
                local_max * 0.7
            ]
            
            for threshold in adaptive_thresholds:
                if threshold <= 0 or threshold >= 255:
                    continue
                
                # Test this configuration
                for target_name, target_bits in targets:
                    # Extract bits
                    extracted_bits = []
                    
                    for i in range(min(len(target_bits), 32)):  # Limit to 32 bits for speed
                        bit_row = i // 8
                        bit_col = i % 8
                        
                        y = row0 + bit_row * row_pitch
                        x = col0 + bit_col * col_pitch
                        
                        if 0 <= y < img.shape[0] - patch_size and 0 <= x < img.shape[1] - patch_size:
                            half = patch_size // 2
                            patch = img[max(0, y-half):min(img.shape[0], y+half+1), 
                                       max(0, x-half):min(img.shape[1], x+half+1)]
                            if patch.size > 0:
                                val = np.median(patch)
                                bit = '1' if val > threshold else '0'
                                extracted_bits.append(bit)
                    
                    if len(extracted_bits) == len(target_bits):
                        extracted_str = ''.join(extracted_bits)
                        matches = sum(1 for i in range(len(target_bits)) if extracted_str[i] == target_bits[i])
                        accuracy = matches / len(target_bits)
                        
                        result = {
                            'position': (row0, col0),
                            'threshold': threshold,
                            'target': target_name,
                            'accuracy': accuracy,
                            'extracted': extracted_str,
                            'target_bits': target_bits
                        }
                        
                        best_results.append(result)
                        
                        if accuracy > 0.8:
                            print(f"    HIGH ACCURACY: {target_name} {accuracy:.1%} (threshold={threshold:.0f})")
                            print(f"      Target:    {target_bits}")
                            print(f"      Extracted: {extracted_str}")
    
    # Sort and show best results
    best_results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print(f"\n=== BEST ADAPTIVE THRESHOLD RESULTS ===")
    for i, result in enumerate(best_results[:15]):
        print(f"{i+1:2d}. {result['accuracy']:.1%} {result['target']:8s} pos=({result['position'][0]:3d},{result['position'][1]:3d}) thresh={result['threshold']:3.0f}")
        
        if result['accuracy'] > 0.9:
            print(f"    *** BREAKTHROUGH: {result['target']} ***")
            print(f"    Target:    {result['target_bits']}")
            print(f"    Extracted: {result['extracted']}")
            
            # Test full extraction at this position
            test_full_extraction(img, result)
            return result
    
    return best_results

def test_inverted_encoding():
    """Test if the message uses inverted encoding (bright=0, dark=1)."""
    
    print(f"\n=== TESTING INVERTED ENCODING ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Test our previous best position but with inverted logic
    row0, col0 = 58, 28
    row_pitch = 31
    col_pitch = 53
    patch_size = 5
    
    targets = [
        ("On", "0100111101101110"),
        ("At", "01000001011101000010000000100000"),
        ("Bitcoin", "0100001001101001011101000110001101101111011010010110111000100000"),
    ]
    
    # Try different thresholds with inverted logic
    for threshold in [20, 30, 40, 50, 75, 100]:
        print(f"\nInverted threshold {threshold}:")
        
        for target_name, target_bits in targets:
            extracted_bits = []
            
            for i in range(len(target_bits)):
                bit_row = i // 8
                bit_col = i % 8
                
                y = row0 + bit_row * row_pitch
                x = col0 + bit_col * col_pitch
                
                if 0 <= y < img.shape[0] - patch_size and 0 <= x < img.shape[1] - patch_size:
                    half = patch_size // 2
                    patch = img[max(0, y-half):min(img.shape[0], y+half+1), 
                               max(0, x-half):min(img.shape[1], x+half+1)]
                    if patch.size > 0:
                        val = np.median(patch)
                        # INVERTED: dark=1, bright=0
                        bit = '0' if val > threshold else '1'
                        extracted_bits.append(bit)
            
            if len(extracted_bits) == len(target_bits):
                extracted_str = ''.join(extracted_bits)
                matches = sum(1 for i in range(len(target_bits)) if extracted_str[i] == target_bits[i])
                accuracy = matches / len(target_bits)
                
                print(f"  {target_name:8s}: {accuracy:.1%}")
                
                if accuracy > 0.9:
                    print(f"    *** INVERTED BREAKTHROUGH: {target_name} ***")
                    print(f"    Target:    {target_bits}")
                    print(f"    Extracted: {extracted_str}")
                    return target_name, threshold, extracted_str

def test_full_extraction(img, best_config):
    """Test full message extraction with the best configuration."""
    
    print(f"\n=== FULL EXTRACTION TEST ===")
    print(f"Position: {best_config['position']}")
    print(f"Threshold: {best_config['threshold']:.0f}")
    print(f"Target: {best_config['target']}")
    
    row0, col0 = best_config['position']
    threshold = best_config['threshold']
    row_pitch = 31
    col_pitch = 53
    patch_size = 5
    
    # Extract several rows
    for r in range(5):
        y = row0 + r * row_pitch
        
        row_bits = []
        for c in range(16):  # 2 characters worth
            x = col0 + c * col_pitch
            
            if 0 <= y < img.shape[0] - patch_size and 0 <= x < img.shape[1] - patch_size:
                half = patch_size // 2
                patch = img[max(0, y-half):min(img.shape[0], y+half+1), 
                           max(0, x-half):min(img.shape[1], x+half+1)]
                if patch.size > 0:
                    val = np.median(patch)
                    bit = '1' if val > threshold else '0'
                    row_bits.append(bit)
        
        # Decode row
        if len(row_bits) >= 16:
            try:
                char1 = chr(int(''.join(row_bits[0:8]), 2))
                char2 = chr(int(''.join(row_bits[8:16]), 2))
                print(f"Row {r}: '{char1}{char2}' ({' '.join(row_bits[0:8])}) ({' '.join(row_bits[8:16])})")
            except:
                print(f"Row {r}: [decode error] {' '.join(row_bits[:16])}")

if __name__ == "__main__":
    print("Adaptive Threshold Search for Hidden Message")
    print("Finding bright regions and testing adaptive thresholds")
    print("="*70)
    
    # Find bright regions first
    bright_regions = find_bright_regions()
    
    # Test adaptive threshold extraction
    best_result = adaptive_threshold_extraction()
    
    # Test inverted encoding
    inverted_result = test_inverted_encoding()
    
    print("\n" + "="*70)
    print("ADAPTIVE SEARCH COMPLETE")
    
    if best_result and hasattr(best_result, '__getitem__') and 'accuracy' in best_result:
        if best_result['accuracy'] > 0.9:
            print(f"BREAKTHROUGH: {best_result['target']} at {best_result['accuracy']:.1%}")
        else:
            print(f"Best result: {best_result['accuracy']:.1%} accuracy")
    
    if inverted_result:
        print(f"Inverted encoding result: {inverted_result[0]} at {inverted_result[1]} threshold")
    
    print("Continue with ML approaches if no breakthrough found")