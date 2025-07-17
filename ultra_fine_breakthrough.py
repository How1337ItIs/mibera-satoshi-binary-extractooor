#!/usr/bin/env python3
"""
Ultra-fine grid search for the final breakthrough.
Testing at 0.1 pixel increments around the promising (60, 30) region.
"""

import cv2
import numpy as np
from scipy import ndimage

def ultra_fine_grid_search():
    """Ultra-fine grid search at 0.1 pixel increments."""
    
    print("=== ULTRA-FINE GRID SEARCH ===")
    print("Testing 0.1 pixel increments around (60, 30)")
    
    # Load the source image
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Use the best known parameters
    row_pitch = 31
    col_pitch = 53
    threshold = 127
    patch_size = 5
    
    target_bits = "0100111101101110"  # "On"
    
    # Ultra-fine search around (60, 30)
    row_range = np.arange(58.0, 63.0, 0.2)  # ±2 pixels, 0.2 increments
    col_range = np.arange(28.0, 33.0, 0.2)  # ±2 pixels, 0.2 increments
    
    best_results = []
    breakthrough_found = False
    
    total_tests = len(row_range) * len(col_range)
    test_count = 0
    
    print(f"Testing {total_tests} positions...")
    
    for row0 in row_range:
        for col0 in col_range:
            test_count += 1
            
            if test_count % 50 == 0:
                print(f"  Progress: {test_count}/{total_tests}")
            
            # Extract 16 bits with sub-pixel precision
            bits = extract_bits_subpixel(img, row0, col0, row_pitch, col_pitch, 16, threshold, patch_size)
            
            if len(bits) == 16:
                bits_str = ''.join(bits)
                
                # Score against "On"
                matches = sum(1 for i in range(16) if bits_str[i] == target_bits[i])
                score = matches / 16
                
                if score > 0.75:  # High threshold for breakthrough
                    try:
                        char1 = chr(int(bits_str[:8], 2))
                        char2 = chr(int(bits_str[8:16], 2))
                        decoded = f"{char1}{char2}"
                        
                        result = {
                            'position': (row0, col0),
                            'score': score,
                            'bits': bits_str,
                            'decoded': decoded
                        }
                        
                        best_results.append(result)
                        
                        if decoded == "On":
                            print(f"\n*** BREAKTHROUGH! ***")
                            print(f"Found 'On' at position ({row0:.1f}, {col0:.1f})")
                            print(f"Score: {score:.1%}")
                            print(f"Bits: {bits_str}")
                            breakthrough_found = True
                            
                            # Extract full message immediately
                            extract_full_message_precise(img, row0, col0, row_pitch, col_pitch)
                            return result
                        
                        elif score > 0.9:
                            print(f"\nHigh score: {score:.1%} -> '{decoded}' at ({row0:.1f}, {col0:.1f})")
                    
                    except:
                        pass
    
    # Sort and show best results
    best_results.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\n=== ULTRA-FINE SEARCH RESULTS ===")
    print(f"Tested {total_tests} positions")
    print(f"Found {len(best_results)} high-scoring results")
    
    if best_results:
        print(f"\nTop results:")
        for i, result in enumerate(best_results[:10]):
            print(f"  {i+1:2d}. ({result['position'][0]:5.1f}, {result['position'][1]:5.1f}) "
                  f"{result['score']:.1%} -> '{result['decoded']}'")
        
        # Test the best one
        if not breakthrough_found:
            best = best_results[0]
            print(f"\n=== TESTING BEST POSITION ===")
            print(f"Position: ({best['position'][0]:.1f}, {best['position'][1]:.1f})")
            print(f"Score: {best['score']:.1%}")
            print(f"Decoded: '{best['decoded']}'")
            
            # Extract more context
            extract_full_message_precise(img, best['position'][0], best['position'][1], row_pitch, col_pitch)
    
    else:
        print("No high-scoring results found in ultra-fine search")
    
    return best_results

def extract_bits_subpixel(img, row0, col0, row_pitch, col_pitch, num_bits, threshold, patch_size):
    """Extract bits with sub-pixel precision using interpolation."""
    
    bits = []
    
    for i in range(num_bits):
        bit_row = i // 8
        bit_col = i % 8
        
        y = row0 + bit_row * row_pitch
        x = col0 + bit_col * col_pitch
        
        # Use bilinear interpolation for sub-pixel sampling
        if 1 <= y < img.shape[0]-1 and 1 <= x < img.shape[1]-1:
            
            # Sample a patch around the sub-pixel position
            half = patch_size // 2
            
            # Get integer bounds
            y_int = int(y)
            x_int = int(x)
            
            # Extract patch
            patch_y_start = max(0, y_int - half)
            patch_y_end = min(img.shape[0], y_int + half + 1)
            patch_x_start = max(0, x_int - half)
            patch_x_end = min(img.shape[1], x_int + half + 1)
            
            patch = img[patch_y_start:patch_y_end, patch_x_start:patch_x_end]
            
            if patch.size > 0:
                # Use bilinear interpolation on the patch center
                if patch.shape[0] >= 2 and patch.shape[1] >= 2:
                    # Get fractional parts
                    y_frac = y - y_int
                    x_frac = x - x_int
                    
                    # Find center of patch
                    center_y = patch.shape[0] // 2
                    center_x = patch.shape[1] // 2
                    
                    if center_y < patch.shape[0]-1 and center_x < patch.shape[1]-1:
                        # Bilinear interpolation
                        top_left = patch[center_y, center_x]
                        top_right = patch[center_y, center_x + 1]
                        bottom_left = patch[center_y + 1, center_x]
                        bottom_right = patch[center_y + 1, center_x + 1]
                        
                        top = top_left * (1 - x_frac) + top_right * x_frac
                        bottom = bottom_left * (1 - x_frac) + bottom_right * x_frac
                        val = top * (1 - y_frac) + bottom * y_frac
                    else:
                        val = np.median(patch)
                else:
                    val = np.median(patch)
                
                bit = '1' if val > threshold else '0'
                bits.append(bit)
    
    return bits

def extract_full_message_precise(img, row0, col0, row_pitch, col_pitch):
    """Extract full message with precise sub-pixel positioning."""
    
    print(f"\n=== PRECISE FULL MESSAGE EXTRACTION ===")
    print(f"Position: ({row0:.1f}, {col0:.1f})")
    print(f"Pitch: {row_pitch}x{col_pitch}")
    
    # Calculate grid
    max_rows = min(50, int((img.shape[0] - row0) / row_pitch))
    max_cols = min(100, int((img.shape[1] - col0) / col_pitch))
    
    print(f"Grid: {max_rows} x {max_cols}")
    
    message_lines = []
    
    for r in range(max_rows):
        y = row0 + r * row_pitch
        
        row_bits = []
        for c in range(max_cols):
            x = col0 + c * col_pitch
            
            if 1 <= y < img.shape[0]-1 and 1 <= x < img.shape[1]-1:
                # Sub-pixel sampling
                bits = extract_bits_subpixel(img, y, x, 0, 1, 1, 127, 5)
                if bits:
                    row_bits.append(bits[0])
        
        # Decode row
        if len(row_bits) >= 8:
            decoded_chars = []
            for i in range(0, len(row_bits), 8):
                if i + 8 <= len(row_bits):
                    byte = ''.join(row_bits[i:i+8])
                    try:
                        val = int(byte, 2)
                        if 32 <= val <= 126:
                            decoded_chars.append(chr(val))
                        else:
                            decoded_chars.append(f'[{val}]')
                    except:
                        decoded_chars.append('?')
            
            line = ''.join(decoded_chars)
            message_lines.append(line)
            
            if r < 20:
                print(f"Row {r:2d}: {line}")
    
    # Save precise results
    with open('ULTRA_FINE_EXTRACTION_RESULTS.txt', 'w') as f:
        f.write("=== ULTRA-FINE PRECISION EXTRACTION ===\n")
        f.write(f"Position: ({row0:.1f}, {col0:.1f})\n")
        f.write(f"Pitch: {row_pitch}x{col_pitch}\n")
        f.write(f"Grid: {max_rows} x {max_cols}\n\n")
        
        for i, line in enumerate(message_lines):
            f.write(f"Row {i:2d}: {line}\n")
    
    print(f"\nPrecise results saved to ULTRA_FINE_EXTRACTION_RESULTS.txt")
    
    return message_lines

def test_alternative_targets():
    """Test for messages other than 'On the'."""
    
    print(f"\n=== TESTING ALTERNATIVE MESSAGE TARGETS ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Common message beginnings
    targets = [
        ("Bitcoin", "0100001001101001011101000110001101101111011010010110111000100000"),
        ("Satoshi", "0101001101100001011101000110111101110011011010000110100100100000"),
        ("Message", "01001101011001010111001101110011011000010110011101100101"),
        ("The", "01010100011010000110010100100000"),
        ("In", "01001001011011100010000000100000"),
        ("To", "01010100011011110010000000100000"),
        ("At", "01000001011101000010000000100000"),
        ("Be", "01000010011001010010000000100000")
    ]
    
    row_pitch = 31
    col_pitch = 53
    
    for target_name, target_bits in targets:
        print(f"\nSearching for '{target_name}'...")
        
        best_score = 0
        best_pos = None
        
        # Quick search around promising area
        for row0 in np.arange(58, 63, 0.5):
            for col0 in np.arange(28, 33, 0.5):
                bits = extract_bits_subpixel(img, row0, col0, row_pitch, col_pitch, len(target_bits), 127, 5)
                
                if len(bits) == len(target_bits):
                    bits_str = ''.join(bits)
                    matches = sum(1 for i in range(len(target_bits)) if bits_str[i] == target_bits[i])
                    score = matches / len(target_bits)
                    
                    if score > best_score:
                        best_score = score
                        best_pos = (row0, col0)
        
        print(f"  Best score: {best_score:.1%} at {best_pos}")
        
        if best_score > 0.8:
            print(f"  *** HIGH SCORE for '{target_name}' ***")
            return target_name, best_pos, best_score
    
    return None, None, 0

if __name__ == "__main__":
    print("Ultra-Fine Breakthrough Search")
    print("0.1 pixel precision around (60, 30)")
    print("="*50)
    
    # Ultra-fine grid search
    results = ultra_fine_grid_search()
    
    # Test alternative targets
    alt_result = test_alternative_targets()
    
    print("\n" + "="*50)
    print("ULTRA-FINE SEARCH COMPLETE")
    
    if results:
        best = results[0]
        if best['decoded'] == "On":
            print("🎉 BREAKTHROUGH: Found 'On' with ultra-fine precision!")
        elif best['score'] > 0.9:
            print(f"🔍 VERY CLOSE: {best['score']:.1%} accuracy - almost there!")
        else:
            print(f"📈 PROGRESS: Best {best['score']:.1%} - precision helps but not enough")
    
    if alt_result[0]:
        print(f"🎯 ALTERNATIVE: Found '{alt_result[0]}' with {alt_result[2]:.1%} accuracy")
    
    print("Next: Test alternative encodings and ML approaches")