#!/usr/bin/env python3
"""
Final breakthrough attempt: exhaustive search with all techniques.
"""

import cv2
import numpy as np
from scipy import ndimage
import json

def load_all_images():
    """Load all available image sources."""
    
    sources = {
        'original': 'original_satoshi_poster.png',
        'bw_mask': 'binary_extractor/output_real_data/bw_mask.png',
        'gaussian': 'binary_extractor/output_real_data/gaussian_subtracted.png',
        'cyan': 'binary_extractor/output_real_data/cyan_channel.png'
    }
    
    images = {}
    for name, path in sources.items():
        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images[name] = img
                print(f"Loaded {name}: {img.shape}, range {img.min()}-{img.max()}")
        except:
            pass
    
    return images

def exhaustive_on_search(img, img_name):
    """Exhaustive search for 'On' pattern."""
    
    print(f"\n=== EXHAUSTIVE SEARCH: {img_name} ===")
    
    target_bits = "0100111101101110"  # "On"
    best_results = []
    
    # Test multiple thresholds
    thresholds = [100, 127, 150, 180, 200]
    
    # Test multiple sampling sizes
    patch_sizes = [3, 5, 7]
    
    # Test multiple grid configurations
    test_configs = [
        (31, 53), (31, 25), (31, 18), (30, 24), (32, 26),
        (5, 52), (25, 50), (28, 20), (35, 15)
    ]
    
    for threshold in thresholds:
        for patch_size in patch_sizes:
            for row_pitch, col_pitch in test_configs:
                
                # Search origins
                for row0 in range(50, 120, 3):
                    for col0 in range(20, 100, 2):
                        
                        # Extract 16 bits
                        bits = extract_bits_config(
                            img, row0, col0, row_pitch, col_pitch, 
                            16, threshold, patch_size
                        )
                        
                        if len(bits) == 16:
                            bits_str = ''.join(bits)
                            
                            # Score
                            matches = sum(1 for i in range(16) 
                                        if bits_str[i] == target_bits[i])
                            score = matches / 16
                            
                            if score > 0.75:  # High threshold
                                try:
                                    char1 = chr(int(bits_str[:8], 2))
                                    char2 = chr(int(bits_str[8:16], 2))
                                    decoded = f"{char1}{char2}"
                                    
                                    if char1.isalnum() and char2.isalnum():
                                        result = {
                                            'image': img_name,
                                            'threshold': threshold,
                                            'patch_size': patch_size,
                                            'pitch': (row_pitch, col_pitch),
                                            'origin': (row0, col0),
                                            'score': score,
                                            'bits': bits_str,
                                            'decoded': decoded
                                        }
                                        
                                        best_results.append(result)
                                        
                                        if decoded == "On":
                                            print(f"FOUND 'On'! Config: {result}")
                                            return result
                                        
                                        if score > 0.9:
                                            print(f"High score: {score:.1%} -> '{decoded}' at {(row0, col0)}")
                                
                                except:
                                    pass
    
    # Sort and return best
    best_results.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"Best results for {img_name}:")
    for i, result in enumerate(best_results[:5]):
        print(f"  {i+1}. {result['score']:.1%} -> '{result['decoded']}' "
              f"thresh={result['threshold']} patch={result['patch_size']} "
              f"pitch={result['pitch']} origin={result['origin']}")
    
    return best_results[0] if best_results else None

def extract_bits_config(img, row0, col0, row_pitch, col_pitch, num_bits, threshold, patch_size):
    """Extract bits with specific configuration."""
    
    bits = []
    
    for i in range(num_bits):
        bit_row = i // 8
        bit_col = i % 8
        
        y = row0 + bit_row * row_pitch
        x = col0 + bit_col * col_pitch
        
        if y < img.shape[0] - patch_size//2 and x < img.shape[1] - patch_size//2:
            half = patch_size // 2
            patch = img[max(0, y-half):min(img.shape[0], y+half+1), 
                       max(0, x-half):min(img.shape[1], x+half+1)]
            
            if patch.size > 0:
                val = np.median(patch)
                bit = '1' if val > threshold else '0'
                bits.append(bit)
    
    return bits

def test_alternative_approaches(img, img_name):
    """Test alternative extraction approaches."""
    
    print(f"\n=== ALTERNATIVE APPROACHES: {img_name} ===")
    
    # Approach 1: Edge detection first
    edges = cv2.Canny(img, 50, 150)
    edge_result = exhaustive_on_search(edges, f"{img_name}_edges")
    
    # Approach 2: Morphological operations
    kernel = np.ones((3,3), np.uint8)
    morphed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    morph_result = exhaustive_on_search(morphed, f"{img_name}_morph")
    
    # Approach 3: Histogram equalization
    equalized = cv2.equalizeHist(img)
    eq_result = exhaustive_on_search(equalized, f"{img_name}_eq")
    
    return [edge_result, morph_result, eq_result]

def try_different_bit_layouts(img, config):
    """Try different ways of interpreting the bit layout."""
    
    if not config:
        return []
    
    print(f"\n=== DIFFERENT BIT LAYOUTS ===")
    
    row0, col0 = config['origin']
    row_pitch, col_pitch = config['pitch']
    threshold = config['threshold']
    patch_size = config['patch_size']
    
    layouts = {
        'standard': extract_standard_layout,
        'column_major': extract_column_major_layout,
        'reverse_bits': extract_reverse_bits_layout,
        'reverse_bytes': extract_reverse_bytes_layout
    }
    
    results = []
    
    for layout_name, layout_func in layouts.items():
        bits = layout_func(img, row0, col0, row_pitch, col_pitch, 
                          24, threshold, patch_size)  # 3 characters
        
        if len(bits) >= 16:
            decoded_chars = []
            for i in range(0, min(len(bits), 24), 8):
                if i + 8 <= len(bits):
                    byte = ''.join(bits[i:i+8])
                    try:
                        val = int(byte, 2)
                        if 32 <= val <= 126:
                            decoded_chars.append(chr(val))
                        else:
                            decoded_chars.append(f'[{val}]')
                    except:
                        decoded_chars.append('?')
            
            decoded = ''.join(decoded_chars)
            results.append({
                'layout': layout_name,
                'decoded': decoded,
                'bits': ''.join(bits)
            })
            
            print(f"  {layout_name}: '{decoded}'")
            
            if 'On' in decoded:
                print(f"FOUND 'On' with {layout_name} layout!")
                return results
    
    return results

def extract_standard_layout(img, row0, col0, row_pitch, col_pitch, num_bits, threshold, patch_size):
    """Standard row-major bit extraction."""
    return extract_bits_config(img, row0, col0, row_pitch, col_pitch, num_bits, threshold, patch_size)

def extract_column_major_layout(img, row0, col0, row_pitch, col_pitch, num_bits, threshold, patch_size):
    """Column-major bit extraction."""
    bits = []
    for i in range(num_bits):
        char_idx = i // 8
        bit_idx = i % 8
        
        # Column-major: read down columns first
        y = row0 + bit_idx * row_pitch
        x = col0 + char_idx * col_pitch
        
        if y < img.shape[0] - patch_size//2 and x < img.shape[1] - patch_size//2:
            half = patch_size // 2
            patch = img[max(0, y-half):min(img.shape[0], y+half+1), 
                       max(0, x-half):min(img.shape[1], x+half+1)]
            
            if patch.size > 0:
                val = np.median(patch)
                bit = '1' if val > threshold else '0'
                bits.append(bit)
    
    return bits

def extract_reverse_bits_layout(img, row0, col0, row_pitch, col_pitch, num_bits, threshold, patch_size):
    """Reverse bit order within each byte."""
    standard_bits = extract_bits_config(img, row0, col0, row_pitch, col_pitch, num_bits, threshold, patch_size)
    
    # Reverse bits within each byte
    reversed_bits = []
    for i in range(0, len(standard_bits), 8):
        if i + 8 <= len(standard_bits):
            byte = standard_bits[i:i+8]
            reversed_bits.extend(byte[::-1])
    
    return reversed_bits

def extract_reverse_bytes_layout(img, row0, col0, row_pitch, col_pitch, num_bits, threshold, patch_size):
    """Reverse byte order."""
    standard_bits = extract_bits_config(img, row0, col0, row_pitch, col_pitch, num_bits, threshold, patch_size)
    
    # Group into bytes and reverse order
    bytes_list = []
    for i in range(0, len(standard_bits), 8):
        if i + 8 <= len(standard_bits):
            bytes_list.append(standard_bits[i:i+8])
    
    # Reverse byte order
    reversed_bits = []
    for byte in reversed(bytes_list):
        reversed_bits.extend(byte)
    
    return reversed_bits

def main():
    print("Final Breakthrough Attempt")
    print("Exhaustive search across all sources and methods")
    print("="*60)
    
    # Load all images
    images = load_all_images()
    
    if not images:
        print("No images available!")
        return
    
    all_results = []
    best_overall = None
    
    # Test each image source
    for img_name, img in images.items():
        
        # Exhaustive search for 'On'
        result = exhaustive_on_search(img, img_name)
        if result:
            all_results.append(result)
            
            if result['decoded'] == "On":
                print(f"\nBREAKTHROUGH: Found 'On' in {img_name}!")
                best_overall = result
                break
        
        # Try alternative preprocessing
        alt_results = test_alternative_approaches(img, img_name)
        all_results.extend([r for r in alt_results if r])
        
        # Try different bit layouts on best result
        if result:
            layout_results = try_different_bit_layouts(img, result)
            for lr in layout_results:
                if 'On' in lr['decoded']:
                    print(f"BREAKTHROUGH with layout: {lr}")
                    best_overall = {**result, **lr}
                    break
    
    # Summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    
    if best_overall:
        print(f"BEST RESULT:")
        print(f"  Image: {best_overall.get('image', 'unknown')}")
        print(f"  Decoded: '{best_overall.get('decoded', '???')}'")
        print(f"  Score: {best_overall.get('score', 0):.1%}")
        print(f"  Config: {best_overall}")
        
        if best_overall.get('decoded') == "On":
            print(f"\n🎉 SUCCESS! Found 'On' - extracting full message...")
            # Extract full message with best config
            
    else:
        # Show top results
        all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        print(f"Top results (no 'On' found):")
        for i, result in enumerate(all_results[:10]):
            print(f"  {i+1}. {result.get('decoded', '???')} "
                  f"({result.get('score', 0):.1%}) "
                  f"from {result.get('image', '???')}")
    
    # Save results
    with open('BREAKTHROUGH_ATTEMPT_RESULTS.json', 'w') as f:
        # Convert to JSON-serializable
        json_results = []
        for result in all_results:
            json_result = {}
            for key, value in result.items():
                if isinstance(value, (np.integer, np.floating)):
                    json_result[key] = float(value)
                else:
                    json_result[key] = value
            json_results.append(json_result)
        
        json.dump({
            'best_overall': best_overall,
            'all_results': json_results[:20]  # Top 20
        }, f, indent=2)
    
    print(f"\nResults saved to BREAKTHROUGH_ATTEMPT_RESULTS.json")

if __name__ == "__main__":
    main()