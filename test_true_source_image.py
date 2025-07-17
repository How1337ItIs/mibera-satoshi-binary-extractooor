#!/usr/bin/env python3
"""
Test the true source Mibera Satoshi poster with all our breakthrough techniques.
"""

import cv2
import numpy as np

def test_true_source():
    """Test the true source image with our best techniques."""
    
    print("=== TESTING TRUE SOURCE MIBERA SATOSHI POSTER ===")
    
    # Load the true source image
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("Could not load true source image!")
        return
    
    print(f"True source image: {img.shape}")
    print(f"Value range: {img.min()} to {img.max()}")
    
    # Test o3's exact one-liner on the true source
    print(f"\n=== O3's One-Liner on True Source ===")
    
    mask = img > 200
    proj = mask.sum(0).astype(float)
    proj -= proj.mean()
    corr = np.correlate(proj, proj, mode='full')[len(proj)-1:]
    peak = np.argmax(corr[5:]) + 5
    
    print(f"Detected column pitch: {peak} pixels")
    
    # Also test row pitch
    row_mask = img > 200
    row_proj = row_mask.sum(1).astype(float)
    row_proj -= row_proj.mean()
    row_corr = np.correlate(row_proj, row_proj, mode='full')[len(row_proj):]
    row_peak = np.argmax(row_corr[5:]) + 5
    
    print(f"Detected row pitch: {row_peak} pixels")
    
    # Now test with actual extraction
    target_bits = "0100111101101110"  # "On"
    best_results = []
    
    # Test multiple configurations based on our learning
    configs = [
        # (name, row_pitch, col_pitch, threshold, patch_size)
        ("detected_true", row_peak, peak, 127, 5),
        ("o3_hypothesis", row_peak, 8, 127, 5),
        ("manual_baseline", 31, 18, 127, 5),
        ("high_contrast", row_peak, peak, 180, 3),
        ("low_contrast", row_peak, peak, 100, 7),
    ]
    
    for name, row_pitch, col_pitch, threshold, patch_size in configs:
        print(f"\nTesting {name}: {row_pitch}x{col_pitch}, thresh={threshold}, patch={patch_size}")
        
        best_score = 0
        best_result = None
        
        # Focused search around promising areas
        for row0 in range(60, 90, 2):
            for col0 in range(30, 60, 2):
                
                bits = extract_bits_robust(img, row0, col0, row_pitch, col_pitch, 16, threshold, patch_size)
                
                if len(bits) == 16:
                    bits_str = ''.join(bits)
                    
                    matches = sum(1 for i in range(16) if bits_str[i] == target_bits[i])
                    score = matches / 16
                    
                    if score > best_score:
                        best_score = score
                        
                        try:
                            char1 = chr(int(bits_str[:8], 2))
                            char2 = chr(int(bits_str[8:16], 2))
                            decoded = f"{char1}{char2}"
                        except:
                            decoded = "??"
                        
                        best_result = {
                            'config': name,
                            'params': (row_pitch, col_pitch, threshold, patch_size),
                            'origin': (row0, col0),
                            'score': score,
                            'bits': bits_str,
                            'decoded': decoded
                        }
        
        if best_result:
            print(f"  Best: {best_result['origin']} -> {best_result['score']:.1%} '{best_result['decoded']}'")
            best_results.append(best_result)
            
            if best_result['decoded'] == "On":
                print(f"*** BREAKTHROUGH! Found 'On' with {name} ***")
                extract_full_message_breakthrough(img, best_result)
                return best_result
    
    # Show all results
    print(f"\n=== ALL RESULTS ===")
    best_results.sort(key=lambda x: x['score'], reverse=True)
    
    for i, result in enumerate(best_results):
        print(f"{i+1}. {result['config']:15s} {result['score']:.1%} -> '{result['decoded']}' at {result['origin']}")
    
    # Test the best one for full extraction
    if best_results:
        print(f"\n=== FULL EXTRACTION WITH BEST ===")
        best = best_results[0]
        full_message = extract_full_message_breakthrough(img, best)
        
        # Save comprehensive results
        with open('TRUE_SOURCE_EXTRACTION_RESULTS.txt', 'w') as f:
            f.write("=== TRUE SOURCE MIBERA SATOSHI POSTER EXTRACTION ===\n")
            f.write(f"Image: {img.shape} pixels\n")
            f.write(f"Detected pitches: row={row_peak}px, col={peak}px\n\n")
            
            f.write("Configuration Results:\n")
            for result in best_results:
                f.write(f"  {result['config']}: {result['score']:.1%} -> '{result['decoded']}'\n")
            
            f.write(f"\nBest configuration: {best['config']}\n")
            f.write(f"Parameters: {best['params']}\n")
            f.write(f"Origin: {best['origin']}\n")
            f.write(f"Score: {best['score']:.1%}\n\n")
            
            if full_message:
                f.write("Extracted message:\n")
                for i, line in enumerate(full_message):
                    f.write(f"Row {i:2d}: {line}\n")
        
        print(f"Results saved to TRUE_SOURCE_EXTRACTION_RESULTS.txt")
    
    return best_results

def extract_bits_robust(img, row0, col0, row_pitch, col_pitch, num_bits, threshold, patch_size):
    """Extract bits with robust patch sampling."""
    
    bits = []
    
    for i in range(num_bits):
        bit_row = i // 8
        bit_col = i % 8
        
        y = row0 + bit_row * row_pitch
        x = col0 + bit_col * col_pitch
        
        if y < img.shape[0] - patch_size and x < img.shape[1] - patch_size:
            half = patch_size // 2
            patch = img[max(0, y-half):min(img.shape[0], y+half+1), 
                       max(0, x-half):min(img.shape[1], x+half+1)]
            
            if patch.size > 0:
                val = np.median(patch)
                bit = '1' if val > threshold else '0'
                bits.append(bit)
    
    return bits

def extract_full_message_breakthrough(img, best_config):
    """Extract full message using the breakthrough configuration."""
    
    row_pitch, col_pitch, threshold, patch_size = best_config['params']
    row0, col0 = best_config['origin']
    
    print(f"Extracting full message with {best_config['config']} configuration...")
    print(f"Parameters: pitch={row_pitch}x{col_pitch}, origin=({row0},{col0}), thresh={threshold}")
    
    # Calculate grid size
    max_rows = min(60, (img.shape[0] - row0) // row_pitch)
    max_cols = min(120, (img.shape[1] - col0) // col_pitch)
    
    print(f"Grid: {max_rows} x {max_cols}")
    
    message_lines = []
    
    for r in range(max_rows):
        y = row0 + r * row_pitch
        if y >= img.shape[0] - patch_size:
            break
        
        # Extract this row
        row_bits = []
        for c in range(max_cols):
            x = col0 + c * col_pitch
            if x >= img.shape[1] - patch_size:
                break
            
            half = patch_size // 2
            patch = img[max(0, y-half):min(img.shape[0], y+half+1), 
                       max(0, x-half):min(img.shape[1], x+half+1)]
            
            if patch.size > 0:
                val = np.median(patch)
                bit = '1' if val > threshold else '0'
                row_bits.append(bit)
        
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
            
            if r < 15:  # Show first 15 rows
                print(f"Row {r:2d}: {line}")
    
    # Look for readable content
    readable_lines = []
    for i, line in enumerate(message_lines):
        readable_chars = sum(1 for c in line if c.isalnum() or c.isspace() or c in '.,!?-')
        if len(line) > 0 and readable_chars / len(line) > 0.4:
            readable_lines.append(f"Row {i}: {line}")
    
    if readable_lines:
        print(f"\nReadable content found:")
        for line in readable_lines[:10]:
            print(f"  {line}")
    
    return message_lines

def compare_with_previous():
    """Compare the true source with previous attempts."""
    
    print(f"\n=== COMPARISON WITH PREVIOUS ATTEMPTS ===")
    
    # Load both images for comparison
    true_source = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    prev_source = cv2.imread('original_satoshi_poster.png', cv2.IMREAD_GRAYSCALE)
    
    if true_source is not None and prev_source is not None:
        # Compare basic properties
        print(f"True source: {true_source.shape}, range {true_source.min()}-{true_source.max()}")
        print(f"Previous source: {prev_source.shape}, range {prev_source.min()}-{prev_source.max()}")
        
        # Check if they're the same
        if np.array_equal(true_source, prev_source):
            print("Images are identical - previous work still valid")
        else:
            print("Images are different - true source may yield better results")
            
            # Show difference stats
            diff = np.abs(true_source.astype(float) - prev_source.astype(float))
            print(f"Difference: mean={np.mean(diff):.1f}, max={np.max(diff):.1f}")
    
    else:
        print("Could not load one or both images for comparison")

if __name__ == "__main__":
    print("Testing True Source Mibera Satoshi Poster")
    print("Final breakthrough attempt with definitive source")
    print("="*70)
    
    # Compare with previous
    compare_with_previous()
    
    # Test the true source
    results = test_true_source()
    
    print("\n" + "="*70)
    print("FINAL ASSESSMENT:")
    
    if results:
        best = results[0]
        if best['decoded'] == "On":
            print("SUCCESS: Found 'On' in true source image!")
        elif best['score'] > 0.8:
            print(f"VERY CLOSE: {best['score']:.1%} accuracy with '{best['decoded']}'")
        elif best['score'] > 0.6:
            print(f"PROMISING: {best['score']:.1%} accuracy - continue refinement")
        else:
            print(f"CHALLENGE: Best was {best['score']:.1%} - may need alternative approach")
    else:
        print("NO RESULTS: Could not extract meaningful patterns")
    
    print("Comprehensive analysis complete.")