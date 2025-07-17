#!/usr/bin/env python3
"""
Debug why we're not getting 'On' extraction.
"""

import cv2
import numpy as np

def debug_on_extraction():
    """Debug the 'On' extraction issue."""
    
    img = cv2.imread('binary_extractor/output_real_data/bw_mask.png', 0)
    
    # Target pattern for "On"
    target = "0100111101101110"  # "On" in binary
    
    print(f"Debugging 'On' extraction...")
    print(f"Target bits: {target}")
    print(f"Target chars: O={ord('O'):08b}, n={ord('n'):08b}")
    
    # Try a systematic search across the image
    best_matches = []
    
    # Test different row/col positions and parameters
    for row_start in range(50, 120, 5):
        for col_start in range(20, 100, 5):
            for row_step in range(25, 40, 2):  # Row pitch variations
                for col_step in range(45, 60, 2):  # Col pitch variations
                    
                    # Extract 16 bits
                    bits = []
                    
                    for i in range(16):
                        y = row_start + (i // 8) * row_step
                        x = col_start + (i % 8) * col_step
                        
                        if y < img.shape[0] - 5 and x < img.shape[1] - 5:
                            # Sample 5x5 region
                            region = img[y-2:y+3, x-2:x+3]
                            if region.size > 0:
                                val = np.median(region)
                                bit = '1' if val > 127 else '0'
                                bits.append(bit)
                    
                    if len(bits) == 16:
                        bits_str = ''.join(bits)
                        
                        # Calculate match score
                        matches = sum(1 for i in range(16) if bits_str[i] == target[i])
                        score = matches / 16
                        
                        if score > 0.5:  # Good match
                            # Try to decode
                            try:
                                char1 = chr(int(bits_str[:8], 2))
                                char2 = chr(int(bits_str[8:16], 2))
                                decoded = f"{char1}{char2}"
                            except:
                                decoded = "??"
                            
                            best_matches.append({
                                'score': score,
                                'pos': (row_start, col_start),
                                'pitch': (row_step, col_step),
                                'bits': bits_str,
                                'decoded': decoded
                            })
    
    # Sort by score
    best_matches.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\nFound {len(best_matches)} decent matches:")
    for i, match in enumerate(best_matches[:20]):
        print(f"  {i+1:2d}. Score {match['score']:.1%} at {match['pos']} "
              f"pitch {match['pitch']} -> '{match['decoded']}' ({match['bits']})")
    
    # Also try some known working coordinates from previous analysis
    print(f"\nTesting known coordinates from previous work...")
    
    known_coords = [
        # From the 77.1% manual extraction
        (69, 37, 31, 18),
        (70, 37, 31, 18),
        (69, 36, 31, 18),
        # From binary extractor output
        (72, 28, 30, 25),
    ]
    
    for row_start, col_start, row_step, col_step in known_coords:
        print(f"\nTesting ({row_start}, {col_start}) with pitch {row_step}x{col_step}:")
        
        # Extract first 16 bits
        bits = []
        for i in range(16):
            y = row_start + (i // 8) * row_step
            x = col_start + (i % 8) * col_step
            
            if y < img.shape[0] - 3 and x < img.shape[1] - 3:
                # Try different sampling methods
                methods = {
                    'center': img[y, x],
                    '3x3_mean': np.mean(img[y-1:y+2, x-1:x+2]),
                    '5x5_median': np.median(img[y-2:y+3, x-2:x+3]),
                }
                
                for method_name, val in methods.items():
                    if len(bits) <= i:
                        bit = '1' if val > 127 else '0'
                        if method_name == 'center':
                            bits.append(bit)
        
        if len(bits) >= 16:
            bits_str = ''.join(bits[:16])
            matches = sum(1 for i in range(16) if bits_str[i] == target[i])
            score = matches / 16
            
            try:
                char1 = chr(int(bits_str[:8], 2))
                char2 = chr(int(bits_str[8:16], 2))
                decoded = f"{char1}{char2}"
            except:
                decoded = "??"
            
            print(f"  Score: {score:.1%}, Decoded: '{decoded}', Bits: {bits_str}")

def try_different_approaches():
    """Try completely different extraction approaches."""
    
    img = cv2.imread('binary_extractor/output_real_data/bw_mask.png', 0)
    
    print(f"\n=== Trying Different Approaches ===")
    
    # Approach 1: Look for ASCII patterns directly
    print("1. Direct ASCII pattern search:")
    
    # Scan for patterns that look like "On" when decoded
    on_candidates = []
    
    for y in range(50, 150, 5):
        for x in range(20, 200, 5):
            # Extract 16 consecutive bits horizontally
            bits = []
            for i in range(16):
                if x + i < img.shape[1]:
                    val = img[y, x + i]
                    bit = '1' if val > 127 else '0'
                    bits.append(bit)
            
            if len(bits) == 16:
                try:
                    char1 = chr(int(''.join(bits[:8]), 2))
                    char2 = chr(int(''.join(bits[8:16]), 2))
                    
                    if char1.isalnum() and char2.isalnum():
                        on_candidates.append({
                            'pos': (y, x),
                            'chars': f"{char1}{char2}",
                            'bits': ''.join(bits)
                        })
                        
                        if char1 == 'O' and char2 == 'n':
                            print(f"  FOUND 'On' at ({y}, {x})!")
                            return (y, x)
                
                except:
                    pass
    
    print(f"  Found {len(on_candidates)} readable character pairs:")
    for cand in on_candidates[:10]:
        print(f"    {cand['pos']}: '{cand['chars']}'")
    
    # Approach 2: Use the extracted cells.csv data
    try:
        import pandas as pd
        cells_df = pd.read_csv('binary_extractor/output_real_data/cells.csv')
        
        print(f"\n2. Using extracted cells.csv:")
        print(f"   Grid: {cells_df['row'].max()+1} x {cells_df['col'].max()+1}")
        
        # Look for "On" in the first few rows
        for row_idx in range(min(10, cells_df['row'].max()+1)):
            row_cells = cells_df[cells_df['row'] == row_idx].sort_values('col')
            row_bits = ''.join(row_cells['bit'].apply(lambda x: x if x in '01' else '0'))
            
            if len(row_bits) >= 16:
                try:
                    for start in range(min(8, len(row_bits)-15)):
                        test_bits = row_bits[start:start+16]
                        char1 = chr(int(test_bits[:8], 2))
                        char2 = chr(int(test_bits[8:16], 2))
                        
                        if char1.isalnum() and char2.isalnum():
                            print(f"   Row {row_idx} start {start}: '{char1}{char2}'")
                            
                            if char1 == 'O' and char2 == 'n':
                                print(f"   FOUND 'On' in cells.csv row {row_idx}!")
                except:
                    pass
    
    except Exception as e:
        print(f"   Could not load cells.csv: {e}")

if __name__ == "__main__":
    print("Debugging 'On' Extraction")
    print("="*50)
    
    debug_on_extraction()
    try_different_approaches()
    
    print("\n" + "="*50)
    print("If no 'On' found, the message might be:")
    print("1. At a different location than expected")
    print("2. Rotated or transformed")
    print("3. Using a different encoding")
    print("4. Corrupted in this processed binary mask")