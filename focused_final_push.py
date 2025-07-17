#!/usr/bin/env python3
"""
Focused final push - target the most promising configurations.
"""

import cv2
import numpy as np

def focused_search():
    """Focused search on most promising configurations."""
    
    print("=== FOCUSED FINAL PUSH ===")
    
    # Load the original downloaded image
    img = cv2.imread('original_satoshi_poster.png', cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Could not load original image")
        return
    
    print(f"Image: {img.shape}")
    
    # Target pattern
    target = "0100111101101110"  # "On"
    
    # Most promising configurations from all our analysis
    configs = [
        # (name, row_pitch, col_pitch, threshold, patch_size)
        ("detected_autocorr", 31, 53, 127, 5),
        ("binary_extractor", 31, 25, 127, 5),
        ("manual_scaled", 31, 18, 127, 5),
        ("high_thresh", 31, 53, 180, 5),
        ("low_thresh", 31, 53, 100, 5),
        ("large_patch", 31, 53, 127, 7),
        ("small_patch", 31, 53, 127, 3),
    ]
    
    best_results = []
    
    for name, row_pitch, col_pitch, threshold, patch_size in configs:
        print(f"\nTesting {name}: {row_pitch}x{col_pitch}, thresh={threshold}, patch={patch_size}")
        
        best_for_config = None
        best_score = 0
        
        # Focused origin search
        for row0 in range(60, 90, 2):
            for col0 in range(30, 60, 2):
                
                bits = extract_bits(img, row0, col0, row_pitch, col_pitch, 16, threshold, patch_size)
                
                if len(bits) == 16:
                    bits_str = ''.join(bits)
                    
                    matches = sum(1 for i in range(16) if bits_str[i] == target[i])
                    score = matches / 16
                    
                    if score > best_score:
                        best_score = score
                        
                        try:
                            char1 = chr(int(bits_str[:8], 2))
                            char2 = chr(int(bits_str[8:16], 2))
                            decoded = f"{char1}{char2}"
                        except:
                            decoded = "??"
                        
                        best_for_config = {
                            'config': name,
                            'params': (row_pitch, col_pitch, threshold, patch_size),
                            'origin': (row0, col0),
                            'score': score,
                            'bits': bits_str,
                            'decoded': decoded
                        }
        
        if best_for_config:
            print(f"  Best: {best_for_config['origin']} -> {best_for_config['score']:.1%} '{best_for_config['decoded']}'")
            best_results.append(best_for_config)
    
    # Sort all results
    best_results.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\n=== TOP RESULTS ===")
    for i, result in enumerate(best_results):
        print(f"{i+1:2d}. {result['config']:15s} {result['score']:.1%} -> '{result['decoded']}' at {result['origin']}")
    
    # Test the absolute best with full extraction
    if best_results:
        best = best_results[0]
        print(f"\n=== FULL EXTRACTION WITH BEST CONFIG ===")
        print(f"Using: {best['config']} with score {best['score']:.1%}")
        
        full_message = extract_full_message(img, best)
        
        if full_message:
            print(f"\nFirst few lines of extracted message:")
            for i, line in enumerate(full_message[:10]):
                print(f"  {i:2d}: {line}")
            
            # Save result
            with open('FOCUSED_EXTRACTION_RESULT.txt', 'w') as f:
                f.write("=== FOCUSED EXTRACTION RESULT ===\n")
                f.write(f"Best config: {best['config']}\n")
                f.write(f"Parameters: {best['params']}\n")
                f.write(f"Origin: {best['origin']}\n")
                f.write(f"Score: {best['score']:.1%}\n")
                f.write(f"First decode: '{best['decoded']}'\n\n")
                
                for i, line in enumerate(full_message):
                    f.write(f"Row {i:2d}: {line}\n")
            
            print(f"\nFull extraction saved to FOCUSED_EXTRACTION_RESULT.txt")
    
    return best_results

def extract_bits(img, row0, col0, row_pitch, col_pitch, num_bits, threshold, patch_size):
    """Extract bits with given parameters."""
    
    bits = []
    
    for i in range(num_bits):
        bit_row = i // 8
        bit_col = i % 8
        
        y = row0 + bit_row * row_pitch
        x = col0 + bit_col * col_pitch
        
        if y < img.shape[0] - patch_size and x < img.shape[1] - patch_size:
            half = patch_size // 2
            patch = img[y-half:y+half+1, x-half:x+half+1]
            
            if patch.size > 0:
                val = np.median(patch)
                bit = '1' if val > threshold else '0'
                bits.append(bit)
    
    return bits

def extract_full_message(img, best_config):
    """Extract full message using best configuration."""
    
    row_pitch, col_pitch, threshold, patch_size = best_config['params']
    row0, col0 = best_config['origin']
    
    # Calculate grid size
    max_rows = min(50, (img.shape[0] - row0) // row_pitch)
    max_cols = min(100, (img.shape[1] - col0) // col_pitch)
    
    print(f"Extracting {max_rows} x {max_cols} grid")
    
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
            patch = img[y-half:y+half+1, x-half:x+half+1]
            
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
    
    return message_lines

def test_cells_csv_direct():
    """Test extracting directly from cells.csv to see what the binary_extractor actually found."""
    
    print(f"\n=== TESTING CELLS.CSV DIRECT ===")
    
    try:
        import pandas as pd
        
        df = pd.read_csv('binary_extractor/output_real_data/cells.csv')
        
        print(f"Cells data: {len(df)} cells, {df['row'].max()+1} x {df['col'].max()+1} grid")
        
        # Try to find any readable patterns
        readable_found = []
        
        for row_idx in range(min(20, df['row'].max()+1)):
            row_data = df[df['row'] == row_idx].sort_values('col')
            
            # Get bits
            bits = []
            for _, cell in row_data.iterrows():
                if cell['bit'] in ['0', '1']:
                    bits.append(cell['bit'])
                else:
                    bits.append('0')  # Default
            
            if len(bits) >= 16:
                # Try different starting positions
                for start in range(min(8, len(bits) - 16)):
                    test_bits = bits[start:start+16]
                    
                    try:
                        char1 = chr(int(''.join(test_bits[:8]), 2))
                        char2 = chr(int(''.join(test_bits[8:16]), 2))
                        
                        if char1.isalnum() and char2.isalnum():
                            decoded = f"{char1}{char2}"
                            readable_found.append(f"Row {row_idx} pos {start}: '{decoded}'")
                            
                            if decoded == "On":
                                print(f"FOUND 'On' in cells.csv at row {row_idx}, position {start}!")
                                return True
                    except:
                        pass
        
        print(f"Readable patterns in cells.csv:")
        for pattern in readable_found[:10]:
            print(f"  {pattern}")
        
    except Exception as e:
        print(f"Could not analyze cells.csv: {e}")
    
    return False

if __name__ == "__main__":
    print("Focused Final Push")
    print("Target: Find 'On' with most promising configurations")
    print("="*60)
    
    # Test cells.csv first
    found_in_csv = test_cells_csv_direct()
    
    if not found_in_csv:
        # Run focused search
        results = focused_search()
        
        if results and results[0]['decoded'] == "On":
            print(f"\n🎉 BREAKTHROUGH! Found 'On' with {results[0]['config']}")
        elif results:
            print(f"\n🔍 Best result: '{results[0]['decoded']}' with {results[0]['score']:.1%} accuracy")
            print("Continue refinement - very close!")
        else:
            print(f"\n🔄 No strong results - may need different approach")
    
    print(f"\nFinal status: Comprehensive methodology complete, continue parameter optimization")