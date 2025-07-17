#!/usr/bin/env python3
"""
Final corrected extractor using actual measured pitches.
Key insight: The binary mask shows ~53px column spacing, 31px row spacing.
"""

import cv2
import numpy as np
from scipy import signal
import json

def extract_with_measured_pitches():
    """Extract using the actual measured pitches from the binary mask."""
    
    # Load binary mask
    bw = cv2.imread('binary_extractor/output_real_data/bw_mask.png', cv2.IMREAD_GRAYSCALE)
    print(f"Image shape: {bw.shape}")
    
    # Use measured pitches
    row_pitch = 31  # This was consistent
    col_pitch = 53  # From manual measurement
    
    print(f"Using pitches: row={row_pitch}px, col={col_pitch}px")
    
    # Find origin by searching for good alignment
    best_score = -1
    best_origin = None
    
    print("\nSearching for optimal origin...")
    
    for row0 in range(50, 90, 2):
        for col0 in range(20, 80, 2):
            score = score_grid_alignment(bw, row0, col0, row_pitch, col_pitch)
            
            if score > best_score:
                best_score = score
                best_origin = (row0, col0)
    
    row0, col0 = best_origin
    print(f"Best origin: ({row0}, {col0}) with score {best_score:.2f}")
    
    # Extract grid
    print("\nExtracting grid...")
    
    rows = []
    cols = []
    
    y = row0
    while y < bw.shape[0] - 15:
        rows.append(y)
        y += row_pitch
    
    x = col0
    while x < bw.shape[1] - 15:
        cols.append(x)
        x += col_pitch
    
    print(f"Grid size: {len(rows)} rows x {len(cols)} columns")
    
    # Extract bits using 6x6 patch sampling
    grid = []
    all_bits = []
    
    for row_idx, y in enumerate(rows):
        row_bits = []
        for col_idx, x in enumerate(cols):
            # Sample 6x6 patch
            patch = bw[max(0, y-3):min(bw.shape[0], y+4), 
                      max(0, x-3):min(bw.shape[1], x+4)]
            
            if patch.size > 0:
                # Use median for robustness
                val = np.median(patch)
                bit = '1' if val > 127 else '0'
            else:
                bit = '0'
            
            row_bits.append(bit)
            all_bits.append(bit)
        
        grid.append(row_bits)
        
        # Check first few rows
        if row_idx < 5:
            row_str = ''.join(row_bits)
            decoded = decode_row(row_str)
            print(f"Row {row_idx:2d}: {row_str[:50]}... -> {decoded[:20]}")
    
    # Statistics
    zeros = all_bits.count('0')
    ones = all_bits.count('1')
    print(f"\nBit statistics:")
    print(f"  Zeros: {zeros} ({zeros/len(all_bits)*100:.1f}%)")
    print(f"  Ones: {ones} ({ones/len(all_bits)*100:.1f}%)")
    
    # Search for "On the " pattern
    search_for_pattern(grid, "On the ")
    
    # Save results
    save_extraction_results(grid, row_pitch, col_pitch, row0, col0)
    
    return grid

def score_grid_alignment(bw, row0, col0, row_pitch, col_pitch):
    """Score how well a grid aligns with actual digit positions."""
    score = 0
    
    # Test first few rows and columns
    for r in range(3):
        y = row0 + r * row_pitch
        if y >= bw.shape[0] - 5:
            break
            
        for c in range(10):
            x = col0 + c * col_pitch
            if x >= bw.shape[1] - 5:
                break
            
            # Sample 5x5 region
            region = bw[y-2:y+3, x-2:x+3]
            if region.size > 0:
                mean_val = np.mean(region)
                # Reward clear decisions (very black or very white)
                if mean_val > 200:
                    score += 2
                elif mean_val < 50:
                    score += 1
                else:
                    score -= 0.5
    
    return score

def decode_row(bits):
    """Decode a row of bits to ASCII."""
    decoded = []
    for i in range(0, len(bits), 8):
        if i + 8 <= len(bits):
            byte = bits[i:i+8]
            try:
                val = int(byte, 2)
                if 32 <= val <= 126:
                    decoded.append(chr(val))
                else:
                    decoded.append(f'[{val}]')
            except:
                decoded.append('?')
    return ''.join(decoded)

def search_for_pattern(grid, target="On the "):
    """Search for target pattern in the grid."""
    target_bits = ''.join(format(ord(c), '08b') for c in target)
    
    print(f"\nSearching for '{target}' pattern...")
    print(f"Target bits: {target_bits}")
    
    best_matches = []
    
    for row_idx, row in enumerate(grid):
        if len(row) >= len(target_bits):
            row_str = ''.join(row)
            
            # Try different starting positions
            for start in range(min(20, len(row_str) - len(target_bits) + 1)):
                test_bits = row_str[start:start + len(target_bits)]
                
                # Calculate match score
                matches = sum(1 for i in range(len(target_bits)) 
                            if i < len(test_bits) and test_bits[i] == target_bits[i])
                score = matches / len(target_bits)
                
                if score > 0.5:
                    decoded = decode_row(test_bits)
                    best_matches.append({
                        'row': row_idx,
                        'col': start,
                        'score': score,
                        'decoded': decoded
                    })
    
    # Sort by score
    best_matches.sort(key=lambda x: x['score'], reverse=True)
    
    if best_matches:
        print(f"Found {len(best_matches)} good matches:")
        for i, match in enumerate(best_matches[:10]):
            print(f"  {i+1}. Row {match['row']}, Col {match['col']}: "
                  f"{match['score']:.1%} -> '{match['decoded']}'")
    else:
        print("No good matches found")
    
    return best_matches

def save_extraction_results(grid, row_pitch, col_pitch, row0, col0):
    """Save the extraction results."""
    
    results = {
        'method': 'measured_pitch_extraction',
        'parameters': {
            'row_pitch': row_pitch,
            'col_pitch': col_pitch,
            'row0': row0,
            'col0': col0,
            'sampling': '6x6_median_patch'
        },
        'grid_size': {
            'rows': len(grid),
            'cols': len(grid[0]) if grid else 0
        },
        'extracted_data': []
    }
    
    # Save each row
    for row_idx, row in enumerate(grid):
        row_str = ''.join(row)
        decoded = decode_row(row_str)
        
        results['extracted_data'].append({
            'row': row_idx,
            'bits': row_str,
            'decoded': decoded[:50]  # Limit length
        })
    
    # Save to file
    with open('final_extraction_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Also save as text
    with open('final_extraction_results.txt', 'w') as f:
        f.write("=== FINAL EXTRACTION RESULTS ===\n")
        f.write(f"Method: Measured pitch extraction\n")
        f.write(f"Grid: {row_pitch}x{col_pitch} pixels, origin ({row0}, {col0})\n")
        f.write(f"Size: {len(grid)} rows x {len(grid[0]) if grid else 0} cols\n\n")
        
        for row_idx, row in enumerate(grid):
            row_str = ''.join(row)
            decoded = decode_row(row_str)
            f.write(f"Row {row_idx:2d}: {row_str}\n")
            f.write(f"       -> {decoded}\n\n")
    
    print(f"\nResults saved to final_extraction_results.json and .txt")

if __name__ == "__main__":
    print("Final Corrected Extractor")
    print("Using measured pitches from actual binary mask")
    print("="*50)
    
    # Extract with measured pitches
    grid = extract_with_measured_pitches()
    
    print("\n" + "="*50)
    print("CONCLUSION:")
    print("This uses the actual measured column pitch (~53px) from")
    print("the binary mask. If o3's advice was for original resolution,")
    print("then the 8px pitch would apply to a different scale image.")
    print("The key insight is using robust 6x6 patch sampling and")
    print("proper grid alignment scoring.")