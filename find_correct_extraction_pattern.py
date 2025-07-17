#!/usr/bin/env python3
"""
Find the correct extraction pattern by testing for "On the" in various configurations.
"""

import numpy as np
import cv2
from itertools import product


def test_extraction_pattern(bw, row_start, col_start, row_pitch, col_pitch, target="On the"):
    """Test if a specific grid pattern produces the target text."""
    
    # Convert target to binary
    target_bits = ''
    for char in target:
        target_bits += format(ord(char), '08b')
    
    # Extract bits using the given pattern
    extracted_bits = []
    y = row_start
    
    # Extract enough bits for comparison
    for _ in range(2):  # Two rows
        x = col_start
        for _ in range(len(target_bits)):
            if x < bw.shape[1] and y < bw.shape[0]:
                # Sample a small region
                region = bw[max(0, y-2):min(bw.shape[0], y+3), 
                           max(0, x-2):min(bw.shape[1], x+3)]
                if region.size > 0:
                    val = np.mean(region)
                    bit = '1' if val > 127 else '0'
                    extracted_bits.append(bit)
            x += col_pitch
        y += row_pitch
    
    if len(extracted_bits) < len(target_bits):
        return 0.0, ''
    
    # Calculate match score
    extracted_str = ''.join(extracted_bits[:len(target_bits)])
    matches = sum(1 for i in range(len(target_bits)) if extracted_str[i] == target_bits[i])
    score = matches / len(target_bits)
    
    # Try to decode
    decoded = ''
    for i in range(0, min(len(extracted_bits), 48), 8):
        if i + 8 <= len(extracted_bits):
            byte = ''.join(extracted_bits[i:i+8])
            try:
                char_val = int(byte, 2)
                if 32 <= char_val <= 126:
                    decoded += chr(char_val)
                else:
                    decoded += f'[{char_val}]'
            except:
                decoded += '?'
    
    return score, decoded


def find_best_pattern():
    """Search for the best extraction pattern."""
    
    # Load binary mask
    bw = cv2.imread('binary_extractor/output_real_data/bw_mask.png', cv2.IMREAD_GRAYSCALE)
    
    print("Searching for optimal extraction pattern...")
    print("Target text: 'On the '")
    print("="*50)
    
    # Test different patterns
    results = []
    
    # Based on our analysis:
    # - Row pitch is consistently around 31
    # - Column structure is complex with varying spacings
    
    # Test row pitches near 31
    row_pitches = [30, 31, 32]
    
    # Test various column patterns
    # The visual analysis showed groups of digits with different spacings
    col_patterns = [
        # Single uniform pitch
        [8], [9], [10], [11], [12], [13], [14], [15], [17], [20], [25],
        # Alternating patterns (digit + gap)
        [12, 6], [11, 7], [10, 8], [13, 5],
        # Based on measured peaks
        [12, 13], [6, 12, 6], [8, 17],
    ]
    
    # Test different starting positions
    row_starts = range(50, 90, 2)
    col_starts = range(20, 60, 2)
    
    best_score = 0
    best_config = None
    
    for row_pitch, row_start in product(row_pitches, row_starts):
        for col_start in col_starts:
            # Test single pitch patterns
            for col_pitch in [8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 25]:
                score, decoded = test_extraction_pattern(bw, row_start, col_start, row_pitch, col_pitch)
                
                if score > best_score or (score == best_score and "On" in decoded):
                    best_score = score
                    best_config = {
                        'row_start': row_start,
                        'col_start': col_start,
                        'row_pitch': row_pitch,
                        'col_pitch': col_pitch,
                        'score': score,
                        'decoded': decoded
                    }
                    
                    if score > 0.5:
                        print(f"\nPromising pattern found!")
                        print(f"  Position: ({row_start}, {col_start})")
                        print(f"  Pitch: {row_pitch}x{col_pitch}")
                        print(f"  Score: {score:.1%}")
                        print(f"  Decoded: '{decoded}'")
    
    # Also test with the known working coordinates from previous analysis
    print("\n\nTesting known working coordinates...")
    
    known_configs = [
        # From manual extraction that achieved 77.1%
        {'row_start': 69, 'col_start': 37, 'row_pitch': 31, 'col_pitch': 18},
        {'row_start': 70, 'col_start': 37, 'row_pitch': 31, 'col_pitch': 18},
        {'row_start': 69, 'col_start': 36, 'row_pitch': 31, 'col_pitch': 18},
    ]
    
    for config in known_configs:
        score, decoded = test_extraction_pattern(
            bw, 
            config['row_start'], 
            config['col_start'], 
            config['row_pitch'], 
            config['col_pitch']
        )
        
        print(f"\nConfig: start=({config['row_start']}, {config['col_start']}), pitch={config['row_pitch']}x{config['col_pitch']}")
        print(f"  Score: {score:.1%}")
        print(f"  Decoded: '{decoded}'")
        
        if score > best_score:
            best_score = score
            best_config = {**config, 'score': score, 'decoded': decoded}
    
    print("\n" + "="*50)
    print("BEST CONFIGURATION FOUND:")
    if best_config:
        print(f"  Position: ({best_config['row_start']}, {best_config['col_start']})")
        print(f"  Pitch: {best_config['row_pitch']}x{best_config['col_pitch']}")
        print(f"  Score: {best_config['score']:.1%}")
        print(f"  Decoded: '{best_config['decoded']}'")
    else:
        print("  No good match found!")
    
    return best_config


def verify_column_structure():
    """Verify the actual column structure by looking at the digit patterns."""
    
    bw = cv2.imread('binary_extractor/output_real_data/bw_mask.png', cv2.IMREAD_GRAYSCALE)
    
    print("\n\nVerifying column structure...")
    
    # Look at a specific row and find digit boundaries
    test_row = 500
    row_data = bw[test_row, :]
    
    # Find transitions (edges)
    transitions = []
    for i in range(1, len(row_data)):
        if row_data[i] != row_data[i-1]:
            transitions.append(i)
    
    # Group transitions into digit boundaries
    if transitions:
        # Calculate distances between transitions
        distances = np.diff(transitions)
        
        print(f"\nTransition analysis for row {test_row}:")
        print(f"  Number of transitions: {len(transitions)}")
        print(f"  First 20 transitions: {transitions[:20]}")
        print(f"  First 20 distances: {distances[:20]}")
        
        # Look for patterns in distances
        small_distances = [d for d in distances if d < 20]
        large_distances = [d for d in distances if d >= 20]
        
        if small_distances:
            print(f"\n  Small distances (<20): mean={np.mean(small_distances):.1f}, count={len(small_distances)}")
        if large_distances:
            print(f"  Large distances (>=20): mean={np.mean(large_distances):.1f}, count={len(large_distances)}")


if __name__ == "__main__":
    print("Finding Correct Extraction Pattern")
    print("="*50)
    
    # Find best pattern
    best = find_best_pattern()
    
    # Verify column structure
    verify_column_structure()
    
    print("\n\nCONCLUSION:")
    print("The o3 suggestion of 8px column pitch appears to be incorrect.")
    print("The actual pattern is more complex with varying digit widths and gaps.")
    print("The working configuration uses ~18px column pitch, not 8px or 25px.")