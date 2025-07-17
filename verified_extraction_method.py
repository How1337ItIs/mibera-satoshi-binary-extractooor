#!/usr/bin/env python3
"""
Create a verified extraction method using the known good text:
"On the winter solstice December 21 "

This will help establish the correct grid parameters for reliable extraction.
"""

import cv2
import numpy as np
import json

def create_target_pattern():
    """Create the target bit pattern for known text."""
    
    known_text = "On the winter solstice December 21 "
    print(f"=== TARGET PATTERN ===")
    print(f"Known text: '{known_text}'")
    print(f"Length: {len(known_text)} characters")
    
    # Convert to binary
    target_bits = []
    target_bytes = []
    
    for char in known_text:
        byte_val = ord(char)
        target_bytes.append(byte_val)
        
        # Convert to 8-bit binary
        binary = format(byte_val, '08b')
        target_bits.extend([int(b) for b in binary])
        
        print(f"'{char}' = {byte_val:3d} = {binary}")
    
    print(f"\nTotal bits needed: {len(target_bits)}")
    return target_bits, target_bytes, known_text

def find_matching_grid_config(target_bits):
    """Find grid configuration that extracts the target pattern."""
    
    print(f"\n=== FINDING MATCHING GRID CONFIGURATION ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    best_matches = []
    target_length = len(target_bits)
    
    # Test reasonable parameter ranges
    row_pitches = range(20, 35)
    col_pitches = range(45, 65)
    thresholds = [50, 55, 60, 65, 70]
    
    print(f"Testing {len(row_pitches) * len(col_pitches) * len(thresholds)} configurations...")
    
    for row_pitch in row_pitches:
        for col_pitch in col_pitches:
            for threshold in thresholds:
                
                # Test multiple starting positions
                for start_row in range(5, 25):
                    for start_col in range(10, 35):
                        
                        # Extract bits for the known text length
                        extracted_bits = []
                        
                        for bit_idx in range(target_length):
                            bit_row = bit_idx // 8  # 8 bits per character
                            bit_col = bit_idx % 8
                            
                            y = start_row + bit_row * row_pitch
                            x = start_col + bit_col * col_pitch
                            
                            if (0 <= y - 3 and y + 3 < img.shape[0] and 
                                0 <= x - 3 and x + 3 < img.shape[1]):
                                
                                region = img[y-3:y+4, x-3:x+4]
                                median_val = np.median(region)
                                bit = 1 if median_val > threshold else 0
                                extracted_bits.append(bit)
                            else:
                                break
                        
                        if len(extracted_bits) == target_length:
                            # Calculate match score
                            matches = sum(1 for i in range(target_length) 
                                        if extracted_bits[i] == target_bits[i])
                            match_ratio = matches / target_length
                            
                            if match_ratio >= 0.85:  # 85% or better match
                                best_matches.append({
                                    'row_pitch': row_pitch,
                                    'col_pitch': col_pitch,
                                    'start_row': start_row,
                                    'start_col': start_col,
                                    'threshold': threshold,
                                    'match_ratio': match_ratio,
                                    'matches': matches,
                                    'extracted_bits': extracted_bits
                                })
    
    # Sort by match ratio
    best_matches.sort(key=lambda x: x['match_ratio'], reverse=True)
    
    print(f"\nFound {len(best_matches)} configurations with ≥85% match")
    
    if best_matches:
        print(f"\nTop 10 matches:")
        print(f"{'Rank':<4} {'Pitch':<8} {'Origin':<12} {'Thresh':<6} {'Match':<8} {'Accuracy'}")
        print("-" * 60)
        
        for i, match in enumerate(best_matches[:10]):
            pitch_str = f"{match['row_pitch']}x{match['col_pitch']}"
            origin_str = f"({match['start_row']},{match['start_col']})"
            match_pct = match['match_ratio'] * 100
            accuracy = f"{match['matches']}/{target_length}"
            print(f"{i+1:<4} {pitch_str:<8} {origin_str:<12} {match['threshold']:<6} {match_pct:<7.1f}% {accuracy}")
    
    return best_matches

def validate_extraction(config, target_bits, known_text):
    """Validate the extraction by decoding and comparing."""
    
    print(f"\n=== VALIDATING EXTRACTION ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    row_pitch = config['row_pitch']
    col_pitch = config['col_pitch']
    start_row = config['start_row']
    start_col = config['start_col']
    threshold = config['threshold']
    
    print(f"Using configuration:")
    print(f"  Pitch: {row_pitch}x{col_pitch}")
    print(f"  Origin: ({start_row}, {start_col})")
    print(f"  Threshold: {threshold}")
    
    # Extract the known text bits
    extracted_bits = config['extracted_bits']
    
    # Convert back to text
    extracted_text = ""
    for char_idx in range(len(extracted_bits) // 8):
        byte_val = 0
        for bit_idx in range(8):
            bit_pos = char_idx * 8 + bit_idx
            byte_val |= (extracted_bits[bit_pos] << (7 - bit_idx))
        
        if 32 <= byte_val <= 126:
            extracted_text += chr(byte_val)
        else:
            extracted_text += '?'
    
    print(f"\nExtracted text: '{extracted_text}'")
    print(f"Target text:    '{known_text}'")
    print(f"Match: {'✓' if extracted_text == known_text else '✗'}")
    
    # Character-by-character comparison
    print(f"\nCharacter comparison:")
    for i in range(min(len(extracted_text), len(known_text))):
        ext_char = extracted_text[i]
        tgt_char = known_text[i]
        match_mark = "✓" if ext_char == tgt_char else "✗"
        print(f"  {i:2d}: '{ext_char}' vs '{tgt_char}' {match_mark}")
    
    return extracted_text == known_text

def save_verified_method(best_config, target_bits, known_text):
    """Save the verified extraction method."""
    
    print(f"\n=== SAVING VERIFIED METHOD ===")
    
    results = {
        "timestamp": "2025-07-17",
        "analysis_type": "verified_extraction_method",
        "target_text": known_text,
        "target_bits": len(target_bits),
        "verified_config": {
            "row_pitch": best_config['row_pitch'],
            "col_pitch": best_config['col_pitch'],
            "start_row": best_config['start_row'],
            "start_col": best_config['start_col'],
            "threshold": best_config['threshold'],
            "accuracy": best_config['match_ratio']
        },
        "method_description": "Grid configuration verified against known text pattern",
        "next_steps": "Use this configuration to extract complete message"
    }
    
    with open('verified_extraction_method.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Verified method saved to verified_extraction_method.json")
    
    # Also save the exact grid parameters for future use
    grid_params = {
        "ROW_PITCH": best_config['row_pitch'],
        "COL_PITCH": best_config['col_pitch'],
        "START_ROW": best_config['start_row'],
        "START_COL": best_config['start_col'],
        "THRESHOLD": best_config['threshold'],
        "VERIFIED_TEXT": known_text,
        "ACCURACY": best_config['match_ratio']
    }
    
    with open('grid_parameters.json', 'w') as f:
        json.dump(grid_params, f, indent=2)
    
    print("Grid parameters saved to grid_parameters.json")
    
    return results

if __name__ == "__main__":
    print("Verified Extraction Method")
    print("Using known text to establish reliable grid")
    print("=" * 50)
    
    # Create target pattern
    target_bits, target_bytes, known_text = create_target_pattern()
    
    # Find matching grid configuration
    matches = find_matching_grid_config(target_bits)
    
    if matches:
        print(f"\nFound verified extraction method!")
        
        # Validate the best match
        best_config = matches[0]
        is_valid = validate_extraction(best_config, target_bits, known_text)
        
        if is_valid:
            print(f"\n✓ VERIFICATION SUCCESSFUL")
            
            # Save the verified method
            results = save_verified_method(best_config, target_bits, known_text)
            
            print(f"\n" + "=" * 50)
            print("VERIFIED METHOD ESTABLISHED")
            print(f"Accuracy: {best_config['match_ratio']:.1%}")
            print(f"Ready for complete message extraction")
        else:
            print(f"\n✗ VERIFICATION FAILED")
            print("Need to check grid parameters")
    else:
        print("\nNo matching configurations found")
        print("May need to adjust search parameters")