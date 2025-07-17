#!/usr/bin/env python3
"""
Manual calibration using the extracted bit sequences to decode the message
and reverse-engineer the correct grid parameters.
"""

import cv2
import numpy as np
import json

def decode_manual_bits():
    """Decode the manually extracted bit sequences."""
    
    print("=== DECODING MANUAL BIT SEQUENCES ===")
    
    # Manual bit sequences from user
    bit_sequences = [
        "01100101 01100011 01100101 01101101 01100010 01100101 01110010 00100000 00110010 00110001 00100000 00110010",
        "0100111 01100110 00100000 01110100 01101000 01100101 0010000 01110111 01101001 01101110 10100 01100101",
        "01110010 0010000 01110011 01101111 01101100 01110011 01110100 01101001 01100011 01100101 00100000 01000100",
        "01100101 01100011 01100101 01101101 01100010 01100101 01110010 00100000 00110010 00110001 00100000 00110010",
        "00110000 00110010 00110010 00100000 01110111 01101000 0101001 01101100 01110011 01110100 00100000 01100100",
        "01100101 01100101 01110000 00100000 01101001 01101110 00100000 01110100 0110100001100101 00100000 01100110"
    ]
    
    decoded_lines = []
    
    for i, bit_line in enumerate(bit_sequences):
        print(f"\nLine {i+1}: {bit_line}")
        
        # Clean and split bits
        bits = bit_line.replace(" ", "")
        
        # Convert to bytes and decode
        text = ""
        for j in range(0, len(bits), 8):
            if j + 8 <= len(bits):
                byte_str = bits[j:j+8]
                if len(byte_str) == 8:
                    byte_val = int(byte_str, 2)
                    if 32 <= byte_val <= 126:
                        text += chr(byte_val)
                    else:
                        text += f"[{byte_val}]"
        
        print(f"Decoded: '{text}'")
        decoded_lines.append(text)
    
    # Combine all lines
    full_message = " ".join(decoded_lines)
    print(f"\n=== FULL MESSAGE ===")
    print(f"'{full_message}'")
    
    return decoded_lines, full_message

def reverse_engineer_grid():
    """Use manual extraction to reverse-engineer correct grid parameters."""
    
    print("\n=== REVERSE ENGINEERING GRID ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # We know the first line should decode to something like "December 21 2"
    target_first_line = "01100101 01100011 01100101 01101101 01100010 01100101 01110010 00100000 00110010 00110001 00100000 00110010"
    target_bits = target_first_line.replace(" ", "")
    
    print(f"Target first line: {len(target_bits)} bits")
    print(f"Target pattern: {target_bits[:32]}...")
    
    # Test parameter ranges to match this pattern
    best_matches = []
    
    # Reasonable parameter ranges based on previous analysis
    row_pitches = range(20, 35, 1)
    col_pitches = range(45, 65, 1)
    
    print(f"Testing {len(row_pitches) * len(col_pitches)} pitch combinations...")
    
    for row_pitch in row_pitches:
        for col_pitch in col_pitches:
            
            # Test multiple starting positions
            for start_row in range(5, 20):
                for start_col in range(10, 30):
                    
                    # Extract first 96 bits (12 characters - enough for first line)
                    extracted_bits = ""
                    
                    for bit_idx in range(96):
                        bit_row = bit_idx // 8  # 8 bits per character
                        bit_col = bit_idx % 8
                        
                        y = start_row + bit_row * row_pitch
                        x = start_col + bit_col * col_pitch
                        
                        if (0 <= y - 3 and y + 3 < img.shape[0] and 
                            0 <= x - 3 and x + 3 < img.shape[1]):
                            
                            region = img[y-3:y+4, x-3:x+4]
                            median_val = np.median(region)
                            
                            # Test multiple thresholds
                            for threshold in [50, 60, 70]:
                                bit = "1" if median_val > threshold else "0"
                                extracted_bits += bit
                                break  # Use first threshold
                    
                    if len(extracted_bits) == 96:
                        # Compare with target
                        matches = sum(1 for i in range(min(len(extracted_bits), len(target_bits)))
                                    if extracted_bits[i] == target_bits[i])
                        
                        match_ratio = matches / len(target_bits)
                        
                        if match_ratio > 0.7:  # 70% or better match
                            best_matches.append({
                                'row_pitch': row_pitch,
                                'col_pitch': col_pitch,
                                'start_row': start_row,
                                'start_col': start_col,
                                'match_ratio': match_ratio,
                                'matches': matches,
                                'extracted_bits': extracted_bits[:len(target_bits)],
                                'threshold': threshold
                            })
    
    # Sort by match ratio
    best_matches.sort(key=lambda x: x['match_ratio'], reverse=True)
    
    print(f"\nFound {len(best_matches)} configurations with >70% match")
    print(f"{'Rank':<4} {'Pitch':<8} {'Origin':<12} {'Match':<6} {'Threshold'}")
    print("-" * 50)
    
    for i, match in enumerate(best_matches[:10]):
        pitch_str = f"{match['row_pitch']}x{match['col_pitch']}"
        origin_str = f"({match['start_row']},{match['start_col']})"
        match_pct = match['match_ratio'] * 100
        print(f"{i+1:<4} {pitch_str:<8} {origin_str:<12} {match_pct:<5.1f}% {match['threshold']}")
    
    return best_matches

def validate_calibration(best_match):
    """Validate the calibrated grid by extracting and decoding text."""
    
    print(f"\n=== VALIDATING CALIBRATION ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    row_pitch = best_match['row_pitch']
    col_pitch = best_match['col_pitch']
    start_row = best_match['start_row']
    start_col = best_match['start_col']
    threshold = best_match['threshold']
    
    print(f"Using calibrated parameters:")
    print(f"  Pitch: {row_pitch}x{col_pitch}")
    print(f"  Origin: ({start_row}, {start_col})")
    print(f"  Threshold: {threshold}")
    
    # Extract larger text sequence
    max_chars = 100
    all_bits = []
    
    for char_idx in range(max_chars):
        for bit_idx in range(8):
            total_bit_idx = char_idx * 8 + bit_idx
            bit_row = total_bit_idx // 8
            bit_col = total_bit_idx % 8
            
            y = start_row + bit_row * row_pitch
            x = start_col + bit_col * col_pitch
            
            if (0 <= y - 3 and y + 3 < img.shape[0] and 
                0 <= x - 3 and x + 3 < img.shape[1]):
                
                region = img[y-3:y+4, x-3:x+4]
                median_val = np.median(region)
                bit = 1 if median_val > threshold else 0
                all_bits.append(bit)
            else:
                break
    
    # Convert to text
    extracted_text = ""
    for char_idx in range(len(all_bits) // 8):
        byte_val = 0
        for bit_idx in range(8):
            bit_pos = char_idx * 8 + bit_idx
            if bit_pos < len(all_bits):
                byte_val |= (all_bits[bit_pos] << (7 - bit_idx))
        
        if 32 <= byte_val <= 126:
            extracted_text += chr(byte_val)
        else:
            extracted_text += '?'
    
    print(f"\nExtracted text ({len(extracted_text)} chars):")
    print(f"'{extracted_text}'")
    
    # Quality metrics
    printable_ratio = sum(1 for c in extracted_text if c != '?') / len(extracted_text) if extracted_text else 0
    ones_ratio = sum(all_bits) / len(all_bits) if all_bits else 0
    
    print(f"Printable ratio: {printable_ratio:.1%}")
    print(f"Ones ratio: {ones_ratio:.1%}")
    
    # Look for the manual patterns
    words = extracted_text.replace('?', ' ').split()
    print(f"Words found: {words[:10]}")
    
    return {
        'extracted_text': extracted_text,
        'bits': all_bits,
        'printable_ratio': printable_ratio,
        'ones_ratio': ones_ratio,
        'config': best_match
    }

def save_manual_calibration_results():
    """Save manual calibration results."""
    
    print(f"\n=== SAVING MANUAL CALIBRATION RESULTS ===")
    
    # Decode manual bits
    decoded_lines, full_message = decode_manual_bits()
    
    # Reverse engineer grid
    matches = reverse_engineer_grid()
    
    if matches:
        # Validate with best match
        validation = validate_calibration(matches[0])
    else:
        validation = {'extracted_text': '', 'printable_ratio': 0}
    
    # Compile results
    results = {
        "timestamp": "2025-07-17",
        "analysis_type": "manual_calibration",
        "manual_extraction": {
            "decoded_lines": decoded_lines,
            "full_message": full_message,
            "total_chars": len(full_message)
        },
        "reverse_engineering": {
            "matches_found": len(matches),
            "best_match_ratio": matches[0]['match_ratio'] if matches else 0,
            "calibrated_config": matches[0] if matches else {}
        },
        "validation": {
            "extracted_text": validation['extracted_text'],
            "printable_ratio": validation['printable_ratio'],
            "ones_ratio": validation.get('ones_ratio', 0)
        },
        "assessment": "Manual bit extraction used to calibrate grid detection parameters"
    }
    
    with open('manual_calibration_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Manual calibration results saved")
    
    return results

if __name__ == "__main__":
    print("Manual Calibration Analysis")
    print("Using manually extracted bits to calibrate grid")
    print("=" * 50)
    
    # Decode manual extractions
    decoded_lines, full_message = decode_manual_bits()
    
    # Reverse engineer grid parameters
    matches = reverse_engineer_grid()
    
    if matches:
        print(f"\nFound calibrated parameters!")
        
        # Validate calibration
        validation = validate_calibration(matches[0])
        
        # Save results
        results = save_manual_calibration_results()
        
        print(f"\n" + "=" * 50)
        print("MANUAL CALIBRATION COMPLETE")
        print(f"Best match: {matches[0]['match_ratio']:.1%}")
        print(f"Extracted quality: {validation['printable_ratio']:.1%}")
    else:
        print("\nNo matching grid configurations found")
        print("Manual extraction may have errors or need different approach")