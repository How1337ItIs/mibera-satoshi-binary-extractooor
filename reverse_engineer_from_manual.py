#!/usr/bin/env python3
"""
Reverse engineer grid parameters from manual extraction.
Work backwards from known text to find exact grid alignment.
"""

import cv2
import numpy as np
import json
from itertools import product

def reverse_engineer_grid_from_manual():
    """Use manual extraction to find the actual grid parameters."""
    
    print("=== REVERSE ENGINEERING FROM MANUAL EXTRACTION ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Known manual extraction
    known_text = "On the winter solstice December 21 "
    print(f"Target text: '{known_text}'")
    
    # Convert to target bit pattern
    target_bits = []
    for char in known_text:
        byte_val = ord(char)
        binary = format(byte_val, '08b')
        target_bits.extend([int(b) for b in binary])
    
    print(f"Target bits: {len(target_bits)}")
    
    # Systematic search with finer granularity
    # Based on your manual success, the grid is likely in a specific region
    
    best_matches = []
    
    # Test ranges - focus on area where text is likely to be
    row_starts = range(80, 130, 2)  # Likely vertical position
    col_starts = range(30, 80, 2)   # Likely horizontal position
    row_pitches = range(20, 40, 1)  # Row spacing
    col_pitches = range(40, 65, 1)  # Column spacing
    thresholds = range(40, 90, 5)   # Threshold values
    
    total_configs = len(row_starts) * len(col_starts) * len(row_pitches) * len(col_pitches) * len(thresholds)
    print(f"Testing {total_configs:,} configurations...")
    
    config_count = 0
    
    for row_start in row_starts:
        for col_start in col_starts:
            for row_pitch in row_pitches:
                for col_pitch in col_pitches:
                    for threshold in thresholds:
                        config_count += 1
                        
                        if config_count % 10000 == 0:
                            print(f"  Progress: {config_count:,}/{total_configs:,} ({config_count/total_configs*100:.1f}%)")
                        
                        # Extract bits for this configuration
                        extracted_bits = []
                        
                        for bit_idx in range(len(target_bits)):
                            bit_row = bit_idx // 8
                            bit_col = bit_idx % 8
                            
                            y = row_start + bit_row * row_pitch
                            x = col_start + bit_col * col_pitch
                            
                            if (0 <= y - 3 and y + 3 < img.shape[0] and 
                                0 <= x - 3 and x + 3 < img.shape[1]):
                                
                                region = img[y-3:y+4, x-3:x+4]
                                median_val = np.median(region)
                                bit = 1 if median_val > threshold else 0
                                extracted_bits.append(bit)
                        
                        if len(extracted_bits) == len(target_bits):
                            # Calculate bit-level match
                            bit_matches = sum(1 for i in range(len(target_bits)) 
                                            if extracted_bits[i] == target_bits[i])
                            bit_accuracy = bit_matches / len(target_bits)
                            
                            # Only consider configurations with >50% bit accuracy
                            if bit_accuracy > 0.5:
                                # Convert to text to verify
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
                                
                                # Calculate character match
                                char_matches = sum(1 for i in range(min(len(extracted_text), len(known_text)))
                                                 if extracted_text[i] == known_text[i])
                                char_accuracy = char_matches / len(known_text)
                                
                                best_matches.append({
                                    'config': (row_start, col_start, row_pitch, col_pitch, threshold),
                                    'bit_accuracy': bit_accuracy,
                                    'char_accuracy': char_accuracy,
                                    'extracted_text': extracted_text,
                                    'total_score': bit_accuracy * 0.6 + char_accuracy * 0.4
                                })
    
    # Sort by total score
    best_matches.sort(key=lambda x: x['total_score'], reverse=True)
    
    print(f"\nCompleted search: {config_count:,} configurations tested")
    print(f"Found {len(best_matches)} promising matches (>50% bit accuracy)")
    
    if best_matches:
        print(f"\nTop 10 matches:")
        print(f"{'Rank':<4} {'Config':<25} {'Bit%':<6} {'Char%':<7} {'Score':<6} {'Text'}")
        print("-" * 80)
        
        for i, match in enumerate(best_matches[:10]):
            config = match['config']
            config_str = f"{config[0]:2d},{config[1]:2d},{config[2]:2d},{config[3]:2d},{config[4]:2d}"
            bit_pct = match['bit_accuracy'] * 100
            char_pct = match['char_accuracy'] * 100
            score = match['total_score']
            text = match['extracted_text'][:20] + ('...' if len(match['extracted_text']) > 20 else '')
            
            print(f"{i+1:<4} {config_str:<25} {bit_pct:<5.1f}% {char_pct:<6.1f}% {score:<5.2f} '{text}'")
    
    return best_matches

def validate_best_match(best_match):
    """Validate the best match by extracting longer text."""
    
    print(f"\n=== VALIDATING BEST MATCH ===")
    
    config = best_match['config']
    row_start, col_start, row_pitch, col_pitch, threshold = config
    
    print(f"Best configuration:")
    print(f"  Position: ({row_start}, {col_start})")
    print(f"  Pitch: {row_pitch}x{col_pitch}")
    print(f"  Threshold: {threshold}")
    print(f"  Accuracy: {best_match['bit_accuracy']:.1%} bits, {best_match['char_accuracy']:.1%} chars")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Extract longer sequence
    max_chars = 80
    all_bits = []
    
    for char_idx in range(max_chars):
        for bit_idx in range(8):
            total_bit_idx = char_idx * 8 + bit_idx
            bit_row = total_bit_idx // 8
            bit_col = total_bit_idx % 8
            
            y = row_start + bit_row * row_pitch
            x = col_start + bit_col * col_pitch
            
            if (0 <= y - 3 and y + 3 < img.shape[0] and 
                0 <= x - 3 and x + 3 < img.shape[1]):
                
                region = img[y-3:y+4, x-3:x+4]
                median_val = np.median(region)
                bit = 1 if median_val > threshold else 0
                all_bits.append(bit)
            else:
                break
    
    # Convert to text
    complete_text = ""
    for char_idx in range(len(all_bits) // 8):
        byte_val = 0
        for bit_idx in range(8):
            bit_pos = char_idx * 8 + bit_idx
            if bit_pos < len(all_bits):
                byte_val |= (all_bits[bit_pos] << (7 - bit_idx))
        
        if 32 <= byte_val <= 126:
            complete_text += chr(byte_val)
        else:
            complete_text += '?'
    
    print(f"\nExtended extraction ({len(complete_text)} chars):")
    print(f"'{complete_text}'")
    
    # Quality metrics
    printable_ratio = sum(1 for c in complete_text if c != '?') / len(complete_text) if complete_text else 0
    ones_ratio = sum(all_bits) / len(all_bits) if all_bits else 0
    
    print(f"\nQuality metrics:")
    print(f"Printable ratio: {printable_ratio:.1%}")
    print(f"Ones ratio: {ones_ratio:.1%}")
    
    # Look for words
    words = complete_text.replace('?', ' ').split()
    valid_words = [w for w in words if len(w) >= 2 and w.isalpha()]
    
    print(f"Valid words found: {valid_words[:10]}")
    
    return {
        'config': config,
        'complete_text': complete_text,
        'printable_ratio': printable_ratio,
        'ones_ratio': ones_ratio,
        'valid_words': valid_words
    }

def save_reverse_engineering_results():
    """Save reverse engineering results."""
    
    print(f"\n=== SAVING REVERSE ENGINEERING RESULTS ===")
    
    # Run reverse engineering search
    matches = reverse_engineer_grid_from_manual()
    
    if matches:
        # Validate best match
        validation = validate_best_match(matches[0])
        
        # Compile results
        results = {
            "timestamp": "2025-07-17",
            "analysis_type": "reverse_engineer_from_manual",
            "approach": "Systematic search to find grid parameters matching manual extraction",
            "target_text": "On the winter solstice December 21",
            "search_results": {
                "total_matches": len(matches),
                "best_bit_accuracy": matches[0]['bit_accuracy'] if matches else 0,
                "best_char_accuracy": matches[0]['char_accuracy'] if matches else 0,
                "best_config": matches[0]['config'] if matches else None
            },
            "validation": validation if matches else {},
            "top_matches": [
                {
                    "config": m['config'],
                    "bit_accuracy": m['bit_accuracy'],
                    "char_accuracy": m['char_accuracy'],
                    "extracted_text": m['extracted_text']
                }
                for m in matches[:5]
            ],
            "assessment": "Reverse engineering approach to find exact grid parameters from manual extraction"
        }
        
        with open('reverse_engineering_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("Results saved to reverse_engineering_results.json")
        
    else:
        print("No matches found - may need to expand search ranges")
    
    return matches

if __name__ == "__main__":
    print("Reverse Engineer Grid from Manual Extraction")
    print("Systematic search to find exact parameters")
    print("=" * 55)
    
    # Run reverse engineering
    matches = reverse_engineer_grid_from_manual()
    
    if matches:
        print(f"\nFound {len(matches)} promising matches!")
        
        # Validate best match
        validation = validate_best_match(matches[0])
        
        # Save results
        save_reverse_engineering_results()
        
        print(f"\n" + "=" * 55)
        print("REVERSE ENGINEERING COMPLETE")
        print(f"Best match: {matches[0]['bit_accuracy']:.1%} bit accuracy")
        print(f"Text quality: {validation['printable_ratio']:.1%} printable")
        
        if validation['valid_words']:
            print(f"Found {len(validation['valid_words'])} valid words")
    else:
        print("\nNo matches found")
        print("Grid parameters may be outside tested ranges")
        print("Consider expanding search space or different approach")