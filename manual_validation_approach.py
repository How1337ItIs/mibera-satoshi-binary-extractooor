#!/usr/bin/env python3
"""
Manual validation approach - use the known text to validate and improve extraction.
Focus on the verified readable portion to establish reliable method.
"""

import cv2
import numpy as np
import json

def test_known_text_extraction():
    """Test extraction against known readable text."""
    
    print("=== MANUAL VALIDATION APPROACH ===")
    print("Using known text: 'On the winter solstice December 21'")
    
    # Load the correct image
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("ERROR: Could not load image")
        return None
    
    print(f"Image loaded: {img.shape}")
    
    # Known text for validation
    known_text = "On the winter solstice December 21 "
    
    # Test systematic grid configurations
    test_configs = [
        # (row_pitch, col_pitch, start_row, start_col, threshold)
        (31, 53, 101, 53, 72),  # Original verification
        (30, 52, 100, 50, 60),  # Variation
        (25, 50, 13, 18, 56),   # From "OF" result
        (28, 55, 95, 45, 65),   # Alternative
        (32, 54, 105, 55, 70),  # Another test
    ]
    
    results = []
    
    for i, (row_pitch, col_pitch, start_row, start_col, threshold) in enumerate(test_configs):
        print(f"\nTesting config {i+1}: {row_pitch}x{col_pitch} at ({start_row},{start_col}) t={threshold}")
        
        # Extract the length of known text
        extracted_bits = []
        
        for char_idx in range(len(known_text)):
            for bit_idx in range(8):
                # Calculate grid position
                total_bit_pos = char_idx * 8 + bit_idx
                grid_row = total_bit_pos // 8
                grid_col = total_bit_pos % 8
                
                # Calculate pixel position
                y = start_row + grid_row * row_pitch
                x = start_col + grid_col * col_pitch
                
                # Extract bit using 6x6 median sampling
                if (0 <= y - 3 and y + 3 < img.shape[0] and 
                    0 <= x - 3 and x + 3 < img.shape[1]):
                    
                    region = img[y-3:y+4, x-3:x+4]
                    median_val = np.median(region)
                    bit = 1 if median_val > threshold else 0
                    extracted_bits.append(bit)
        
        # Convert bits to text
        extracted_text = ""
        for char_idx in range(len(extracted_bits) // 8):
            byte_val = 0
            for bit_idx in range(8):
                bit_pos = char_idx * 8 + bit_idx
                if bit_pos < len(extracted_bits):
                    byte_val |= (extracted_bits[bit_pos] << (7 - bit_idx))
            
            if 32 <= byte_val <= 126:
                extracted_text += chr(byte_val)
            else:
                extracted_text += '?'
        
        # Calculate match score
        char_matches = 0
        for j in range(min(len(extracted_text), len(known_text))):
            if extracted_text[j] == known_text[j]:
                char_matches += 1
        
        match_ratio = char_matches / len(known_text) if known_text else 0
        printable_ratio = sum(1 for c in extracted_text if c != '?') / len(extracted_text) if extracted_text else 0
        
        print(f"  Extracted: '{extracted_text}'")
        print(f"  Match: {char_matches}/{len(known_text)} ({match_ratio:.1%})")
        print(f"  Printable: {printable_ratio:.1%}")
        
        results.append({
            'config_index': i,
            'config': (row_pitch, col_pitch, start_row, start_col, threshold),
            'extracted_text': extracted_text,
            'match_ratio': match_ratio,
            'char_matches': char_matches,
            'printable_ratio': printable_ratio,
            'extracted_bits': extracted_bits
        })
    
    # Sort by match ratio
    results.sort(key=lambda x: x['match_ratio'], reverse=True)
    
    print(f"\n=== RESULTS SUMMARY ===")
    for i, result in enumerate(results):
        config = result['config']
        match = result['match_ratio']
        printable = result['printable_ratio']
        text_preview = result['extracted_text'][:15] + ('...' if len(result['extracted_text']) > 15 else '')
        print(f"{i+1}. Config {result['config_index']+1}: {match:.1%} match, {printable:.1%} printable, '{text_preview}'")
    
    return results

def extract_complete_message_with_best_config(best_config):
    """Extract complete message using the best validated configuration."""
    
    print(f"\n=== COMPLETE MESSAGE EXTRACTION ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    row_pitch, col_pitch, start_row, start_col, threshold = best_config['config']
    
    print(f"Using best config: {row_pitch}x{col_pitch} at ({start_row},{start_col}) thresh={threshold}")
    print(f"Validation accuracy: {best_config['match_ratio']:.1%}")
    
    # Extract longer sequence
    max_chars = 150
    all_bits = []
    
    for char_idx in range(max_chars):
        for bit_idx in range(8):
            total_bit_pos = char_idx * 8 + bit_idx
            grid_row = total_bit_pos // 8
            grid_col = total_bit_pos % 8
            
            y = start_row + grid_row * row_pitch
            x = start_col + grid_col * col_pitch
            
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
    
    print(f"\nComplete message ({len(complete_text)} chars):")
    print(f"'{complete_text}'")
    
    # Quality analysis
    printable_ratio = sum(1 for c in complete_text if c != '?') / len(complete_text) if complete_text else 0
    ones_ratio = sum(all_bits) / len(all_bits) if all_bits else 0
    
    print(f"\nQuality metrics:")
    print(f"Printable ratio: {printable_ratio:.1%}")
    print(f"Ones ratio: {ones_ratio:.1%}")
    print(f"Total bits: {len(all_bits)}")
    
    # Look for meaningful content
    words = complete_text.replace('?', ' ').split()
    meaningful_words = [w for w in words if len(w) >= 3 and w.isalpha()]
    
    print(f"Meaningful words: {meaningful_words[:10]}")
    
    return {
        'complete_text': complete_text,
        'bits': all_bits,
        'printable_ratio': printable_ratio,
        'ones_ratio': ones_ratio,
        'meaningful_words': meaningful_words
    }

def save_manual_validation_results(results, complete_extraction=None):
    """Save manual validation results."""
    
    print(f"\n=== SAVING MANUAL VALIDATION RESULTS ===")
    
    best_config = results[0] if results else None
    
    output = {
        "timestamp": "2025-07-17",
        "analysis_type": "manual_validation_approach",
        "target_text": "On the winter solstice December 21",
        "configurations_tested": len(results),
        "best_match": {
            "config": best_config['config'] if best_config else None,
            "match_ratio": best_config['match_ratio'] if best_config else 0,
            "extracted_text": best_config['extracted_text'] if best_config else ""
        },
        "complete_extraction": complete_extraction,
        "assessment": "Validation against known text to establish reliable extraction method"
    }
    
    with open('manual_validation_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("Manual validation results saved")
    
    return output

if __name__ == "__main__":
    print("Manual Validation Approach")
    print("Testing extraction against known readable text")
    print("=" * 50)
    
    # Test known text extraction
    results = test_known_text_extraction()
    
    if results and results[0]['match_ratio'] > 0.1:
        print(f"\nFound working configuration!")
        
        # Extract complete message with best config
        complete_extraction = extract_complete_message_with_best_config(results[0])
        
        # Save results
        save_manual_validation_results(results, complete_extraction)
        
        print(f"\n" + "=" * 50)
        print("MANUAL VALIDATION COMPLETE")
        print(f"Best match ratio: {results[0]['match_ratio']:.1%}")
        print(f"Complete message quality: {complete_extraction['printable_ratio']:.1%}")
        
        if complete_extraction['meaningful_words']:
            print(f"Meaningful words found: {len(complete_extraction['meaningful_words'])}")
    else:
        print("\nNo working configuration found with current parameters")
        print("May need to expand search ranges or adjust sampling method")
        
        # Save results anyway
        save_manual_validation_results(results if results else [])