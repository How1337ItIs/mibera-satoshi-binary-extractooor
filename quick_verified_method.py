#!/usr/bin/env python3
"""
Quick verified method using focused search around the manual findings.
"""

import cv2
import numpy as np
import json

def test_known_good_text():
    """Test extraction of known good text with focused parameters."""
    
    known_text = "On the winter solstice December 21 "
    print(f"=== TESTING KNOWN TEXT ===")
    print(f"Target: '{known_text}' ({len(known_text)} chars)")
    
    # Convert to target bits
    target_bits = []
    for char in known_text:
        byte_val = ord(char)
        binary = format(byte_val, '08b')
        target_bits.extend([int(b) for b in binary])
    
    print(f"Target bits: {len(target_bits)}")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Test focused parameter ranges based on previous results
    configs_to_test = [
        (25, 50, 13, 18, 56),  # From refined analysis "OF" result
        (31, 53, 12, 11, 60),  # From verification data
        (30, 52, 15, 20, 55),  # Close variation
        (28, 55, 10, 15, 60),  # Alternative
        (23, 48, 10, 15, 55),  # From text variations
    ]
    
    results = []
    
    for row_pitch, col_pitch, start_row, start_col, threshold in configs_to_test:
        print(f"\nTesting: {row_pitch}x{col_pitch} at ({start_row},{start_col}) thresh={threshold}")
        
        # Extract bits
        extracted_bits = []
        for bit_idx in range(len(target_bits)):
            bit_row = bit_idx // 8
            bit_col = bit_idx % 8
            
            y = start_row + bit_row * row_pitch
            x = start_col + bit_col * col_pitch
            
            if (0 <= y - 3 and y + 3 < img.shape[0] and 
                0 <= x - 3 and x + 3 < img.shape[1]):
                
                region = img[y-3:y+4, x-3:x+4]
                median_val = np.median(region)
                bit = 1 if median_val > threshold else 0
                extracted_bits.append(bit)
        
        if len(extracted_bits) == len(target_bits):
            # Convert to text
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
            
            # Calculate accuracy
            char_matches = sum(1 for i in range(min(len(extracted_text), len(known_text)))
                             if extracted_text[i] == known_text[i])
            char_accuracy = char_matches / len(known_text)
            
            print(f"  Extracted: '{extracted_text}'")
            print(f"  Accuracy: {char_matches}/{len(known_text)} ({char_accuracy:.1%})")
            
            results.append({
                'config': (row_pitch, col_pitch, start_row, start_col, threshold),
                'extracted_text': extracted_text,
                'accuracy': char_accuracy,
                'char_matches': char_matches
            })
    
    # Sort by accuracy
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print(f"\n=== RESULTS SUMMARY ===")
    for i, result in enumerate(results):
        config = result['config']
        text = result['extracted_text'][:20] + ('...' if len(result['extracted_text']) > 20 else '')
        acc = result['accuracy']
        print(f"{i+1}. {config[0]}x{config[1]} at ({config[2]},{config[3]}) t={config[4]}: {acc:.1%} '{text}'")
    
    return results

def extract_complete_message_with_best():
    """Extract complete message using the best verified configuration."""
    
    print(f"\n=== EXTRACTING COMPLETE MESSAGE ===")
    
    # Test configurations first
    results = test_known_good_text()
    
    if not results or results[0]['accuracy'] < 0.8:
        print("No good configuration found")
        return None
    
    best_result = results[0]
    row_pitch, col_pitch, start_row, start_col, threshold = best_result['config']
    
    print(f"Using best config: {row_pitch}x{col_pitch} at ({start_row},{start_col}) thresh={threshold}")
    print(f"Verified accuracy: {best_result['accuracy']:.1%}")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Extract much longer sequence
    max_chars = 200
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
    
    # Analysis
    printable_ratio = sum(1 for c in complete_text if c != '?') / len(complete_text) if complete_text else 0
    words = complete_text.replace('?', ' ').split()
    valid_words = [w for w in words if len(w) >= 2 and w.isalpha()]
    
    print(f"\nQuality metrics:")
    print(f"Printable ratio: {printable_ratio:.1%}")
    print(f"Total bits: {len(all_bits)}")
    print(f"Valid words: {valid_words[:10]}")
    
    return {
        'config': best_result['config'],
        'complete_text': complete_text,
        'bits': all_bits,
        'printable_ratio': printable_ratio,
        'valid_words': valid_words
    }

if __name__ == "__main__":
    print("Quick Verified Method")
    print("Testing focused configurations")
    print("=" * 40)
    
    # Extract complete message
    result = extract_complete_message_with_best()
    
    if result:
        print(f"\n" + "=" * 40)
        print("VERIFIED EXTRACTION COMPLETE")
        
        config = result['config']
        print(f"Final config: {config[0]}x{config[1]} at ({config[2]},{config[3]}) thresh={config[4]}")
        print(f"Message quality: {result['printable_ratio']:.1%}")
        print(f"Words found: {len(result['valid_words'])}")
        
        # Save configuration
        with open('final_grid_config.json', 'w') as f:
            json.dump({
                'row_pitch': config[0],
                'col_pitch': config[1], 
                'start_row': config[2],
                'start_col': config[3],
                'threshold': config[4],
                'complete_message': result['complete_text'],
                'quality': result['printable_ratio']
            }, f, indent=2)
        
        print("Configuration saved to final_grid_config.json")
    else:
        print("Unable to establish verified method")