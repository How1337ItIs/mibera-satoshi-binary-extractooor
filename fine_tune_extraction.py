#!/usr/bin/env python3
"""
Fine-tune the extraction around the best match area to improve accuracy.
"""

import cv2
import numpy as np

def fine_tune_extraction():
    """Fine-tune extraction around the best match coordinates."""
    
    # Load the poster image
    img = cv2.imread('satoshi.png')
    if img is None:
        img = cv2.imread('satoshi (1).png')
    if img is None:
        print("ERROR: Could not load poster image")
        return
    
    print(f"Loaded poster image: {img.shape}")
    
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # The manually verified pattern
    target_pattern = "010011110110111000100000011101000110100001100101"
    
    # Fine-tune around the best match area: y=70, x_start=35, spacing=18, threshold=80
    y_range = range(65, 76)  # y: 65-75
    x_range = range(30, 41)  # x_start: 30-40  
    spacing_range = [17, 17.5, 18, 18.5, 19]  # spacing: 17-19
    threshold_range = range(75, 86)  # threshold: 75-85
    
    best_match = None
    best_score = 0
    
    print("Fine-tuning parameters around best match area...")
    
    for y in y_range:
        for x_start in x_range:
            for spacing in spacing_range:
                for threshold in threshold_range:
                    
                    # Extract bits using these parameters
                    bits = []
                    for bit_pos in range(len(target_pattern)):
                        x = x_start + (bit_pos * spacing)
                        if x >= img.shape[1] or y >= img.shape[0]:
                            break
                            
                        # Handle fractional spacing
                        x_int = int(round(x))
                        if x_int >= img.shape[1]:
                            break
                            
                        v_value = hsv[y, x_int, 2]
                        bit = '1' if v_value > threshold else '0'
                        bits.append(bit)
                    
                    if len(bits) < len(target_pattern):
                        continue
                    
                    extracted_pattern = ''.join(bits[:len(target_pattern)])
                    
                    # Calculate match score
                    matches = sum(1 for i in range(len(target_pattern)) 
                                 if extracted_pattern[i] == target_pattern[i])
                    score = matches / len(target_pattern)
                    
                    if score > best_score:
                        best_score = score
                        best_match = {
                            'y': y,
                            'x_start': x_start,
                            'spacing': spacing,
                            'threshold': threshold,
                            'score': score,
                            'extracted': extracted_pattern
                        }
                        
                        print(f"New best: {score:.1%} at y={y}, x={x_start}, spacing={spacing}, threshold={threshold}")
    
    print(f"\n=== FINE-TUNED BEST MATCH ===")
    if best_match:
        print(f"Score: {best_match['score']:.1%}")
        print(f"Position: y={best_match['y']}, x_start={best_match['x_start']}")
        print(f"Parameters: spacing={best_match['spacing']}, threshold={best_match['threshold']}")
        
        # Compare patterns bit by bit
        target = target_pattern
        extracted = best_match['extracted']
        
        print(f"\nBit-by-bit comparison:")
        print(f"Target:    {target}")
        print(f"Extracted: {extracted}")
        print(f"Match:     {''.join('✓' if t==e else '✗' for t,e in zip(target, extracted))}")
        
        # Decode both
        def decode_pattern(pattern):
            chars = []
            for i in range(0, len(pattern), 8):
                if i + 8 <= len(pattern):
                    byte = pattern[i:i+8]
                    try:
                        char_val = int(byte, 2)
                        if 32 <= char_val <= 126:
                            chars.append(chr(char_val))
                        else:
                            chars.append(f'[{char_val}]')
                    except:
                        chars.append('?')
            return ''.join(chars)
        
        target_text = decode_pattern(target)
        extracted_text = decode_pattern(extracted)
        
        print(f"\nTarget text: {target_text}")
        print(f"Extracted text: {extracted_text}")
        
        # Save results
        with open('fine_tuned_results.txt', 'w') as f:
            f.write("=== FINE-TUNED EXTRACTION RESULTS ===\n")
            f.write(f"Best match score: {best_match['score']:.1%}\n")
            f.write(f"Position: y={best_match['y']}, x_start={best_match['x_start']}\n")
            f.write(f"Parameters: spacing={best_match['spacing']}, threshold={best_match['threshold']}\n")
            f.write(f"\nTarget pattern:    {target}\n")
            f.write(f"Extracted pattern: {extracted}\n")
            f.write(f"Match pattern:     {''.join('✓' if t==e else '✗' for t,e in zip(target, extracted))}\n")
            f.write(f"\nTarget text: {target_text}\n")
            f.write(f"Extracted text: {extracted_text}\n")
        
        # Now extract the complete line with these optimized parameters
        extract_complete_line(best_match, hsv)
        
        return best_match
    else:
        print("No improvement found")
        return None

def extract_complete_line(best_match, hsv):
    """Extract the complete line using optimized parameters."""
    
    y = best_match['y']
    x_start = best_match['x_start']
    spacing = best_match['spacing']
    threshold = best_match['threshold']
    
    print(f"\nExtracting complete line with optimized parameters...")
    
    # Extract many more bits to get the full message
    bits = []
    for bit_pos in range(400):  # Extract up to 400 bits (50 characters)
        x = x_start + (bit_pos * spacing)
        x_int = int(round(x))
        
        if x_int >= hsv.shape[1]:
            break
            
        v_value = hsv[y, x_int, 2]
        bit = '1' if v_value > threshold else '0'
        bits.append(bit)
    
    bit_string = ''.join(bits)
    
    # Decode as ASCII
    chars = []
    for i in range(0, len(bit_string), 8):
        if i + 8 <= len(bit_string):
            byte = bit_string[i:i+8]
            try:
                char_val = int(byte, 2)
                if 32 <= char_val <= 126:
                    chars.append(chr(char_val))
                else:
                    chars.append(f'[{char_val}]')
            except:
                chars.append('?')
    
    text = ''.join(chars)
    
    print(f"Complete line: {text}")
    
    # Save complete line results
    with open('complete_line_results.txt', 'w') as f:
        f.write("=== COMPLETE LINE EXTRACTION ===\n")
        f.write(f"Parameters: y={y}, x_start={x_start}, spacing={spacing}, threshold={threshold}\n")
        f.write(f"Score on known pattern: {best_match['score']:.1%}\n")
        f.write(f"Raw bits: {bit_string}\n")
        f.write(f"Decoded text: {text}\n")
        f.write(f"Text length: {len([c for c in chars if c.isalnum() or c.isspace()])} readable characters\n")

if __name__ == "__main__":
    fine_tune_extraction()