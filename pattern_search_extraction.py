#!/usr/bin/env python3
"""
Search for the manually verified bit pattern in the poster to find exact coordinates,
then extract the complete message from there.
"""

import cv2
import numpy as np

def search_for_pattern():
    """Search for the manually verified bit pattern in the poster."""
    
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
    
    # The manually verified pattern: "01001111 01101110 00100000 01110100 01101000 01100101"
    # This decodes to "On the"
    target_pattern = "010011110110111000100000011101000110100001100101"
    
    print(f"Searching for pattern: {target_pattern}")
    print(f"Pattern length: {len(target_pattern)} bits")
    
    # Test different parameter combinations
    spacings = [15, 16, 17, 18, 19, 20]
    thresholds = [70, 75, 80, 85, 90, 95]
    
    best_match = None
    best_score = 0
    
    # Search in the top region of the poster (where text is expected)
    for y in range(10, 100, 5):  # Search y positions 10 to 95
        for x_start in range(0, 50, 5):  # Search x starting positions 0 to 45
            for spacing in spacings:
                for threshold in thresholds:
                    
                    # Extract bits using these parameters
                    bits = []
                    for bit_pos in range(len(target_pattern)):
                        x = x_start + (bit_pos * spacing)
                        if x >= img.shape[1] or y >= img.shape[0]:
                            break
                            
                        v_value = hsv[y, x, 2]
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
                        
                        if score > 0.8:  # If we get >80% match, print it
                            print(f"Good match ({score:.1%}) at y={y}, x={x_start}, spacing={spacing}, threshold={threshold}")
                            
                            # Decode the extracted pattern
                            chars = []
                            for i in range(0, len(extracted_pattern), 8):
                                if i + 8 <= len(extracted_pattern):
                                    byte = extracted_pattern[i:i+8]
                                    try:
                                        char_val = int(byte, 2)
                                        if 32 <= char_val <= 126:
                                            chars.append(chr(char_val))
                                        else:
                                            chars.append(f'[{char_val}]')
                                    except:
                                        chars.append('?')
                            
                            text = ''.join(chars)
                            print(f"  Decoded: {text}")
    
    print(f"\n=== BEST MATCH FOUND ===")
    if best_match:
        print(f"Score: {best_match['score']:.1%}")
        print(f"Position: y={best_match['y']}, x_start={best_match['x_start']}")
        print(f"Parameters: spacing={best_match['spacing']}, threshold={best_match['threshold']}")
        print(f"Target:    {target_pattern}")
        print(f"Extracted: {best_match['extracted']}")
        
        # Decode the best match
        chars = []
        pattern = best_match['extracted']
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
        
        text = ''.join(chars)
        print(f"Decoded: {text}")
        
        # Save results
        with open('pattern_search_results.txt', 'w') as f:
            f.write("=== PATTERN SEARCH RESULTS ===\n")
            f.write(f"Best match score: {best_match['score']:.1%}\n")
            f.write(f"Position: y={best_match['y']}, x_start={best_match['x_start']}\n")
            f.write(f"Parameters: spacing={best_match['spacing']}, threshold={best_match['threshold']}\n")
            f.write(f"Target pattern:    {target_pattern}\n")
            f.write(f"Extracted pattern: {best_match['extracted']}\n")
            f.write(f"Decoded text: {text}\n")
        
        return best_match
    else:
        print("No pattern match found")
        return None

def extract_extended_message(best_match):
    """Extract extended message using the found parameters."""
    if not best_match:
        print("No parameters to use for extended extraction")
        return
    
    # Load the poster image
    img = cv2.imread('satoshi.png')
    if img is None:
        img = cv2.imread('satoshi (1).png')
    if img is None:
        print("ERROR: Could not load poster image")
        return
    
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Use the found parameters
    y = best_match['y']
    x_start = best_match['x_start']
    spacing = best_match['spacing']
    threshold = best_match['threshold']
    
    print(f"\nExtracting extended message with found parameters:")
    print(f"y={y}, x_start={x_start}, spacing={spacing}, threshold={threshold}")
    
    # Extract more bits to get the complete line
    bits = []
    for bit_pos in range(200):  # Extract up to 200 bits (25 characters)
        x = x_start + (bit_pos * spacing)
        if x >= img.shape[1]:
            break
            
        v_value = hsv[y, x, 2]
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
    
    print(f"Extended text: {text}")
    
    # Save extended results
    with open('extended_message_results.txt', 'w') as f:
        f.write("=== EXTENDED MESSAGE EXTRACTION ===\n")
        f.write(f"Parameters: y={y}, x_start={x_start}, spacing={spacing}, threshold={threshold}\n")
        f.write(f"Raw bits: {bit_string}\n")
        f.write(f"Decoded text: {text}\n")
    
    return text

if __name__ == "__main__":
    # Search for the pattern
    best_match = search_for_pattern()
    
    # Extract extended message
    if best_match and best_match['score'] > 0.6:  # If we found a decent match
        extended_text = extract_extended_message(best_match)
    else:
        print("Pattern match score too low for extended extraction")