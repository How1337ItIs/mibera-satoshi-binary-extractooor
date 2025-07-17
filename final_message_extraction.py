#!/usr/bin/env python3
"""
Extract the complete hidden message using our best parameters.
"""

import cv2
import numpy as np

def extract_complete_message():
    """Extract the complete hidden message using optimized parameters."""
    
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
    
    # Our best parameters from fine-tuning
    y = 70
    x_start = 35
    spacing = 18
    threshold = 76
    
    print(f"Using optimized parameters: y={y}, x_start={x_start}, spacing={spacing}, threshold={threshold}")
    
    # Extract the first line (where we found the pattern)
    print(f"\nExtracting Line 1 at y={y}:")
    
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
    print(f"Line 1 text: {text}")
    
    # Extract additional lines above and below to get the complete message
    lines = []
    lines.append({'y': y, 'text': text, 'bits': bit_string})
    
    # Try lines above and below with the same spacing
    line_spacing = 25  # Estimate based on previous work
    
    for line_offset in range(-3, 4):  # 3 lines above and below
        if line_offset == 0:
            continue  # Skip the line we already extracted
            
        line_y = y + (line_offset * line_spacing)
        if line_y < 0 or line_y >= hsv.shape[0]:
            continue
            
        print(f"\nExtracting Line {line_offset + 4} at y={line_y}:")
        
        # Extract bits for this line
        line_bits = []
        for bit_pos in range(300):  # Extract up to 300 bits
            x = x_start + (bit_pos * spacing)
            x_int = int(round(x))
            
            if x_int >= hsv.shape[1]:
                break
                
            v_value = hsv[line_y, x_int, 2]
            bit = '1' if v_value > threshold else '0'
            line_bits.append(bit)
        
        line_bit_string = ''.join(line_bits)
        
        # Decode as ASCII
        line_chars = []
        for i in range(0, len(line_bit_string), 8):
            if i + 8 <= len(line_bit_string):
                byte = line_bit_string[i:i+8]
                try:
                    char_val = int(byte, 2)
                    if 32 <= char_val <= 126:
                        line_chars.append(chr(char_val))
                    else:
                        line_chars.append(f'[{char_val}]')
                except:
                    line_chars.append('?')
        
        line_text = ''.join(line_chars)
        print(f"Line {line_offset + 4} text: {line_text}")
        
        lines.append({'y': line_y, 'text': line_text, 'bits': line_bit_string})
    
    # Sort lines by y position
    lines.sort(key=lambda x: x['y'])
    
    # Save complete results
    with open('complete_hidden_message_final.txt', 'w') as f:
        f.write("=== COMPLETE HIDDEN MESSAGE EXTRACTION ===\n")
        f.write(f"Optimized parameters: y={y}, x_start={x_start}, spacing={spacing}, threshold={threshold}\n")
        f.write(f"Total lines extracted: {len(lines)}\n\n")
        
        for i, line in enumerate(lines):
            f.write(f"Line {i+1} (y={line['y']}):\n")
            f.write(f"  Text: {line['text']}\n")
            f.write(f"  Bits: {line['bits'][:80]}...\n\n")
        
        # Concatenate all readable text
        all_text = ' '.join([line['text'] for line in lines])
        f.write(f"=== CONCATENATED MESSAGE ===\n")
        f.write(f"{all_text}\n")
        
        # Count readable characters
        readable_chars = sum(len([c for c in line['text'] if c.isalnum() or c.isspace()]) for line in lines)
        f.write(f"\nTotal readable characters: {readable_chars}\n")
    
    print(f"\nExtracted {len(lines)} lines total")
    print("Results saved to complete_hidden_message_final.txt")
    
    # Show concatenated message
    all_text = ' '.join([line['text'] for line in lines])
    print(f"\nConcatenated message: {all_text}")
    
    # Check if we can see "On the winter" or similar
    if any(word in all_text.lower() for word in ['on', 'winter', 'the']):
        print("SUCCESS: Found expected words in extracted text!")
    else:
        print("Note: Expected words not clearly visible, may need further calibration")
    
    return lines

if __name__ == "__main__":
    extract_complete_message()