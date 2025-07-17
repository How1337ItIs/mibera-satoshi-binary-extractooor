#!/usr/bin/env python3
"""
Refined poster extraction using calibrated parameters from cropped image success.
Apply the exact working parameters (spacing=18, threshold=80) to extract readable text.
"""

import cv2
import numpy as np

def extract_hidden_text_refined():
    """Extract hidden text using calibrated parameters from cropped image success."""
    
    # Load the poster image
    img = cv2.imread('satoshi.png')
    if img is None:
        img = cv2.imread('satoshi (1).png')
    if img is None:
        print("ERROR: Could not load poster image")
        return
    
    print(f"Loaded poster image: {img.shape}")
    
    # Convert to HSV for better threshold control
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Calibrated parameters from successful cropped extraction
    spacing = 18  # Pixel spacing between bits
    threshold = 80  # V-channel threshold for 1/0 classification
    start_y = 35  # Start from top region where text was found
    line_height = 25  # Spacing between text lines
    
    print(f"Using calibrated parameters: spacing={spacing}, threshold={threshold}")
    
    extracted_lines = []
    
    # Extract from top region first (where we know text exists)
    for line_num in range(10):  # Try first 10 lines
        y = start_y + (line_num * line_height)
        if y >= img.shape[0]:
            break
            
        print(f"\nExtracting line {line_num} at y={y}")
        
        # Extract bits from this line
        bits = []
        for bit_pos in range(100):  # Extract up to 100 bits per line
            x = 15 + (bit_pos * spacing)  # Start at x=15 with spacing
            if x >= img.shape[1]:
                break
                
            # Sample the V channel value at this position
            v_value = hsv[y, x, 2]
            
            # Classify as 1 or 0 based on threshold
            bit = '1' if v_value > threshold else '0'
            bits.append(bit)
        
        bit_string = ''.join(bits)
        print(f"Raw bits: {bit_string[:64]}...")  # Show first 64 bits
        
        # Try to decode as ASCII (8-bit chunks)
        chars = []
        for i in range(0, len(bit_string), 8):
            if i + 8 <= len(bit_string):
                byte = bit_string[i:i+8]
                try:
                    char_val = int(byte, 2)
                    if 32 <= char_val <= 126:  # Printable ASCII
                        chars.append(chr(char_val))
                    else:
                        chars.append(f'[{char_val}]')
                except:
                    chars.append('?')
        
        text = ''.join(chars)
        print(f"Decoded: {text}")
        
        extracted_lines.append({
            'line': line_num,
            'y': y,
            'bits': bit_string,
            'text': text,
            'readable_chars': len([c for c in chars if c.isalnum() or c.isspace()])
        })
    
    # Save results
    with open('refined_extraction_results.txt', 'w') as f:
        f.write("=== REFINED EXTRACTION USING CALIBRATED PARAMETERS ===\n")
        f.write(f"Parameters: spacing={spacing}, threshold={threshold}, start_y={start_y}\n\n")
        
        for line_data in extracted_lines:
            f.write(f"Line {line_data['line']} (y={line_data['y']}):\n")
            f.write(f"  Bits: {line_data['bits'][:80]}...\n")
            f.write(f"  Text: {line_data['text']}\n")
            f.write(f"  Readable chars: {line_data['readable_chars']}\n\n")
        
        # Show concatenated readable text
        all_text = ' '.join([line['text'] for line in extracted_lines])
        f.write(f"\n=== CONCATENATED TEXT ===\n{all_text}\n")
    
    print(f"\nExtracted {len(extracted_lines)} lines")
    print("Results saved to refined_extraction_results.txt")
    
    # Check for the expected "On the winter" pattern
    all_text = ' '.join([line['text'] for line in extracted_lines])
    if "On" in all_text or "winter" in all_text:
        print("✅ SUCCESS: Found expected 'On'/'winter' text pattern!")
    else:
        print("⚠️  Need parameter adjustment - expected text not clearly visible")
    
    return extracted_lines

if __name__ == "__main__":
    extract_hidden_text_refined()