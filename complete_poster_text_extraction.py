#!/usr/bin/env python3
"""
Complete Poster Text Extraction - Claude Code Agent
Extract all lines of hidden text from the entire poster
"""

import cv2
import numpy as np

def extract_complete_poster_text():
    """Extract all lines of text from the poster"""
    
    img = cv2.imread('satoshi (1).png')
    if img is None:
        print("ERROR: Could not load poster")
        return
        
    print(f"Poster size: {img.shape}")
    print("=== EXTRACTING ALL HIDDEN TEXT LINES ===")
    
    spacing = 11  # Bit spacing
    threshold = 85  # Blue channel threshold
    line_height = 25  # Estimated line spacing
    
    all_text_lines = []
    
    # Extract multiple lines down the poster
    for line_num in range(40):  # Try up to 40 lines
        y = 5 + line_num * line_height  # Start at y=5, go down by line_height
        
        if y >= img.shape[0] - 10:  # Don't go past bottom
            break
            
        print(f"\nLine {line_num} (y={y}):")
        
        # Try different x starting positions for this line
        best_text = ""
        best_x = 0
        
        for x_start in [0, 10, 20, 30, 40]:
            
            # Extract bits for this line
            extracted_bits = []
            max_chars = min(50, (img.shape[1] - x_start) // (spacing * 8))  # Max characters
            
            for char_i in range(max_chars):
                char_bits = []
                
                for bit_i in range(8):  # 8 bits per character
                    bit_x = x_start + (char_i * 8 + bit_i) * spacing
                    bit_y = y
                    
                    if bit_x >= img.shape[1]:
                        break
                        
                    pixel_val = img[bit_y, bit_x, 0]  # Blue channel
                    bit = '1' if pixel_val > threshold else '0'
                    char_bits.append(bit)
                
                if len(char_bits) == 8:
                    extracted_bits.extend(char_bits)
            
            # Decode to ASCII
            text_chars = []
            for i in range(0, len(extracted_bits), 8):
                if i + 8 <= len(extracted_bits):
                    byte_str = ''.join(extracted_bits[i:i+8])
                    try:
                        byte_val = int(byte_str, 2)
                        if 32 <= byte_val <= 126:  # Printable ASCII
                            text_chars.append(chr(byte_val))
                        else:
                            text_chars.append(f'[{byte_val}]')
                    except:
                        text_chars.append('?')
            
            decoded_text = ''.join(text_chars)
            
            # Count readable characters
            readable_chars = sum(1 for c in decoded_text if c.isalnum() or c in ' .,!?-')
            
            if readable_chars > len(best_text.replace('[', '').replace(']', '')):
                best_text = decoded_text
                best_x = x_start
        
        if best_text and len(best_text.strip()) > 3:  # Only show lines with content
            print(f"  x={best_x}: \"{best_text[:60]}\"")
            all_text_lines.append({
                'line': line_num,
                'y': y,
                'x': best_x,
                'text': best_text
            })
    
    return all_text_lines

def save_extracted_text(text_lines):
    """Save all extracted text to file"""
    
    with open('complete_hidden_message.txt', 'w') as f:
        f.write("=== COMPLETE HIDDEN MESSAGE FROM SATOSHI POSTER ===\n")
        f.write(f"Extracted {len(text_lines)} lines of text\n\n")
        
        for line_data in text_lines:
            f.write(f"Line {line_data['line']:2d} (y={line_data['y']:3d}, x={line_data['x']:2d}): {line_data['text']}\n")
        
        f.write(f"\n=== CONCATENATED MESSAGE ===\n")
        full_message = ' '.join(line_data['text'].strip() for line_data in text_lines)
        f.write(full_message)
    
    print(f"\nComplete message saved to: complete_hidden_message.txt")
    
    # Show summary
    print(f"\n=== EXTRACTION SUMMARY ===")
    print(f"Lines extracted: {len(text_lines)}")
    
    if text_lines:
        full_text = ' '.join(line_data['text'] for line_data in text_lines)
        readable_chars = sum(1 for c in full_text if c.isalnum() or c in ' .,!?-')
        print(f"Total characters: {len(full_text)}")
        print(f"Readable characters: {readable_chars}")
        
        print(f"\nFirst few lines:")
        for line_data in text_lines[:5]:
            print(f"  \"{line_data['text'][:50]}\"")

if __name__ == "__main__":
    text_lines = extract_complete_poster_text()
    save_extracted_text(text_lines)