#!/usr/bin/env python3
"""
Grid pattern tester - systematically test different grid reading patterns
to match the user's manual extraction results.
"""

import cv2
import numpy as np
import pandas as pd

def test_grid_patterns():
    """Test different grid reading patterns."""
    
    print("=== TESTING GRID READING PATTERNS ===")
    
    IMG = "mibera_satoshi_poster_highres.png"
    img = cv2.imread(IMG, 0)
    
    # User's expected pattern for "On the"
    expected_bits = "010011110110111000100000011101000110100001100101"
    print(f"Expected 'On the...': {expected_bits}")
    
    # Proven parameters
    row0, col0 = 110, 56
    row_pitch, col_pitch = 30, 52
    threshold = 80  # This threshold showed some promise
    
    print(f"Testing with: pos=({row0},{col0}), pitch={row_pitch}x{col_pitch}, threshold={threshold}")
    
    # Test different grid reading patterns
    patterns = [
        ("row-major", "Read row by row, left to right"),
        ("col-major", "Read column by column, top to bottom"),
        ("zigzag", "Zigzag pattern"),
        ("spiral", "Spiral pattern"),
        ("diagonal", "Diagonal pattern"),
        ("reverse-row", "Row by row, right to left"),
        ("reverse-col", "Column by column, bottom to top")
    ]
    
    for pattern_name, description in patterns:
        print(f"\n--- Testing {pattern_name}: {description} ---")
        
        bits = extract_grid_pattern(img, row0, col0, row_pitch, col_pitch, threshold, pattern_name)
        
        if len(bits) >= 48:
            bits_str = ''.join(map(str, bits[:48]))
            print(f"First 48 bits: {bits_str}")
            
            # Check for match
            if bits_str.startswith(expected_bits):
                print(f"✅ MATCH FOUND with {pattern_name}!")
                return pattern_name, bits
            else:
                # Check inverted
                inverted_str = ''.join(map(str, [1-b for b in bits[:48]]))
                if inverted_str.startswith(expected_bits):
                    print(f"✅ MATCH FOUND with {pattern_name} (inverted)!")
                    return pattern_name, [1-b for b in bits]
    
    print("❌ No pattern matched the expected sequence")
    return None, None

def extract_grid_pattern(img, row0, col0, row_pitch, col_pitch, threshold, pattern):
    """Extract bits using different grid patterns."""
    
    bits = []
    max_rows = 10
    max_cols = 10
    
    if pattern == "row-major":
        # Read row by row, left to right
        for row in range(max_rows):
            for col in range(max_cols):
                y = row0 + row * row_pitch
                x = col0 + col * col_pitch
                if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                    pixel_val = img[y, x]
                    bit = 1 if pixel_val > threshold else 0
                    bits.append(bit)
    
    elif pattern == "col-major":
        # Read column by column, top to bottom
        for col in range(max_cols):
            for row in range(max_rows):
                y = row0 + row * row_pitch
                x = col0 + col * col_pitch
                if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                    pixel_val = img[y, x]
                    bit = 1 if pixel_val > threshold else 0
                    bits.append(bit)
    
    elif pattern == "zigzag":
        # Zigzag pattern
        for row in range(max_rows):
            if row % 2 == 0:
                # Left to right
                for col in range(max_cols):
                    y = row0 + row * row_pitch
                    x = col0 + col * col_pitch
                    if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                        pixel_val = img[y, x]
                        bit = 1 if pixel_val > threshold else 0
                        bits.append(bit)
            else:
                # Right to left
                for col in range(max_cols-1, -1, -1):
                    y = row0 + row * row_pitch
                    x = col0 + col * col_pitch
                    if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                        pixel_val = img[y, x]
                        bit = 1 if pixel_val > threshold else 0
                        bits.append(bit)
    
    elif pattern == "reverse-row":
        # Row by row, right to left
        for row in range(max_rows):
            for col in range(max_cols-1, -1, -1):
                y = row0 + row * row_pitch
                x = col0 + col * col_pitch
                if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                    pixel_val = img[y, x]
                    bit = 1 if pixel_val > threshold else 0
                    bits.append(bit)
    
    elif pattern == "reverse-col":
        # Column by column, bottom to top
        for col in range(max_cols):
            for row in range(max_rows-1, -1, -1):
                y = row0 + row * row_pitch
                x = col0 + col * col_pitch
                if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                    pixel_val = img[y, x]
                    bit = 1 if pixel_val > threshold else 0
                    bits.append(bit)
    
    return bits

def test_different_positions():
    """Test different starting positions around the proven coordinates."""
    
    print(f"\n=== TESTING DIFFERENT STARTING POSITIONS ===")
    
    IMG = "mibera_satoshi_poster_highres.png"
    img = cv2.imread(IMG, 0)
    
    # User's expected pattern
    expected_bits = "010011110110111000100000011101000110100001100101"
    
    # Base position
    base_row, base_col = 110, 56
    row_pitch, col_pitch = 30, 52
    threshold = 80
    
    # Test positions around the base
    for row_offset in range(-5, 6):
        for col_offset in range(-5, 6):
            row0 = base_row + row_offset
            col0 = base_col + col_offset
            
            # Extract first 48 bits using row-major pattern
            bits = []
            for i in range(48):
                bit_row = i // 8
                bit_col = i % 8
                
                y = row0 + bit_row * row_pitch
                x = col0 + bit_col * col_pitch
                
                if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                    pixel_val = img[y, x]
                    bit = 1 if pixel_val > threshold else 0
                    bits.append(bit)
            
            if len(bits) == 48:
                bits_str = ''.join(map(str, bits))
                
                # Check for match
                if bits_str.startswith(expected_bits):
                    print(f"✅ MATCH FOUND at position ({row0}, {col0})!")
                    return row0, col0, bits
                
                # Check inverted
                inverted_str = ''.join(map(str, [1-b for b in bits]))
                if inverted_str.startswith(expected_bits):
                    print(f"✅ MATCH FOUND at position ({row0}, {col0}) (inverted)!")
                    return row0, col0, [1-b for b in bits]
    
    print("❌ No position matched the expected sequence")
    return None, None, None

if __name__ == "__main__":
    print("Grid Pattern Tester")
    print("=" * 60)
    
    # Test different grid patterns
    pattern, bits = test_grid_patterns()
    
    if pattern is None:
        # Test different positions
        row0, col0, bits = test_different_positions()
        
        if row0 is not None:
            print(f"\n✅ Found correct position: ({row0}, {col0})")
            print(f"First 48 bits: {''.join(map(str, bits[:48]))}")
            
            # Show ASCII preview
            ascii_bytes = []
            for i in range(0, len(bits), 8):
                if i + 8 <= len(bits):
                    byte_str = ''.join(map(str, bits[i:i+8]))
                    byte_val = int(byte_str, 2)
                    ascii_bytes.append(byte_val)
            
            try:
                ascii_text = bytes(ascii_bytes).decode('ascii', errors='replace')
                print(f"ASCII preview: {ascii_text}")
            except:
                print(f"Raw bytes: {ascii_bytes}")
        else:
            print("❌ Could not find correct pattern or position")
    else:
        print(f"\n✅ Found correct pattern: {pattern}")
        print(f"First 48 bits: {''.join(map(str, bits[:48]))}") 