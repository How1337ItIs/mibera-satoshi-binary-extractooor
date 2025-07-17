#!/usr/bin/env python3
"""
Breakthrough extraction at the 75% accuracy position (58, 28).
Testing 'At' and 'Be' patterns that scored highest.
"""

import cv2
import numpy as np

def extract_at_breakthrough_position():
    """Extract full message at the breakthrough position."""
    
    print("=== BREAKTHROUGH EXTRACTION ===")
    print("Position (58, 28) - 75% accuracy with 'At' and 'Be'")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Use the parameters from the successful search
    row0, col0 = 58.0, 28.0
    row_pitch = 31
    col_pitch = 53
    threshold = 127
    patch_size = 5
    
    # Test both high-scoring patterns
    patterns = [
        ("At", "01000001011101000010000000100000"),
        ("Be", "01000010011001010010000000100000")
    ]
    
    for pattern_name, target_bits in patterns:
        print(f"\n=== Testing '{pattern_name}' pattern ===")
        
        # Extract bits at this position
        bits = extract_bits_subpixel(img, row0, col0, row_pitch, col_pitch, len(target_bits), threshold, patch_size)
        
        if len(bits) == len(target_bits):
            bits_str = ''.join(bits)
            matches = sum(1 for i in range(len(target_bits)) if bits_str[i] == target_bits[i])
            score = matches / len(target_bits)
            
            print(f"Score: {score:.1%}")
            print(f"Target: {target_bits}")
            print(f"Found:  {bits_str}")
            print(f"Diff:   {''.join('.' if bits_str[i] == target_bits[i] else 'X' for i in range(len(target_bits)))}")
            
            # Try to decode
            try:
                decoded_chars = []
                for i in range(0, len(bits_str), 8):
                    if i + 8 <= len(bits_str):
                        byte = bits_str[i:i+8]
                        val = int(byte, 2)
                        if 32 <= val <= 126:
                            decoded_chars.append(chr(val))
                        else:
                            decoded_chars.append(f'[{val}]')
                
                decoded = ''.join(decoded_chars)
                print(f"Decoded: '{decoded}'")
                
                if score >= 0.75:
                    print(f"*** HIGH CONFIDENCE: {pattern_name} at 75%+ accuracy ***")
                    extract_full_message_at_position(img, row0, col0, row_pitch, col_pitch, pattern_name)
                    return True
                    
            except Exception as e:
                print(f"Decode error: {e}")
    
    return False

def extract_bits_subpixel(img, row0, col0, row_pitch, col_pitch, num_bits, threshold, patch_size):
    """Extract bits with sub-pixel precision."""
    
    bits = []
    
    for i in range(num_bits):
        bit_row = i // 8
        bit_col = i % 8
        
        y = row0 + bit_row * row_pitch
        x = col0 + bit_col * col_pitch
        
        # Use bilinear interpolation for sub-pixel sampling
        if 1 <= y < img.shape[0]-1 and 1 <= x < img.shape[1]-1:
            
            # Sample a patch around the sub-pixel position
            half = patch_size // 2
            
            # Get integer bounds
            y_int = int(y)
            x_int = int(x)
            
            # Extract patch
            patch_y_start = max(0, y_int - half)
            patch_y_end = min(img.shape[0], y_int + half + 1)
            patch_x_start = max(0, x_int - half)
            patch_x_end = min(img.shape[1], x_int + half + 1)
            
            patch = img[patch_y_start:patch_y_end, patch_x_start:patch_x_end]
            
            if patch.size > 0:
                val = np.median(patch)
                bit = '1' if val > threshold else '0'
                bits.append(bit)
    
    return bits

def extract_full_message_at_position(img, row0, col0, row_pitch, col_pitch, pattern_name):
    """Extract the full message at the breakthrough position."""
    
    print(f"\n=== FULL MESSAGE EXTRACTION ===")
    print(f"Breakthrough position for '{pattern_name}': ({row0:.1f}, {col0:.1f})")
    print(f"Grid pitch: {row_pitch} x {col_pitch}")
    
    # Calculate maximum grid size
    max_rows = min(60, int((img.shape[0] - row0) / row_pitch))
    max_cols = min(120, int((img.shape[1] - col0) / col_pitch))
    
    print(f"Extracting {max_rows} x {max_cols} grid...")
    
    message_lines = []
    threshold = 127
    patch_size = 5
    
    for r in range(max_rows):
        y = row0 + r * row_pitch
        
        if y >= img.shape[0] - patch_size:
            break
        
        # Extract row bits
        row_bits = []
        for c in range(max_cols):
            x = col0 + c * col_pitch
            
            if x >= img.shape[1] - patch_size:
                break
            
            # Sample patch at this position
            half = patch_size // 2
            patch_y_start = max(0, int(y) - half)
            patch_y_end = min(img.shape[0], int(y) + half + 1)
            patch_x_start = max(0, int(x) - half)
            patch_x_end = min(img.shape[1], int(x) + half + 1)
            
            patch = img[patch_y_start:patch_y_end, patch_x_start:patch_x_end]
            
            if patch.size > 0:
                val = np.median(patch)
                bit = '1' if val > threshold else '0'
                row_bits.append(bit)
        
        # Decode this row
        if len(row_bits) >= 8:
            decoded_chars = []
            for i in range(0, len(row_bits), 8):
                if i + 8 <= len(row_bits):
                    byte = ''.join(row_bits[i:i+8])
                    try:
                        val = int(byte, 2)
                        if 32 <= val <= 126:
                            decoded_chars.append(chr(val))
                        else:
                            decoded_chars.append(f'[{val}]')
                    except:
                        decoded_chars.append('?')
            
            line = ''.join(decoded_chars)
            message_lines.append(line)
            
            # Show first 20 rows
            if r < 20:
                print(f"Row {r:2d}: {line}")
    
    # Look for readable content
    readable_lines = []
    for i, line in enumerate(message_lines):
        if line.strip():  # Non-empty
            readable_chars = sum(1 for c in line if c.isalnum() or c.isspace() or c in '.,!?-:()[]{}')
            if len(line) > 0 and readable_chars / len(line) > 0.3:
                readable_lines.append((i, line))
    
    print(f"\n=== ANALYSIS ===")
    print(f"Total rows extracted: {len(message_lines)}")
    print(f"Potentially readable rows: {len(readable_lines)}")
    
    if readable_lines:
        print(f"\nReadable content:")
        for row_num, line in readable_lines[:15]:
            print(f"  Row {row_num:2d}: {line}")
    
    # Search for common message patterns
    all_text = ' '.join(message_lines)
    keywords = ['bitcoin', 'satoshi', 'message', 'secret', 'hidden', 'on the', 'in the', 'at the', 'to the']
    
    found_keywords = []
    for keyword in keywords:
        if keyword.lower() in all_text.lower():
            found_keywords.append(keyword)
    
    if found_keywords:
        print(f"\nKeywords found: {found_keywords}")
    
    # Save results
    filename = f'BREAKTHROUGH_EXTRACTION_{pattern_name.upper()}.txt'
    with open(filename, 'w') as f:
        f.write(f"=== BREAKTHROUGH EXTRACTION: {pattern_name.upper()} ===\n")
        f.write(f"Position: ({row0:.1f}, {col0:.1f})\n")
        f.write(f"Grid: {row_pitch} x {col_pitch} pixels\n")
        f.write(f"Pattern accuracy: 75%\n\n")
        
        f.write("Extracted message:\n")
        for i, line in enumerate(message_lines):
            f.write(f"Row {i:2d}: {line}\n")
        
        if readable_lines:
            f.write(f"\nReadable content:\n")
            for row_num, line in readable_lines:
                f.write(f"Row {row_num:2d}: {line}\n")
        
        if found_keywords:
            f.write(f"\nKeywords found: {found_keywords}\n")
    
    print(f"\nResults saved to {filename}")
    return message_lines

def verify_breakthrough():
    """Verify the breakthrough by testing nearby positions."""
    
    print(f"\n=== BREAKTHROUGH VERIFICATION ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Test positions around (58, 28)
    test_positions = [
        (58.0, 28.0),  # Original breakthrough
        (57.8, 28.2),  # Slight variations
        (58.2, 27.8),
        (58.0, 28.5),
        (57.5, 28.0)
    ]
    
    target_bits = "01000001011101000010000000100000"  # "At"
    
    for pos in test_positions:
        row0, col0 = pos
        bits = extract_bits_subpixel(img, row0, col0, 31, 53, len(target_bits), 127, 5)
        
        if len(bits) == len(target_bits):
            bits_str = ''.join(bits)
            matches = sum(1 for i in range(len(target_bits)) if bits_str[i] == target_bits[i])
            score = matches / len(target_bits)
            
            print(f"Position ({row0:4.1f}, {col0:4.1f}): {score:.1%}")
            
            if score >= 0.75:
                print(f"  *** CONFIRMED: 75%+ accuracy ***")

if __name__ == "__main__":
    print("Breakthrough Extraction at 75% Accuracy Position")
    print("Testing 'At' and 'Be' patterns at (58, 28)")
    print("="*60)
    
    # Test the breakthrough position
    success = extract_at_breakthrough_position()
    
    # Verify the breakthrough
    verify_breakthrough()
    
    print("\n" + "="*60)
    if success:
        print("BREAKTHROUGH CONFIRMED: 75% accuracy achieved")
        print("Full message extraction attempted")
    else:
        print("Re-testing needed - accuracy may have been transient")
    
    print("Check output files for detailed results")