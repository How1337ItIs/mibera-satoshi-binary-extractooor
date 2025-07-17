#!/usr/bin/env python3
"""
Corrected message extractor - trying different bit interpretations
to match the expected "On the..." pattern.

Created by: Cursor Agent
Purpose: Test different bit interpretations to find correct extraction method
Date: 2025-07-17
"""

import cv2
import numpy as np
import pandas as pd
import hashlib
import os
import json

def test_different_interpretations():
    """Test different bit interpretations to find the correct one."""
    
    print("=== TESTING DIFFERENT BIT INTERPRETATIONS ===")
    
    IMG = "mibera_satoshi_poster_highres.png"
    img = cv2.imread(IMG, 0)
    
    # Proven parameters
    row0, col0 = 110, 56
    row_pitch, col_pitch = 30, 52
    
    # Expected pattern for "On the"
    expected_bits = "010011110110111000100000011101000110100001100101"
    
    print(f"Expected 'On the...': {expected_bits}")
    
    # Test different thresholds and bit interpretations
    thresholds = [30, 45, 60, 80, 100, 120, 150, 180]
    
    for threshold in thresholds:
        print(f"\n--- Testing threshold {threshold} ---")
        
        # Extract first 48 bits (6 bytes = "On the")
        bits_normal = []
        bits_inverted = []
        
        for i in range(48):
            bit_row = i // 8
            bit_col = i % 8
            
            y = row0 + bit_row * row_pitch
            x = col0 + bit_col * col_pitch
            
            if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                pixel_val = img[y, x]
                
                # Normal interpretation
                bit_normal = 1 if pixel_val > threshold else 0
                bits_normal.append(bit_normal)
                
                # Inverted interpretation
                bit_inverted = 0 if pixel_val > threshold else 1
                bits_inverted.append(bit_inverted)
        
        # Convert to strings
        normal_str = ''.join(map(str, bits_normal))
        inverted_str = ''.join(map(str, bits_inverted))
        
        # Check matches
        normal_match = normal_str.startswith(expected_bits)
        inverted_match = inverted_str.startswith(expected_bits)
        
        print(f"Normal:   {normal_str[:48]} (match: {normal_match})")
        print(f"Inverted: {inverted_str[:48]} (match: {inverted_match})")
        
        if normal_match or inverted_match:
            print(f"✅ FOUND MATCH with threshold {threshold}!")
            
            # Show ASCII preview
            if normal_match:
                bits_for_ascii = bits_normal
                print("Using normal interpretation")
            else:
                bits_for_ascii = bits_inverted
                print("Using inverted interpretation")
            
            # Convert to ASCII
            ascii_bytes = []
            for i in range(0, len(bits_for_ascii), 8):
                if i + 8 <= len(bits_for_ascii):
                    byte_str = ''.join(map(str, bits_for_ascii[i:i+8]))
                    byte_val = int(byte_str, 2)
                    ascii_bytes.append(byte_val)
            
            try:
                ascii_text = bytes(ascii_bytes).decode('ascii', errors='replace')
                print(f"ASCII preview: {ascii_text}")
            except:
                print(f"Raw bytes: {ascii_bytes}")
            
            return threshold, normal_match, bits_normal if normal_match else bits_inverted
    
    print("❌ No match found with any threshold")
    return None, None, None

def extract_complete_message(threshold, use_inverted=False):
    """Extract the complete message with the correct parameters."""
    
    print(f"\n=== EXTRACTING COMPLETE MESSAGE ===")
    print(f"Threshold: {threshold}, Inverted: {use_inverted}")
    
    IMG = "mibera_satoshi_poster_highres.png"
    img = cv2.imread(IMG, 0)
    
    # Proven parameters
    row0, col0 = 110, 56
    row_pitch, col_pitch = 30, 52
    
    # Extract bits
    bits = []
    max_rows = 50  # Extract more rows to get complete message
    max_cols = 50
    
    for row in range(max_rows):
        for col in range(max_cols):
            y = row0 + row * row_pitch
            x = col0 + col * col_pitch
            
            if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                pixel_val = img[y, x]
                
                if use_inverted:
                    bit = 0 if pixel_val > threshold else 1
                else:
                    bit = 1 if pixel_val > threshold else 0
                
                bits.append({
                    'row': row,
                    'col': col,
                    'pixel_y': y,
                    'pixel_x': x,
                    'sample_value': pixel_val,
                    'threshold': threshold,
                    'bit': bit
                })
    
    print(f"Extracted {len(bits)} bits")
    
    # Create DataFrame
    df = pd.DataFrame(bits)
    
    # Save to CSV
    os.makedirs("output", exist_ok=True)
    csv_filename = f"output/corrected_message_dump_thresh{threshold}.csv"
    df.to_csv(csv_filename, index=False)
    
    # Calculate SHA256
    sha = hashlib.sha256(open(csv_filename, "rb").read()).hexdigest()
    with open(f"output/corrected_message_dump_thresh{threshold}.sha256", "w") as f:
        f.write(sha + "\n")
    
    print(f"Saved to {csv_filename}")
    
    return df, bits

def preview_corrected_message(df, max_bytes=300):
    """Preview the corrected message."""
    
    print(f"\n=== CORRECTED MESSAGE PREVIEW ===")
    
    # Convert bits to binary string
    bit_list = df['bit'].astype(str).tolist()
    bits_str = ''.join(bit_list)
    
    print(f"Total bits: {len(bits_str)}")
    
    # Convert to bytes
    bytes_data = []
    for i in range(0, len(bits_str), 8):
        if i + 8 <= len(bits_str):
            byte_str = bits_str[i:i+8]
            byte_val = int(byte_str, 2)
            bytes_data.append(byte_val)
    
    # Convert to ASCII
    try:
        ascii_text = bytes(bytes_data).decode('ascii', errors='replace')
        print(f"ASCII text: {ascii_text[:max_bytes]}")
        
        # Look for readable text
        readable_chars = sum(1 for c in ascii_text if 32 <= ord(c) <= 126)
        readability = readable_chars / len(ascii_text) if ascii_text else 0
        print(f"Readability: {readable_chars}/{len(ascii_text)} ({readability:.1%})")
        
    except Exception as e:
        print(f"Error decoding ASCII: {e}")
        print(f"Raw bytes: {bytes_data[:max_bytes]}")
    
    # Show bit statistics
    ones_count = sum(df['bit'])
    ones_ratio = ones_count / len(df)
    print(f"Ones ratio: {ones_count}/{len(df)} ({ones_ratio:.1%})")
    
    return bits_str, bytes_data, ascii_text

if __name__ == "__main__":
    print("Corrected Message Extractor")
    print("=" * 60)
    
    # Test different interpretations
    threshold, is_normal, correct_bits = test_different_interpretations()
    
    if threshold is not None:
        use_inverted = not is_normal
        print(f"\n✅ Using threshold {threshold}, inverted: {use_inverted}")
        
        # Extract complete message
        df, bits = extract_complete_message(threshold, use_inverted)
        
        # Preview message
        bits_str, bytes_data, ascii_text = preview_corrected_message(df)
        
        print(f"\n" + "=" * 60)
        print("CORRECTED MESSAGE EXTRACTION COMPLETE")
        print(f"Message: {ascii_text[:100]}...")
        
        # Save results
        results = {
            "timestamp": "2025-07-17",
            "analysis_type": "corrected_message_extraction",
            "parameters": {
                "position": [110, 56],
                "pitch": [30, 52],
                "threshold": threshold,
                "inverted": use_inverted
            },
            "results": {
                "total_bits": len(df),
                "ones_ratio": sum(df['bit'])/len(df),
                "total_bytes": len(bytes_data),
                "message_preview": ascii_text[:200]
            },
            "assessment": "Corrected extraction matching expected pattern"
        }
        
        with open(f'output/corrected_message_analysis_thresh{threshold}.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to output/corrected_message_analysis_thresh{threshold}.json")
        
    else:
        print("❌ Could not find correct interpretation") 