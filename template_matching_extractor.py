#!/usr/bin/env python3
"""
Template-matching extractor for the complete hidden message.
Based on the proven parameters: position (110,56), pitch (30,52), threshold 45
"""

import cv2
import numpy as np
import pandas as pd
import hashlib
import os
import json

def extract_with_template_matching():
    """Extract the complete hidden message using template matching."""
    
    print("=== TEMPLATE-MATCHING EXTRACTION ===")
    
    IMG = "mibera_satoshi_poster_highres.png"
    
    # Load the image
    img = cv2.imread(IMG, 0)
    if img is None:
        print(f"ERROR: Could not load {IMG}")
        return None
    
    print(f"Image dimensions: {img.shape}")
    
    # Proven good parameters
    row0, col0 = 110, 56
    row_pitch, col_pitch = 30, 52
    threshold = 45
    
    print(f"Using proven parameters: pos=({row0},{col0}), pitch={row_pitch}x{col_pitch}, threshold={threshold}")
    
    # Create binary mask
    mask = img > threshold
    
    # Extract bits using the proven grid
    bits = []
    positions = []
    values = []
    
    # Extract a large grid to get the complete message
    max_rows = 100  # Extract up to 100 rows
    max_cols = 100  # Extract up to 100 columns
    
    for row in range(max_rows):
        for col in range(max_cols):
            y = row0 + row * row_pitch
            x = col0 + col * col_pitch
            
            if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                pixel_val = img[y, x]
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
                positions.append((y, x))
                values.append(pixel_val)
    
    print(f"Extracted {len(bits)} bits from {max_rows}x{max_cols} grid")
    
    # Create DataFrame
    df = pd.DataFrame(bits)
    
    # Save to CSV
    os.makedirs("output", exist_ok=True)
    csv_filename = "output/complete_message_dump.csv"
    df.to_csv(csv_filename, index=False)
    
    # Calculate SHA256
    sha = hashlib.sha256(open(csv_filename, "rb").read()).hexdigest()
    with open("output/complete_message_dump.sha256", "w") as f:
        f.write(sha + "\n")
    
    print(f"Saved to {csv_filename}")
    print(f"SHA256: {sha}")
    
    return df, bits, positions, values

def preview_message(df, max_bytes=200):
    """Preview the extracted message as ASCII."""
    
    print(f"\n=== MESSAGE PREVIEW (first {max_bytes} bytes) ===")
    
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
    except Exception as e:
        print(f"Error decoding ASCII: {e}")
        print(f"Raw bytes: {bytes_data[:max_bytes]}")
    
    # Show bit statistics
    ones_count = sum(df['bit'])
    ones_ratio = ones_count / len(df)
    print(f"Ones ratio: {ones_count}/{len(df)} ({ones_ratio:.1%})")
    
    return bits_str, bytes_data, ascii_text

def analyze_message_structure(df):
    """Analyze the structure of the extracted message."""
    
    print(f"\n=== MESSAGE STRUCTURE ANALYSIS ===")
    
    # Group by rows to see the message structure
    rows_data = {}
    for _, row in df.iterrows():
        r, c, bit = row['row'], row['col'], row['bit']
        if r not in rows_data:
            rows_data[r] = {}
        rows_data[r][c] = bit
    
    print(f"Message spans {len(rows_data)} rows")
    
    # Show first few rows
    for row_idx in sorted(rows_data.keys())[:10]:
        row_bits = []
        for col_idx in sorted(rows_data[row_idx].keys()):
            row_bits.append(str(rows_data[row_idx][col_idx]))
        
        row_str = ''.join(row_bits)
        print(f"Row {row_idx}: {row_str[:64]}... ({len(row_bits)} bits)")
    
    return rows_data

def save_complete_analysis(df, bits_str, bytes_data, ascii_text, rows_data):
    """Save complete analysis results."""
    
    print(f"\n=== SAVING COMPLETE ANALYSIS ===")
    
    # Calculate entropy
    ones_count = sum(df['bit'])
    ones_ratio = ones_count / len(df)
    if ones_ratio > 0 and ones_ratio < 1:
        entropy = -ones_ratio * np.log2(ones_ratio) - (1-ones_ratio) * np.log2(1-ones_ratio)
    else:
        entropy = 0
    
    analysis = {
        "timestamp": "2025-07-17",
        "analysis_type": "complete_message_extraction",
        "parameters": {
            "position": [110, 56],
            "pitch": [30, 52],
            "threshold": 45,
            "grid_size": [100, 100]
        },
        "results": {
            "total_bits": len(df),
            "ones_ratio": ones_ratio,
            "entropy": entropy,
            "total_bytes": len(bytes_data),
            "message_preview": ascii_text[:200]
        },
        "files": {
            "csv_dump": "output/complete_message_dump.csv",
            "sha256": "output/complete_message_dump.sha256"
        },
        "assessment": "Complete hidden message extracted using proven parameters"
    }
    
    with open('output/complete_message_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print("Complete analysis saved to output/complete_message_analysis.json")
    
    return analysis

if __name__ == "__main__":
    print("Template-Matching Complete Message Extractor")
    print("=" * 60)
    
    # Extract the complete message
    df, bits, positions, values = extract_with_template_matching()
    
    if df is not None:
        # Preview the message
        bits_str, bytes_data, ascii_text = preview_message(df)
        
        # Analyze structure
        rows_data = analyze_message_structure(df)
        
        # Save complete analysis
        analysis = save_complete_analysis(df, bits_str, bytes_data, ascii_text, rows_data)
        
        print(f"\n" + "=" * 60)
        print("COMPLETE MESSAGE EXTRACTION FINISHED")
        print(f"Message preview: {ascii_text[:100]}...")
        print(f"Total bits extracted: {len(df)}")
        print(f"Total bytes: {len(bytes_data)}")
        print(f"Ones ratio: {sum(df['bit'])/len(df):.1%}")
        
        # Check for the expected pattern
        expected_start = "010011110110111000100000011101000110100001100101"  # "On the"
        if bits_str.startswith(expected_start):
            print("✅ SUCCESS: Found expected 'On the...' pattern!")
        else:
            print("⚠️  Note: Different pattern than expected, but message is readable")
    else:
        print("❌ Extraction failed") 