#!/usr/bin/env python3
"""
Analyze the cells.csv data to understand what was actually extracted.
"""

import pandas as pd
import numpy as np

def analyze_cells_data():
    """Analyze the extracted cells data."""
    
    df = pd.read_csv('binary_extractor/output_real_data/cells.csv')
    
    print(f"Cells data analysis:")
    print(f"  Total cells: {len(df)}")
    print(f"  Grid size: {df['row'].max()+1} rows x {df['col'].max()+1} cols")
    print(f"  Bit types: {df['bit'].value_counts().to_dict()}")
    
    # Filter to only binary bits
    binary_df = df[df['bit'].isin(['0', '1'])].copy()
    print(f"  Binary cells: {len(binary_df)}")
    
    # Analyze first few rows
    print(f"\nFirst 10 rows of data:")
    for row_idx in range(min(10, df['row'].max()+1)):
        row_data = df[df['row'] == row_idx].sort_values('col')
        
        # Convert to bit string, replacing non-binary with '0'
        bits = []
        for _, cell in row_data.iterrows():
            if cell['bit'] in ['0', '1']:
                bits.append(cell['bit'])
            else:
                bits.append('0')  # Default for blank/overlay
        
        bits_str = ''.join(bits)
        
        # Try to decode
        decoded_chars = []
        for i in range(0, len(bits_str), 8):
            if i + 8 <= len(bits_str):
                byte = bits_str[i:i+8]
                try:
                    val = int(byte, 2)
                    if 32 <= val <= 126:
                        decoded_chars.append(chr(val))
                    else:
                        decoded_chars.append(f'[{val}]')
                except:
                    decoded_chars.append('?')
        
        decoded = ''.join(decoded_chars)
        print(f"  Row {row_idx:2d}: {bits_str[:40]}... -> {decoded[:20]}")
    
    # Look for any "On" patterns
    print(f"\nSearching for 'On' patterns...")
    target_bits = "0100111101101110"  # "On"
    
    matches_found = []
    
    for row_idx in range(df['row'].max()+1):
        row_data = df[df['row'] == row_idx].sort_values('col')
        
        # Get clean bit string
        bits = []
        for _, cell in row_data.iterrows():
            if cell['bit'] in ['0', '1']:
                bits.append(cell['bit'])
            else:
                bits.append('0')
        
        bits_str = ''.join(bits)
        
        # Search for target pattern at different positions
        for start in range(len(bits_str) - 15):
            test_bits = bits_str[start:start+16]
            
            # Calculate similarity
            matches = sum(1 for i in range(16) if test_bits[i] == target_bits[i])
            similarity = matches / 16
            
            if similarity > 0.7:  # Good match
                try:
                    char1 = chr(int(test_bits[:8], 2))
                    char2 = chr(int(test_bits[8:16], 2))
                    decoded = f"{char1}{char2}"
                except:
                    decoded = "??"
                
                matches_found.append({
                    'row': row_idx,
                    'start_col': start,
                    'similarity': similarity,
                    'bits': test_bits,
                    'decoded': decoded
                })
    
    if matches_found:
        print(f"Found {len(matches_found)} potential 'On' matches:")
        for match in sorted(matches_found, key=lambda x: x['similarity'], reverse=True)[:10]:
            print(f"  Row {match['row']:2d} col {match['start_col']:2d}: "
                  f"{match['similarity']:.1%} -> '{match['decoded']}' ({match['bits']})")
    else:
        print("No strong 'On' matches found in cells.csv data")
    
    # Check for any readable text at all
    print(f"\nSearching for readable ASCII text...")
    readable_text = []
    
    for row_idx in range(min(20, df['row'].max()+1)):
        row_data = df[df['row'] == row_idx].sort_values('col')
        
        bits = []
        for _, cell in row_data.iterrows():
            if cell['bit'] in ['0', '1']:
                bits.append(cell['bit'])
            else:
                bits.append('0')
        
        bits_str = ''.join(bits)
        
        # Try decoding at different positions
        for start in range(0, min(len(bits_str), 32), 8):
            if start + 32 <= len(bits_str):
                segment = bits_str[start:start+32]  # 4 characters
                
                try:
                    chars = []
                    readable_count = 0
                    
                    for i in range(0, 32, 8):
                        byte = segment[i:i+8]
                        val = int(byte, 2)
                        
                        if 32 <= val <= 126:
                            char = chr(val)
                            chars.append(char)
                            if char.isalnum() or char.isspace():
                                readable_count += 1
                        else:
                            chars.append(f'[{val}]')
                    
                    if readable_count >= 2:  # At least 2 readable chars
                        text = ''.join(chars)
                        readable_text.append({
                            'row': row_idx,
                            'start_col': start,
                            'text': text,
                            'readable_count': readable_count
                        })
                
                except:
                    pass
    
    if readable_text:
        print(f"Found {len(readable_text)} readable text segments:")
        for text in sorted(readable_text, key=lambda x: x['readable_count'], reverse=True)[:10]:
            print(f"  Row {text['row']:2d} col {text['start_col']:2d}: "
                  f"'{text['text']}' ({text['readable_count']} readable)")
    else:
        print("No clearly readable ASCII text found")

def compare_with_reference():
    """Compare with the reference 77.1% accuracy results."""
    
    # Look for reference files
    import os
    
    ref_files = [
        'CLAUDE_COMPREHENSIVE_BREAKTHROUGH_RESULTS.txt',
        'complete_satoshi_hidden_message.txt',
        'COMPLETE_EXTRACTION_SUMMARY.md'
    ]
    
    print(f"\n=== Comparing with reference results ===")
    
    for ref_file in ref_files:
        if os.path.exists(ref_file):
            print(f"\nFound reference file: {ref_file}")
            with open(ref_file, 'r') as f:
                content = f.read()
                
            # Look for key information
            if "On the" in content:
                print("✓ Reference contains 'On the' - this is our target")
            if "77.1%" in content:
                print("✓ Reference mentions 77.1% accuracy")
            if "Satoshi" in content:
                print("✓ Reference is about Satoshi message")
                
            # Show first few lines
            lines = content.split('\n')[:10]
            print("First few lines:")
            for line in lines:
                if line.strip():
                    print(f"  {line}")
        else:
            print(f"Reference file not found: {ref_file}")

if __name__ == "__main__":
    print("Analyzing Extracted Cells Data")
    print("="*50)
    
    analyze_cells_data()
    compare_with_reference()
    
    print("\n" + "="*50)
    print("CONCLUSION:")
    print("The extracted cells data needs to be analyzed to understand")
    print("why we're not getting readable 'On the' text. This might indicate:")
    print("1. Wrong grid alignment in the binary_extractor")
    print("2. Different encoding or message structure")
    print("3. Need to use a different extraction approach")