#!/usr/bin/env python3
"""
Final message extraction attempt.
Using all insights gained from the pitch debate resolution.
"""

import cv2
import numpy as np
import json

def final_message_extraction():
    """Final attempt to extract the hidden message."""
    
    print("=== FINAL MESSAGE EXTRACTION ===")
    print("Using refined methodology post-pitch-debate")
    
    # Load the binary mask
    img = cv2.imread('binary_extractor/output_real_data/bw_mask.png', 0)
    print(f"Image: {img.shape}")
    
    best_results = []
    
    # Test various grid configurations
    test_configs = [
        (31, 53, "Measured from autocorr"),
        (31, 25, "Binary extractor original"),
        (30, 24, "Slight variation"),
        (32, 26, "Slight variation"),
        (31, 18, "Manual 77.1% config scaled"),
        (5, 52, "Detected in optimizer"),
    ]
    
    for row_pitch, col_pitch, desc in test_configs:
        print(f"\nTesting: {desc} ({row_pitch}x{col_pitch})")
        
        # Find best origin for this pitch
        best_score = -1
        best_origin = None
        best_bits = None
        
        # Search origins
        row_range = range(max(30, 70 - row_pitch), min(120, 70 + row_pitch), 2)
        col_range = range(max(10, 40 - col_pitch//2), min(100, 40 + col_pitch//2), 2)
        
        for row0 in row_range:
            for col0 in col_range:
                # Extract first 24 bits (3 characters)
                bits = extract_bits_at(img, row0, col0, row_pitch, col_pitch, 24)
                
                if len(bits) == 24:
                    # Score this extraction
                    score = score_extraction(bits)
                    
                    if score > best_score:
                        best_score = score
                        best_origin = (row0, col0)
                        best_bits = bits
        
        if best_origin:
            row0, col0 = best_origin
            decoded = decode_bits(best_bits)
            
            print(f"  Best: ({row0}, {col0}) score={best_score:.2f}")
            print(f"  Text: '{decoded}'")
            
            best_results.append({
                'config': (row_pitch, col_pitch, desc),
                'origin': best_origin,
                'score': best_score,
                'bits': best_bits,
                'decoded': decoded
            })
    
    # Find overall best
    if best_results:
        best_results.sort(key=lambda x: x['score'], reverse=True)
        best = best_results[0]
        
        print(f"\n=== BEST CONFIGURATION ===")
        print(f"Config: {best['config'][2]} {best['config'][:2]}")
        print(f"Origin: {best['origin']}")
        print(f"Score: {best['score']:.2f}")
        print(f"Decoded: '{best['decoded']}'")
        
        # Extract full message with best config
        row_pitch, col_pitch = best['config'][:2]
        row0, col0 = best['origin']
        
        full_message = extract_full_message(img, row0, col0, row_pitch, col_pitch)
        
        return full_message
    
    return None

def extract_bits_at(img, row0, col0, row_pitch, col_pitch, num_bits):
    """Extract specific number of bits at given configuration."""
    
    bits = []
    
    for i in range(num_bits):
        # Calculate position
        bit_row = i // 8  # Which character row
        bit_col = i % 8   # Which bit in character
        
        y = row0 + bit_row * row_pitch
        x = col0 + bit_col * col_pitch
        
        if y < img.shape[0] - 3 and x < img.shape[1] - 3:
            # Sample 5x5 region
            region = img[max(0, y-2):min(img.shape[0], y+3), 
                        max(0, x-2):min(img.shape[1], x+3)]
            
            if region.size > 0:
                val = np.median(region)
                bit = '1' if val > 127 else '0'
                bits.append(bit)
    
    return bits

def score_extraction(bits):
    """Score an extraction based on how likely it is to be readable text."""
    
    if len(bits) < 8:
        return 0
    
    score = 0
    
    # Try to decode and score
    for i in range(0, len(bits), 8):
        if i + 8 <= len(bits):
            byte = ''.join(bits[i:i+8])
            try:
                val = int(byte, 2)
                
                # Score based on character likelihood
                if 65 <= val <= 90:  # Uppercase letters
                    score += 3
                elif 97 <= val <= 122:  # Lowercase letters
                    score += 3
                elif val == 32:  # Space
                    score += 2
                elif 48 <= val <= 57:  # Digits
                    score += 2
                elif 33 <= val <= 126:  # Other printable
                    score += 1
                else:  # Non-printable
                    score -= 2
                    
            except:
                score -= 1
    
    return score

def decode_bits(bits):
    """Decode bits to ASCII string."""
    
    chars = []
    for i in range(0, len(bits), 8):
        if i + 8 <= len(bits):
            byte = ''.join(bits[i:i+8])
            try:
                val = int(byte, 2)
                if 32 <= val <= 126:
                    chars.append(chr(val))
                else:
                    chars.append(f'[{val}]')
            except:
                chars.append('?')
    
    return ''.join(chars)

def extract_full_message(img, row0, col0, row_pitch, col_pitch):
    """Extract the complete message using best configuration."""
    
    print(f"\n=== EXTRACTING FULL MESSAGE ===")
    print(f"Using: origin ({row0}, {col0}), pitch {row_pitch}x{col_pitch}")
    
    # Calculate maximum grid size
    max_rows = (img.shape[0] - row0) // row_pitch
    max_cols = (img.shape[1] - col0) // col_pitch
    
    print(f"Maximum grid: {max_rows} x {max_cols}")
    
    # Extract all bits
    message_rows = []
    
    for r in range(min(max_rows, 50)):  # Limit to reasonable size
        y = row0 + r * row_pitch
        if y >= img.shape[0] - 3:
            break
        
        row_bits = []
        for c in range(min(max_cols, 100)):  # Limit columns too
            x = col0 + c * col_pitch
            if x >= img.shape[1] - 3:
                break
            
            # Sample 5x5 region
            region = img[max(0, y-2):min(img.shape[0], y+3), 
                        max(0, x-2):min(img.shape[1], x+3)]
            
            if region.size > 0:
                val = np.median(region)
                bit = '1' if val > 127 else '0'
                row_bits.append(bit)
        
        message_rows.append(row_bits)
    
    print(f"Extracted: {len(message_rows)} rows")
    
    # Decode each row
    decoded_rows = []
    for i, row_bits in enumerate(message_rows):
        if len(row_bits) >= 8:  # At least one character
            decoded = decode_bits(row_bits)
            decoded_rows.append(decoded)
            
            if i < 10:  # Show first 10 rows
                print(f"Row {i:2d}: {decoded[:40]}")
    
    # Look for readable content
    readable_lines = []
    for i, decoded in enumerate(decoded_rows):
        # Count readable characters
        readable_chars = sum(1 for c in decoded 
                           if c.isalnum() or c.isspace() or c in '.,!?-')
        
        if len(decoded) > 0 and readable_chars / len(decoded) > 0.3:
            readable_lines.append(f"Row {i}: {decoded}")
    
    if readable_lines:
        print(f"\n=== READABLE CONTENT FOUND ===")
        for line in readable_lines[:10]:
            print(line)
    else:
        print(f"\n=== NO CLEARLY READABLE CONTENT ===")
    
    # Save results
    results = {
        'method': 'final_comprehensive_extraction',
        'configuration': {
            'row_pitch': int(row_pitch),
            'col_pitch': int(col_pitch),
            'origin': [int(row0), int(col0)]
        },
        'grid_size': [len(message_rows), len(message_rows[0]) if message_rows else 0],
        'decoded_rows': decoded_rows[:20],  # Limit size
        'readable_lines': readable_lines[:10]
    }
    
    with open('FINAL_EXTRACTION_RESULTS.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to FINAL_EXTRACTION_RESULTS.json")
    
    return results

if __name__ == "__main__":
    print("Final Message Extraction Attempt")
    print("Post-pitch-debate comprehensive approach")
    print("="*60)
    
    try:
        result = final_message_extraction()
        
        if result and result['readable_lines']:
            print(f"\nSUCCESS: Found readable content!")
        else:
            print(f"\nChallenge remains: Grid alignment needs further refinement")
            
    except Exception as e:
        print(f"Error: {e}")