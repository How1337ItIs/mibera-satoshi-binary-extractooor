#!/usr/bin/env python3
"""
Test the original image with the 8px hypothesis.
This should finally resolve whether o3's advice was correct.
"""

import cv2
import numpy as np

def test_original_with_8px():
    """Test the original image with 8px column pitch."""
    
    # Load original image
    img = cv2.imread('original_satoshi_poster.png', cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("Could not load original image!")
        return
    
    print(f"Original image loaded: {img.shape}")
    print(f"Value range: {img.min()} to {img.max()}")
    
    # Test o3's exact one-liner
    print(f"\n=== O3's Exact One-Liner Test ===")
    
    mask = img > 200  # bright digits only
    proj = mask.sum(0).astype(float)
    proj -= proj.mean()
    corr = np.correlate(proj, proj, mode='full')[len(proj)-1:]
    peak = np.argmax(corr[5:]) + 5
    
    print(f"Detected column pitch: {peak} pixels")
    
    # Scale check
    scale = img.shape[1] / 4096
    true_pitch_4k = peak / scale if scale > 0 else peak
    
    print(f"Scale factor: {scale:.3f}")
    print(f"True pitch in 4K terms: {true_pitch_4k:.1f} pixels")
    
    if abs(true_pitch_4k - 8) < 2:
        print("✓ O3's hypothesis CONFIRMED on original image!")
    else:
        print(f"✗ O3's hypothesis doesn't match - got {true_pitch_4k:.1f}px, expected ~8px")
    
    # Also test row pitch
    row_mask = img > 200
    row_proj = row_mask.sum(1).astype(float)
    row_proj -= row_proj.mean()
    row_corr = np.correlate(row_proj, row_proj, mode='full')[len(row_proj):]
    row_peak = np.argmax(row_corr[5:]) + 5
    
    print(f"Detected row pitch: {row_peak} pixels")
    
    return peak, row_peak

def extract_with_original_8px():
    """Extract using 8px pitch on original image."""
    
    img = cv2.imread('original_satoshi_poster.png', cv2.IMREAD_GRAYSCALE)
    
    print(f"\n=== EXTRACTION WITH 8PX PITCH ===")
    
    # Use 8px column pitch as o3 suggested
    col_pitch = 8
    row_pitch = 31  # Scale appropriately
    
    # Scale row pitch if needed
    if img.shape[1] != 4096:
        scale = img.shape[1] / 4096
        row_pitch = int(row_pitch * scale)
        print(f"Scaled row pitch to {row_pitch} for {img.shape[1]}px width")
    
    print(f"Using pitches: row={row_pitch}px, col={col_pitch}px")
    
    # Search for best origin
    target_bits = "0100111101101110"  # "On"
    best_score = 0
    best_result = None
    
    print("Searching for optimal origin...")
    
    # Search in reasonable range
    row_range = range(50, 150, 2)
    col_range = range(20, 80, 1)
    
    for row0 in row_range:
        for col0 in col_range:
            # Extract first 16 bits
            bits = extract_bits_robust(img, row0, col0, row_pitch, col_pitch, 16)
            
            if len(bits) == 16:
                bits_str = ''.join(bits)
                
                # Score against "On"
                matches = sum(1 for i in range(16) if bits_str[i] == target_bits[i])
                score = matches / 16
                
                if score > best_score:
                    best_score = score
                    
                    try:
                        char1 = chr(int(bits_str[:8], 2))
                        char2 = chr(int(bits_str[8:16], 2))
                        decoded = f"{char1}{char2}"
                    except:
                        decoded = "??"
                    
                    best_result = {
                        'origin': (row0, col0),
                        'score': score,
                        'bits': bits_str,
                        'decoded': decoded
                    }
                    
                    if score > 0.8:
                        print(f"  Excellent match at ({row0}, {col0}): {score:.1%} -> '{decoded}'")
    
    if best_result:
        print(f"\nBest result:")
        print(f"  Origin: {best_result['origin']}")
        print(f"  Score: {best_result['score']:.1%}")
        print(f"  Bits: {best_result['bits']}")
        print(f"  Decoded: '{best_result['decoded']}'")
        
        if best_result['decoded'] == "On":
            print("🎉 SUCCESS! Found 'On' in original image!")
            
            # Extract more of the message
            extract_full_message_original(img, best_result['origin'], row_pitch, col_pitch)
            
        return best_result
    
    print("Could not find 'On' pattern")
    return None

def extract_bits_robust(img, row0, col0, row_pitch, col_pitch, num_bits):
    """Extract bits with robust 6x6 patch sampling."""
    
    bits = []
    
    for i in range(num_bits):
        bit_row = i // 8
        bit_col = i % 8
        
        y = row0 + bit_row * row_pitch
        x = col0 + bit_col * col_pitch
        
        if y < img.shape[0] - 3 and x < img.shape[1] - 3:
            # Sample 6x6 patch
            patch = img[max(0, y-3):min(img.shape[0], y+4), 
                       max(0, x-3):min(img.shape[1], x+4)]
            
            if patch.size > 0:
                val = np.median(patch)
                bit = '1' if val > 127 else '0'
                bits.append(bit)
    
    return bits

def extract_full_message_original(img, origin, row_pitch, col_pitch):
    """Extract the full message from original image."""
    
    print(f"\n=== EXTRACTING FULL MESSAGE FROM ORIGINAL ===")
    
    row0, col0 = origin
    
    # Calculate maximum grid
    max_rows = min(100, (img.shape[0] - row0) // row_pitch)
    max_cols = min(150, (img.shape[1] - col0) // col_pitch)
    
    print(f"Extracting {max_rows} x {max_cols} grid from original image")
    
    # Extract full grid
    message_rows = []
    
    for r in range(max_rows):
        y = row0 + r * row_pitch
        if y >= img.shape[0] - 3:
            break
        
        row_bits = []
        for c in range(max_cols):
            x = col0 + c * col_pitch
            if x >= img.shape[1] - 3:
                break
            
            # Sample 6x6 patch
            patch = img[max(0, y-3):min(img.shape[0], y+4), 
                       max(0, x-3):min(img.shape[1], x+4)]
            
            if patch.size > 0:
                val = np.median(patch)
                bit = '1' if val > 127 else '0'
                row_bits.append(bit)
        
        message_rows.append(row_bits)
    
    # Decode each row
    print(f"\nDecoded message:")
    full_message = []
    
    for i, row_bits in enumerate(message_rows):
        if len(row_bits) >= 8:
            # Decode row
            decoded_chars = []
            for j in range(0, len(row_bits), 8):
                if j + 8 <= len(row_bits):
                    byte = ''.join(row_bits[j:j+8])
                    try:
                        val = int(byte, 2)
                        if 32 <= val <= 126:
                            decoded_chars.append(chr(val))
                        else:
                            decoded_chars.append(f'[{val}]')
                    except:
                        decoded_chars.append('?')
            
            decoded_row = ''.join(decoded_chars)
            full_message.append(decoded_row)
            
            if i < 20:  # Show first 20 rows
                print(f"Row {i:2d}: {decoded_row}")
    
    # Save the extracted message
    with open('ORIGINAL_EXTRACTED_MESSAGE.txt', 'w') as f:
        f.write("=== EXTRACTED MESSAGE FROM ORIGINAL SATOSHI POSTER ===\n")
        f.write(f"Origin: {origin}\n")
        f.write(f"Pitches: {row_pitch}x{col_pitch}\n")
        f.write(f"Grid: {len(message_rows)} x {len(message_rows[0]) if message_rows else 0}\n\n")
        
        for i, row in enumerate(full_message):
            f.write(f"Row {i:2d}: {row}\n")
    
    print(f"\nFull message saved to ORIGINAL_EXTRACTED_MESSAGE.txt")
    
    # Look for readable content
    readable_lines = []
    for i, row in enumerate(full_message):
        readable_chars = sum(1 for c in row if c.isalnum() or c.isspace() or c in '.,!?-')
        if len(row) > 0 and readable_chars / len(row) > 0.5:
            readable_lines.append(f"Row {i}: {row}")
    
    if readable_lines:
        print(f"\n🎉 READABLE CONTENT FOUND:")
        for line in readable_lines[:10]:
            print(f"  {line}")
    
    return full_message

if __name__ == "__main__":
    print("Testing Original Image with O3's 8px Hypothesis")
    print("="*60)
    
    # Test pitch detection
    col_pitch, row_pitch = test_original_with_8px()
    
    # Test extraction
    result = extract_with_original_8px()
    
    print("\n" + "="*60)
    if result and result['decoded'] == "On":
        print("🎉 BREAKTHROUGH: Found 'On' in original image!")
        print("O3's advice was correct - needed original resolution!")
    elif result:
        print(f"🔍 Partial success: Best result was '{result['decoded']}'")
        print("Continue refinement on original image")
    else:
        print("🔄 Continue iteration - may need different parameters")