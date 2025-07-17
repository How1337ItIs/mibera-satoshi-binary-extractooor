#!/usr/bin/env python3
"""
Debug the extraction accuracy discrepancy.
75% accuracy against 'At' but extracting all zeros suggests a logic error.
"""

import cv2
import numpy as np

def debug_extraction_logic():
    """Debug why we get 75% accuracy with all-zero extraction."""
    
    print("=== DEBUGGING EXTRACTION LOGIC ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Test position that showed 75% accuracy
    row0, col0 = 58.0, 28.0
    row_pitch = 31
    col_pitch = 53
    threshold = 127
    patch_size = 5
    
    # The "At" pattern in binary
    target_bits = "01000001011101000010000000100000"  # "At  "
    
    print(f"Target 'At': {target_bits}")
    print(f"Position: ({row0}, {col0})")
    print(f"Threshold: {threshold}")
    
    # Extract bits step by step with debugging
    extracted_bits = []
    bit_values = []
    
    for i in range(len(target_bits)):
        bit_row = i // 8
        bit_col = i % 8
        
        y = row0 + bit_row * row_pitch
        x = col0 + bit_col * col_pitch
        
        # Sample patch
        half = patch_size // 2
        y_int = int(y)
        x_int = int(x)
        
        patch_y_start = max(0, y_int - half)
        patch_y_end = min(img.shape[0], y_int + half + 1)
        patch_x_start = max(0, x_int - half) 
        patch_x_end = min(img.shape[1], x_int + half + 1)
        
        patch = img[patch_y_start:patch_y_end, patch_x_start:patch_x_end]
        
        if patch.size > 0:
            val = np.median(patch)
            bit = '1' if val > threshold else '0'
            
            bit_values.append(val)
            extracted_bits.append(bit)
            
            if i < 8:  # Show first 8 bits in detail
                print(f"Bit {i:2d}: pos=({y:5.1f},{x:5.1f}) patch_val={val:3.0f} bit='{bit}' target='{target_bits[i]}'")
    
    extracted_str = ''.join(extracted_bits)
    
    print(f"\nExtracted: {extracted_str}")
    print(f"Target:    {target_bits}")
    
    # Calculate actual matches
    matches = sum(1 for i in range(len(target_bits)) if extracted_str[i] == target_bits[i])
    accuracy = matches / len(target_bits)
    
    print(f"Matches: {matches}/{len(target_bits)} = {accuracy:.1%}")
    
    # Check if all bits are actually zero
    if extracted_str == '0' * len(target_bits):
        print("*** ALL BITS ARE ZERO - this explains the discrepancy ***")
        print("The 75% accuracy is because 'At' has many zeros in it")
        
        # Count zeros in target
        target_zeros = target_bits.count('0')
        print(f"Target 'At' has {target_zeros}/{len(target_bits)} zeros = {target_zeros/len(target_bits):.1%}")
        
        if target_zeros / len(target_bits) == accuracy:
            print("*** CONFIRMED: We're matching zeros only, not actual pattern ***")
    
    # Test different thresholds
    print(f"\n=== THRESHOLD TESTING ===")
    print(f"Patch values: min={min(bit_values):.0f}, max={max(bit_values):.0f}, median={np.median(bit_values):.0f}")
    
    for test_threshold in [50, 100, 127, 150, 200]:
        test_bits = ['1' if val > test_threshold else '0' for val in bit_values]
        test_str = ''.join(test_bits)
        test_matches = sum(1 for i in range(len(target_bits)) if test_str[i] == target_bits[i])
        test_accuracy = test_matches / len(target_bits)
        
        print(f"Threshold {test_threshold:3d}: {test_str[:16]}... -> {test_accuracy:.1%}")
        
        if test_accuracy > 0.8:
            print(f"    *** HIGH ACCURACY with threshold {test_threshold} ***")

def test_inverted_bits():
    """Test if the bits should be inverted (1s and 0s swapped)."""
    
    print(f"\n=== TESTING INVERTED BITS ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    row0, col0 = 58.0, 28.0
    row_pitch = 31
    col_pitch = 53
    patch_size = 5
    
    # Extract raw values first
    bit_values = []
    for i in range(32):  # Test first 32 bits
        bit_row = i // 8
        bit_col = i % 8
        
        y = row0 + bit_row * row_pitch
        x = col0 + bit_col * col_pitch
        
        half = patch_size // 2
        y_int = int(y)
        x_int = int(x)
        
        patch_y_start = max(0, y_int - half)
        patch_y_end = min(img.shape[0], y_int + half + 1)
        patch_x_start = max(0, x_int - half)
        patch_x_end = min(img.shape[1], x_int + half + 1)
        
        patch = img[patch_y_start:patch_y_end, patch_x_start:patch_x_end]
        
        if patch.size > 0:
            val = np.median(patch)
            bit_values.append(val)
    
    # Test both normal and inverted with different thresholds
    targets = [
        ("On", "0100111101101110"),
        ("At", "01000001011101000010000000100000"),
        ("Be", "01000010011001010010000000100000"),
        ("The", "01010100011010000110010100100000")
    ]
    
    for target_name, target_bits in targets:
        print(f"\nTesting '{target_name}':")
        
        best_accuracy = 0
        best_config = None
        
        for threshold in [50, 100, 127, 150, 200]:
            for inverted in [False, True]:
                # Convert values to bits
                if inverted:
                    test_bits = ['0' if val > threshold else '1' for val in bit_values[:len(target_bits)]]
                else:
                    test_bits = ['1' if val > threshold else '0' for val in bit_values[:len(target_bits)]]
                
                test_str = ''.join(test_bits)
                matches = sum(1 for i in range(len(target_bits)) if test_str[i] == target_bits[i])
                accuracy = matches / len(target_bits)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_config = (threshold, inverted, test_str)
        
        if best_config:
            threshold, inverted, test_str = best_config
            print(f"  Best: {best_accuracy:.1%} (threshold={threshold}, inverted={inverted})")
            print(f"  Target: {target_bits}")
            print(f"  Found:  {test_str}")
            
            if best_accuracy > 0.8:
                print(f"    *** BREAKTHROUGH: {target_name} at {best_accuracy:.1%} ***")
                return target_name, best_config

def test_alternative_bit_orders():
    """Test column-major vs row-major bit ordering."""
    
    print(f"\n=== TESTING ALTERNATIVE BIT ORDERS ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    row0, col0 = 58.0, 28.0
    row_pitch = 31
    col_pitch = 53
    threshold = 127
    patch_size = 5
    
    # Extract a 4x4 grid of bits (for 2 characters)
    grid_bits = []
    for r in range(2):  # 2 character rows
        for c in range(8):  # 8 bits per character
            y = row0 + r * row_pitch
            x = col0 + c * col_pitch
            
            half = patch_size // 2
            y_int = int(y)
            x_int = int(x)
            
            patch_y_start = max(0, y_int - half)
            patch_y_end = min(img.shape[0], y_int + half + 1)
            patch_x_start = max(0, x_int - half)
            patch_x_end = min(img.shape[1], x_int + half + 1)
            
            patch = img[patch_y_start:patch_y_end, patch_x_start:patch_x_end]
            
            if patch.size > 0:
                val = np.median(patch)
                bit = '1' if val > threshold else '0'
                grid_bits.append((r, c, bit, val))
    
    # Test different orderings
    orderings = [
        ("row-major", lambda bits: [b[2] for b in sorted(bits, key=lambda x: (x[0], x[1]))]),
        ("col-major", lambda bits: [b[2] for b in sorted(bits, key=lambda x: (x[1], x[0]))]),
        ("row-major-inverted", lambda bits: ['0' if b[2]=='1' else '1' for b in sorted(bits, key=lambda x: (x[0], x[1]))]),
        ("col-major-inverted", lambda bits: ['0' if b[2]=='1' else '1' for b in sorted(bits, key=lambda x: (x[1], x[0]))])
    ]
    
    target_patterns = [
        ("On", "0100111101101110"),
        ("At", "01000001011101000010000000100000"[:16]),  # First 16 bits
        ("Be", "01000010011001010010000000100000"[:16])   # First 16 bits
    ]
    
    for order_name, order_func in orderings:
        ordered_bits = order_func(grid_bits)
        ordered_str = ''.join(ordered_bits)
        
        print(f"\n{order_name}: {ordered_str}")
        
        for target_name, target_bits in target_patterns:
            if len(ordered_str) >= len(target_bits):
                test_str = ordered_str[:len(target_bits)]
                matches = sum(1 for i in range(len(target_bits)) if test_str[i] == target_bits[i])
                accuracy = matches / len(target_bits)
                
                print(f"  vs {target_name}: {accuracy:.1%}")
                
                if accuracy > 0.8:
                    print(f"    *** BREAKTHROUGH: {target_name} with {order_name} ***")
                    return order_name, target_name, test_str

if __name__ == "__main__":
    print("Debug Extraction Accuracy Discrepancy")
    print("Investigating 75% accuracy with all-zero extraction")
    print("="*60)
    
    # Debug the basic extraction logic
    debug_extraction_logic()
    
    # Test inverted bits
    inverted_result = test_inverted_bits()
    
    # Test alternative bit orders
    order_result = test_alternative_bit_orders()
    
    print("\n" + "="*60)
    print("DEBUGGING COMPLETE")
    
    if inverted_result:
        print(f"Inverted bits breakthrough: {inverted_result[0]}")
    
    if order_result:
        print(f"Alternative ordering breakthrough: {order_result[1]} with {order_result[0]}")
    
    print("Results show the nature of the extraction issue")