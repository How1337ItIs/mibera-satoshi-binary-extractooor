#!/usr/bin/env python3
"""
Test the promising position (100, 50) with 71.9% accuracy for 'At'.
Fine-tune around this position for breakthrough.
"""

import cv2
import numpy as np

def test_promising_position():
    """Test and refine the (100, 50) position that showed 71.9% accuracy."""
    
    print("=== TESTING PROMISING POSITION (100, 50) ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Base parameters from the promising result
    base_row = 100
    base_col = 50
    base_threshold = 68
    row_pitch = 31
    col_pitch = 53
    patch_size = 5
    
    # Target patterns
    targets = [
        ("At", "01000001011101000010000000100000"),
        ("On", "0100111101101110"),
        ("The", "01010100011010000110010100100000"),
        ("Be", "01000010011001010010000000100000"),
        ("In", "01001001011011100010000000100000")
    ]
    
    best_results = []
    
    # Fine-tune around the promising position
    print(f"Fine-tuning around ({base_row}, {base_col}) with threshold {base_threshold}")
    
    for row_offset in range(-5, 6, 1):
        for col_offset in range(-5, 6, 1):
            for threshold_offset in range(-10, 11, 2):
                
                test_row = base_row + row_offset
                test_col = base_col + col_offset
                test_threshold = base_threshold + threshold_offset
                
                if test_threshold <= 0 or test_threshold >= 255:
                    continue
                
                for target_name, target_bits in targets:
                    # Extract bits at this configuration
                    extracted_bits = []
                    
                    for i in range(len(target_bits)):
                        bit_row = i // 8
                        bit_col = i % 8
                        
                        y = test_row + bit_row * row_pitch
                        x = test_col + bit_col * col_pitch
                        
                        if 0 <= y < img.shape[0] - patch_size and 0 <= x < img.shape[1] - patch_size:
                            half = patch_size // 2
                            patch = img[max(0, y-half):min(img.shape[0], y+half+1), 
                                       max(0, x-half):min(img.shape[1], x+half+1)]
                            if patch.size > 0:
                                val = np.median(patch)
                                bit = '1' if val > test_threshold else '0'
                                extracted_bits.append(bit)
                    
                    if len(extracted_bits) == len(target_bits):
                        extracted_str = ''.join(extracted_bits)
                        matches = sum(1 for i in range(len(target_bits)) if extracted_str[i] == target_bits[i])
                        accuracy = matches / len(target_bits)
                        
                        result = {
                            'position': (test_row, test_col),
                            'threshold': test_threshold,
                            'target': target_name,
                            'accuracy': accuracy,
                            'extracted': extracted_str,
                            'target_bits': target_bits
                        }
                        
                        best_results.append(result)
    
    # Sort and show best results
    best_results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print(f"\n=== FINE-TUNING RESULTS ===")
    breakthrough_found = False
    
    for i, result in enumerate(best_results[:20]):
        accuracy_percent = result['accuracy'] * 100
        print(f"{i+1:2d}. {accuracy_percent:5.1f}% {result['target']:8s} pos=({result['position'][0]:3d},{result['position'][1]:3d}) thresh={result['threshold']:3.0f}")
        
        if result['accuracy'] >= 0.8:
            print(f"    *** BREAKTHROUGH CANDIDATE ***")
            print(f"    Target:    {result['target_bits']}")
            print(f"    Extracted: {result['extracted']}")
            print(f"    Diff:      {''.join('.' if result['extracted'][i] == result['target_bits'][i] else 'X' for i in range(len(result['target_bits'])))}")
            
            if not breakthrough_found:
                # Test full extraction with this configuration
                test_full_message_extraction(img, result)
                breakthrough_found = True
        
        elif result['accuracy'] >= 0.75:
            print(f"    HIGH ACCURACY - showing bits:")
            print(f"    Target:    {result['target_bits'][:32]}...")
            print(f"    Extracted: {result['extracted'][:32]}...")
    
    return best_results

def test_full_message_extraction(img, config):
    """Extract full message using the best configuration."""
    
    print(f"\n=== FULL MESSAGE EXTRACTION ===")
    print(f"Using configuration: {config['target']} at ({config['position'][0]}, {config['position'][1]}) thresh={config['threshold']}")
    
    row0, col0 = config['position']
    threshold = config['threshold']
    row_pitch = 31
    col_pitch = 53
    patch_size = 5
    
    # Extract a larger grid
    max_rows = min(30, (img.shape[0] - row0) // row_pitch)
    max_cols = min(60, (img.shape[1] - col0) // col_pitch)
    
    print(f"Extracting {max_rows} x {max_cols} grid...")
    
    message_lines = []
    
    for r in range(max_rows):
        y = row0 + r * row_pitch
        
        if y >= img.shape[0] - patch_size:
            break
        
        # Extract this row
        row_bits = []
        for c in range(max_cols):
            x = col0 + c * col_pitch
            
            if x >= img.shape[1] - patch_size:
                break
            
            half = patch_size // 2
            patch = img[max(0, y-half):min(img.shape[0], y+half+1), 
                       max(0, x-half):min(img.shape[1], x+half+1)]
            
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
            
            # Show first 15 rows
            if r < 15:
                print(f"Row {r:2d}: {line}")
    
    # Look for meaningful content
    print(f"\n=== CONTENT ANALYSIS ===")
    
    # Check for readable text
    readable_lines = []
    for i, line in enumerate(message_lines):
        if line.strip():
            readable_chars = sum(1 for c in line if c.isalnum() or c.isspace() or c in '.,!?-:()[]{}')
            readability = readable_chars / len(line) if len(line) > 0 else 0
            
            if readability > 0.4 and len(line.strip()) > 2:
                readable_lines.append((i, line, readability))
    
    if readable_lines:
        print(f"Potentially readable content ({len(readable_lines)} lines):")
        for row_num, line, readability in readable_lines[:10]:
            print(f"  Row {row_num:2d} ({readability:.0%}): {line}")
    
    # Search for keywords
    all_text = ' '.join(message_lines).lower()
    keywords = ['bitcoin', 'satoshi', 'nakamoto', 'blockchain', 'crypto', 'message', 'hidden', 'secret', 
               'on the', 'in the', 'at the', 'to the', 'from the', 'genesis', 'block', 'hash']
    
    found_keywords = []
    for keyword in keywords:
        if keyword in all_text:
            found_keywords.append(keyword)
    
    if found_keywords:
        print(f"Keywords found: {found_keywords}")
    
    # Save results
    filename = f'PROMISING_POSITION_EXTRACTION.txt'
    with open(filename, 'w') as f:
        f.write(f"=== PROMISING POSITION EXTRACTION ===\n")
        f.write(f"Configuration: {config['target']} pattern\n")
        f.write(f"Position: ({row0}, {col0})\n")
        f.write(f"Threshold: {threshold}\n")
        f.write(f"Accuracy: {config['accuracy']:.1%}\n")
        f.write(f"Grid: {max_rows} x {max_cols}\n\n")
        
        f.write("Raw extraction:\n")
        for i, line in enumerate(message_lines):
            f.write(f"Row {i:2d}: {line}\n")
        
        if readable_lines:
            f.write(f"\nReadable content:\n")
            for row_num, line, readability in readable_lines:
                f.write(f"Row {row_num:2d} ({readability:.0%}): {line}\n")
        
        if found_keywords:
            f.write(f"\nKeywords found: {found_keywords}\n")
    
    print(f"\nResults saved to {filename}")
    return message_lines

def analyze_bit_patterns():
    """Analyze the bit patterns at the promising position for insights."""
    
    print(f"\n=== BIT PATTERN ANALYSIS ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Use the promising configuration
    row0, col0 = 100, 50
    threshold = 68
    row_pitch = 31
    col_pitch = 53
    patch_size = 5
    
    # Extract a 8x8 grid of raw values and bits
    print(f"8x8 grid at ({row0}, {col0}) with threshold {threshold}:")
    print(f"Position values and bits:")
    
    for r in range(8):
        y = row0 + r * row_pitch
        row_vals = []
        row_bits = []
        
        for c in range(8):
            x = col0 + c * col_pitch
            
            if 0 <= y < img.shape[0] - patch_size and 0 <= x < img.shape[1] - patch_size:
                half = patch_size // 2
                patch = img[max(0, y-half):min(img.shape[0], y+half+1), 
                           max(0, x-half):min(img.shape[1], x+half+1)]
                if patch.size > 0:
                    val = np.median(patch)
                    bit = '1' if val > threshold else '0'
                    row_vals.append(f"{val:3.0f}")
                    row_bits.append(bit)
                else:
                    row_vals.append("---")
                    row_bits.append("-")
            else:
                row_vals.append("---")
                row_bits.append("-")
        
        vals_str = ' '.join(row_vals)
        bits_str = ''.join(row_bits)
        print(f"Row {r}: {vals_str} -> {bits_str}")
    
    print(f"\nThreshold analysis:")
    print(f"Values above {threshold}: represent '1' bits")
    print(f"Values below {threshold}: represent '0' bits")

if __name__ == "__main__":
    print("Testing Promising Position (100, 50)")
    print("Fine-tuning for breakthrough accuracy")
    print("="*60)
    
    # Test the promising position with fine-tuning
    results = test_promising_position()
    
    # Analyze bit patterns
    analyze_bit_patterns()
    
    print("\n" + "="*60)
    
    if results and results[0]['accuracy'] >= 0.8:
        print(f"BREAKTHROUGH: {results[0]['accuracy']:.1%} accuracy achieved!")
        print(f"Configuration: {results[0]['target']} at {results[0]['position']} thresh={results[0]['threshold']}")
    elif results and results[0]['accuracy'] >= 0.75:
        print(f"HIGH ACCURACY: {results[0]['accuracy']:.1%} - very close to breakthrough")
        print(f"Best: {results[0]['target']} pattern")
    else:
        print(f"Best accuracy: {results[0]['accuracy']:.1%} - continue refinement needed")
    
    print("Check PROMISING_POSITION_EXTRACTION.txt for detailed results")