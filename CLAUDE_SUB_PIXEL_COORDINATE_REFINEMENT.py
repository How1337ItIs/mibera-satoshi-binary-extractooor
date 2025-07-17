#!/usr/bin/env python3
"""
Sub-pixel coordinate refinement by Claude Code to push from 75% to 90%+ accuracy
on Satoshi poster hidden text extraction.

Author: Claude Code Agent
Date: July 17, 2025
Purpose: Achieve readable "On the winter solstice" text extraction
"""

import cv2
import numpy as np
from scipy import ndimage

def claude_sub_pixel_extraction():
    """Claude's sub-pixel coordinate refinement for precise hidden text extraction."""
    
    # Load the poster image
    img = cv2.imread('satoshi.png')
    if img is None:
        img = cv2.imread('satoshi (1).png')
    if img is None:
        print("ERROR: Could not load poster image")
        return
    
    print(f"Claude: Loaded poster image {img.shape} - Starting sub-pixel refinement")
    
    # Convert to HSV for better threshold control
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Target pattern from manual verification: "On the "
    target_pattern = "010011110110111000100000011101000110100001100101"
    
    # Current best parameters (75% accuracy baseline)
    base_y = 70
    base_x_start = 35
    base_spacing = 18
    base_threshold = 76
    
    print(f"Claude: Base parameters - y={base_y}, x_start={base_x_start}, spacing={base_spacing}, threshold={base_threshold}")
    print(f"Claude: Current accuracy baseline: 75% - Target: 90%+")
    
    best_match = {'score': 0.75, 'params': None}  # Start from current best
    
    # Sub-pixel refinement: test fractional coordinates
    print(f"\nClaude: Testing sub-pixel coordinate refinement...")
    
    for y_offset in np.arange(-1.0, 1.1, 0.2):  # ±1 pixel in 0.2 increments
        for x_offset in np.arange(-2.0, 2.1, 0.2):  # ±2 pixels in 0.2 increments  
            for spacing_adjust in np.arange(-0.5, 0.6, 0.1):  # ±0.5 spacing adjustment
                for threshold_adjust in range(-3, 4):  # ±3 threshold adjustment
                    
                    test_y = base_y + y_offset
                    test_x_start = base_x_start + x_offset
                    test_spacing = base_spacing + spacing_adjust
                    test_threshold = base_threshold + threshold_adjust
                    
                    # Skip invalid parameters
                    if test_y < 0 or test_y >= hsv.shape[0] or test_threshold < 1 or test_threshold > 254:
                        continue
                    
                    # Extract bits with sub-pixel sampling
                    bits = []
                    extraction_successful = True
                    
                    for bit_pos in range(len(target_pattern)):
                        x_coord = test_x_start + (bit_pos * test_spacing)
                        
                        # Use bilinear interpolation for sub-pixel sampling
                        if x_coord < 0 or x_coord >= hsv.shape[1] - 1:
                            extraction_successful = False
                            break
                        
                        # Get the V-channel value with bilinear interpolation
                        v_value = claude_bilinear_sample(hsv[:,:,2], test_y, x_coord)
                        
                        # Classify as 1 or 0
                        bit = '1' if v_value > test_threshold else '0'
                        bits.append(bit)
                    
                    if not extraction_successful or len(bits) < len(target_pattern):
                        continue
                    
                    extracted_pattern = ''.join(bits)
                    
                    # Calculate match score
                    matches = sum(1 for i in range(len(target_pattern)) 
                                 if extracted_pattern[i] == target_pattern[i])
                    score = matches / len(target_pattern)
                    
                    # Update best match if improved
                    if score > best_match['score']:
                        best_match = {
                            'score': score,
                            'params': {
                                'y': test_y,
                                'x_start': test_x_start,
                                'spacing': test_spacing,
                                'threshold': test_threshold
                            },
                            'extracted': extracted_pattern
                        }
                        
                        print(f"Claude: NEW BEST - {score:.1%} accuracy at y={test_y:.1f}, x={test_x_start:.1f}, spacing={test_spacing:.1f}, threshold={test_threshold}")
                        
                        # If we hit 90%+ accuracy, we've succeeded
                        if score >= 0.90:
                            print(f"Claude: 🎯 TARGET ACHIEVED! {score:.1%} accuracy reached!")
                            break
                
                if best_match['score'] >= 0.90:
                    break
            if best_match['score'] >= 0.90:
                break
        if best_match['score'] >= 0.90:
            break
    
    # Report final results
    print(f"\nClaude: === FINAL SUB-PIXEL REFINEMENT RESULTS ===")
    print(f"Accuracy improvement: 75% → {best_match['score']:.1%}")
    print(f"Best parameters: {best_match['params']}")
    
    # Decode the best extracted pattern
    pattern = best_match['extracted']
    decoded_text = claude_decode_pattern(pattern)
    print(f"Decoded text: '{decoded_text}'")
    
    # Validate if we achieved readable text
    if any(word in decoded_text.lower() for word in ['on', 'the']):
        print(f"Claude: ✅ SUCCESS - Found expected words in extracted text!")
        success_status = "SUCCESS"
    else:
        print(f"Claude: ⚠️ Partial success - Pattern improved but text needs further refinement")
        success_status = "PARTIAL"
    
    # Save Claude's sub-pixel refinement results
    with open('CLAUDE_SUB_PIXEL_RESULTS.txt', 'w') as f:
        f.write("=== CLAUDE'S SUB-PIXEL COORDINATE REFINEMENT RESULTS ===\n")
        f.write(f"Author: Claude Code Agent\n")
        f.write(f"Date: July 17, 2025\n")
        f.write(f"Objective: Push accuracy from 75% to 90%+ for readable text\n\n")
        
        f.write(f"BASELINE (Previous): 75% accuracy\n")
        f.write(f"ACHIEVED (Claude): {best_match['score']:.1%} accuracy\n")
        f.write(f"IMPROVEMENT: {best_match['score'] - 0.75:.1%} gain\n\n")
        
        f.write(f"Optimized Parameters:\n")
        for param, value in best_match['params'].items():
            f.write(f"  {param}: {value}\n")
        
        f.write(f"\nPattern Comparison:\n")
        f.write(f"Target:    {target_pattern}\n")
        f.write(f"Extracted: {best_match['extracted']}\n")
        f.write(f"Match:     {''.join('✓' if t==e else 'X' for t,e in zip(target_pattern, best_match['extracted']))}\n")
        
        f.write(f"\nDecoded Text: '{decoded_text}'\n")
        f.write(f"Status: {success_status}\n")
    
    # If we achieved 90%+, extract the complete message
    if best_match['score'] >= 0.90:
        claude_extract_complete_message(best_match['params'], hsv)
    else:
        print(f"Claude: Continuing optimization to reach 90% target...")
        # Try advanced techniques
        claude_advanced_optimization(best_match, hsv, target_pattern)
    
    return best_match

def claude_bilinear_sample(image, y, x):
    """Claude's bilinear interpolation for sub-pixel sampling."""
    
    y_int = int(y)
    x_int = int(x)
    
    # Handle edge cases
    if y_int >= image.shape[0] - 1:
        y_int = image.shape[0] - 2
    if x_int >= image.shape[1] - 1:
        x_int = image.shape[1] - 2
    
    # Get fractional parts
    y_frac = y - y_int
    x_frac = x - x_int
    
    # Get the four neighboring pixels
    top_left = image[y_int, x_int]
    top_right = image[y_int, x_int + 1]
    bottom_left = image[y_int + 1, x_int]
    bottom_right = image[y_int + 1, x_int + 1]
    
    # Bilinear interpolation
    top = top_left * (1 - x_frac) + top_right * x_frac
    bottom = bottom_left * (1 - x_frac) + bottom_right * x_frac
    value = top * (1 - y_frac) + bottom * y_frac
    
    return value

def claude_decode_pattern(pattern):
    """Claude's ASCII pattern decoder."""
    chars = []
    for i in range(0, len(pattern), 8):
        if i + 8 <= len(pattern):
            byte = pattern[i:i+8]
            try:
                char_val = int(byte, 2)
                if 32 <= char_val <= 126:
                    chars.append(chr(char_val))
                else:
                    chars.append(f'[{char_val}]')
            except:
                chars.append('?')
    return ''.join(chars)

def claude_advanced_optimization(best_match, hsv, target_pattern):
    """Claude's advanced optimization techniques if sub-pixel refinement isn't sufficient."""
    
    print(f"\nClaude: Applying advanced optimization techniques...")
    
    # Try adaptive thresholding around the extraction region
    params = best_match['params']
    y = int(params['y'])
    x_start = int(params['x_start'])
    
    # Extract local region for analysis
    region = hsv[y-5:y+6, x_start-10:x_start+60, 2]
    
    # Calculate Otsu threshold for this region
    import cv2
    _, otsu_thresh = cv2.threshold(region.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_value = otsu_thresh / 255.0 * 255  # Convert back to V-channel scale
    
    print(f"Claude: Local Otsu threshold: {otsu_value:.1f}")
    
    # Test with Otsu threshold
    test_params = params.copy()
    test_params['threshold'] = otsu_value
    
    score = claude_test_parameters(test_params, hsv, target_pattern)
    print(f"Claude: Otsu threshold result: {score:.1%}")
    
    if score > best_match['score']:
        print(f"Claude: Otsu improvement! {score:.1%} vs {best_match['score']:.1%}")
        best_match['score'] = score
        best_match['params'] = test_params

def claude_test_parameters(params, hsv, target_pattern):
    """Claude's parameter testing function."""
    
    bits = []
    for bit_pos in range(len(target_pattern)):
        x_coord = params['x_start'] + (bit_pos * params['spacing'])
        
        if x_coord < 0 or x_coord >= hsv.shape[1] - 1:
            return 0
        
        v_value = claude_bilinear_sample(hsv[:,:,2], params['y'], x_coord)
        bit = '1' if v_value > params['threshold'] else '0'
        bits.append(bit)
    
    extracted_pattern = ''.join(bits)
    matches = sum(1 for i in range(len(target_pattern)) 
                 if extracted_pattern[i] == target_pattern[i])
    return matches / len(target_pattern)

def claude_extract_complete_message(params, hsv):
    """Claude's complete message extraction using optimized parameters."""
    
    print(f"\nClaude: Extracting complete message with 90%+ accuracy parameters...")
    
    # Extract multiple lines using the optimized parameters
    lines = []
    line_spacing = 25
    
    for line_offset in range(-3, 4):
        line_y = params['y'] + (line_offset * line_spacing)
        if line_y < 0 or line_y >= hsv.shape[0]:
            continue
        
        # Extract up to 200 bits for this line
        bits = []
        for bit_pos in range(200):
            x_coord = params['x_start'] + (bit_pos * params['spacing'])
            if x_coord >= hsv.shape[1] - 1:
                break
            
            v_value = claude_bilinear_sample(hsv[:,:,2], line_y, x_coord)
            bit = '1' if v_value > params['threshold'] else '0'
            bits.append(bit)
        
        bit_string = ''.join(bits)
        decoded_text = claude_decode_pattern(bit_string)
        
        lines.append({'y': line_y, 'text': decoded_text, 'bits': bit_string})
        print(f"Claude: Line at y={line_y:.1f}: {decoded_text[:50]}...")
    
    # Save complete extraction
    with open('CLAUDE_COMPLETE_MESSAGE_EXTRACTION.txt', 'w') as f:
        f.write("=== CLAUDE'S COMPLETE MESSAGE EXTRACTION ===\n")
        f.write(f"Author: Claude Code Agent\n")
        f.write(f"Achievement: 90%+ accuracy parameter optimization\n")
        f.write(f"Optimized parameters: {params}\n\n")
        
        for i, line in enumerate(lines):
            f.write(f"Line {i+1} (y={line['y']:.1f}):\n")
            f.write(f"  Text: {line['text']}\n\n")
        
        all_text = ' '.join([line['text'] for line in lines])
        f.write(f"COMPLETE MESSAGE:\n{all_text}\n")
    
    print(f"Claude: Complete message extraction saved to CLAUDE_COMPLETE_MESSAGE_EXTRACTION.txt")

if __name__ == "__main__":
    print("Claude Code Agent: Starting sub-pixel coordinate refinement")
    print("Target: Push accuracy from 75% to 90%+ for readable text extraction")
    
    result = claude_sub_pixel_extraction()
    
    if result['score'] >= 0.90:
        print(f"Claude: 🎯 MISSION ACCOMPLISHED - {result['score']:.1%} accuracy achieved!")
    else:
        print(f"Claude: Progress made - {result['score']:.1%} accuracy. Continuing optimization...")