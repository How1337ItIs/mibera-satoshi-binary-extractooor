#!/usr/bin/env python3
"""
Test extraction on the cropped top row image to verify our parameters work there,
then map those successful parameters to the full poster.
"""

import cv2
import numpy as np

def test_cropped_extraction():
    """Test extraction on cropped image to confirm working parameters."""
    
    # Load the cropped top row image
    img = cv2.imread('top row.png')
    if img is None:
        print("ERROR: Could not load top row.png")
        return
    
    print(f"Loaded cropped image: {img.shape}")
    
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Test different parameter combinations to find the best one
    test_params = [
        {'spacing': 17, 'threshold': 85, 'start_y': 20},
        {'spacing': 18, 'threshold': 80, 'start_y': 22},
        {'spacing': 18, 'threshold': 85, 'start_y': 25},
        {'spacing': 19, 'threshold': 80, 'start_y': 20},
        {'spacing': 16, 'threshold': 75, 'start_y': 25},
    ]
    
    best_result = None
    best_score = 0
    
    # Expected text from manual reading
    target_text = "On the winter"
    
    for params in test_params:
        spacing = params['spacing']
        threshold = params['threshold']
        start_y = params['start_y']
        
        print(f"\nTesting spacing={spacing}, threshold={threshold}, start_y={start_y}")
        
        # Extract bits from the line
        bits = []
        for bit_pos in range(80):  # Extract enough bits for the expected text
            x = 10 + (bit_pos * spacing)  # Start at x=10
            if x >= img.shape[1]:
                break
                
            # Sample the V channel value
            v_value = hsv[start_y, x, 2]
            
            # Classify as 1 or 0
            bit = '1' if v_value > threshold else '0'
            bits.append(bit)
        
        bit_string = ''.join(bits)
        
        # Decode as ASCII
        chars = []
        for i in range(0, len(bit_string), 8):
            if i + 8 <= len(bit_string):
                byte = bit_string[i:i+8]
                try:
                    char_val = int(byte, 2)
                    if 32 <= char_val <= 126:  # Printable ASCII
                        chars.append(chr(char_val))
                    else:
                        chars.append(f'[{char_val}]')
                except:
                    chars.append('?')
        
        text = ''.join(chars)
        
        # Score based on how many target characters we find
        score = 0
        for target_char in target_text.lower():
            if target_char in text.lower():
                score += 1
        
        print(f"  Decoded: {text[:40]}...")
        print(f"  Score: {score}/{len(target_text)}")
        
        if score > best_score:
            best_score = score
            best_result = {
                'params': params,
                'text': text,
                'bits': bit_string,
                'score': score
            }
    
    print(f"\n=== BEST RESULT ===")
    if best_result:
        print(f"Parameters: {best_result['params']}")
        print(f"Text: {best_result['text']}")
        print(f"Score: {best_result['score']}")
        print(f"Bits: {best_result['bits'][:64]}...")
        
        # Save the best result
        with open('cropped_calibration_results.txt', 'w') as f:
            f.write("=== CROPPED IMAGE CALIBRATION RESULTS ===\n")
            f.write(f"Best parameters: {best_result['params']}\n")
            f.write(f"Extracted text: {best_result['text']}\n")
            f.write(f"Score: {best_result['score']}/{len(target_text)}\n")
            f.write(f"Raw bits: {best_result['bits']}\n")
        
        return best_result['params']
    else:
        print("No successful extraction found")
        return None

def apply_to_full_poster(params):
    """Apply successful parameters to the full poster."""
    if not params:
        print("No parameters to apply")
        return
    
    # Load the poster image
    img = cv2.imread('satoshi.png')
    if img is None:
        img = cv2.imread('satoshi (1).png')
    if img is None:
        print("ERROR: Could not load poster image")
        return
    
    print(f"\nApplying to full poster: {img.shape}")
    
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    spacing = params['spacing']
    threshold = params['threshold']
    
    # The cropped image was from the top, so map coordinates
    # Try different y positions in the poster top region
    for poster_y in range(20, 60, 5):  # Test y positions 20, 25, 30, 35, 40, 45, 50, 55
        print(f"\nTesting poster y={poster_y}")
        
        # Extract bits
        bits = []
        for bit_pos in range(100):
            x = 15 + (bit_pos * spacing)
            if x >= img.shape[1]:
                break
                
            v_value = hsv[poster_y, x, 2]
            bit = '1' if v_value > threshold else '0'
            bits.append(bit)
        
        bit_string = ''.join(bits)
        
        # Decode as ASCII
        chars = []
        for i in range(0, len(bit_string), 8):
            if i + 8 <= len(bit_string):
                byte = bit_string[i:i+8]
                try:
                    char_val = int(byte, 2)
                    if 32 <= char_val <= 126:
                        chars.append(chr(char_val))
                    else:
                        chars.append(f'[{char_val}]')
                except:
                    chars.append('?')
        
        text = ''.join(chars)
        
        # Check for target text
        if any(word in text.lower() for word in ['on', 'the', 'winter']):
            print(f"  ✅ FOUND TARGET TEXT: {text}")
        else:
            print(f"  Text: {text[:40]}...")
    
    print("\nFull poster test complete")

if __name__ == "__main__":
    # First test on cropped image
    best_params = test_cropped_extraction()
    
    # Then apply to full poster
    apply_to_full_poster(best_params)