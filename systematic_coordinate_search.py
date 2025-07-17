#!/usr/bin/env python3
"""
Systematic Coordinate Search - Claude Code Agent
Search for exact coordinates of "On the winte" text in poster
"""

import cv2
import numpy as np

def systematic_search():
    """Systematically search poster for the hidden text coordinates"""
    
    # Load poster
    img = cv2.imread('satoshi (1).png')
    if img is None:
        print("ERROR: Could not load poster")
        return
        
    print(f"Poster dimensions: {img.shape}")
    
    # Target pattern from manual reading
    target_bits = '010011110110111000100000011101000110100001100101'  # "On the "
    print(f"Searching for: {target_bits} (\"On the \")")
    
    best_matches = []
    
    # Search parameters
    search_step = 20  # Search every 20 pixels
    spacing_options = [6, 8, 10, 12, 15]  # Different pixel spacings
    threshold_options = [70, 80, 90, 100, 110, 120]
    
    print("Starting systematic search...")
    
    for y in range(50, min(500, img.shape[0]), search_step):
        for x in range(50, min(800, img.shape[1]), search_step):
            for spacing in spacing_options:
                for threshold in threshold_options:
                    
                    # Extract bits at this location
                    extracted_bits = []
                    valid_extraction = True
                    
                    for i in range(len(target_bits)):
                        bit_x = x + i * spacing
                        bit_y = y
                        
                        if bit_x >= img.shape[1] or bit_y >= img.shape[0]:
                            valid_extraction = False
                            break
                            
                        # Sample small region around coordinate
                        sample_region = img[max(0, bit_y-2):bit_y+3, 
                                          max(0, bit_x-2):bit_x+3, 0]  # Blue channel
                        
                        if sample_region.size > 0:
                            blue_mean = np.mean(sample_region)
                            bit = '1' if blue_mean > threshold else '0'
                            extracted_bits.append(bit)
                        else:
                            valid_extraction = False
                            break
                    
                    if valid_extraction and len(extracted_bits) == len(target_bits):
                        # Calculate match percentage
                        matches = sum(1 for i in range(len(target_bits)) 
                                    if extracted_bits[i] == target_bits[i])
                        match_pct = matches / len(target_bits) * 100
                        
                        if match_pct >= 75:  # High accuracy threshold
                            best_matches.append({
                                'x': x,
                                'y': y,
                                'spacing': spacing,
                                'threshold': threshold,
                                'match_pct': match_pct,
                                'extracted': ''.join(extracted_bits)
                            })
    
    # Sort by match percentage
    best_matches.sort(key=lambda m: m['match_pct'], reverse=True)
    
    print(f"\n=== SEARCH RESULTS ===")
    print(f"Found {len(best_matches)} high-quality matches (>75%)")
    
    # Show top matches
    for i, match in enumerate(best_matches[:10]):
        print(f"\nMatch {i+1}: {match['match_pct']:.1f}% accuracy")
        print(f"  Location: ({match['x']}, {match['y']})")
        print(f"  Spacing: {match['spacing']} pixels")
        print(f"  Threshold: {match['threshold']}")
        print(f"  Extracted: {match['extracted']}")
        print(f"  Target:    {target_bits}")
    
    if best_matches:
        return best_matches[0]  # Return best match
    else:
        print("No high-quality matches found")
        return None

def extract_extended_message(best_match, length=200):
    """Extract extended message using best parameters"""
    
    if not best_match:
        print("No parameters available for extended extraction")
        return
        
    img = cv2.imread('satoshi (1).png')
    
    print(f"\n=== EXTENDED MESSAGE EXTRACTION ===")
    print(f"Using parameters: x={best_match['x']}, y={best_match['y']}")
    print(f"Spacing: {best_match['spacing']}, Threshold: {best_match['threshold']}")
    
    # Extract longer sequence
    extended_bits = []
    x_start = best_match['x']
    y_start = best_match['y']
    spacing = best_match['spacing']
    threshold = best_match['threshold']
    
    for i in range(length):
        bit_x = x_start + i * spacing
        bit_y = y_start
        
        if bit_x >= img.shape[1]:
            break
            
        sample_region = img[max(0, bit_y-2):bit_y+3, 
                          max(0, bit_x-2):bit_x+3, 0]
        
        if sample_region.size > 0:
            blue_mean = np.mean(sample_region)
            bit = '1' if blue_mean > threshold else '0'
            extended_bits.append(bit)
    
    # Convert to ASCII
    bit_string = ''.join(extended_bits)
    print(f"Extracted {len(extended_bits)} bits")
    print(f"Bit string: {bit_string[:80]}...")
    
    # Decode as ASCII
    ascii_chars = []
    for i in range(0, len(bit_string), 8):
        if i + 8 <= len(bit_string):
            byte_str = bit_string[i:i+8]
            try:
                byte_val = int(byte_str, 2)
                if 32 <= byte_val <= 126:  # Printable ASCII
                    ascii_chars.append(chr(byte_val))
                else:
                    ascii_chars.append(f'[{byte_val}]')
            except:
                ascii_chars.append('?')
    
    extracted_text = ''.join(ascii_chars)
    print(f"\nExtracted text: \"{extracted_text}\"")
    
    return extracted_text, bit_string

if __name__ == "__main__":
    # Search for coordinates
    best_match = systematic_search()
    
    # Extract extended message if we found a good match
    if best_match:
        extended_text, bit_string = extract_extended_message(best_match)
        
        # Save results
        with open('hidden_message_extraction.txt', 'w') as f:
            f.write(f"=== HIDDEN MESSAGE EXTRACTION RESULTS ===\n")
            f.write(f"Best match coordinates: ({best_match['x']}, {best_match['y']})\n")
            f.write(f"Parameters: spacing={best_match['spacing']}, threshold={best_match['threshold']}\n")
            f.write(f"Accuracy: {best_match['match_pct']:.1f}%\n")
            f.write(f"\nExtracted text: {extended_text}\n")
            f.write(f"\nBit string: {bit_string}\n")
            
        print(f"\nResults saved to: hidden_message_extraction.txt")
    else:
        print("Could not find reliable extraction parameters")