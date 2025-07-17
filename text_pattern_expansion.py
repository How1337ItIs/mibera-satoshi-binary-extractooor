#!/usr/bin/env python3
"""
Expand text pattern testing beyond 'On' to find the actual message start.
Test variations around the best configuration to improve accuracy.
"""

import cv2
import numpy as np
import pandas as pd
import json

def test_text_pattern_variations():
    """Test variations around best configuration to find actual text."""
    
    print("=== TESTING TEXT PATTERN VARIATIONS ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Best configuration from previous analysis
    base_row_pitch, base_col_pitch = 25, 50
    base_row, base_col = 13, 18
    
    # Test fine adjustments around best position
    results = []
    
    for row_adj in range(-3, 4):
        for col_adj in range(-3, 4):
            for pitch_row_adj in range(-2, 3):
                for pitch_col_adj in range(-2, 3):
                    
                    row_pitch = base_row_pitch + pitch_row_adj
                    col_pitch = base_col_pitch + pitch_col_adj
                    start_row = base_row + row_adj
                    start_col = base_col + col_adj
                    
                    # Extract first 32 bits (4 characters)
                    bits = []
                    for i in range(32):
                        bit_row = i // 8  # 8 bits per character
                        bit_col = i % 8
                        
                        y = start_row + bit_row * row_pitch
                        x = start_col + bit_col * col_pitch
                        
                        if (0 <= y - 3 and y + 3 < img.shape[0] and 
                            0 <= x - 3 and x + 3 < img.shape[1]):
                            
                            region = img[y-3:y+4, x-3:x+4]
                            median_val = np.median(region)
                            bit = 1 if median_val > 56 else 0  # Use threshold from previous analysis
                            bits.append(bit)
                    
                    if len(bits) == 32:
                        # Convert to 4 characters
                        chars = []
                        for char_idx in range(4):
                            byte_val = 0
                            for bit_idx in range(8):
                                bit_pos = char_idx * 8 + bit_idx
                                byte_val |= (bits[bit_pos] << (7 - bit_idx))
                            
                            if 32 <= byte_val <= 126:
                                chars.append(chr(byte_val))
                            else:
                                chars.append('?')
                        
                        text = ''.join(chars)
                        
                        # Score based on printable characters and common patterns
                        printable_score = sum(1 for c in text if c.isalnum() or c in ' .,!?')
                        
                        # Bonus for common English patterns
                        pattern_score = 0
                        if text.lower().startswith('on '):
                            pattern_score += 10
                        elif text.lower().startswith('the'):
                            pattern_score += 8
                        elif text.lower().startswith('of '):
                            pattern_score += 6
                        elif text.lower().startswith('in '):
                            pattern_score += 6
                        elif text.lower().startswith('to '):
                            pattern_score += 6
                        elif text.lower().startswith('and'):
                            pattern_score += 6
                        elif text.lower().startswith('for'):
                            pattern_score += 6
                        elif text.lower().startswith('are'):
                            pattern_score += 6
                        elif text.lower().startswith('all'):
                            pattern_score += 6
                        
                        total_score = printable_score + pattern_score
                        
                        results.append({
                            'row_pitch': row_pitch,
                            'col_pitch': col_pitch,
                            'start_row': start_row,
                            'start_col': start_col,
                            'text': text,
                            'printable_score': printable_score,
                            'pattern_score': pattern_score,
                            'total_score': total_score
                        })
    
    # Sort by total score
    results.sort(key=lambda x: x['total_score'], reverse=True)
    
    print(f"Tested {len(results)} configurations")
    print(f"\nTop 10 results:")
    print(f"{'Rank':<4} {'Pitch':<8} {'Origin':<12} {'Score':<5} {'Text'}")
    print("-" * 50)
    
    for i, result in enumerate(results[:10]):
        pitch_str = f"{result['row_pitch']}x{result['col_pitch']}"
        origin_str = f"({result['start_row']},{result['start_col']})"
        score = result['total_score']
        text = result['text']
        print(f"{i+1:<4} {pitch_str:<8} {origin_str:<12} {score:<5} '{text}'")
    
    return results

def extract_longer_text(config):
    """Extract longer text sequence using best configuration."""
    
    print(f"\n=== EXTRACTING LONGER TEXT SEQUENCE ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    row_pitch = config['row_pitch']
    col_pitch = config['col_pitch']
    start_row = config['start_row']
    start_col = config['start_col']
    
    print(f"Using configuration: pitch {row_pitch}x{col_pitch}, origin ({start_row}, {start_col})")
    
    # Extract longer sequence - up to 200 characters
    max_chars = 200
    all_bits = []
    
    for char_idx in range(max_chars):
        char_bits = []
        
        for bit_idx in range(8):
            # Calculate position for this bit
            total_bit_idx = char_idx * 8 + bit_idx
            bit_row = total_bit_idx // 16  # 16 bits per row (2 characters)
            bit_col = total_bit_idx % 16
            
            y = start_row + bit_row * row_pitch
            x = start_col + bit_col * col_pitch
            
            if (0 <= y - 3 and y + 3 < img.shape[0] and 
                0 <= x - 3 and x + 3 < img.shape[1]):
                
                region = img[y-3:y+4, x-3:x+4]
                median_val = np.median(region)
                bit = 1 if median_val > 56 else 0
                char_bits.append(bit)
                all_bits.append(bit)
            else:
                break  # Outside image bounds
        
        if len(char_bits) < 8:
            break
    
    print(f"Extracted {len(all_bits)} bits ({len(all_bits)//8} characters)")
    
    # Convert to text
    text_chars = []
    for i in range(0, len(all_bits) - 7, 8):
        byte_val = 0
        for j in range(8):
            byte_val |= (all_bits[i + j] << (7 - j))
        
        if 32 <= byte_val <= 126:
            text_chars.append(chr(byte_val))
        else:
            text_chars.append('?')
    
    full_text = ''.join(text_chars)
    
    # Analyze text quality
    printable_ratio = sum(1 for c in full_text if c != '?') / len(full_text) if full_text else 0
    
    print(f"Full text ({len(full_text)} chars): '{full_text[:100]}{'...' if len(full_text) > 100 else ''}'")
    print(f"Printable ratio: {printable_ratio:.1%}")
    
    # Look for word boundaries and common patterns
    words = full_text.replace('?', ' ').split()
    valid_words = [word for word in words if len(word) >= 2 and word.isalpha()]
    
    print(f"Potential words found: {valid_words[:10]}")
    
    return full_text, all_bits, printable_ratio

def test_alternative_message_starts():
    """Test alternative message starting patterns."""
    
    print(f"\n=== TESTING ALTERNATIVE MESSAGE STARTS ===")
    
    # Common message starting patterns to test
    test_patterns = [
        "The ", "This", "In a", "On t", "At t", "For ", "All ", "We a", "I am", "You ",
        "Dear", "To a", "From", "With", "Here", "Now ", "Then", "When", "What", "Why ",
        "How ", "Who ", "Many", "Some", "Each", "Most", "Both", "Such", "That", "More"
    ]
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Test around the best known configuration
    base_row_pitch, base_col_pitch = 25, 50
    
    best_matches = []
    
    for start_row in range(5, 30, 2):
        for start_col in range(10, 40, 2):
            
            # Extract first 32 bits
            bits = []
            for i in range(32):
                bit_row = i // 8
                bit_col = i % 8
                
                y = start_row + bit_row * base_row_pitch
                x = start_col + bit_col * base_col_pitch
                
                if (0 <= y - 3 and y + 3 < img.shape[0] and 
                    0 <= x - 3 and x + 3 < img.shape[1]):
                    
                    region = img[y-3:y+4, x-3:x+4]
                    median_val = np.median(region)
                    bit = 1 if median_val > 56 else 0
                    bits.append(bit)
            
            if len(bits) == 32:
                # Convert to 4 characters
                text = ""
                for char_idx in range(4):
                    byte_val = 0
                    for bit_idx in range(8):
                        bit_pos = char_idx * 8 + bit_idx
                        byte_val |= (bits[bit_pos] << (7 - bit_idx))
                    
                    if 32 <= byte_val <= 126:
                        text += chr(byte_val)
                    else:
                        text += '?'
                
                # Check against test patterns
                for pattern in test_patterns:
                    if len(text) >= len(pattern):
                        # Calculate similarity
                        matches = sum(1 for i in range(len(pattern)) 
                                    if i < len(text) and text[i].lower() == pattern[i].lower())
                        similarity = matches / len(pattern)
                        
                        if similarity >= 0.75:  # 75% or better match
                            best_matches.append({
                                'position': (start_row, start_col),
                                'text': text,
                                'pattern': pattern,
                                'similarity': similarity,
                                'matches': matches
                            })
    
    # Sort by similarity
    best_matches.sort(key=lambda x: x['similarity'], reverse=True)
    
    print(f"Found {len(best_matches)} potential pattern matches:")
    for i, match in enumerate(best_matches[:10]):
        pos_str = f"({match['position'][0]}, {match['position'][1]})"
        sim_pct = match['similarity'] * 100
        print(f"{i+1:2d}. {pos_str:<10} '{match['text']}' vs '{match['pattern']}' ({sim_pct:.0f}%)")
    
    return best_matches

def save_text_expansion_results():
    """Save comprehensive text pattern expansion results."""
    
    print(f"\n=== SAVING TEXT EXPANSION RESULTS ===")
    
    # Run all analyses
    variations = test_text_pattern_variations()
    alternative_starts = test_alternative_message_starts()
    
    if variations:
        best_config = variations[0]
        full_text, all_bits, printable_ratio = extract_longer_text(best_config)
    else:
        full_text, all_bits, printable_ratio = "", [], 0
        best_config = {}
    
    # Compile results
    results = {
        "timestamp": "2025-07-17",
        "analysis_type": "text_pattern_expansion",
        "fine_tuning": {
            "configurations_tested": len(variations),
            "best_config": best_config,
            "top_scores": [r['total_score'] for r in variations[:5]]
        },
        "text_extraction": {
            "full_text_length": len(full_text),
            "printable_ratio": printable_ratio,
            "preview": full_text[:100] if full_text else "",
            "total_bits": len(all_bits)
        },
        "alternative_patterns": {
            "matches_found": len(alternative_starts),
            "best_matches": alternative_starts[:5] if alternative_starts else []
        },
        "assessment": "Expanded text pattern testing beyond 'On' to find actual message content"
    }
    
    with open('text_expansion_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Text expansion analysis saved to text_expansion_analysis.json")
    
    # Save refined bit dump if we have good results
    if best_config and printable_ratio > 0.15:
        bit_data = []
        for i, bit in enumerate(all_bits):
            char_idx = i // 8
            bit_idx = i % 8
            bit_row = i // 16
            bit_col = i % 16
            
            y = best_config['start_row'] + bit_row * best_config['row_pitch']
            x = best_config['start_col'] + bit_col * best_config['col_pitch']
            
            bit_data.append({
                'bit_index': i,
                'char_index': char_idx,
                'bit_in_char': bit_idx,
                'grid_row': bit_row,
                'grid_col': bit_col,
                'pixel_y': y,
                'pixel_x': x,
                'bit': bit
            })
        
        df = pd.DataFrame(bit_data)
        df.to_csv('text_expansion_bit_dump.csv', index=False)
        print("Saved text_expansion_bit_dump.csv")
    
    return results

if __name__ == "__main__":
    print("Text Pattern Expansion Analysis")
    print("Testing variations around best configuration")
    print("=" * 50)
    
    # Test fine variations around best configuration
    variations = test_text_pattern_variations()
    
    if variations and variations[0]['total_score'] > 4:
        print(f"\nFound improved configuration!")
        
        # Extract longer text with best configuration
        best_config = variations[0]
        full_text, all_bits, printable_ratio = extract_longer_text(best_config)
        
        # Test alternative message starts
        alternative_starts = test_alternative_message_starts()
        
        # Save comprehensive results
        results = save_text_expansion_results()
        
        print(f"\n" + "=" * 50)
        print("TEXT EXPANSION COMPLETE")
        print(f"Best configuration score: {best_config['total_score']}")
        print(f"Text length: {len(full_text)} characters")
        print(f"Printable ratio: {printable_ratio:.1%}")
        
        if alternative_starts:
            print(f"Alternative patterns found: {len(alternative_starts)}")
    else:
        print("\nNo significant improvements found")
        print("Current best remains: 'OF' pattern")