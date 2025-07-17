#!/usr/bin/env python3
"""
Focused pattern search around promising configurations.
Target specific high-value patterns efficiently.
"""

import cv2
import numpy as np
import json

def focused_text_search():
    """Focused search around most promising parameter ranges."""
    
    print("=== FOCUSED TEXT SEARCH ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Focus on most promising ranges based on previous results
    promising_configs = [
        (25, 50),  # From previous "OF" match
        (35, 60),  # From refined analysis
        (31, 53),  # From original verification
        (30, 52),  # Close variation
        (28, 55),  # Alternative
        (20, 45),  # Smaller spacing
    ]
    
    # High-value target patterns
    target_patterns = [
        "On the",
        "The ",
        "This",
        "To a",
        "In a", 
        "For",
        "All",
        "Bitcoin",
        "Satoshi",
        "Block",
        "Genesis",
        "Hash",
        "Proof"
    ]
    
    best_matches = []
    
    print(f"Testing {len(promising_configs)} pitch configurations...")
    
    for row_pitch, col_pitch in promising_configs:
        print(f"\nTesting pitch {row_pitch}x{col_pitch}")
        
        config_matches = []
        
        # Test starting positions in reasonable range
        for start_row in range(5, 25, 1):
            for start_col in range(10, 30, 1):
                
                # Extract enough bits for longest pattern
                max_pattern_len = max(len(p) for p in target_patterns)
                bits_needed = max_pattern_len * 8
                
                bits = []
                positions = []
                
                for i in range(bits_needed):
                    bit_row = i // 8  # 8 bits per character
                    bit_col = i % 8
                    
                    y = start_row + bit_row * row_pitch
                    x = start_col + bit_col * col_pitch
                    
                    if (0 <= y - 3 and y + 3 < img.shape[0] and 
                        0 <= x - 3 and x + 3 < img.shape[1]):
                        
                        region = img[y-3:y+4, x-3:x+4]
                        median_val = np.median(region)
                        
                        # Test multiple thresholds
                        bit = 1 if median_val > 56 else 0  # Use refined threshold
                        bits.append(bit)
                        positions.append((y, x))
                
                if len(bits) >= bits_needed:
                    # Convert to text
                    text = ""
                    for char_idx in range(len(bits) // 8):
                        byte_val = 0
                        for bit_idx in range(8):
                            bit_pos = char_idx * 8 + bit_idx
                            if bit_pos < len(bits):
                                byte_val |= (bits[bit_pos] << (7 - bit_idx))
                        
                        if 32 <= byte_val <= 126:
                            text += chr(byte_val)
                        else:
                            text += '?'
                    
                    # Test against all patterns
                    for pattern in target_patterns:
                        if len(text) >= len(pattern):
                            # Calculate exact match score
                            matches = 0
                            for i in range(len(pattern)):
                                if i < len(text) and text[i].lower() == pattern[i].lower():
                                    matches += 1
                            
                            match_ratio = matches / len(pattern)
                            
                            if match_ratio >= 0.5:  # 50% or better
                                config_matches.append({
                                    'row_pitch': row_pitch,
                                    'col_pitch': col_pitch,
                                    'start_row': start_row,
                                    'start_col': start_col,
                                    'text': text[:len(pattern)*2],  # Show more context
                                    'pattern': pattern,
                                    'match_ratio': match_ratio,
                                    'matches': matches,
                                    'bits': bits[:len(pattern)*8],
                                    'positions': positions[:len(pattern)*8]
                                })
        
        # Show best matches for this configuration
        if config_matches:
            config_matches.sort(key=lambda x: x['match_ratio'], reverse=True)
            print(f"  Found {len(config_matches)} matches, best: '{config_matches[0]['pattern']}' ({config_matches[0]['match_ratio']:.1%})")
            best_matches.extend(config_matches[:3])  # Take top 3 from each config
    
    # Global sort
    best_matches.sort(key=lambda x: x['match_ratio'], reverse=True)
    
    print(f"\n=== TOP MATCHES ACROSS ALL CONFIGURATIONS ===")
    print(f"{'Rank':<4} {'Pitch':<8} {'Origin':<12} {'Pattern':<8} {'Text':<15} {'Match'}")
    print("-" * 70)
    
    for i, match in enumerate(best_matches[:15]):
        pitch_str = f"{match['row_pitch']}x{match['col_pitch']}"
        origin_str = f"({match['start_row']},{match['start_col']})"
        pattern = match['pattern']
        text = match['text'][:12]
        match_pct = match['match_ratio'] * 100
        print(f"{i+1:<4} {pitch_str:<8} {origin_str:<12} {pattern:<8} '{text:<13}' {match_pct:.0f}%")
    
    return best_matches

def extract_complete_message(match):
    """Extract complete message using best match configuration."""
    
    print(f"\n=== EXTRACTING COMPLETE MESSAGE ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    row_pitch = match['row_pitch']
    col_pitch = match['col_pitch']
    start_row = match['start_row']
    start_col = match['start_col']
    
    print(f"Using configuration: {row_pitch}x{col_pitch} at ({start_row}, {start_col})")
    print(f"Pattern matched: '{match['pattern']}' ({match['match_ratio']:.1%})")
    
    # Extract large sequence
    max_chars = 100
    all_bits = []
    all_positions = []
    
    for char_idx in range(max_chars):
        for bit_idx in range(8):
            total_bit_idx = char_idx * 8 + bit_idx
            bit_row = total_bit_idx // 8
            bit_col = total_bit_idx % 8
            
            y = start_row + bit_row * row_pitch
            x = start_col + bit_col * col_pitch
            
            if (0 <= y - 3 and y + 3 < img.shape[0] and 
                0 <= x - 3 and x + 3 < img.shape[1]):
                
                region = img[y-3:y+4, x-3:x+4]
                median_val = np.median(region)
                bit = 1 if median_val > 56 else 0
                
                all_bits.append(bit)
                all_positions.append((y, x, median_val))
            else:
                break
    
    # Convert to text
    full_text = ""
    for char_idx in range(len(all_bits) // 8):
        byte_val = 0
        for bit_idx in range(8):
            bit_pos = char_idx * 8 + bit_idx
            if bit_pos < len(all_bits):
                byte_val |= (all_bits[bit_pos] << (7 - bit_idx))
        
        if 32 <= byte_val <= 126:
            full_text += chr(byte_val)
        else:
            full_text += '?'
    
    # Analyze message quality
    printable_ratio = sum(1 for c in full_text if c != '?') / len(full_text) if full_text else 0
    ones_ratio = sum(all_bits) / len(all_bits) if all_bits else 0
    
    # Extract words
    words = full_text.replace('?', ' ').split()
    valid_words = [w for w in words if len(w) >= 2 and w.isalpha()]
    
    print(f"\nExtracted {len(all_bits)} bits ({len(full_text)} characters)")
    print(f"Full message: '{full_text}'")
    print(f"Printable ratio: {printable_ratio:.1%}")
    print(f"Ones ratio: {ones_ratio:.1%}")
    print(f"Valid words: {valid_words}")
    
    # Look for sentence structure
    sentences = full_text.replace('?', '.').split('.')
    valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    
    if valid_sentences:
        print(f"Potential sentences:")
        for i, sentence in enumerate(valid_sentences[:3]):
            print(f"  {i+1}. '{sentence}'")
    
    return {
        'full_text': full_text,
        'bits': all_bits,
        'positions': all_positions,
        'printable_ratio': printable_ratio,
        'ones_ratio': ones_ratio,
        'valid_words': valid_words,
        'sentences': valid_sentences
    }

def save_focused_search_results():
    """Save focused search results."""
    
    print(f"\n=== SAVING FOCUSED SEARCH RESULTS ===")
    
    # Run focused search
    matches = focused_text_search()
    
    if matches:
        # Extract complete message with best match
        best_match = matches[0]
        complete_message = extract_complete_message(best_match)
    else:
        complete_message = {'full_text': '', 'printable_ratio': 0}
    
    # Compile results
    results = {
        "timestamp": "2025-07-17",
        "analysis_type": "focused_pattern_search",
        "search_summary": {
            "total_matches": len(matches),
            "best_match_ratio": matches[0]['match_ratio'] if matches else 0,
            "best_pattern": matches[0]['pattern'] if matches else "",
            "configurations_tested": 6
        },
        "complete_message": {
            "text": complete_message['full_text'],
            "length": len(complete_message['full_text']),
            "printable_ratio": complete_message['printable_ratio'],
            "valid_words": complete_message.get('valid_words', []),
            "sentences": complete_message.get('sentences', [])
        },
        "top_matches": matches[:5] if matches else [],
        "assessment": "Focused search on most promising configurations for text patterns"
    }
    
    with open('focused_search_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Focused search results saved to focused_search_results.json")
    
    return results

if __name__ == "__main__":
    print("Focused Pattern Search")
    print("Targeting most promising configurations")
    print("=" * 45)
    
    # Run focused search
    matches = focused_text_search()
    
    if matches:
        print(f"\nFound {len(matches)} pattern matches!")
        
        # Extract complete message with best match
        best_match = matches[0]
        complete_message = extract_complete_message(best_match)
        
        # Save results
        results = save_focused_search_results()
        
        print(f"\n" + "=" * 45)
        print("FOCUSED SEARCH COMPLETE")
        print(f"Best match: '{best_match['pattern']}' ({best_match['match_ratio']:.1%})")
        print(f"Message length: {len(complete_message['full_text'])} chars")
        print(f"Quality: {complete_message['printable_ratio']:.1%} printable")
    else:
        print("\nNo pattern matches found")
        print("Grid detection may need fundamental recalibration")