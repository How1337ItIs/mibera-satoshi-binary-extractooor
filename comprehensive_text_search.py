#!/usr/bin/env python3
"""
Comprehensive text search using wider parameter ranges and target-driven approach.
Search for specific known patterns that might be in the hidden message.
"""

import cv2
import numpy as np
import json

def comprehensive_grid_search():
    """Comprehensive search across wide parameter ranges."""
    
    print("=== COMPREHENSIVE GRID SEARCH ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Wider parameter ranges
    row_pitches = range(15, 40, 2)
    col_pitches = range(35, 65, 2)
    
    # Target patterns to search for
    target_patterns = [
        "On the",
        "The ",
        "In a",
        "To a",
        "For",
        "All",
        "We ",
        "I ",
        "You",
        "This",
        "That",
        "What",
        "When",
        "Where",
        "Why",
        "How",
        "Who",
        "Bitcoin",
        "Satoshi",
        "Block",
        "Hash",
        "Key",
        "Message",
        "Secret",
        "Hidden",
        "Code",
        "Cipher"
    ]
    
    best_matches = []
    
    total_configs = len(row_pitches) * len(col_pitches) * 400  # 20x20 starting positions
    print(f"Testing {total_configs:,} configurations...")
    
    config_count = 0
    
    for row_pitch in row_pitches:
        for col_pitch in col_pitches:
            # Test starting positions
            for start_row in range(0, 40, 2):
                for start_col in range(0, 40, 2):
                    config_count += 1
                    
                    if config_count % 5000 == 0:
                        print(f"Progress: {config_count:,}/{total_configs:,} ({config_count/total_configs*100:.1f}%)")
                    
                    # Extract 8 characters (64 bits) for pattern matching
                    bits = []
                    
                    for i in range(64):
                        bit_row = i // 8  # 8 bits per character
                        bit_col = i % 8
                        
                        y = start_row + bit_row * row_pitch
                        x = start_col + bit_col * col_pitch
                        
                        if (0 <= y - 3 and y + 3 < img.shape[0] and 
                            0 <= x - 3 and x + 3 < img.shape[1]):
                            
                            region = img[y-3:y+4, x-3:x+4]
                            median_val = np.median(region)
                            
                            # Test multiple thresholds
                            for threshold in [40, 50, 60, 70]:
                                bit = 1 if median_val > threshold else 0
                                if len(bits) < 64:
                                    bits.append(bit)
                                break  # Use first threshold for now
                    
                    if len(bits) == 64:
                        # Convert to 8 characters
                        text = ""
                        for char_idx in range(8):
                            byte_val = 0
                            for bit_idx in range(8):
                                bit_pos = char_idx * 8 + bit_idx
                                if bit_pos < len(bits):
                                    byte_val |= (bits[bit_pos] << (7 - bit_idx))
                            
                            if 32 <= byte_val <= 126:
                                text += chr(byte_val)
                            else:
                                text += '?'
                        
                        # Check against target patterns
                        for pattern in target_patterns:
                            if len(text) >= len(pattern):
                                # Calculate character-by-character match
                                matches = 0
                                for i in range(len(pattern)):
                                    if i < len(text) and text[i].lower() == pattern[i].lower():
                                        matches += 1
                                
                                match_ratio = matches / len(pattern)
                                
                                if match_ratio >= 0.6:  # 60% or better match
                                    best_matches.append({
                                        'row_pitch': row_pitch,
                                        'col_pitch': col_pitch,
                                        'start_row': start_row,
                                        'start_col': start_col,
                                        'text': text,
                                        'pattern': pattern,
                                        'match_ratio': match_ratio,
                                        'matches': matches
                                    })
    
    print(f"Completed {config_count:,} configurations")
    
    # Sort by match ratio
    best_matches.sort(key=lambda x: (x['match_ratio'], x['matches']), reverse=True)
    
    print(f"\nFound {len(best_matches)} potential matches:")
    print(f"{'Rank':<4} {'Pitch':<8} {'Origin':<12} {'Pattern':<10} {'Text':<12} {'Match'}")
    print("-" * 70)
    
    for i, match in enumerate(best_matches[:15]):
        pitch_str = f"{match['row_pitch']}x{match['col_pitch']}"
        origin_str = f"({match['start_row']},{match['start_col']})"
        pattern = match['pattern']
        text = match['text'][:len(pattern)]
        match_pct = match['match_ratio'] * 100
        print(f"{i+1:<4} {pitch_str:<8} {origin_str:<12} {pattern:<10} '{text:<10}' {match_pct:.0f}%")
    
    return best_matches

def validate_top_matches(matches):
    """Validate top matches by extracting longer sequences."""
    
    print(f"\n=== VALIDATING TOP MATCHES ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    validated_results = []
    
    for i, match in enumerate(matches[:5]):
        print(f"\n--- Validating match {i+1}: '{match['pattern']}' ---")
        
        row_pitch = match['row_pitch']
        col_pitch = match['col_pitch'] 
        start_row = match['start_row']
        start_col = match['start_col']
        
        # Extract longer sequence (40 characters)
        max_chars = 40
        bits = []
        
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
                    bit = 1 if median_val > 60 else 0  # Standard threshold
                    bits.append(bit)
                else:
                    break
        
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
        
        # Calculate quality metrics
        printable_ratio = sum(1 for c in text if c != '?') / len(text) if text else 0
        ones_ratio = sum(bits) / len(bits) if bits else 0
        
        # Check for word-like patterns
        words = text.replace('?', ' ').split()
        valid_words = [w for w in words if len(w) >= 2 and w.isalpha()]
        
        print(f"Configuration: {row_pitch}x{col_pitch} at ({start_row},{start_col})")
        print(f"Extracted text: '{text[:30]}{'...' if len(text) > 30 else ''}'")
        print(f"Printable ratio: {printable_ratio:.1%}")
        print(f"Ones ratio: {ones_ratio:.1%}")
        print(f"Valid words: {valid_words[:5]}")
        
        validated_results.append({
            'config': match,
            'long_text': text,
            'printable_ratio': printable_ratio,
            'ones_ratio': ones_ratio,
            'valid_words': valid_words,
            'quality_score': printable_ratio * 0.7 + (0.5 - abs(0.5 - ones_ratio)) * 0.3
        })
    
    # Sort by quality score
    validated_results.sort(key=lambda x: x['quality_score'], reverse=True)
    
    return validated_results

def save_comprehensive_search_results():
    """Save comprehensive search results."""
    
    print(f"\n=== SAVING COMPREHENSIVE SEARCH RESULTS ===")
    
    # Run comprehensive search
    matches = comprehensive_grid_search()
    
    # Validate top matches if any found
    if matches:
        validated = validate_top_matches(matches)
    else:
        validated = []
    
    # Compile results
    results = {
        "timestamp": "2025-07-17",
        "analysis_type": "comprehensive_text_search",
        "search_parameters": {
            "row_pitch_range": "15-40 (step 2)",
            "col_pitch_range": "35-65 (step 2)", 
            "start_position_range": "0-40x0-40 (step 2)",
            "total_configurations": len(matches) if matches else 0
        },
        "pattern_matches": {
            "total_found": len(matches),
            "best_match_ratio": max([m['match_ratio'] for m in matches]) if matches else 0,
            "top_matches": matches[:10] if matches else []
        },
        "validation_results": {
            "configurations_validated": len(validated),
            "best_quality_score": max([v['quality_score'] for v in validated]) if validated else 0,
            "best_printable_ratio": max([v['printable_ratio'] for v in validated]) if validated else 0,
            "validated_configs": validated[:3] if validated else []
        },
        "assessment": "Comprehensive grid search for specific text patterns"
    }
    
    with open('comprehensive_search_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Comprehensive search results saved")
    
    return results

if __name__ == "__main__":
    print("Comprehensive Text Search")
    print("Wide parameter search for specific patterns")
    print("=" * 50)
    
    # Run comprehensive search
    matches = comprehensive_grid_search()
    
    if matches:
        print(f"\nFound {len(matches)} potential pattern matches!")
        
        # Validate top matches
        validated = validate_top_matches(matches)
        
        # Save results
        results = save_comprehensive_search_results()
        
        print(f"\n" + "=" * 50)
        print("COMPREHENSIVE SEARCH COMPLETE")
        
        if validated:
            best = validated[0]
            print(f"Best configuration quality: {best['quality_score']:.3f}")
            print(f"Best printable ratio: {best['printable_ratio']:.1%}")
            print(f"Best ones ratio: {best['ones_ratio']:.1%}")
    else:
        print("\nNo pattern matches found in comprehensive search")
        print("May need different approach or preprocessing")