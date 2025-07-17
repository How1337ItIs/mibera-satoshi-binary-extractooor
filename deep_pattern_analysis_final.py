#!/usr/bin/env python3
"""
Deep pattern analysis based on cryptographic findings.
Focus on the 56.2% entropy and LSB randomness patterns.
"""

import cv2
import numpy as np
from collections import Counter
import json

def analyze_entropy_distribution():
    """Analyze entropy distribution across different image regions."""
    
    print("=== ENTROPY DISTRIBUTION ANALYSIS ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Test entropy across different regions
    regions = [
        ("Breakthrough", 101, 53),
        ("Upper left", 50, 30),
        ("Center", 200, 200),
        ("Lower right", 300, 400),
        ("Random sample", 150, 150)
    ]
    
    entropy_results = []
    
    for region_name, row0, col0 in regions:
        print(f"\n--- {region_name} region ({row0}, {col0}) ---")
        
        # Extract 200 bits from this region
        region_bits = []
        patch_size = 5
        
        for r in range(25):  # 25 rows
            for c in range(8):   # 8 columns
                y = row0 + r * 31
                x = col0 + c * 53
                
                if 0 <= y < img.shape[0] - patch_size and 0 <= x < img.shape[1] - patch_size:
                    half = patch_size // 2
                    patch = img[max(0, y-half):min(img.shape[0], y+half+1), 
                               max(0, x-half):min(img.shape[1], x+half+1)]
                    
                    if patch.size > 0:
                        val = np.median(patch)
                        bit_val = 1 if val > 72 else 0
                        region_bits.append(bit_val)
        
        # Convert bits to bytes
        bytes_data = []
        for i in range(0, len(region_bits) - 7, 8):
            byte_val = 0
            for j in range(8):
                byte_val |= (region_bits[i+j] << (7-j))
            bytes_data.append(byte_val)
        
        # Calculate entropy
        if len(bytes_data) > 0:
            freq = Counter(bytes_data)
            entropy = 0
            for count in freq.values():
                prob = count / len(bytes_data)
                if prob > 0:
                    entropy -= prob * np.log2(prob)
            
            ones_count = sum(region_bits)
            bit_entropy = 0
            if ones_count > 0 and ones_count < len(region_bits):
                p1 = ones_count / len(region_bits)
                p0 = 1 - p1
                bit_entropy = -p1 * np.log2(p1) - p0 * np.log2(p0)
            
            print(f"  Bytes extracted: {len(bytes_data)}")
            print(f"  Byte entropy: {entropy:.3f}")
            print(f"  Bit entropy: {bit_entropy:.3f}")
            print(f"  Ones ratio: {ones_count}/{len(region_bits)} ({ones_count/len(region_bits):.1%})")
            
            entropy_results.append({
                'region': region_name,
                'position': (row0, col0),
                'byte_entropy': entropy,
                'bit_entropy': bit_entropy,
                'ones_ratio': ones_count/len(region_bits) if region_bits else 0,
                'sample_size': len(bytes_data)
            })
    
    # Find region with highest entropy
    if entropy_results:
        best_entropy = max(entropy_results, key=lambda x: x['byte_entropy'])
        print(f"\nHighest entropy region: {best_entropy['region']} ({best_entropy['byte_entropy']:.3f})")
    
    return entropy_results

def advanced_template_matching():
    """Advanced template matching using known Bitcoin/Satoshi patterns."""
    
    print(f"\n=== ADVANCED TEMPLATE MATCHING ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Known Bitcoin-related binary patterns
    known_patterns = {
        "bitcoin_b": "01100010",  # 'b' in binary
        "satoshi_s": "01110011",  # 's' in binary
        "nakamoto_n": "01101110", # 'n' in binary
        "timestamp": "01010100",  # 'T' for timestamp
        "version_1": "00000001",  # Version 1
        "difficulty": "11111111", # Max difficulty marker
        "genesis_start": "00000000", # Common start
        "message_m": "01101101",  # 'm' in binary
    }
    
    # Test multiple positions and thresholds
    test_configs = [
        (101, 53, 72),   # Breakthrough config
        (103, 55, 75),   # Variation 1
        (99, 51, 69),    # Variation 2
        (101, 53, 80),   # Higher threshold
        (101, 53, 64),   # Lower threshold
    ]
    
    best_matches = []
    
    for config_idx, (row0, col0, threshold) in enumerate(test_configs):
        print(f"\n--- Config {config_idx+1}: ({row0}, {col0}) thresh={threshold} ---")
        
        # Extract longer bit sequence
        extracted_bits = []
        for r in range(20):
            for c in range(32):  # 4 bytes per row
                y = row0 + r * 31
                x = col0 + c * 53
                
                if 0 <= y < img.shape[0] - 5 and 0 <= x < img.shape[1] - 5:
                    patch = img[max(0, y-2):min(img.shape[0], y+3), 
                               max(0, x-2):min(img.shape[1], x+3)]
                    
                    if patch.size > 0:
                        val = np.median(patch)
                        bit = '1' if val > threshold else '0'
                        extracted_bits.append(bit)
        
        extracted_str = ''.join(extracted_bits)
        
        # Test against known patterns
        for pattern_name, pattern in known_patterns.items():
            best_match_score = 0
            best_position = -1
            
            # Sliding window search
            for start in range(len(extracted_str) - len(pattern) + 1):
                test_segment = extracted_str[start:start + len(pattern)]
                matches = sum(1 for i in range(len(pattern)) if test_segment[i] == pattern[i])
                score = matches / len(pattern)
                
                if score > best_match_score:
                    best_match_score = score
                    best_position = start
            
            if best_match_score > 0.75:  # High confidence threshold
                print(f"  HIGH MATCH: {pattern_name} at position {best_position} ({best_match_score:.1%})")
                best_matches.append({
                    'pattern': pattern_name,
                    'score': best_match_score,
                    'position': best_position,
                    'config': (row0, col0, threshold),
                    'extracted': extracted_str[best_position:best_position + len(pattern)],
                    'target': pattern
                })
            elif best_match_score > 0.65:  # Medium confidence
                print(f"  GOOD MATCH: {pattern_name} ({best_match_score:.1%})")
    
    # Summary of best matches
    if best_matches:
        best_matches.sort(key=lambda x: x['score'], reverse=True)
        print(f"\n=== BEST PATTERN MATCHES ===")
        for match in best_matches[:5]:
            print(f"{match['score']:.1%} {match['pattern']} at config {match['config']}")
            print(f"  Target:    {match['target']}")
            print(f"  Extracted: {match['extracted']}")
    
    return best_matches

def comprehensive_bit_analysis():
    """Comprehensive analysis of bit patterns and structure."""
    
    print(f"\n=== COMPREHENSIVE BIT ANALYSIS ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Extract large bit grid from breakthrough position
    row0, col0 = 101, 53
    threshold = 72
    
    bit_grid = []
    
    for r in range(30):
        bit_row = []
        
        for c in range(40):
            y = row0 + r * 31
            x = col0 + c * 53
            
            if 0 <= y < img.shape[0] - 5 and 0 <= x < img.shape[1] - 5:
                patch = img[max(0, y-2):min(img.shape[0], y+3), 
                           max(0, x-2):min(img.shape[1], x+3)]
                
                if patch.size > 0:
                    val = np.median(patch)
                    bit = 1 if val > threshold else 0
                    bit_row.append(bit)
                else:
                    bit_row.append(0)
            else:
                bit_row.append(0)
        
        bit_grid.append(bit_row)
    
    bit_array = np.array(bit_grid)
    
    print(f"Extracted {bit_array.shape[0]}x{bit_array.shape[1]} comprehensive bit grid")
    
    # Structural analysis
    print(f"\n--- Structural Analysis ---")
    
    # Column-wise analysis
    col_entropies = []
    for c in range(bit_array.shape[1]):
        col_bits = bit_array[:, c]
        ones = np.sum(col_bits)
        if ones > 0 and ones < len(col_bits):
            p1 = ones / len(col_bits)
            p0 = 1 - p1
            entropy = -p1 * np.log2(p1) - p0 * np.log2(p0)
        else:
            entropy = 0
        col_entropies.append(entropy)
    
    # Row-wise analysis  
    row_entropies = []
    for r in range(bit_array.shape[0]):
        row_bits = bit_array[r, :]
        ones = np.sum(row_bits)
        if ones > 0 and ones < len(row_bits):
            p1 = ones / len(row_bits)
            p0 = 1 - p1
            entropy = -p1 * np.log2(p1) - p0 * np.log2(p0)
        else:
            entropy = 0
        row_entropies.append(entropy)
    
    print(f"Average column entropy: {np.mean(col_entropies):.3f}")
    print(f"Average row entropy: {np.mean(row_entropies):.3f}")
    
    # Find most structured regions
    high_entropy_cols = [i for i, e in enumerate(col_entropies) if e > 0.8]
    high_entropy_rows = [i for i, e in enumerate(row_entropies) if e > 0.8]
    
    print(f"High entropy columns: {high_entropy_cols}")
    print(f"High entropy rows: {high_entropy_rows}")
    
    # Save comprehensive analysis
    analysis_data = {
        'grid_shape': [int(x) for x in bit_array.shape],
        'total_ones': int(np.sum(bit_array)),
        'total_bits': int(bit_array.size),
        'overall_entropy': float(np.mean(col_entropies + row_entropies)),
        'column_entropies': [float(e) for e in col_entropies],
        'row_entropies': [float(e) for e in row_entropies],
        'high_entropy_regions': {
            'columns': high_entropy_cols,
            'rows': high_entropy_rows
        }
    }
    
    with open('comprehensive_bit_analysis.json', 'w') as f:
        json.dump(analysis_data, f, indent=2)
    
    print(f"\nComprehensive analysis saved to comprehensive_bit_analysis.json")
    
    return bit_array, analysis_data

def final_extraction_attempt():
    """Final attempt using all insights from deep analysis."""
    
    print(f"\n=== FINAL EXTRACTION ATTEMPT ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Use all insights from analysis
    configs = [
        ("Breakthrough optimized", 101, 53, 72),
        ("High entropy region", 103, 55, 75),
        ("Alternative threshold", 101, 53, 80),
        ("Lower threshold", 101, 53, 64),
        ("Shifted position", 99, 51, 72)
    ]
    
    extraction_results = []
    
    for config_name, row0, col0, threshold in configs:
        print(f"\n--- {config_name} ---")
        
        # Extract message with this configuration
        message_lines = []
        for r in range(10):  # First 10 rows
            row_bits = []
            for c in range(32):  # 4 bytes per row
                y = row0 + r * 31
                x = col0 + c * 53
                
                if 0 <= y < img.shape[0] - 5 and 0 <= x < img.shape[1] - 5:
                    patch = img[max(0, y-2):min(img.shape[0], y+3), 
                               max(0, x-2):min(img.shape[1], x+3)]
                    
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
                            elif val == 0:
                                decoded_chars.append(' ')
                            else:
                                decoded_chars.append(f'[{val}]')
                        except:
                            decoded_chars.append('?')
                
                line = ''.join(decoded_chars)
                message_lines.append(line)
                
                if r < 5:  # Show first 5 rows
                    print(f"  Row {r}: {line}")
        
        # Analyze readability
        readable_chars = 0
        total_chars = 0
        for line in message_lines:
            for char in line:
                total_chars += 1
                if char.isalnum() or char.isspace() or char in '.,!?-':
                    readable_chars += 1
        
        readability = readable_chars / total_chars if total_chars > 0 else 0
        
        # Look for keywords
        all_text = ' '.join(message_lines).lower()
        keywords = ['bitcoin', 'satoshi', 'nakamoto', 'on the', 'at the', 'in the', 'message', 'genesis', 'block']
        found_keywords = [kw for kw in keywords if kw in all_text]
        
        score = readability * 100 + len(found_keywords) * 10
        
        extraction_results.append({
            'config': config_name,
            'readability': readability,
            'keywords': found_keywords,
            'score': score,
            'first_line': message_lines[0] if message_lines else '',
            'total_lines': len(message_lines)
        })
        
        print(f"  Readability: {readability:.1%}")
        if found_keywords:
            print(f"  Keywords: {found_keywords}")
        print(f"  Score: {score:.1f}")
    
    # Show best results
    extraction_results.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\n=== FINAL EXTRACTION RANKING ===")
    for i, result in enumerate(extraction_results):
        print(f"{i+1}. {result['config']:25s} Score: {result['score']:5.1f} "
              f"(readability: {result['readability']:.1%})")
        if result['keywords']:
            print(f"   Keywords: {result['keywords']}")
        if result['first_line'].strip():
            print(f"   First line: {result['first_line'][:50]}...")
    
    return extraction_results

if __name__ == "__main__":
    print("Deep Pattern Analysis - Final Phase")
    print("Advanced entropy, template matching, and final extraction")
    print("="*70)
    
    # Entropy distribution analysis
    entropy_results = analyze_entropy_distribution()
    
    # Advanced template matching
    template_matches = advanced_template_matching()
    
    # Comprehensive bit analysis
    bit_grid, structural_analysis = comprehensive_bit_analysis()
    
    # Final extraction attempt
    final_results = final_extraction_attempt()
    
    print("\n" + "="*70)
    print("DEEP PATTERN ANALYSIS COMPLETE")
    
    # Summary findings
    if entropy_results:
        best_entropy = max(entropy_results, key=lambda x: x['byte_entropy'])
        print(f"Highest entropy region: {best_entropy['region']} ({best_entropy['byte_entropy']:.3f})")
    
    if template_matches:
        best_template = max(template_matches, key=lambda x: x['score'])
        print(f"Best template match: {best_template['pattern']} ({best_template['score']:.1%})")
    
    if final_results:
        best_final = final_results[0]
        print(f"Best final extraction: {best_final['config']} (score: {best_final['score']:.1f})")
    
    total_high_entropy = len(structural_analysis['high_entropy_regions']['columns']) + len(structural_analysis['high_entropy_regions']['rows'])
    print(f"High entropy regions found: {total_high_entropy}")
    
    print("\nDeep analysis complete - comprehensive extraction framework established")