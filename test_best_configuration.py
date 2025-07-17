#!/usr/bin/env python3
"""
Test the best configuration found in alternative analysis.
Extract with optimal settings and analyze for any meaningful patterns.
"""

import cv2
import numpy as np
import json
import hashlib
from collections import Counter
import string

def extract_with_best_config():
    """Extract using the best configuration from alternative analysis."""
    
    print("=== TESTING BEST CONFIGURATION ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Best config from alternative analysis
    row0, col0 = 110, 56
    row_pitch, col_pitch = 30, 52
    threshold = 45  # Optimal threshold found
    
    print(f"Configuration: pos=({row0},{col0}), pitch={row_pitch}x{col_pitch}, threshold={threshold}")
    
    # Extract larger sequence for analysis
    all_bits = []
    positions = []
    values = []
    
    for i in range(1000):  # Extract 1000 bits
        bit_row = i // 8
        bit_col = i % 8
        
        y = row0 + bit_row * row_pitch
        x = col0 + bit_col * col_pitch
        
        if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
            pixel_val = img[y, x]
            bit = 1 if pixel_val > threshold else 0
            
            all_bits.append(bit)
            positions.append((y, x))
            values.append(pixel_val)
    
    print(f"Extracted {len(all_bits)} bits")
    
    # Basic statistics
    ones_count = sum(all_bits)
    ones_ratio = ones_count / len(all_bits)
    
    print(f"Ones ratio: {ones_count}/{len(all_bits)} ({ones_ratio:.1%})")
    
    # Calculate entropy
    if ones_count > 0 and ones_count < len(all_bits):
        p1 = ones_ratio
        p0 = 1 - p1
        entropy = -p1 * np.log2(p1) - p0 * np.log2(p0)
    else:
        entropy = 0
    
    print(f"Bit entropy: {entropy:.3f}")
    
    return all_bits, positions, values

def analyze_bit_patterns(bits):
    """Analyze bit patterns for meaningful structure."""
    
    print("\n=== BIT PATTERN ANALYSIS ===")
    
    bit_string = ''.join(map(str, bits))
    
    # Test various pattern lengths
    pattern_lengths = [8, 16, 32, 64]
    
    for length in pattern_lengths:
        print(f"\n--- {length}-bit patterns ---")
        
        patterns = {}
        for i in range(len(bit_string) - length + 1):
            pattern = bit_string[i:i+length]
            patterns[pattern] = patterns.get(pattern, 0) + 1
        
        # Most common patterns
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
        
        print(f"Unique patterns: {len(patterns)}")
        print(f"Top 5 patterns:")
        
        for i, (pattern, count) in enumerate(sorted_patterns[:5]):
            if length <= 16:
                print(f"  {i+1}. {pattern}: {count} occurrences")
            else:
                print(f"  {i+1}. {pattern[:16]}...: {count} occurrences")

def test_ascii_interpretation(bits):
    """Test ASCII interpretation of bit sequences."""
    
    print("\n=== ASCII INTERPRETATION ===")
    
    if len(bits) < 8:
        print("Insufficient bits for ASCII test")
        return
    
    # Convert to bytes
    byte_data = []
    for i in range(0, len(bits) - 7, 8):
        byte_val = 0
        for j in range(8):
            byte_val |= (bits[i + j] << (7 - j))
        byte_data.append(byte_val)
    
    print(f"Converted to {len(byte_data)} bytes")
    
    # Test different ASCII interpretations
    interpretations = []
    
    # Standard ASCII
    ascii_chars = []
    for byte_val in byte_data[:50]:  # First 50 bytes
        if 32 <= byte_val <= 126:  # Printable ASCII
            ascii_chars.append(chr(byte_val))
        else:
            ascii_chars.append('.')
    
    ascii_text = ''.join(ascii_chars)
    printable_ratio = sum(1 for c in ascii_text if c != '.') / len(ascii_text)
    
    print(f"ASCII interpretation (first 50 bytes):")
    print(f"Text: '{ascii_text}'")
    print(f"Printable ratio: {printable_ratio:.1%}")
    
    interpretations.append(('ASCII', ascii_text, printable_ratio))
    
    # Test reversed bit order
    reversed_bits = []
    for i in range(0, len(bits) - 7, 8):
        byte_bits = bits[i:i+8]
        reversed_byte_bits = byte_bits[::-1]
        byte_val = 0
        for j, bit in enumerate(reversed_byte_bits):
            byte_val |= (bit << j)
        reversed_bits.append(byte_val)
    
    ascii_chars_rev = []
    for byte_val in reversed_bits[:50]:
        if 32 <= byte_val <= 126:
            ascii_chars_rev.append(chr(byte_val))
        else:
            ascii_chars_rev.append('.')
    
    ascii_text_rev = ''.join(ascii_chars_rev)
    printable_ratio_rev = sum(1 for c in ascii_text_rev if c != '.') / len(ascii_text_rev)
    
    print(f"\nReversed bit order (first 50 bytes):")
    print(f"Text: '{ascii_text_rev}'")
    print(f"Printable ratio: {printable_ratio_rev:.1%}")
    
    interpretations.append(('Reversed', ascii_text_rev, printable_ratio_rev))
    
    return interpretations, byte_data

def test_non_ascii_encodings(byte_data):
    """Test non-ASCII encodings and patterns."""
    
    print("\n=== NON-ASCII ENCODING TESTS ===")
    
    # Test hex patterns
    hex_string = bytes(byte_data[:100]).hex()
    print(f"Hex (first 100 bytes): {hex_string[:100]}...")
    
    # Look for hex patterns
    hex_patterns = {
        'null_bytes': hex_string.count('00'),
        'ff_bytes': hex_string.count('ff'),
        'repeated_pairs': sum(1 for i in range(0, len(hex_string)-2, 2) 
                             if hex_string[i:i+2] == hex_string[i+2:i+4])
    }
    
    print(f"Hex patterns:")
    for pattern, count in hex_patterns.items():
        print(f"  {pattern}: {count}")
    
    # Test base64-like patterns
    try:
        import base64
        base64_chars = string.ascii_letters + string.digits + '+/='
        base64_compatible = all(chr(b) in base64_chars for b in byte_data if 32 <= b <= 126)
        print(f"Base64 compatible: {base64_compatible}")
    except:
        pass
    
    # Test for numeric patterns
    numeric_bytes = [b for b in byte_data if 48 <= b <= 57]  # ASCII digits
    if numeric_bytes:
        numeric_string = ''.join(chr(b) for b in numeric_bytes)
        print(f"Numeric sequences found: {numeric_string[:50]}...")
    
    return hex_string

def search_for_known_strings(bits):
    """Search for known string patterns without bias."""
    
    print("\n=== KNOWN STRING SEARCH ===")
    
    # Convert to bytes for string searching
    byte_data = []
    for i in range(0, len(bits) - 7, 8):
        byte_val = 0
        for j in range(8):
            byte_val |= (bits[i + j] << (7 - j))
        byte_data.append(byte_val)
    
    # Test strings (not just Bitcoin-related)
    test_strings = [
        "satoshi", "bitcoin", "nakamoto", "genesis", "block", "hash",
        "the", "and", "for", "this", "that", "with", "have", "will",
        "hello", "world", "test", "message", "hidden", "secret",
        "key", "private", "public", "address", "wallet", "coin",
        "on", "at", "be", "to", "in", "of", "is", "was", "are",
        "digital", "crypto", "cipher", "code", "decode", "puzzle"
    ]
    
    found_strings = []
    
    # Test both normal and reversed bit order
    for reverse in [False, True]:
        if reverse:
            test_bytes = []
            for i in range(0, len(bits) - 7, 8):
                byte_bits = bits[i:i+8]
                reversed_byte_bits = byte_bits[::-1]
                byte_val = 0
                for j, bit in enumerate(reversed_byte_bits):
                    byte_val |= (bit << j)
                test_bytes.append(byte_val)
        else:
            test_bytes = byte_data
        
        # Convert to string (allowing non-printable)
        try:
            full_string = ''.join(chr(b) if 32 <= b <= 126 else '?' for b in test_bytes)
            
            for test_string in test_strings:
                positions = []
                start = 0
                while True:
                    pos = full_string.lower().find(test_string.lower(), start)
                    if pos == -1:
                        break
                    positions.append(pos)
                    start = pos + 1
                
                if positions:
                    found_strings.append({
                        'string': test_string,
                        'positions': positions,
                        'count': len(positions),
                        'reversed': reverse
                    })
        except:
            pass
    
    # Report findings
    if found_strings:
        print("String matches found:")
        for match in sorted(found_strings, key=lambda x: x['count'], reverse=True):
            print(f"  '{match['string']}': {match['count']} occurrences "
                  f"{'(reversed)' if match['reversed'] else ''}")
    else:
        print("No common string patterns found")
    
    return found_strings

def statistical_randomness_tests(bits):
    """Perform statistical tests for randomness."""
    
    print("\n=== STATISTICAL RANDOMNESS TESTS ===")
    
    if len(bits) < 100:
        print("Insufficient data for statistical tests")
        return
    
    # Runs test
    runs = 1
    for i in range(1, len(bits)):
        if bits[i] != bits[i-1]:
            runs += 1
    
    expected_runs = (2 * sum(bits) * (len(bits) - sum(bits))) / len(bits) + 1
    
    print(f"Runs test:")
    print(f"  Observed runs: {runs}")
    print(f"  Expected runs: {expected_runs:.1f}")
    print(f"  Difference: {abs(runs - expected_runs):.1f}")
    
    # Autocorrelation test
    autocorr_1 = 0
    for i in range(len(bits) - 1):
        autocorr_1 += bits[i] * bits[i+1]
    
    expected_autocorr = (len(bits) - 1) * (sum(bits) / len(bits)) ** 2
    
    print(f"Autocorrelation test:")
    print(f"  Lag-1 correlation: {autocorr_1}")
    print(f"  Expected: {expected_autocorr:.1f}")
    
    # Frequency test in blocks
    block_size = 20
    if len(bits) >= block_size * 5:
        block_freqs = []
        for i in range(0, len(bits) - block_size + 1, block_size):
            block = bits[i:i+block_size]
            block_freq = sum(block) / len(block)
            block_freqs.append(block_freq)
        
        block_variance = np.var(block_freqs)
        
        print(f"Block frequency test:")
        print(f"  Block size: {block_size}")
        print(f"  Blocks tested: {len(block_freqs)}")
        print(f"  Frequency variance: {block_variance:.4f}")

def save_comprehensive_analysis():
    """Save comprehensive analysis results."""
    
    print("\n=== SAVING COMPREHENSIVE ANALYSIS ===")
    
    # Run all analyses
    bits, positions, values = extract_with_best_config()
    analyze_bit_patterns(bits)
    ascii_results, byte_data = test_ascii_interpretation(bits)
    hex_data = test_non_ascii_encodings(byte_data)
    string_matches = search_for_known_strings(bits)
    statistical_randomness_tests(bits)
    
    # Compile results
    results = {
        "timestamp": "2024-01-17",
        "analysis_type": "comprehensive_best_config_test",
        "extraction_config": {
            "position": [110, 56],
            "pitch": [30, 52],
            "threshold": 45,
            "bits_extracted": len(bits)
        },
        "basic_statistics": {
            "ones_ratio": sum(bits) / len(bits),
            "entropy": -sum(bits)/len(bits) * np.log2(sum(bits)/len(bits)) - 
                      (1-sum(bits)/len(bits)) * np.log2(1-sum(bits)/len(bits)) 
                      if 0 < sum(bits) < len(bits) else 0
        },
        "ascii_interpretation": {
            "best_printable_ratio": max(result[2] for result in ascii_results),
            "interpretations": [{"type": r[0], "text": r[1][:50], "ratio": r[2]} for r in ascii_results]
        },
        "string_matches": string_matches,
        "hex_preview": hex_data[:100] if hex_data else "",
        "assessment": "Comprehensive analysis of optimal extraction configuration"
    }
    
    with open('comprehensive_best_config_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Comprehensive analysis saved to comprehensive_best_config_analysis.json")
    
    return results

if __name__ == "__main__":
    print("Testing Best Configuration from Alternative Analysis")
    print("Comprehensive extraction and pattern analysis")
    print("="*60)
    
    # Extract with best configuration
    bits, positions, values = extract_with_best_config()
    
    # Analyze bit patterns
    analyze_bit_patterns(bits)
    
    # Test ASCII interpretation
    ascii_results, byte_data = test_ascii_interpretation(bits)
    
    # Test non-ASCII encodings
    hex_data = test_non_ascii_encodings(byte_data)
    
    # Search for known strings
    string_matches = search_for_known_strings(bits)
    
    # Statistical tests
    statistical_randomness_tests(bits)
    
    # Save comprehensive results
    analysis_results = save_comprehensive_analysis()
    
    print("\n" + "="*60)
    print("COMPREHENSIVE ANALYSIS COMPLETE")
    
    print(f"\nKey findings:")
    print(f"Bits extracted: {len(bits)}")
    print(f"Ones ratio: {sum(bits)/len(bits):.1%}")
    if ascii_results:
        best_ascii = max(ascii_results, key=lambda x: x[2])
        print(f"Best ASCII ratio: {best_ascii[2]:.1%} ({best_ascii[0]})")
    if string_matches:
        print(f"String matches: {len(string_matches)} patterns found")
    
    print(f"\nObjective analysis complete - no preconceptions about content type")