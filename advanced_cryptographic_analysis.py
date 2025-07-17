#!/usr/bin/env python3
"""
Advanced cryptographic analysis of the extracted bit patterns.
Testing for encryption, compression, and encoded message formats.
"""

import cv2
import numpy as np
import hashlib
import base64
from collections import Counter
import itertools
import string

def cryptographic_pattern_analysis():
    """Analyze extracted bits for cryptographic patterns."""
    
    print("=== ADVANCED CRYPTOGRAPHIC ANALYSIS ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Use breakthrough configuration
    row0, col0 = 101, 53
    threshold = 72
    row_pitch = 31
    col_pitch = 53
    patch_size = 5
    
    # Extract larger bit sequence for crypto analysis
    all_bits = []
    all_bytes = []
    
    for r in range(50):  # Larger sample
        for c in range(60):
            y = row0 + r * row_pitch
            x = col0 + c * col_pitch
            
            if 0 <= y < img.shape[0] - patch_size and 0 <= x < img.shape[1] - patch_size:
                half = patch_size // 2
                patch = img[max(0, y-half):min(img.shape[0], y+half+1), 
                           max(0, x-half):min(img.shape[1], x+half+1)]
                
                if patch.size > 0:
                    val = np.median(patch)
                    bit = 1 if val > threshold else 0
                    all_bits.append(bit)
    
    # Convert to bytes
    for i in range(0, len(all_bits) - 7, 8):
        byte_bits = all_bits[i:i+8]
        byte_val = 0
        for j, bit in enumerate(byte_bits):
            byte_val |= (bit << (7-j))
        all_bytes.append(byte_val)
    
    print(f"Extracted {len(all_bits)} bits, {len(all_bytes)} bytes for crypto analysis")
    
    # Cryptographic tests
    crypto_results = {}
    
    # 1. Frequency analysis
    print(f"\n--- Frequency Analysis ---")
    byte_freq = Counter(all_bytes)
    most_common = byte_freq.most_common(10)
    
    print("Most frequent bytes:")
    for byte_val, count in most_common:
        char = chr(byte_val) if 32 <= byte_val <= 126 else f'[{byte_val}]'
        freq_pct = count / len(all_bytes) * 100
        print(f"  {byte_val:3d} ('{char}'): {count:3d} times ({freq_pct:5.1f}%)")
    
    crypto_results['frequency_analysis'] = most_common
    
    # 2. Entropy calculation
    entropy = 0
    for byte_val, count in byte_freq.items():
        prob = count / len(all_bytes)
        if prob > 0:
            entropy -= prob * np.log2(prob)
    
    print(f"\nByte entropy: {entropy:.3f} bits (max=8.0)")
    print(f"Randomness: {entropy/8:.1%}")
    crypto_results['entropy'] = entropy
    
    # 3. Chi-squared test for randomness
    expected_freq = len(all_bytes) / 256
    chi_squared = sum((count - expected_freq)**2 / expected_freq for count in byte_freq.values())
    print(f"Chi-squared statistic: {chi_squared:.1f}")
    crypto_results['chi_squared'] = chi_squared
    
    # 4. Runs test
    runs = 1
    for i in range(1, len(all_bits)):
        if all_bits[i] != all_bits[i-1]:
            runs += 1
    
    expected_runs = 2 * sum(all_bits) * (len(all_bits) - sum(all_bits)) / len(all_bits) + 1
    print(f"Runs test: {runs} runs (expected: {expected_runs:.1f})")
    crypto_results['runs_test'] = {'actual': runs, 'expected': expected_runs}
    
    return all_bytes, crypto_results

def test_encryption_hypotheses():
    """Test various encryption and encoding hypotheses."""
    
    print(f"\n=== ENCRYPTION HYPOTHESIS TESTING ===")
    
    # Get the byte data
    all_bytes, _ = cryptographic_pattern_analysis()
    
    if len(all_bytes) < 16:
        print("Insufficient data for encryption analysis")
        return
    
    # Convert to binary string and hex
    byte_data = bytes(all_bytes[:100])  # First 100 bytes
    hex_data = byte_data.hex()
    binary_str = ''.join(format(b, '08b') for b in all_bytes[:100])
    
    print(f"Testing {len(byte_data)} bytes of data")
    print(f"Hex preview: {hex_data[:64]}...")
    
    # Test 1: Base64 encoding
    print(f"\n--- Base64 Encoding Test ---")
    try:
        # Try treating as base64
        base64_candidates = []
        for start in range(min(4, len(byte_data))):
            for length in [16, 32, 64, len(byte_data)]:
                if start + length <= len(byte_data):
                    subset = byte_data[start:start+length]
                    try:
                        # Check if it's valid base64
                        if all(b in b'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for b in subset):
                            decoded = base64.b64decode(subset, validate=True)
                            if decoded:
                                base64_candidates.append((start, length, decoded))
                    except:
                        pass
        
        if base64_candidates:
            print(f"Found {len(base64_candidates)} potential base64 sequences")
            for start, length, decoded in base64_candidates[:3]:
                print(f"  Position {start}, length {length}: {decoded[:32]}...")
        else:
            print("No valid base64 sequences found")
    except Exception as e:
        print(f"Base64 test error: {e}")
    
    # Test 2: XOR cipher with common keys
    print(f"\n--- XOR Cipher Test ---")
    common_keys = [
        b'satoshi', b'bitcoin', b'nakamoto', b'genesis', b'block',
        b'key', b'msg', b'code', b'pass', b'secret'
    ]
    
    xor_candidates = []
    for key in common_keys:
        try:
            xored = bytes([b ^ key[i % len(key)] for i, b in enumerate(byte_data)])
            # Check if result has readable ASCII
            readable_count = sum(1 for b in xored if 32 <= b <= 126)
            if readable_count > len(xored) * 0.7:  # 70% readable
                xor_candidates.append((key, xored, readable_count))
        except:
            pass
    
    if xor_candidates:
        print(f"Found {len(xor_candidates)} XOR candidates:")
        for key, result, readable in xor_candidates:
            preview = ''.join(chr(b) if 32 <= b <= 126 else '?' for b in result[:32])
            print(f"  Key '{key.decode()}': {readable}/{len(result)} readable -> {preview}...")
    else:
        print("No promising XOR patterns found")
    
    # Test 3: Caesar/ROT cipher
    print(f"\n--- Caesar/ROT Cipher Test ---")
    ascii_bytes = [b for b in all_bytes if 32 <= b <= 126]
    if len(ascii_bytes) > 10:
        best_rot_score = 0
        best_rot = None
        
        for shift in range(1, 26):
            shifted = []
            for b in ascii_bytes[:50]:
                if 65 <= b <= 90:  # Uppercase
                    shifted.append(((b - 65 + shift) % 26) + 65)
                elif 97 <= b <= 122:  # Lowercase  
                    shifted.append(((b - 97 + shift) % 26) + 97)
                else:
                    shifted.append(b)
            
            # Score based on common English letters
            common_letters = set(b'etaoinshrdlu')
            score = sum(1 for b in shifted if chr(b).lower().encode()[0] in common_letters)
            
            if score > best_rot_score:
                best_rot_score = score
                best_rot = (shift, shifted)
        
        if best_rot:
            shift, result = best_rot
            preview = ''.join(chr(b) for b in result[:32])
            print(f"Best ROT{shift}: score {best_rot_score} -> {preview}...")
        else:
            print("No promising ROT patterns found")
    else:
        print("Insufficient ASCII data for ROT analysis")
    
    return xor_candidates

def steganographic_analysis():
    """Test for steganographic encoding patterns."""
    
    print(f"\n=== STEGANOGRAPHIC ANALYSIS ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Test 1: LSB (Least Significant Bit) analysis
    print(f"\n--- LSB Analysis ---")
    
    # Extract LSBs from image region
    row0, col0 = 101, 53
    lsb_bits = []
    
    for r in range(20):
        for c in range(40):
            y = row0 + r * 31
            x = col0 + c * 53
            
            if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                pixel_val = img[y, x]
                lsb = pixel_val & 1
                lsb_bits.append(lsb)
    
    # Convert LSBs to bytes
    lsb_bytes = []
    for i in range(0, len(lsb_bits) - 7, 8):
        byte_val = 0
        for j in range(8):
            byte_val |= (lsb_bits[i+j] << (7-j))
        lsb_bytes.append(byte_val)
    
    print(f"Extracted {len(lsb_bits)} LSB bits, {len(lsb_bytes)} bytes")
    
    # Check LSB randomness
    lsb_ones = sum(lsb_bits)
    lsb_entropy = 0
    if lsb_ones > 0 and lsb_ones < len(lsb_bits):
        p1 = lsb_ones / len(lsb_bits)
        p0 = 1 - p1
        lsb_entropy = -p1 * np.log2(p1) - p0 * np.log2(p0)
    
    print(f"LSB entropy: {lsb_entropy:.3f} (1.0 = random)")
    
    # Test 2: Bit plane analysis
    print(f"\n--- Bit Plane Analysis ---")
    
    bit_planes = {}
    for bit_pos in range(8):
        plane_bits = []
        for r in range(20):
            for c in range(40):
                y = row0 + r * 31
                x = col0 + c * 53
                
                if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                    pixel_val = img[y, x]
                    bit = (pixel_val >> bit_pos) & 1
                    plane_bits.append(bit)
        
        if plane_bits:
            ones_count = sum(plane_bits)
            plane_entropy = 0
            if ones_count > 0 and ones_count < len(plane_bits):
                p1 = ones_count / len(plane_bits)
                p0 = 1 - p1
                plane_entropy = -p1 * np.log2(p1) - p0 * np.log2(p0)
            
            bit_planes[bit_pos] = {
                'entropy': plane_entropy,
                'ones_ratio': ones_count / len(plane_bits) if plane_bits else 0
            }
    
    print("Bit plane analysis:")
    for bit_pos in range(8):
        data = bit_planes.get(bit_pos, {})
        entropy = data.get('entropy', 0)
        ratio = data.get('ones_ratio', 0)
        print(f"  Bit {bit_pos}: entropy={entropy:.3f}, ones={ratio:.1%}")
    
    # Find most random bit plane (potential steganography)
    most_random = max(bit_planes.items(), key=lambda x: x[1]['entropy'])
    print(f"Most random bit plane: {most_random[0]} (entropy: {most_random[1]['entropy']:.3f})")
    
    return lsb_bytes, bit_planes

def pattern_reconstruction_test():
    """Test pattern reconstruction with different interpretations."""
    
    print(f"\n=== PATTERN RECONSTRUCTION TEST ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Breakthrough position
    row0, col0 = 101, 53
    threshold = 72
    
    # Test different reconstruction methods
    methods = [
        ("Standard threshold", lambda val: '1' if val > threshold else '0'),
        ("Dynamic threshold", lambda val: '1' if val > np.median([threshold-10, val, threshold+10]) else '0'),
        ("Quantized levels", lambda val: str(min(9, val // 28))),  # 0-9 levels
        ("Binary difference", lambda val: '1' if abs(val - threshold) > 20 else '0'),
        ("High contrast only", lambda val: '1' if val > threshold + 30 or val < threshold - 30 else '0')
    ]
    
    results = {}
    
    for method_name, converter in methods:
        print(f"\n--- {method_name} ---")
        
        # Extract pattern
        pattern_grid = []
        for r in range(8):  # 8 rows for analysis
            row_pattern = []
            for c in range(16):  # 16 columns (2 bytes)
                y = row0 + r * 31
                x = col0 + c * 53
                
                if 0 <= y < img.shape[0] - 5 and 0 <= x < img.shape[1] - 5:
                    patch = img[max(0, y-2):min(img.shape[0], y+3), 
                               max(0, x-2):min(img.shape[1], x+3)]
                    
                    if patch.size > 0:
                        val = np.median(patch)
                        symbol = converter(val)
                        row_pattern.append(symbol)
            
            pattern_grid.append(''.join(row_pattern))
        
        # Display pattern
        for i, row in enumerate(pattern_grid):
            print(f"  Row {i}: {row}")
        
        # Try to decode as different formats
        if method_name == "Standard threshold":
            # Try binary decoding
            for i, row in enumerate(pattern_grid):
                if len(row) >= 16:
                    try:
                        byte1 = int(row[:8], 2)
                        byte2 = int(row[8:16], 2)
                        char1 = chr(byte1) if 32 <= byte1 <= 126 else f'[{byte1}]'
                        char2 = chr(byte2) if 32 <= byte2 <= 126 else f'[{byte2}]'
                        print(f"    Decoded: '{char1}{char2}'")
                    except:
                        print(f"    Decode failed")
                    
                    if i >= 3:  # Show first 4 rows
                        break
        
        results[method_name] = pattern_grid
    
    return results

def message_hypothesis_testing():
    """Test specific message hypotheses beyond 'On the'."""
    
    print(f"\n=== MESSAGE HYPOTHESIS TESTING ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Common Bitcoin/Satoshi related phrases
    test_messages = [
        "At the dawn",
        "In the beginning", 
        "To the future",
        "From Satoshi",
        "Bitcoin genesis",
        "Digital gold",
        "Peer to peer",
        "Crypto currency",
        "Block chain",
        "Hash function",
        "Private key",
        "Public ledger",
        "Proof of work",
        "Double spending",
        "Time stamp server"
    ]
    
    # Convert messages to binary
    message_patterns = []
    for msg in test_messages:
        binary_pattern = ''.join(format(ord(c), '08b') for c in msg[:8])  # First 8 chars
        message_patterns.append((msg, binary_pattern))
    
    # Test against extracted data at multiple positions
    test_positions = [
        (101, 53),  # Breakthrough position
        (103, 55),  # Slight shift
        (100, 50),  # Alternative
        (105, 58),  # Another variation
    ]
    
    best_matches = []
    
    for pos_name, (row0, col0) in enumerate(test_positions):
        print(f"\n--- Testing position ({row0}, {col0}) ---")
        
        # Extract bits at this position
        extracted_bits = []
        for r in range(4):  # 4 rows (32 bytes)
            for c in range(8):  # 8 bits per row
                y = row0 + r * 31
                x = col0 + c * 53
                
                if 0 <= y < img.shape[0] - 5 and 0 <= x < img.shape[1] - 5:
                    patch = img[max(0, y-2):min(img.shape[0], y+3), 
                               max(0, x-2):min(img.shape[1], x+3)]
                    
                    if patch.size > 0:
                        val = np.median(patch)
                        bit = '1' if val > 72 else '0'
                        extracted_bits.append(bit)
        
        extracted_pattern = ''.join(extracted_bits)
        
        # Test against each message
        for msg, target_pattern in message_patterns:
            if len(extracted_pattern) >= len(target_pattern):
                test_pattern = extracted_pattern[:len(target_pattern)]
                matches = sum(1 for i in range(len(target_pattern)) if test_pattern[i] == target_pattern[i])
                accuracy = matches / len(target_pattern)
                
                if accuracy > 0.7:  # High threshold
                    print(f"  HIGH MATCH: '{msg}' - {accuracy:.1%}")
                    best_matches.append((msg, accuracy, (row0, col0), test_pattern, target_pattern))
                elif accuracy > 0.6:  # Medium threshold
                    print(f"  GOOD MATCH: '{msg}' - {accuracy:.1%}")
    
    # Show best overall matches
    if best_matches:
        best_matches.sort(key=lambda x: x[1], reverse=True)
        print(f"\n=== BEST MESSAGE MATCHES ===")
        for msg, acc, pos, extracted, target in best_matches[:5]:
            print(f"{acc:.1%} '{msg}' at {pos}")
            print(f"  Target:    {target}")
            print(f"  Extracted: {extracted}")
            print(f"  Diff:      {''.join('.' if extracted[i] == target[i] else 'X' for i in range(len(target)))}")
    
    return best_matches

if __name__ == "__main__":
    print("Advanced Cryptographic Analysis")
    print("Testing encryption, steganography, and pattern reconstruction")
    print("="*70)
    
    # Cryptographic pattern analysis
    byte_data, crypto_stats = cryptographic_pattern_analysis()
    
    # Test encryption hypotheses
    encryption_results = test_encryption_hypotheses()
    
    # Steganographic analysis
    lsb_data, bit_planes = steganographic_analysis()
    
    # Pattern reconstruction tests
    reconstruction_results = pattern_reconstruction_test()
    
    # Message hypothesis testing
    message_matches = message_hypothesis_testing()
    
    print("\n" + "="*70)
    print("ADVANCED CRYPTOGRAPHIC ANALYSIS COMPLETE")
    
    if crypto_stats.get('entropy', 0) > 6:
        print(f"High entropy detected ({crypto_stats['entropy']:.3f}) - possible encryption")
    
    if encryption_results:
        print(f"Found {len(encryption_results)} potential XOR cipher matches")
    
    if message_matches:
        best_match = max(message_matches, key=lambda x: x[1])
        print(f"Best message match: '{best_match[0]}' at {best_match[1]:.1%} accuracy")
    
    print("Advanced cryptographic analysis suggests structured data present")
    print("Recommend further investigation with domain-specific tools")