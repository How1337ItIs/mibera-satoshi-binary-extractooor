#!/usr/bin/env python3
"""
Rigorous hash verification with exact bit provenance and entropy sanity checks.
"""

import cv2
import numpy as np
import hashlib
import random
import json

def extract_with_exact_provenance():
    """Extract bits with exact position tracking for verification."""
    
    print("=== EXACT BIT PROVENANCE EXTRACTION ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Use the claimed breakthrough configuration
    row0, col0 = 101, 53
    row_pitch = 31
    col_pitch = 53
    threshold = 72
    
    extraction_log = []
    all_bits = []
    
    print(f"Configuration: pos=({row0},{col0}), pitch={row_pitch}x{col_pitch}, threshold={threshold}")
    
    # Extract first 32 bytes (256 bits) with full provenance
    for i in range(256):
        bit_row = i // 8
        bit_col = i % 8
        
        y = row0 + bit_row * row_pitch
        x = col0 + bit_col * col_pitch
        
        if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
            pixel_val = img[y, x]
            bit = 1 if pixel_val > threshold else 0
            
            extraction_log.append({
                'bit_index': i,
                'grid_pos': (bit_row, bit_col),
                'pixel_pos': (y, x),
                'pixel_value': int(pixel_val),
                'threshold': threshold,
                'bit': bit
            })
            
            all_bits.append(bit)
    
    print(f"Extracted {len(all_bits)} bits with full provenance")
    
    # Show first 16 bits as example
    print("\nFirst 16 bits with provenance:")
    for i in range(16):
        log = extraction_log[i]
        print(f"Bit {i:2d}: grid({log['grid_pos'][0]},{log['grid_pos'][1]}) "
              f"pixel({log['pixel_pos'][0]},{log['pixel_pos'][1]}) "
              f"val={log['pixel_value']:3d} -> {log['bit']}")
    
    return extraction_log, all_bits

def verify_hash_with_endianness():
    """Verify the 9 leading zeros hash with exact bit positions and endianness."""
    
    print(f"\n=== HASH VERIFICATION WITH ENDIANNESS ===")
    
    extraction_log, all_bits = extract_with_exact_provenance()
    
    if len(all_bits) < 256:
        print("Insufficient bits for 32-byte hash verification")
        return
    
    # Convert first 256 bits to 32 bytes
    byte_data = []
    for i in range(0, 256, 8):
        byte_val = 0
        for j in range(8):
            byte_val |= (all_bits[i + j] << (7 - j))
        byte_data.append(byte_val)
    
    print(f"Converted 256 bits to 32 bytes")
    
    # Look for the claimed 9 leading zeros hash
    claimed_position = 8  # Position where 9 zeros were claimed
    
    if claimed_position + 32 <= len(byte_data):
        hash_bytes = bytes(byte_data[claimed_position:claimed_position + 32])
        
        print(f"\nHash candidate at byte position {claimed_position}:")
        print(f"Raw bytes: {hash_bytes}")
        
        # Big-endian (network byte order)
        be_hex = hash_bytes.hex()
        be_zeros = len(be_hex) - len(be_hex.lstrip('0'))
        
        # Little-endian
        le_bytes = hash_bytes[::-1]  # Reverse byte order
        le_hex = le_bytes.hex()
        le_zeros = len(le_hex) - len(le_hex.lstrip('0'))
        
        print(f"Big-endian:    {be_hex}")
        print(f"Little-endian: {le_hex}")
        print(f"BE leading zeros: {be_zeros}")
        print(f"LE leading zeros: {le_zeros}")
        
        # Show exact bit positions that created this hash
        print(f"\nExact bit positions for this hash:")
        bit_start = claimed_position * 8
        bit_end = bit_start + 256
        
        print(f"Bits {bit_start}-{bit_end-1} (32 bytes)")
        for i in range(bit_start, min(bit_end, bit_start + 64)):  # Show first 8 bytes
            if i < len(extraction_log):
                log = extraction_log[i]
                print(f"  Bit {i:3d}: grid({log['grid_pos'][0]:2d},{log['grid_pos'][1]:2d}) "
                      f"pixel({log['pixel_pos'][0]:4d},{log['pixel_pos'][1]:4d}) "
                      f"val={log['pixel_value']:3d} -> {log['bit']}")
        
        # Calculate probability of this occurring by chance
        if be_zeros >= 6:
            prob = 1 / (16 ** be_zeros)
            print(f"\nProbability of {be_zeros} leading zeros: {prob:.2e} (1 in {16**be_zeros:,})")
        
        return hash_bytes, be_hex, le_hex, be_zeros, le_zeros
    
    else:
        print(f"Cannot extract 32 bytes starting at position {claimed_position}")
        return None

def bitcoin_header_validation():
    """Validate against actual Bitcoin header format."""
    
    print(f"\n=== BITCOIN HEADER VALIDATION ===")
    
    hash_result = verify_hash_with_endianness()
    
    if not hash_result:
        return
    
    hash_bytes, be_hex, le_hex, be_zeros, le_zeros = hash_result
    
    # Bitcoin block hashes are typically little-endian in the protocol
    # but displayed as big-endian in block explorers
    
    print("Testing against Bitcoin header format:")
    
    # Known Bitcoin genesis block hash (displayed format)
    genesis_display = "000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f"
    
    # Compare with our extracted hash
    print(f"Genesis (display): {genesis_display}")
    print(f"Extracted (BE):    {be_hex}")
    print(f"Extracted (LE):    {le_hex}")
    
    # Character-by-character comparison
    be_similarity = sum(1 for i in range(min(len(be_hex), len(genesis_display))) 
                       if be_hex[i] == genesis_display[i])
    le_similarity = sum(1 for i in range(min(le_hex), len(genesis_display)) 
                       if le_hex[i] == genesis_display[i])
    
    print(f"BE similarity: {be_similarity}/64 characters ({be_similarity/64:.1%})")
    print(f"LE similarity: {le_similarity}/64 characters ({le_similarity/64:.1%})")
    
    # Test if this could be a SHA-256 hash of some data
    print(f"\n--- SHA-256 Validation Test ---")
    
    # Bitcoin block hashes are SHA-256(SHA-256(header))
    # We can't reverse this, but we can check format compliance
    
    if be_zeros >= 8:
        print(f"*** 8+ leading zeros suggests high-difficulty block ***")
        print(f"*** Timeframe estimate: 2010-2012 based on difficulty ***")
    elif be_zeros >= 6:
        print(f"*** 6+ leading zeros suggests valid Bitcoin hash ***")
        print(f"*** Could be block hash, transaction hash, or merkle root ***")
    else:
        print(f"*** Insufficient leading zeros for Bitcoin block hash ***")
    
    # Check if it matches any known pattern
    known_prefixes = [
        ("000000000019", "Genesis block pattern"),
        ("00000000001", "Early block pattern"),
        ("000000000a", "Block 10-15 range"),
        ("00000000083", "Block 1 pattern")
    ]
    
    for prefix, description in known_prefixes:
        if be_hex.startswith(prefix.lower()):
            print(f"*** MATCHES {description} ***")
            return True
    
    return False

def entropy_sanity_check():
    """Compare extraction against shuffled data to test for pattern artifacts."""
    
    print(f"\n=== ENTROPY SANITY CHECK ===")
    
    extraction_log, original_bits = extract_with_exact_provenance()
    
    if len(original_bits) < 100:
        print("Insufficient bits for entropy test")
        return
    
    # Test patterns on original data
    test_patterns = {
        "satoshi_s": "01110011",
        "version_1": "00000001", 
        "genesis_start": "00000000",
        "difficulty": "11111111"
    }
    
    original_scores = {}
    shuffled_scores = {}
    
    original_str = ''.join(map(str, original_bits))
    
    print(f"Testing {len(test_patterns)} patterns on {len(original_bits)} bits")
    
    # Score original data
    for pattern_name, pattern in test_patterns.items():
        best_score = 0
        for start in range(len(original_str) - len(pattern) + 1):
            segment = original_str[start:start + len(pattern)]
            matches = sum(1 for i in range(len(pattern)) if segment[i] == pattern[i])
            score = matches / len(pattern)
            best_score = max(best_score, score)
        
        original_scores[pattern_name] = best_score
        print(f"Original {pattern_name:12s}: {best_score:.1%}")
    
    # Test on shuffled data (multiple runs)
    print(f"\nTesting on shuffled data (10 runs):")
    
    for pattern_name in test_patterns:
        shuffled_scores[pattern_name] = []
    
    for run in range(10):
        # Shuffle the bits
        shuffled_bits = original_bits.copy()
        random.shuffle(shuffled_bits)
        shuffled_str = ''.join(map(str, shuffled_bits))
        
        for pattern_name, pattern in test_patterns.items():
            best_score = 0
            for start in range(len(shuffled_str) - len(pattern) + 1):
                segment = shuffled_str[start:start + len(pattern)]
                matches = sum(1 for i in range(len(pattern)) if segment[i] == pattern[i])
                score = matches / len(pattern)
                best_score = max(best_score, score)
            
            shuffled_scores[pattern_name].append(best_score)
    
    # Compare results
    print(f"\nSanity check results:")
    
    for pattern_name in test_patterns:
        orig_score = original_scores[pattern_name]
        shuffled_avg = np.mean(shuffled_scores[pattern_name])
        shuffled_max = max(shuffled_scores[pattern_name])
        
        print(f"{pattern_name:12s}: Original={orig_score:.1%}, "
              f"Shuffled avg={shuffled_avg:.1%}, max={shuffled_max:.1%}")
        
        # Sanity check evaluation
        if orig_score > 0.9 and shuffled_avg > 0.8:
            print(f"    *** PATTERN ARTIFACT WARNING - HIGH SCORES ON SHUFFLED DATA ***")
        elif orig_score > 0.9 and shuffled_avg < 0.6:
            print(f"    *** GENUINE PATTERN - LOW SCORES ON SHUFFLED DATA ***")
        elif orig_score < 0.7:
            print(f"    *** WEAK PATTERN - NOT SIGNIFICANT ***")

def generate_raw_bit_dump():
    """Generate raw CSV bit dump for independent verification."""
    
    print(f"\n=== GENERATING RAW BIT DUMP ===")
    
    extraction_log, all_bits = extract_with_exact_provenance()
    
    # Create CSV with full provenance
    csv_content = "bit_index,grid_row,grid_col,pixel_y,pixel_x,pixel_value,threshold,bit\n"
    
    for log in extraction_log:
        csv_content += (f"{log['bit_index']},{log['grid_pos'][0]},{log['grid_pos'][1]},"
                       f"{log['pixel_pos'][0]},{log['pixel_pos'][1]},"
                       f"{log['pixel_value']},{log['threshold']},{log['bit']}\n")
    
    # Save to file
    with open('raw_bit_dump.csv', 'w') as f:
        f.write(csv_content)
    
    # Calculate checksum
    checksum = hashlib.sha256(csv_content.encode()).hexdigest()
    
    print(f"Raw bit dump saved to raw_bit_dump.csv")
    print(f"Entries: {len(extraction_log)}")
    print(f"SHA-256 checksum: {checksum}")
    
    # Summary statistics
    ones_count = sum(all_bits)
    print(f"Ones: {ones_count}/{len(all_bits)} ({ones_count/len(all_bits):.1%})")
    
    return csv_content, checksum

def save_verification_results():
    """Save comprehensive verification results."""
    
    print(f"\n=== SAVING VERIFICATION RESULTS ===")
    
    # Run all verifications
    extraction_log, all_bits = extract_with_exact_provenance()
    hash_result = verify_hash_with_endianness()
    bitcoin_valid = bitcoin_header_validation()
    csv_content, checksum = generate_raw_bit_dump()
    
    verification_summary = {
        "verification_timestamp": "2024-01-17",
        "extraction_config": {
            "position": (101, 53),
            "pitch": (31, 53),
            "threshold": 72
        },
        "bits_extracted": len(all_bits),
        "hash_verification": {
            "hash_found": hash_result is not None,
            "leading_zeros_be": hash_result[3] if hash_result else 0,
            "leading_zeros_le": hash_result[4] if hash_result else 0,
            "bitcoin_format_valid": bitcoin_valid
        },
        "raw_data": {
            "csv_checksum": checksum,
            "ones_ratio": sum(all_bits) / len(all_bits),
            "total_bits": len(all_bits)
        },
        "verification_status": "CONFIRMED" if (hash_result and bitcoin_valid) else "QUESTIONABLE"
    }
    
    with open('rigorous_verification.json', 'w') as f:
        json.dump(verification_summary, f, indent=2)
    
    print("Verification results saved to rigorous_verification.json")
    
    return verification_summary

if __name__ == "__main__":
    print("Rigorous Hash Verification")
    print("Exact bit provenance and entropy sanity checks")
    print("="*60)
    
    # Extract with exact provenance
    extraction_log, all_bits = extract_with_exact_provenance()
    
    # Verify hash with endianness
    hash_result = verify_hash_with_endianness()
    
    # Bitcoin header validation
    bitcoin_valid = bitcoin_header_validation()
    
    # Entropy sanity check
    entropy_sanity_check()
    
    # Generate raw bit dump
    csv_content, checksum = generate_raw_bit_dump()
    
    # Save verification results
    verification_summary = save_verification_results()
    
    print("\n" + "="*60)
    print("RIGOROUS VERIFICATION COMPLETE")
    
    # Final assessment
    if hash_result:
        print(f"Hash found: {hash_result[3]} leading zeros (BE)")
        print(f"Bitcoin format: {'VALID' if bitcoin_valid else 'INVALID'}")
    
    print(f"Verification status: {verification_summary['verification_status']}")
    print(f"Raw data checksum: {checksum[:16]}...")
    
    print("\nAll extraction data saved for independent verification")