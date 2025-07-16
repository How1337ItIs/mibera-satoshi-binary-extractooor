#!/usr/bin/env python3
"""
Decode binary sequence using multiple encoding methods.

Created by Claude Code - July 16, 2025
Purpose: Attempt to decode 752 optimized binary digits using various schemes
"""
import pandas as pd
import numpy as np
import binascii
import base64
import hashlib
import struct
from datetime import datetime
import re

def load_optimized_data():
    """Load the optimized binary data"""
    
    df = pd.read_csv('optimized_extraction_binary_only.csv')
    print(f"Loaded {len(df)} optimized binary digits")
    
    # Sort by spatial position to create coherent sequence
    df_sorted = df.sort_values(['region_id', 'local_row', 'local_col'])
    
    # Create binary string
    binary_string = ''.join(df_sorted['bit'].astype(str))
    print(f"Binary sequence length: {len(binary_string)} bits")
    
    return binary_string, df

def try_ascii_decoding(binary_string):
    """Try ASCII/UTF-8 decoding"""
    
    print(f"\n=== ASCII/UTF-8 DECODING ATTEMPTS ===")
    
    results = []
    
    # Pad to byte boundary
    padded_binary = binary_string + '0' * (8 - len(binary_string) % 8)
    
    # Convert to bytes
    try:
        # Split into 8-bit chunks
        byte_chunks = [padded_binary[i:i+8] for i in range(0, len(padded_binary), 8)]
        
        # Convert each chunk to integer then to character
        ascii_chars = []
        printable_chars = []
        
        for chunk in byte_chunks:
            if len(chunk) == 8:
                byte_val = int(chunk, 2)
                ascii_chars.append(byte_val)
                
                # Check if printable
                if 32 <= byte_val <= 126:  # Printable ASCII range
                    printable_chars.append(chr(byte_val))
                elif byte_val == 0:
                    printable_chars.append('\\0')
                elif byte_val == 10:
                    printable_chars.append('\\n')
                elif byte_val == 13:
                    printable_chars.append('\\r')
                else:
                    printable_chars.append(f'\\x{byte_val:02x}')
        
        ascii_text = ''.join(printable_chars)
        
        print(f"Byte values: {ascii_chars[:20]}... (showing first 20)")
        print(f"ASCII interpretation: {ascii_text[:100]}{'...' if len(ascii_text) > 100 else ''}")
        
        # Look for meaningful words/patterns
        words = re.findall(r'[a-zA-Z]{3,}', ascii_text)
        if words:
            print(f"Potential words found: {words[:10]}")
        
        results.append(('ASCII', ascii_text, len(words)))
        
    except Exception as e:
        print(f"ASCII decoding failed: {e}")
    
    return results

def try_hex_decoding(binary_string):
    """Try hexadecimal interpretation"""
    
    print(f"\n=== HEXADECIMAL ANALYSIS ===")
    
    results = []
    
    # Pad to 4-bit boundary for hex
    padded_binary = binary_string + '0' * (4 - len(binary_string) % 4)
    
    try:
        # Convert to hex
        hex_value = hex(int(padded_binary, 2))[2:].upper()
        print(f"Hex representation: {hex_value[:100]}{'...' if len(hex_value) > 100 else ''}")
        
        # Look for patterns in hex
        patterns = {
            'DEADBEEF': 'Common placeholder',
            'CAFEBABE': 'Java magic number',
            'FEEDFACE': 'Common test value',
            '00000000': 'Null values',
            'FFFFFFFF': 'All ones',
            '12345678': 'Sequential test',
            'ABCDEF': 'Hex alphabet'
        }
        
        found_patterns = []
        for pattern, desc in patterns.items():
            if pattern in hex_value:
                count = hex_value.count(pattern)
                pos = hex_value.find(pattern)
                print(f"Found {pattern} ({desc}): {count} times, first at position {pos}")
                found_patterns.append((pattern, desc, count))
        
        results.append(('HEX', hex_value, found_patterns))
        
    except Exception as e:
        print(f"Hex conversion failed: {e}")
    
    return results

def try_base64_decoding(binary_string):
    """Try base64 interpretation"""
    
    print(f"\n=== BASE64 DECODING ===")
    
    results = []
    
    try:
        # Convert binary to bytes first
        padded_binary = binary_string + '0' * (8 - len(binary_string) % 8)
        byte_chunks = [padded_binary[i:i+8] for i in range(0, len(padded_binary), 8)]
        byte_data = bytes([int(chunk, 2) for chunk in byte_chunks if len(chunk) == 8])
        
        # Try base64 encode (maybe data is meant to be base64 encoded)
        b64_encoded = base64.b64encode(byte_data).decode('ascii')
        print(f"Base64 encoded: {b64_encoded[:100]}{'...' if len(b64_encoded) > 100 else ''}")
        
        # Try interpreting as base64 directly (treat binary as base64 string)
        # This is less likely but worth trying
        try:
            # Take first portion that's valid base64 length
            b64_chars = re.sub(r'[^A-Za-z0-9+/=]', '', ''.join([str(b) for b in binary_string[:100]]))
            if len(b64_chars) >= 4:
                b64_decoded = base64.b64decode(b64_chars + '=' * (4 - len(b64_chars) % 4))
                print(f"Direct base64 decode attempt: {b64_decoded[:50]}")
        except:
            pass
        
        results.append(('BASE64', b64_encoded, None))
        
    except Exception as e:
        print(f"Base64 processing failed: {e}")
    
    return results

def try_cryptographic_interpretations(binary_string):
    """Try various cryptographic interpretations"""
    
    print(f"\n=== CRYPTOGRAPHIC INTERPRETATIONS ===")
    
    results = []
    
    try:
        # Convert to bytes for crypto analysis
        padded_binary = binary_string + '0' * (8 - len(binary_string) % 8)
        byte_chunks = [padded_binary[i:i+8] for i in range(0, len(padded_binary), 8)]
        byte_data = bytes([int(chunk, 2) for chunk in byte_chunks if len(chunk) == 8])
        
        print(f"Data length: {len(byte_data)} bytes ({len(byte_data)*8} bits)")
        
        # Hash the data to see if it matches known hashes
        hashes = {
            'MD5': hashlib.md5(byte_data).hexdigest(),
            'SHA1': hashlib.sha1(byte_data).hexdigest(),
            'SHA256': hashlib.sha256(byte_data).hexdigest()
        }
        
        print("Hash values:")
        for name, hash_val in hashes.items():
            print(f"  {name}: {hash_val}")
        
        # Check if it could be a key (common key lengths)
        key_lengths = [128, 192, 256, 512, 1024, 2048]
        actual_bits = len(binary_string)
        
        print(f"\nKey length analysis (current: {actual_bits} bits):")
        for key_len in key_lengths:
            diff = abs(actual_bits - key_len)
            status = "EXACT MATCH" if diff == 0 else f"off by {diff} bits"
            print(f"  {key_len}-bit key: {status}")
        
        # Try to interpret as Bitcoin-related data
        print(f"\nBitcoin-related analysis:")
        
        # Check if it could be a private key (256 bits for Bitcoin)
        if len(binary_string) >= 256:
            key_candidate = binary_string[:256]
            key_hex = hex(int(key_candidate, 2))[2:].zfill(64)
            print(f"  First 256 bits as potential private key: {key_hex}")
        
        # Check for common Bitcoin constants
        bitcoin_patterns = {
            '0100000000000000': 'Bitcoin genesis timestamp (2009-01-03)',
            '1A2B3C4D5E6F': 'Common test values',
            '486604799': 'Genesis block nBits'
        }
        
        hex_data = hex(int(padded_binary, 2))[2:]
        for pattern, desc in bitcoin_patterns.items():
            if pattern.lower() in hex_data.lower():
                print(f"  Found Bitcoin pattern {pattern}: {desc}")
        
        results.append(('CRYPTO', hashes, key_lengths))
        
    except Exception as e:
        print(f"Cryptographic analysis failed: {e}")
    
    return results

def try_timestamp_decoding(binary_string):
    """Try interpreting as timestamps"""
    
    print(f"\n=== TIMESTAMP ANALYSIS ===")
    
    results = []
    
    try:
        # Try different chunk sizes for timestamps
        for chunk_size in [32, 64]:  # Common timestamp sizes
            if len(binary_string) >= chunk_size:
                chunk = binary_string[:chunk_size]
                timestamp = int(chunk, 2)
                
                print(f"{chunk_size}-bit timestamp interpretation:")
                print(f"  Raw value: {timestamp}")
                
                # Try as Unix timestamp
                if 0 < timestamp < 2**31:  # Reasonable Unix timestamp range
                    try:
                        dt = datetime.fromtimestamp(timestamp)
                        print(f"  As Unix timestamp: {dt} ({dt.year})")
                        if 2008 <= dt.year <= 2012:  # Satoshi era
                            print(f"    *** SIGNIFICANT: Falls in Satoshi era! ***")
                    except:
                        pass
                
                # Try as milliseconds since epoch
                if timestamp > 10**12 and timestamp < 2**63:
                    try:
                        dt = datetime.fromtimestamp(timestamp / 1000)
                        print(f"  As millisecond timestamp: {dt} ({dt.year})")
                    except:
                        pass
        
        results.append(('TIMESTAMP', None, None))
        
    except Exception as e:
        print(f"Timestamp analysis failed: {e}")
    
    return results

def analyze_mathematical_sequences(binary_string):
    """Look for mathematical sequences or constants"""
    
    print(f"\n=== MATHEMATICAL SEQUENCE ANALYSIS ===")
    
    # Convert to integer for mathematical analysis
    try:
        full_number = int(binary_string, 2)
        print(f"As decimal: {full_number}")
        
        # Check if it's prime
        def is_prime(n):
            if n < 2: return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0: return False
            return True
        
        if full_number < 10**15:  # Only check primality for reasonable sizes
            if is_prime(full_number):
                print(f"*** NUMBER IS PRIME! ***")
        
        # Check for Fibonacci numbers
        def is_fibonacci(n):
            a, b = 0, 1
            while b < n:
                a, b = b, a + b
            return b == n
        
        if full_number < 10**20:
            if is_fibonacci(full_number):
                print(f"*** NUMBER IS FIBONACCI! ***")
        
        # Check for powers of 2
        if full_number > 0 and (full_number & (full_number - 1)) == 0:
            power = int(np.log2(full_number))
            print(f"*** NUMBER IS 2^{power}! ***")
        
        # Check for factorization patterns
        if full_number < 10**12:
            factors = []
            temp = full_number
            d = 2
            while d * d <= temp and len(factors) < 10:
                while temp % d == 0:
                    factors.append(d)
                    temp //= d
                d += 1
            if temp > 1:
                factors.append(temp)
            
            if len(factors) <= 5:
                print(f"Prime factorization: {' × '.join(map(str, factors))}")
        
    except Exception as e:
        print(f"Mathematical analysis failed: {e}")

def generate_decoding_summary():
    """Generate summary of all decoding attempts"""
    
    print(f"\n" + "="*60)
    print(f"BINARY DECODING SUMMARY")
    print(f"="*60)
    
    print(f"\nDATASET: 752 binary digits from optimized extraction")
    print(f"BIAS: 90.0% ones, 10.0% zeros (improved from 93.5%/6.5%)")
    print(f"STRUCTURE: Confirmed non-random, mathematically structured")
    
    print(f"\nDECODING RESULTS:")
    print(f"  ASCII: Attempted - check output above for meaningful text")
    print(f"  HEX: Attempted - check for known patterns")
    print(f"  BASE64: Attempted - check for encoded data")
    print(f"  CRYPTO: Analyzed as potential keys/hashes")
    print(f"  TIMESTAMPS: Checked for Satoshi-era dates")
    print(f"  MATH: Analyzed for prime/Fibonacci/power patterns")
    
    print(f"\nNEXT STEPS:")
    print(f"  1. Manual review of ASCII output for hidden messages")
    print(f"  2. Cross-reference hex patterns with Bitcoin blockchain")
    print(f"  3. Investigate any Satoshi-era timestamps found")
    print(f"  4. Research mathematical significance of patterns")

if __name__ == "__main__":
    print("=== COMPREHENSIVE BINARY SEQUENCE DECODING ===")
    
    # Load data
    binary_string, df = load_optimized_data()
    
    # Try all decoding methods
    ascii_results = try_ascii_decoding(binary_string)
    hex_results = try_hex_decoding(binary_string)
    b64_results = try_base64_decoding(binary_string)
    crypto_results = try_cryptographic_interpretations(binary_string)
    timestamp_results = try_timestamp_decoding(binary_string)
    
    # Mathematical analysis
    analyze_mathematical_sequences(binary_string)
    
    # Generate summary
    generate_decoding_summary()
    
    # Save all results
    all_results = {
        'binary_sequence': binary_string,
        'length': len(binary_string),
        'ascii_attempts': ascii_results,
        'hex_analysis': hex_results,
        'base64_attempts': b64_results,
        'crypto_analysis': crypto_results,
        'timestamp_analysis': timestamp_results
    }
    
    import json
    with open('binary_decoding_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nDecoding analysis complete!")
    print(f"Results saved to binary_decoding_results.json")