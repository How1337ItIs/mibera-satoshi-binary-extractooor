#!/usr/bin/env python3
"""
CURSOR AGENT: COMPREHENSIVE BINARY DECODING & PATTERN SEARCH
===========================================================

Methodical approach to decode the 1,440-bit optimized extraction:
1. Canonical byte split (row-major, 8-bit, no parity)
2. All eight bit-shifts (0-7) of the entire stream
3. Reverse-bit-order per byte & repeat shifts
4. Endianness swaps (little-/big-endian 16-bit chunks)
5. Base64/Base58 window scan
6. WIF/key prefix brute force
7. QR-matrix guess
8. Entropy drop monitoring

Created by: Cursor Agent
Date: July 16, 2025
"""

import base64
import binascii
import textwrap
import hashlib
import csv
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import islice
from collections import defaultdict
import logging
from datetime import datetime
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('decode_attempts.log'),
        logging.StreamHandler()
    ]
)

class BinaryDecoder:
    def __init__(self, csv_path="cursor_optimized_extraction.csv"):
        """Initialize with the optimized binary extraction data"""
        self.csv_path = csv_path
        self.bits = self.load_binary_sequence()
        self.bit_length = len(self.bits)
        self.results = []
        self.candidates = []
        
        logging.info(f"Loaded {self.bit_length} bits from {csv_path}")
        logging.info(f"Bit distribution: {self.bits.count('0')} zeros, {self.bits.count('1')} ones")
        
    def load_binary_sequence(self):
        """Load binary sequence from CSV file"""
        try:
            df = pd.read_csv(self.csv_path)
            # Convert bit column to string and join
            bits = ''.join(df['bit'].astype(str))
            return bits
        except Exception as e:
            logging.error(f"Failed to load binary sequence: {e}")
            return None
            
    def bits_to_bytes(self, bitstr, reverse_bits=False):
        """Convert bit string to bytes, optionally reversing bit order per byte"""
        if reverse_bits:
            # Reverse bit order in each byte
            bitstr = ''.join(bs[::-1] for bs in textwrap.wrap(bitstr, 8))
        
        # Convert to bytes
        bytes_list = []
        for i in range(0, len(bitstr), 8):
            if i + 8 <= len(bitstr):
                byte_str = bitstr[i:i+8]
                bytes_list.append(int(byte_str, 2))
        
        return bytes(bytes_list)
    
    def calculate_entropy(self, data):
        """Calculate Shannon entropy of byte data"""
        if not data:
            return 0
        
        # Count byte frequencies
        byte_counts = defaultdict(int)
        for byte in data:
            byte_counts[byte] += 1
        
        # Calculate entropy
        total = len(data)
        entropy = 0
        for count in byte_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def is_printable_ascii(self, data, min_ratio=0.7):
        """Check if data is mostly printable ASCII"""
        if not data:
            return False
        
        printable_count = sum(1 for byte in data if 32 <= byte <= 126)
        ratio = printable_count / len(data)
        return ratio >= min_ratio
    
    def try_ascii_decode(self, data, offset, shift, reverse_bits=False, endianness="big"):
        """Try ASCII decoding and look for meaningful patterns"""
        if not data:
            return None
        
        # Convert to ASCII string
        ascii_str = ''.join(chr(byte) if 32 <= byte <= 126 else '.' for byte in data)
        
        # Look for interesting patterns
        interesting_patterns = [
            "The", "Bitcoin", "Satoshi", "Nakamoto", "Times", "Genesis",
            "Block", "Chain", "Crypto", "Key", "Address", "Private",
            "Public", "Hash", "Mining", "Wallet", "Transaction"
        ]
        
        found_patterns = []
        for pattern in interesting_patterns:
            if pattern.lower() in ascii_str.lower():
                found_patterns.append(pattern)
        
        # Calculate metrics
        printable_ratio = sum(1 for c in ascii_str if c != '.') / len(ascii_str)
        entropy = self.calculate_entropy(data)
        
        result = {
            'offset': offset,
            'shift': shift,
            'reverse_bits': reverse_bits,
            'endianness': endianness,
            'printable_ratio': printable_ratio,
            'entropy': entropy,
            'ascii_preview': ascii_str[:100],
            'found_patterns': found_patterns,
            'data_length': len(data)
        }
        
        # Log if interesting
        if found_patterns or printable_ratio > 0.8:
            logging.info(f"[ASCII HIT] Offset {offset}, Shift {shift}, Patterns: {found_patterns}")
            logging.info(f"  ASCII: {ascii_str[:120]}")
            self.candidates.append(result)
        
        return result
    
    def try_base64_decode(self, bitstr, offset, window_size=24):
        """Try Base64 decoding with sliding windows"""
        results = []
        
        # Base64 uses 6-bit chunks, so window_size should be multiple of 6
        for i in range(0, len(bitstr) - window_size, 6):
            chunk = bitstr[i:i+window_size]
            if len(chunk) < 6:
                continue
                
            try:
                # Convert to bytes
                bytes_data = self.bits_to_bytes(chunk)
                
                # Try Base64 decode
                decoded = base64.b64decode(bytes_data, validate=True)
                
                if decoded and self.is_printable_ascii(decoded):
                    result = {
                        'offset': offset + i,
                        'window_size': window_size,
                        'decode_type': 'base64',
                        'decoded': decoded.decode('ascii', errors='ignore'),
                        'entropy': self.calculate_entropy(decoded)
                    }
                    results.append(result)
                    logging.info(f"[B64] Offset {offset + i}, Decoded: {decoded[:60]}")
                    
            except (binascii.Error, UnicodeDecodeError):
                continue
        
        return results
    
    def try_base58_decode(self, bitstr, offset, window_size=40):
        """Try Base58 decoding with sliding windows"""
        results = []
        
        # Base58 uses ~5.86 bits per character, so we'll try different window sizes
        for i in range(0, len(bitstr) - window_size, 8):
            chunk = bitstr[i:i+window_size]
            if len(chunk) < 8:
                continue
                
            try:
                # Convert to bytes
                bytes_data = self.bits_to_bytes(chunk)
                
                # Try Base58 decode (simplified - would need proper Base58 library)
                # For now, just check if it looks like Base58 (alphanumeric, no 0, O, I, l)
                decoded_str = bytes_data.decode('ascii', errors='ignore')
                if all(c in '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz' for c in decoded_str):
                    result = {
                        'offset': offset + i,
                        'window_size': window_size,
                        'decode_type': 'base58',
                        'decoded': decoded_str,
                        'entropy': self.calculate_entropy(bytes_data)
                    }
                    results.append(result)
                    logging.info(f"[B58] Offset {offset + i}, Decoded: {decoded_str[:60]}")
                    
            except UnicodeDecodeError:
                continue
        
        return results
    
    def check_wif_key(self, bitstr, offset, window_size=52):
        """Check for WIF (Wallet Import Format) private keys"""
        results = []
        
        # WIF keys are typically 51-52 characters in Base58
        for i in range(0, len(bitstr) - window_size, 8):
            chunk = bitstr[i:i+window_size]
            if len(chunk) < 32:  # Minimum for WIF
                continue
                
            try:
                bytes_data = self.bits_to_bytes(chunk)
                
                # Check for WIF version byte (0x80 for mainnet)
                if len(bytes_data) >= 1 and bytes_data[0] == 0x80:
                    # This could be a WIF key - would need proper Base58Check validation
                    result = {
                        'offset': offset + i,
                        'window_size': window_size,
                        'decode_type': 'wif_candidate',
                        'hex_data': bytes_data.hex(),
                        'entropy': self.calculate_entropy(bytes_data)
                    }
                    results.append(result)
                    logging.info(f"[WIF CANDIDATE] Offset {offset + i}, Hex: {bytes_data.hex()[:20]}")
                    
            except Exception:
                continue
        
        return results
    
    def try_qr_matrix(self, bitstr, matrix_sizes=[37, 29, 25, 21]):
        """Try reshaping bits into QR code matrices"""
        results = []
        
        for size in matrix_sizes:
            if len(bitstr) >= size * size:
                # Extract square matrix
                matrix_bits = bitstr[:size*size]
                
                # Reshape into matrix
                matrix = []
                for i in range(size):
                    row = []
                    for j in range(size):
                        idx = i * size + j
                        if idx < len(matrix_bits):
                            row.append(int(matrix_bits[idx]))
                    matrix.append(row)
                
                # Calculate QR-like metrics
                black_ratio = sum(sum(row) for row in matrix) / (size * size)
                
                result = {
                    'matrix_size': size,
                    'black_ratio': black_ratio,
                    'matrix': matrix,
                    'entropy': self.calculate_entropy(self.bits_to_bytes(matrix_bits))
                }
                results.append(result)
                
                logging.info(f"[QR MATRIX] Size {size}x{size}, Black ratio: {black_ratio:.3f}")
        
        return results
    
    def step1_canonical_byte_split(self):
        """Step 1: Canonical byte split - row-major, 8-bit, no parity"""
        logging.info("=== STEP 1: CANONICAL BYTE SPLIT ===")
        
        # Convert to bytes without any shifts
        bytes_data = self.bits_to_bytes(self.bits)
        result = self.try_ascii_decode(bytes_data, 0, 0, False, "big")
        self.results.append(result)
        
        return result
    
    def step2_bit_shifts(self):
        """Step 2: All eight bit-shifts (0-7) of the entire stream"""
        logging.info("=== STEP 2: BIT SHIFTS ===")
        
        results = []
        for shift in range(8):
            shifted_bits = self.bits[shift:]
            bytes_data = self.bits_to_bytes(shifted_bits)
            result = self.try_ascii_decode(bytes_data, shift, shift, False, "big")
            results.append(result)
            self.results.append(result)
        
        return results
    
    def step3_reverse_bit_order(self):
        """Step 3: Reverse-bit-order per byte & repeat shifts"""
        logging.info("=== STEP 3: REVERSE BIT ORDER ===")
        
        results = []
        for shift in range(8):
            shifted_bits = self.bits[shift:]
            bytes_data = self.bits_to_bytes(shifted_bits, reverse_bits=True)
            result = self.try_ascii_decode(bytes_data, shift, shift, True, "big")
            results.append(result)
            self.results.append(result)
        
        return results
    
    def step4_endianness_swaps(self):
        """Step 4: Endianness swaps - little-/big-endian 16-bit chunks"""
        logging.info("=== STEP 4: ENDIANNESS SWAPS ===")
        
        results = []
        
        # Try 16-bit chunks with different endianness
        for shift in range(8):
            shifted_bits = self.bits[shift:]
            
            # Big endian (default)
            bytes_data = self.bits_to_bytes(shifted_bits)
            result = self.try_ascii_decode(bytes_data, shift, shift, False, "big")
            results.append(result)
            self.results.append(result)
            
            # Little endian (reverse byte order)
            if len(shifted_bits) >= 16:
                # Convert to 16-bit chunks and reverse
                chunks = [shifted_bits[i:i+16] for i in range(0, len(shifted_bits), 16)]
                reversed_chunks = [chunk[::-1] for chunk in chunks]
                reversed_bits = ''.join(reversed_chunks)
                bytes_data = self.bits_to_bytes(reversed_bits)
                result = self.try_ascii_decode(bytes_data, shift, shift, False, "little")
                results.append(result)
                self.results.append(result)
        
        return results
    
    def step5_base64_base58_scan(self):
        """Step 5: Base64/Base58 window scan"""
        logging.info("=== STEP 5: BASE64/BASE58 WINDOW SCAN ===")
        
        results = []
        
        # Base64 with different window sizes (multiples of 6 bits)
        for window_size in [24, 32, 48, 64]:
            base64_results = self.try_base64_decode(self.bits, 0, window_size)
            results.extend(base64_results)
            self.candidates.extend(base64_results)
        
        # Base58 with different window sizes
        for window_size in [40, 50, 60]:
            base58_results = self.try_base58_decode(self.bits, 0, window_size)
            results.extend(base58_results)
            self.candidates.extend(base58_results)
        
        return results
    
    def step6_wif_key_brute(self):
        """Step 6: WIF/key prefix brute force"""
        logging.info("=== STEP 6: WIF/KEY PREFIX BRUTE FORCE ===")
        
        results = []
        
        # Check for WIF keys with different window sizes
        for window_size in [37, 52, 64]:
            wif_results = self.check_wif_key(self.bits, 0, window_size)
            results.extend(wif_results)
            self.candidates.extend(wif_results)
        
        return results
    
    def step7_qr_matrix_guess(self):
        """Step 7: QR-matrix guess"""
        logging.info("=== STEP 7: QR MATRIX GUESS ===")
        
        results = self.try_qr_matrix(self.bits)
        self.candidates.extend(results)
        
        return results
    
    def step8_entropy_monitoring(self):
        """Step 8: Entropy drop monitoring"""
        logging.info("=== STEP 8: ENTROPY MONITORING ===")
        
        # Calculate entropy for all results
        entropy_data = []
        for result in self.results:
            if result:
                entropy_data.append({
                    'offset': result['offset'],
                    'shift': result['shift'],
                    'reverse_bits': result['reverse_bits'],
                    'endianness': result['endianness'],
                    'entropy': result['entropy'],
                    'printable_ratio': result['printable_ratio']
                })
        
        # Find significant entropy drops
        if entropy_data:
            avg_entropy = np.mean([d['entropy'] for d in entropy_data])
            significant_drops = [d for d in entropy_data if d['entropy'] < avg_entropy * 0.8]
            
            for drop in significant_drops:
                logging.info(f"[ENTROPY DROP] Offset {drop['offset']}, Entropy: {drop['entropy']:.3f}")
        
        return entropy_data
    
    def create_heatmap(self):
        """Create decode heatmap visualization"""
        logging.info("=== CREATING DECODE HEATMAP ===")
        
        # Prepare data for heatmap
        offsets = []
        printable_ratios = []
        entropies = []
        
        for result in self.results:
            if result:
                offsets.append(result['offset'])
                printable_ratios.append(result['printable_ratio'])
                entropies.append(result['entropy'])
        
        if not offsets:
            logging.warning("No results to create heatmap")
            return
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Printable ratio by offset
        plt.subplot(2, 1, 1)
        plt.scatter(offsets, printable_ratios, alpha=0.6)
        plt.xlabel('Bit Offset')
        plt.ylabel('Printable Ratio')
        plt.title('Printable ASCII Ratio by Offset')
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Entropy by offset
        plt.subplot(2, 1, 2)
        plt.scatter(offsets, entropies, alpha=0.6, color='red')
        plt.xlabel('Bit Offset')
        plt.ylabel('Entropy (bits)')
        plt.title('Entropy by Offset')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('decode_heatmap.png', dpi=300, bbox_inches='tight')
        logging.info("Heatmap saved: decode_heatmap.png")
    
    def save_results(self):
        """Save all results to files"""
        logging.info("=== SAVING RESULTS ===")
        
        # Save candidate hits (filter out non-serializable fields)
        if self.candidates:
            serializable_candidates = []
            for candidate in self.candidates:
                # Remove non-serializable fields for CSV
                serializable = {k: v for k, v in candidate.items() 
                               if k not in ['matrix'] and not isinstance(v, (list, dict))}
                serializable_candidates.append(serializable)
            
            if serializable_candidates:
                with open('candidate_hits.csv', 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=serializable_candidates[0].keys())
                    writer.writeheader()
                    writer.writerows(serializable_candidates)
                logging.info(f"Saved {len(serializable_candidates)} candidate hits to candidate_hits.csv")
        
        # Save all results
        if self.results:
            with open('decode_attempts.json', 'w') as f:
                json.dump(self.results, f, indent=2)
            logging.info(f"Saved {len(self.results)} decode attempts to decode_attempts.json")
    
    def run_complete_analysis(self):
        """Run the complete methodical decoding analysis"""
        logging.info("="*60)
        logging.info("CURSOR AGENT: COMPREHENSIVE BINARY DECODING")
        logging.info("="*60)
        
        # Run all steps
        self.step1_canonical_byte_split()
        self.step2_bit_shifts()
        self.step3_reverse_bit_order()
        self.step4_endianness_swaps()
        self.step5_base64_base58_scan()
        self.step6_wif_key_brute()
        self.step7_qr_matrix_guess()
        self.step8_entropy_monitoring()
        
        # Create visualizations and save results
        self.create_heatmap()
        self.save_results()
        
        # Summary
        logging.info(f"\n=== DECODING ANALYSIS COMPLETE ===")
        logging.info(f"Total decode attempts: {len(self.results)}")
        logging.info(f"Candidate hits found: {len(self.candidates)}")
        logging.info(f"Files generated:")
        logging.info(f"  - decode_attempts.log")
        logging.info(f"  - candidate_hits.csv")
        logging.info(f"  - decode_attempts.json")
        logging.info(f"  - decode_heatmap.png")
        
        # Success assessment
        if self.candidates:
            logging.info("✅ SUCCESS: Found potential candidates!")
            for candidate in self.candidates[:5]:  # Show first 5
                logging.info(f"  - {candidate.get('decode_type', 'unknown')}: {candidate.get('decoded', candidate.get('hex_data', 'N/A'))[:50]}")
        else:
            logging.info("⚠️ NO OBVIOUS DECODES: Stream may be compressed/encrypted")
            logging.info("   Recommend moving to Direction #2 (crypto-stat tests)")

def main():
    """Main function"""
    decoder = BinaryDecoder()
    decoder.run_complete_analysis()

if __name__ == "__main__":
    main() 