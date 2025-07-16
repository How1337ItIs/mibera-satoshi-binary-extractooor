#!/usr/bin/env python3
"""
Comprehensive analysis of the extracted binary data from the Satoshi poster.
Tests for cryptographic patterns, Bitcoin connections, and hidden meanings.
"""

import pandas as pd
import numpy as np
import hashlib
import binascii
import struct
import os
import json
from datetime import datetime
import re
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path

class SatoshiBinaryAnalyzer:
    """Comprehensive analysis of extracted binary data."""
    
    def __init__(self, cells_csv_path):
        self.cells_csv_path = cells_csv_path
        self.cells_df = pd.read_csv(cells_csv_path)
        self.analysis_results = {}
        
        # Extract binary data
        self.binary_cells = self.cells_df[self.cells_df['bit'].isin(['0', '1'])].copy()
        self.binary_cells = self.binary_cells.sort_values(['row', 'col'])
        
        # Create binary string
        self.binary_string = ''.join(self.binary_cells['bit'].values)
        self.total_bits = len(self.binary_string)
        
        print(f"Loaded {self.total_bits} binary bits for analysis")
        
    def basic_statistical_analysis(self):
        """Basic statistical analysis of the binary data."""
        
        ones_count = self.binary_string.count('1')
        zeros_count = self.binary_string.count('0')
        
        stats = {
            'total_bits': self.total_bits,
            'ones_count': ones_count,
            'zeros_count': zeros_count,
            'ones_percentage': (ones_count / self.total_bits) * 100,
            'zeros_percentage': (zeros_count / self.total_bits) * 100,
            'entropy': self.calculate_entropy(self.binary_string),
            'longest_run_0': self.longest_run(self.binary_string, '0'),
            'longest_run_1': self.longest_run(self.binary_string, '1'),
            'randomness_score': self.calculate_randomness_score()
        }
        
        return stats
    
    def calculate_entropy(self, binary_string):
        """Calculate Shannon entropy of binary string."""
        if not binary_string:
            return 0
        
        ones = binary_string.count('1')
        zeros = binary_string.count('0')
        total = len(binary_string)
        
        if ones == 0 or zeros == 0:
            return 0
        
        p1 = ones / total
        p0 = zeros / total
        
        entropy = -p1 * np.log2(p1) - p0 * np.log2(p0)
        return entropy
    
    def longest_run(self, binary_string, char):
        """Find longest consecutive run of character."""
        max_run = 0
        current_run = 0
        
        for c in binary_string:
            if c == char:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        
        return max_run
    
    def calculate_randomness_score(self):
        """Calculate randomness score based on runs and patterns."""
        # Count runs
        runs = []
        current_run = 1
        
        for i in range(1, len(self.binary_string)):
            if self.binary_string[i] == self.binary_string[i-1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)
        
        # Expected runs for random data
        expected_runs = (self.total_bits + 1) / 2
        actual_runs = len(runs)
        
        # Runs test score (closer to 1 = more random)
        runs_score = min(actual_runs, expected_runs) / max(actual_runs, expected_runs)
        
        return runs_score
    
    def find_recurring_patterns(self):
        """Find recurring binary patterns."""
        
        patterns = {}
        
        # Test different pattern lengths
        for length in range(4, 17):  # 4 to 16 bits
            pattern_counts = Counter()
            
            for i in range(len(self.binary_string) - length + 1):
                pattern = self.binary_string[i:i+length]
                pattern_counts[pattern] += 1
            
            # Find patterns that occur more than expected by chance
            expected_frequency = 1 / (2 ** length)
            significant_patterns = []
            
            for pattern, count in pattern_counts.items():
                actual_frequency = count / (len(self.binary_string) - length + 1)
                if actual_frequency > expected_frequency * 3:  # 3x more than expected
                    significant_patterns.append({
                        'pattern': pattern,
                        'count': count,
                        'expected': expected_frequency * (len(self.binary_string) - length + 1),
                        'significance': actual_frequency / expected_frequency
                    })
            
            if significant_patterns:
                patterns[length] = sorted(significant_patterns, key=lambda x: x['significance'], reverse=True)
        
        return patterns
    
    def test_cryptographic_hashes(self):
        """Test if binary data matches known cryptographic hash patterns."""
        
        hash_tests = {}
        
        # Convert binary to bytes for hashing
        if len(self.binary_string) % 8 == 0:
            # Perfect byte alignment
            byte_data = self.binary_to_bytes(self.binary_string)
        else:
            # Pad to byte boundary
            padded = self.binary_string + '0' * (8 - len(self.binary_string) % 8)
            byte_data = self.binary_to_bytes(padded)
        
        # Test common hash lengths
        hash_functions = {
            'MD5': (hashlib.md5, 128),
            'SHA1': (hashlib.sha1, 160),
            'SHA256': (hashlib.sha256, 256),
            'SHA512': (hashlib.sha512, 512),
            'RIPEMD160': (hashlib.new('ripemd160'), 160) if 'ripemd160' in hashlib.algorithms_available else None
        }
        
        for hash_name, (hash_func, expected_bits) in hash_functions.items():
            if hash_func is None:
                continue
                
            if len(self.binary_string) == expected_bits:
                hash_tests[hash_name] = {
                    'length_match': True,
                    'hex_representation': binascii.hexlify(byte_data).decode('ascii'),
                    'potential_hash': True
                }
            else:
                hash_tests[hash_name] = {
                    'length_match': False,
                    'expected_bits': expected_bits,
                    'actual_bits': len(self.binary_string)
                }
        
        return hash_tests
    
    def binary_to_bytes(self, binary_string):
        """Convert binary string to bytes."""
        byte_data = bytearray()
        for i in range(0, len(binary_string), 8):
            byte = binary_string[i:i+8]
            if len(byte) == 8:
                byte_data.append(int(byte, 2))
        return bytes(byte_data)
    
    def test_bitcoin_patterns(self):
        """Test for Bitcoin-related patterns."""
        
        bitcoin_tests = {}
        
        # Test for Bitcoin address patterns (Base58 when converted)
        try:
            byte_data = self.binary_to_bytes(self.binary_string)
            hex_string = binascii.hexlify(byte_data).decode('ascii')
            
            bitcoin_tests['hex_representation'] = hex_string
            bitcoin_tests['possible_private_key'] = len(hex_string) == 64  # 32 bytes = 256 bits
            bitcoin_tests['possible_public_key'] = len(hex_string) == 66 or len(hex_string) == 130  # Compressed or uncompressed
            
            # Test for specific Bitcoin constants
            bitcoin_constants = {
                'genesis_block_hash': '000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f',
                'genesis_block_merkle': '4a5e1e4baab89f3a32518a88c31bc87f618f76673e2cc77ab2127b7afdeda33b',
                'satoshi_coinbase': '04678afdb0fe5548271967f1a67130b7105cd6a828e03909a67962e0ea1f61deb649f6bc3f4cef38c4f35504e51ec112de5c384df7ba0b8d578a4c702b6bf11d5f'
            }
            
            for const_name, const_value in bitcoin_constants.items():
                if const_value in hex_string.lower():
                    bitcoin_tests[f'contains_{const_name}'] = True
                else:
                    bitcoin_tests[f'contains_{const_name}'] = False
                    
        except Exception as e:
            bitcoin_tests['error'] = str(e)
        
        return bitcoin_tests
    
    def test_ascii_encoding(self):
        """Test for ASCII text encoding."""
        
        ascii_tests = {}
        
        try:
            byte_data = self.binary_to_bytes(self.binary_string)
            
            # Test direct ASCII conversion
            ascii_text = ''
            printable_chars = 0
            
            for byte in byte_data:
                if 32 <= byte <= 126:  # Printable ASCII range
                    ascii_text += chr(byte)
                    printable_chars += 1
                else:
                    ascii_text += '.'
            
            ascii_tests['ascii_text'] = ascii_text
            ascii_tests['printable_percentage'] = (printable_chars / len(byte_data)) * 100
            ascii_tests['likely_text'] = printable_chars > len(byte_data) * 0.7
            
            # Look for common words
            common_words = ['bitcoin', 'satoshi', 'nakamoto', 'crypto', 'key', 'hash', 'block', 'chain']
            found_words = []
            
            for word in common_words:
                if word.lower() in ascii_text.lower():
                    found_words.append(word)
            
            ascii_tests['found_words'] = found_words
            
        except Exception as e:
            ascii_tests['error'] = str(e)
        
        return ascii_tests
    
    def test_mathematical_constants(self):
        """Test for mathematical constants."""
        
        math_tests = {}
        
        # Convert to decimal for comparison
        if len(self.binary_string) <= 64:  # Prevent overflow
            decimal_value = int(self.binary_string, 2)
            math_tests['decimal_value'] = decimal_value
        
        # Test for known constants in binary representation
        known_constants = {
            'pi_decimal': '3141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067',
            'e_decimal': '2718281828459045235360287471352662497757247093699959574966967627724076630353547594571382178525166427',
            'phi_decimal': '1618033988749894848204586834365638117720309179805762862135448622705260462818902449707207204189391137'
        }
        
        binary_str = self.binary_string
        for const_name, const_value in known_constants.items():
            # Check if any part of the binary string matches the constant
            const_binary = bin(int(const_value[:min(len(const_value), 20)]))[2:]  # Take first 20 digits
            
            if const_binary in binary_str:
                math_tests[f'contains_{const_name}'] = True
            else:
                math_tests[f'contains_{const_name}'] = False
        
        return math_tests
    
    def analyze_grid_structure(self):
        """Analyze the 2D grid structure for patterns."""
        
        grid_analysis = {}
        
        # Reconstruct 2D grid
        grid = np.full((54, 50), -1, dtype=int)  # -1 for missing data
        
        for _, row in self.binary_cells.iterrows():
            r, c = row['row'], row['col']
            if 0 <= r < 54 and 0 <= c < 50:
                grid[r][c] = int(row['bit'])
        
        # Analyze rows and columns
        row_patterns = []
        col_patterns = []
        
        for r in range(54):
            row_data = grid[r]
            valid_bits = row_data[row_data != -1]
            if len(valid_bits) > 0:
                row_patterns.append({
                    'row': r,
                    'bits': ''.join(map(str, valid_bits)),
                    'ones_count': np.sum(valid_bits == 1),
                    'zeros_count': np.sum(valid_bits == 0),
                    'completeness': len(valid_bits) / 50
                })
        
        for c in range(50):
            col_data = grid[:, c]
            valid_bits = col_data[col_data != -1]
            if len(valid_bits) > 0:
                col_patterns.append({
                    'col': c,
                    'bits': ''.join(map(str, valid_bits)),
                    'ones_count': np.sum(valid_bits == 1),
                    'zeros_count': np.sum(valid_bits == 0),
                    'completeness': len(valid_bits) / 54
                })
        
        grid_analysis['row_patterns'] = row_patterns
        grid_analysis['col_patterns'] = col_patterns
        grid_analysis['grid_completeness'] = len(self.binary_cells) / (54 * 50)
        
        return grid_analysis
    
    def comprehensive_analysis(self):
        """Run all analysis methods."""
        
        print("Running comprehensive binary data analysis...")
        
        # Basic statistics
        print("1. Basic statistical analysis...")
        self.analysis_results['basic_stats'] = self.basic_statistical_analysis()
        
        # Pattern analysis
        print("2. Finding recurring patterns...")
        self.analysis_results['recurring_patterns'] = self.find_recurring_patterns()
        
        # Cryptographic tests
        print("3. Testing cryptographic hash patterns...")
        self.analysis_results['crypto_hashes'] = self.test_cryptographic_hashes()
        
        # Bitcoin analysis
        print("4. Testing Bitcoin-related patterns...")
        self.analysis_results['bitcoin_patterns'] = self.test_bitcoin_patterns()
        
        # ASCII encoding
        print("5. Testing ASCII encoding...")
        self.analysis_results['ascii_encoding'] = self.test_ascii_encoding()
        
        # Mathematical constants
        print("6. Testing mathematical constants...")
        self.analysis_results['math_constants'] = self.test_mathematical_constants()
        
        # Grid structure
        print("7. Analyzing grid structure...")
        self.analysis_results['grid_structure'] = self.analyze_grid_structure()
        
        return self.analysis_results
    
    def generate_analysis_report(self):
        """Generate comprehensive analysis report."""
        
        report = []
        report.append("# Comprehensive Binary Data Analysis Report")
        report.append("## Satoshi Poster Hidden Data Analysis")
        report.append("")
        report.append(f"**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Total Bits Analyzed**: {self.total_bits}")
        report.append("")
        
        # Basic Statistics
        stats = self.analysis_results['basic_stats']
        report.append("## 1. Basic Statistical Analysis")
        report.append("")
        report.append(f"- **Total Bits**: {stats['total_bits']}")
        report.append(f"- **Ones**: {stats['ones_count']} ({stats['ones_percentage']:.1f}%)")
        report.append(f"- **Zeros**: {stats['zeros_count']} ({stats['zeros_percentage']:.1f}%)")
        report.append(f"- **Entropy**: {stats['entropy']:.4f} (max: 1.0)")
        report.append(f"- **Longest Run of 0s**: {stats['longest_run_0']}")
        report.append(f"- **Longest Run of 1s**: {stats['longest_run_1']}")
        report.append(f"- **Randomness Score**: {stats['randomness_score']:.4f}")
        report.append("")
        
        # Significant Patterns
        patterns = self.analysis_results['recurring_patterns']
        if patterns:
            report.append("## 2. Significant Recurring Patterns")
            report.append("")
            
            for length, pattern_list in patterns.items():
                if pattern_list:
                    report.append(f"### {length}-bit Patterns")
                    report.append("")
                    report.append("| Pattern | Count | Expected | Significance |")
                    report.append("|---------|-------|----------|--------------|")
                    
                    for pattern_info in pattern_list[:5]:  # Top 5 patterns
                        report.append(f"| {pattern_info['pattern']} | {pattern_info['count']} | {pattern_info['expected']:.1f} | {pattern_info['significance']:.1f}x |")
                    
                    report.append("")
        else:
            report.append("## 2. Recurring Patterns")
            report.append("")
            report.append("No significant recurring patterns found above random chance.")
            report.append("")
        
        # Cryptographic Analysis
        crypto = self.analysis_results['crypto_hashes']
        report.append("## 3. Cryptographic Hash Analysis")
        report.append("")
        report.append("| Hash Type | Length Match | Potential Match |")
        report.append("|-----------|--------------|-----------------|")
        
        for hash_name, hash_info in crypto.items():
            length_match = hash_info.get('length_match', False)
            potential = hash_info.get('potential_hash', False)
            report.append(f"| {hash_name} | {length_match} | {potential} |")
        
        report.append("")
        
        # Bitcoin Analysis
        bitcoin = self.analysis_results['bitcoin_patterns']
        report.append("## 4. Bitcoin Pattern Analysis")
        report.append("")
        
        if 'hex_representation' in bitcoin:
            report.append(f"**Hex Representation**: `{bitcoin['hex_representation'][:100]}...`")
            report.append("")
        
        report.append("| Test | Result |")
        report.append("|------|--------|")
        report.append(f"| Possible Private Key (32 bytes) | {bitcoin.get('possible_private_key', False)} |")
        report.append(f"| Possible Public Key | {bitcoin.get('possible_public_key', False)} |")
        
        # Check for Bitcoin constants
        for key, value in bitcoin.items():
            if key.startswith('contains_'):
                const_name = key.replace('contains_', '').replace('_', ' ').title()
                report.append(f"| Contains {const_name} | {value} |")
        
        report.append("")
        
        # ASCII Analysis
        ascii_data = self.analysis_results['ascii_encoding']
        report.append("## 5. ASCII Encoding Analysis")
        report.append("")
        
        if 'ascii_text' in ascii_data:
            report.append(f"**Printable Characters**: {ascii_data['printable_percentage']:.1f}%")
            report.append(f"**Likely Text**: {ascii_data['likely_text']}")
            report.append("")
            
            if ascii_data['found_words']:
                report.append(f"**Found Keywords**: {', '.join(ascii_data['found_words'])}")
            else:
                report.append("**Found Keywords**: None")
            
            report.append("")
            report.append("**ASCII Preview**:")
            report.append(f"```")
            report.append(ascii_data['ascii_text'][:200] + "..." if len(ascii_data['ascii_text']) > 200 else ascii_data['ascii_text'])
            report.append("```")
            report.append("")
        
        # Mathematical Constants
        math_data = self.analysis_results['math_constants']
        report.append("## 6. Mathematical Constants Analysis")
        report.append("")
        
        if 'decimal_value' in math_data:
            report.append(f"**Decimal Value**: {math_data['decimal_value']}")
        
        report.append("")
        report.append("| Constant | Found in Data |")
        report.append("|----------|---------------|")
        
        for key, value in math_data.items():
            if key.startswith('contains_'):
                const_name = key.replace('contains_', '').replace('_', ' ').title()
                report.append(f"| {const_name} | {value} |")
        
        report.append("")
        
        # Grid Structure
        grid_data = self.analysis_results['grid_structure']
        report.append("## 7. Grid Structure Analysis")
        report.append("")
        report.append(f"**Grid Completeness**: {grid_data['grid_completeness']:.1f}%")
        report.append("")
        
        # Conclusions
        report.append("## 8. Analysis Conclusions")
        report.append("")
        
        # Determine most likely interpretation
        if stats['entropy'] > 0.9:
            report.append("### High Entropy Data")
            report.append("- Data appears highly random or encrypted")
            report.append("- Could be cryptographic hash or encrypted content")
            report.append("- Unlikely to be plain text")
        elif stats['entropy'] > 0.7:
            report.append("### Medium Entropy Data")
            report.append("- Data shows some structure but significant randomness")
            report.append("- Could be compressed data or mixed content")
        else:
            report.append("### Low Entropy Data")
            report.append("- Data shows clear patterns and structure")
            report.append("- More likely to be readable content or simple encoding")
        
        report.append("")
        
        # Recommendations
        report.append("## 9. Recommendations for Further Analysis")
        report.append("")
        report.append("1. **Cryptographic Analysis**: Test against known Bitcoin transactions and blocks")
        report.append("2. **Pattern Matching**: Compare with Satoshi's known writings and code")
        report.append("3. **Steganographic Analysis**: Check for hidden layers or encoding")
        report.append("4. **Historical Context**: Correlate with Bitcoin development timeline")
        report.append("5. **Expert Review**: Have cryptography experts analyze the data")
        
        return "\n".join(report)

def main():
    """Main analysis function."""
    
    cells_csv = "output_final/cells.csv"
    
    if not os.path.exists(cells_csv):
        print("Error: cells.csv not found")
        return
    
    # Create output directory
    os.makedirs("test_results/binary_analysis", exist_ok=True)
    
    # Initialize analyzer
    analyzer = SatoshiBinaryAnalyzer(cells_csv)
    
    # Run comprehensive analysis
    results = analyzer.comprehensive_analysis()
    
    # Generate report
    report = analyzer.generate_analysis_report()
    
    # Save results
    with open("test_results/binary_analysis/analysis_results.json", 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Deep convert the results
        json_results = json.loads(json.dumps(results, default=convert_numpy))
        json.dump(json_results, f, indent=2)
    
    # Save report
    with open("test_results/binary_analysis/BINARY_ANALYSIS_REPORT.md", 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n=== Binary Data Analysis Complete ===")
    print(f"Analyzed {analyzer.total_bits} bits")
    print(f"Results saved to: test_results/binary_analysis/")
    print(f"Report: test_results/binary_analysis/BINARY_ANALYSIS_REPORT.md")

if __name__ == "__main__":
    main()