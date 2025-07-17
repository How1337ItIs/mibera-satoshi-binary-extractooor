#!/usr/bin/env python3
"""
CURSOR AGENT: Cryptographic Analysis Tool
========================================

Analyzes the 1,440 optimized binary digits for cryptographic patterns,
entropy, and potential hidden information.

Created by: Cursor Agent
Date: July 16, 2025
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import hashlib
import binascii
from scipy import stats
import re

class CryptographicAnalyzer:
    def __init__(self):
        self.optimized_data = pd.read_csv("cursor_optimized_extraction.csv")
        self.binary_sequence = self.load_binary_sequence()
        
    def load_binary_sequence(self):
        """Load the binary sequence from optimized data"""
        # Sort by region_id, local_row, local_col to maintain order
        sorted_data = self.optimized_data.sort_values(['region_id', 'local_row', 'local_col'])
        binary_string = ''.join(str(int(bit)) for bit in sorted_data['bit'])
        return binary_string
        
    def analyze_entropy(self):
        """Analyze entropy and randomness of the binary sequence"""
        print("=== ENTROPY ANALYSIS ===")
        
        binary_sequence = self.binary_sequence
        length = len(binary_sequence)
        
        # Calculate Shannon entropy
        ones = binary_sequence.count('1')
        zeros = binary_sequence.count('0')
        
        if ones > 0 and zeros > 0:
            p1 = ones / length
            p0 = zeros / length
            entropy = -p1 * np.log2(p1) - p0 * np.log2(p0)
        else:
            entropy = 0
            
        print(f"Sequence length: {length} bits")
        print(f"Zeros: {zeros} ({zeros/length*100:.1f}%)")
        print(f"Ones: {ones} ({ones/length*100:.1f}%)")
        print(f"Shannon entropy: {entropy:.3f} bits")
        print(f"Expected for random: 1.000 bits")
        print(f"Entropy ratio: {entropy:.1%}")
        
        # Entropy assessment
        if entropy > 0.9:
            print("✅ HIGH ENTROPY: Very random-like distribution")
        elif entropy > 0.7:
            print("✅ GOOD ENTROPY: Reasonably random distribution")
        elif entropy > 0.5:
            print("⚠️ MODERATE ENTROPY: Some randomness, some structure")
        else:
            print("🚨 LOW ENTROPY: Highly structured or biased")
            
        return entropy
        
    def analyze_patterns(self):
        """Analyze patterns in the binary sequence"""
        print("\n=== PATTERN ANALYSIS ===")
        
        binary_sequence = self.binary_sequence
        length = len(binary_sequence)
        
        # Analyze consecutive patterns
        consecutive_ones = 0
        consecutive_zeros = 0
        alternating = 0
        
        for i in range(length - 1):
            if binary_sequence[i] == binary_sequence[i+1]:
                if binary_sequence[i] == '1':
                    consecutive_ones += 1
                else:
                    consecutive_zeros += 1
            else:
                alternating += 1
                
        print(f"Consecutive ones: {consecutive_ones} ({consecutive_ones/(length-1)*100:.1f}%)")
        print(f"Consecutive zeros: {consecutive_zeros} ({consecutive_zeros/(length-1)*100:.1f}%)")
        print(f"Alternating: {alternating} ({alternating/(length-1)*100:.1f}%)")
        
        # Look for repeating patterns
        print(f"\nRepeating pattern analysis:")
        
        for pattern_length in [2, 3, 4, 8, 16]:
            patterns = {}
            for i in range(length - pattern_length + 1):
                pattern = binary_sequence[i:i+pattern_length]
                patterns[pattern] = patterns.get(pattern, 0) + 1
                
            # Find most common patterns
            most_common = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:3]
            
            print(f"  {pattern_length}-bit patterns:")
            for pattern, count in most_common:
                percentage = count / (length - pattern_length + 1) * 100
                print(f"    '{pattern}': {count} times ({percentage:.1f}%)")
                
    def analyze_cryptographic_properties(self):
        """Analyze cryptographic properties of the sequence"""
        print("\n=== CRYPTOGRAPHIC PROPERTIES ===")
        
        binary_sequence = self.binary_sequence
        
        # Convert to bytes for cryptographic analysis
        # Pad to multiple of 8 if needed
        if len(binary_sequence) % 8 != 0:
            binary_sequence += '0' * (8 - len(binary_sequence) % 8)
            
        # Convert to bytes
        try:
            bytes_data = bytes(int(binary_sequence[i:i+8], 2) for i in range(0, len(binary_sequence), 8))
            
            # Calculate various hashes
            md5_hash = hashlib.md5(bytes_data).hexdigest()
            sha1_hash = hashlib.sha1(bytes_data).hexdigest()
            sha256_hash = hashlib.sha256(bytes_data).hexdigest()
            
            print(f"MD5 hash: {md5_hash}")
            print(f"SHA1 hash: {sha1_hash}")
            print(f"SHA256 hash: {sha256_hash}")
            
            # Check for known patterns in hashes
            print(f"\nHash analysis:")
            
            # Look for patterns in hash
            if re.search(r'(.)\1{3,}', md5_hash):
                print("  ⚠️ MD5 contains repeating characters")
            if re.search(r'(.)\1{3,}', sha1_hash):
                print("  ⚠️ SHA1 contains repeating characters")
            if re.search(r'(.)\1{3,}', sha256_hash):
                print("  ⚠️ SHA256 contains repeating characters")
                
        except Exception as e:
            print(f"Error in cryptographic analysis: {e}")
            
    def analyze_bit_distribution_by_region(self):
        """Analyze bit distribution across different regions"""
        print("\n=== REGION-BASED ANALYSIS ===")
        
        region_stats = defaultdict(lambda: {'zeros': 0, 'ones': 0, 'total': 0})
        
        for _, row in self.optimized_data.iterrows():
            region_id = row['region_id']
            bit = int(row['bit'])
            region_stats[region_id]['total'] += 1
            if bit == 0:
                region_stats[region_id]['zeros'] += 1
            else:
                region_stats[region_id]['ones'] += 1
                
        print("Region | Zeros | Ones | Total | % Zeros | Entropy")
        print("-" * 55)
        
        region_entropies = {}
        
        for region_id in sorted(region_stats.keys()):
            stats = region_stats[region_id]
            zeros = stats['zeros']
            ones = stats['ones']
            total = stats['total']
            
            if zeros > 0 and ones > 0:
                p0 = zeros / total
                p1 = ones / total
                entropy = -p0 * np.log2(p0) - p1 * np.log2(p1)
            else:
                entropy = 0
                
            region_entropies[region_id] = entropy
            
            print(f"{int(region_id):6d} | {zeros:5d} | {ones:4d} | {total:5d} | {zeros/total*100:7.1f}% | {entropy:6.3f}")
            
        # Find regions with highest entropy
        high_entropy_regions = sorted(region_entropies.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\nHighest entropy regions:")
        for region_id, entropy in high_entropy_regions:
            print(f"  Region {region_id}: {entropy:.3f} bits")
            
        return region_entropies
        
    def look_for_known_patterns(self):
        """Look for known cryptographic or encoding patterns"""
        print("\n=== KNOWN PATTERN SEARCH ===")
        
        binary_sequence = self.binary_sequence
        
        # Look for Bitcoin-related patterns
        bitcoin_patterns = {
            'private_key_start': '1',  # Bitcoin private keys often start with 1
            'public_key_start': '02',  # Compressed public keys start with 02
            'address_start': '1',      # Bitcoin addresses start with 1
        }
        
        print("Bitcoin-related pattern search:")
        for pattern_name, pattern in bitcoin_patterns.items():
            count = binary_sequence.count(pattern)
            print(f"  {pattern_name}: {count} occurrences")
            
        # Look for ASCII patterns (8-bit chunks)
        print(f"\nASCII pattern analysis:")
        ascii_chars = []
        for i in range(0, len(binary_sequence), 8):
            if i + 8 <= len(binary_sequence):
                byte_str = binary_sequence[i:i+8]
                try:
                    ascii_val = int(byte_str, 2)
                    if 32 <= ascii_val <= 126:  # Printable ASCII
                        ascii_chars.append(chr(ascii_val))
                    else:
                        ascii_chars.append('.')
                except:
                    ascii_chars.append('.')
                    
        ascii_string = ''.join(ascii_chars)
        print(f"ASCII representation (first 100 chars): {ascii_string[:100]}")
        
        # Look for repeating sequences
        print(f"\nRepeating sequence search:")
        for length in [4, 8, 16, 32]:
            sequences = {}
            for i in range(len(binary_sequence) - length + 1):
                seq = binary_sequence[i:i+length]
                sequences[seq] = sequences.get(seq, 0) + 1
                
            # Find sequences that repeat
            repeating = {seq: count for seq, count in sequences.items() if count > 1}
            if repeating:
                most_repeated = max(repeating.items(), key=lambda x: x[1])
                print(f"  {length}-bit sequences: {len(repeating)} repeating, most common: '{most_repeated[0]}' ({most_repeated[1]} times)")
            else:
                print(f"  {length}-bit sequences: No repeating patterns")
                
    def create_visualization(self):
        """Create visualizations of the binary sequence"""
        print("\n=== CREATING VISUALIZATIONS ===")
        
        binary_sequence = self.binary_sequence
        length = len(binary_sequence)
        
        # Create bit distribution plot
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Bit distribution
        plt.subplot(2, 2, 1)
        bit_counts = [binary_sequence.count('0'), binary_sequence.count('1')]
        plt.pie(bit_counts, labels=['Zeros', 'Ones'], autopct='%1.1f%%', startangle=90)
        plt.title('Bit Distribution')
        
        # Subplot 2: Entropy by region
        plt.subplot(2, 2, 2)
        region_entropies = self.analyze_bit_distribution_by_region()
        regions = list(region_entropies.keys())
        entropies = list(region_entropies.values())
        plt.bar(regions, entropies)
        plt.xlabel('Region ID')
        plt.ylabel('Entropy (bits)')
        plt.title('Entropy by Region')
        
        # Subplot 3: Bit sequence visualization (first 200 bits)
        plt.subplot(2, 2, 3)
        sample_length = min(200, length)
        sample_bits = [int(bit) for bit in binary_sequence[:sample_length]]
        plt.plot(range(sample_length), sample_bits, 'b-', linewidth=0.5)
        plt.xlabel('Bit Position')
        plt.ylabel('Bit Value')
        plt.title(f'Bit Sequence (first {sample_length} bits)')
        plt.ylim(-0.1, 1.1)
        
        # Subplot 4: Pattern frequency
        plt.subplot(2, 2, 4)
        patterns = {}
        for i in range(0, length - 7, 8):
            pattern = binary_sequence[i:i+8]
            patterns[pattern] = patterns.get(pattern, 0) + 1
            
        if patterns:
            most_common = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:10]
            pattern_names = [f"'{p[0]}'" for p in most_common]
            frequencies = [p[1] for p in most_common]
            plt.barh(range(len(pattern_names)), frequencies)
            plt.yticks(range(len(pattern_names)), pattern_names)
            plt.xlabel('Frequency')
            plt.title('Most Common 8-bit Patterns')
            
        plt.tight_layout()
        plt.savefig('cursor_cryptographic_analysis.png', dpi=300, bbox_inches='tight')
        print("Visualization saved: cursor_cryptographic_analysis.png")
        
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n=== GENERATING SUMMARY REPORT ===")
        
        # Calculate key metrics
        entropy = self.analyze_entropy()
        region_entropies = self.analyze_bit_distribution_by_region()
        
        # Create summary
        summary = {
            'sequence_length': len(self.binary_sequence),
            'zeros': self.binary_sequence.count('0'),
            'ones': self.binary_sequence.count('1'),
            'zero_percentage': self.binary_sequence.count('0') / len(self.binary_sequence) * 100,
            'entropy': entropy,
            'entropy_ratio': entropy,
            'region_entropies': dict(region_entropies),
            'average_region_entropy': np.mean(list(region_entropies.values())),
            'max_region_entropy': max(region_entropies.values()),
            'min_region_entropy': min(region_entropies.values()),
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Save summary
        with open('cursor_cryptographic_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
            
        print("Summary report saved: cursor_cryptographic_summary.json")
        
        # Print key findings
        print(f"\n=== KEY FINDINGS ===")
        print(f"Sequence length: {summary['sequence_length']} bits")
        print(f"Bit distribution: {summary['zeros']} zeros, {summary['ones']} ones")
        print(f"Overall entropy: {summary['entropy']:.3f} bits ({summary['entropy_ratio']:.1%} of max)")
        print(f"Average region entropy: {summary['average_region_entropy']:.3f} bits")
        print(f"Entropy range: {summary['min_region_entropy']:.3f} - {summary['max_region_entropy']:.3f} bits")
        
        # Assessment
        if summary['entropy'] > 0.8:
            print("✅ HIGH QUALITY: Sequence shows good randomness")
        elif summary['entropy'] > 0.6:
            print("✅ MODERATE QUALITY: Sequence has some randomness")
        else:
            print("⚠️ LOW QUALITY: Sequence appears highly structured")
            
        return summary
        
    def run_complete_analysis(self):
        """Run the complete cryptographic analysis"""
        print("="*60)
        print("CURSOR AGENT: CRYPTOGRAPHIC ANALYSIS")
        print("="*60)
        
        # Run all analyses
        self.analyze_entropy()
        self.analyze_patterns()
        self.analyze_cryptographic_properties()
        self.analyze_bit_distribution_by_region()
        self.look_for_known_patterns()
        
        # Create visualizations
        self.create_visualization()
        
        # Generate summary
        summary = self.generate_summary_report()
        
        print(f"\n=== ANALYSIS COMPLETE ===")
        print("Check the generated files for detailed results:")
        print("  - cursor_cryptographic_analysis.png")
        print("  - cursor_cryptographic_summary.json")

def main():
    """Main analysis function"""
    analyzer = CryptographicAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main() 