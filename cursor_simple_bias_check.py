#!/usr/bin/env python3
"""
CURSOR AGENT: Simple Bit Bias Check
==================================

Quick analysis of the bit bias issue to understand the 93.5% ones vs 6.5% zeros problem.

Created by: Cursor Agent
Date: July 16, 2025
"""

import json
import pandas as pd
import numpy as np
from PIL import Image
from collections import defaultdict

def main():
    print("="*50)
    print("CURSOR AGENT: SIMPLE BIT BIAS CHECK")
    print("="*50)
    
    # Load data
    try:
        binary_data = pd.read_csv("complete_extraction_binary_only.csv")
        print(f"Loaded {len(binary_data)} binary classifications")
    except Exception as e:
        print(f"Error loading binary data: {e}")
        return
        
    # Basic bit distribution
    print("\n=== BASIC BIT DISTRIBUTION ===")
    bit_counts = binary_data['bit'].value_counts()
    print(f"Bit counts: {bit_counts.to_dict()}")
    
    total_bits = len(binary_data)
    zeros = bit_counts.get(0.0, 0)
    ones = bit_counts.get(1.0, 0)
    
    print(f"Total bits: {total_bits}")
    print(f"Zeros: {zeros} ({zeros/total_bits*100:.1f}%)")
    print(f"Ones: {ones} ({ones/total_bits*100:.1f}%)")
    print(f"Ratio (1s:0s): {ones/zeros:.1f}" if zeros > 0 else "Ratio: ∞ (no zeros)")
    
    # Check for bias
    if zeros/total_bits < 0.1:
        print("\n🚨 CRITICAL ISSUE: Severe bit bias detected!")
        print("   - Less than 10% zeros suggests classification error")
        print("   - May be extracting background instead of digits")
    else:
        print("\n✅ Bit distribution appears reasonable")
        
    # Region analysis
    print("\n=== REGION ANALYSIS ===")
    region_bits = defaultdict(lambda: {'0': 0, '1': 0})
    
    for _, row in binary_data.iterrows():
        region_id = int(row['region_id'])
        bit = int(row['bit'])
        region_bits[region_id][str(bit)] += 1
        
    print("Region | Zeros | Ones | Total | % Zeros")
    print("-" * 45)
    
    for region_id in sorted(region_bits.keys()):
        zeros = region_bits[region_id]['0']
        ones = region_bits[region_id]['1']
        total = zeros + ones
        pct_zeros = zeros/total*100 if total > 0 else 0
        
        print(f"{region_id:6d} | {zeros:5d} | {ones:4d} | {total:5d} | {pct_zeros:6.1f}%")
        
    # Confidence analysis
    print("\n=== CONFIDENCE ANALYSIS ===")
    confidences = binary_data['confidence']
    print(f"Confidence range: {confidences.min():.1f} - {confidences.max():.1f}")
    print(f"Mean confidence: {confidences.mean():.1f}")
    print(f"Median confidence: {confidences.median():.1f}")
    
    # Check if zeros have different confidence than ones
    zero_conf = binary_data[binary_data['bit'] == 0.0]['confidence']
    one_conf = binary_data[binary_data['bit'] == 1.0]['confidence']
    
    if len(zero_conf) > 0 and len(one_conf) > 0:
        print(f"\nConfidence by bit value:")
        print(f"  Zeros: {zero_conf.mean():.1f} ± {zero_conf.std():.1f}")
        print(f"  Ones: {one_conf.mean():.1f} ± {one_conf.std():.1f}")
        
    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Overall clarity rate: {total_bits/1440*100:.1f}%")
    print(f"Bit bias severity: {ones/zeros:.1f}x more ones than zeros" if zeros > 0 else "Infinite bias (no zeros)")
    
    if zeros/total_bits < 0.1:
        print("\n🔧 RECOMMENDATIONS:")
        print("1. Check if blue channel threshold is too low")
        print("2. Verify we're extracting actual digits, not background")
        print("3. Test alternative color channels or thresholds")
        print("4. Manually inspect some extracted cells")

if __name__ == "__main__":
    main() 