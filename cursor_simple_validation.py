#!/usr/bin/env python3
"""
CURSOR AGENT: Simple Validation Check
====================================

Quick validation to check if optimization results are rational.

Created by: Cursor Agent
Date: July 16, 2025
"""

import pandas as pd
import numpy as np

def main():
    print("="*50)
    print("CURSOR AGENT: SIMPLE VALIDATION CHECK")
    print("="*50)
    
    # Load data
    try:
        original_data = pd.read_csv("complete_extraction_binary_only.csv")
        optimized_data = pd.read_csv("cursor_optimized_extraction.csv")
        print("✅ Data files loaded successfully")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
        
    # Basic statistics
    print(f"\n=== BASIC STATISTICS ===")
    print(f"Original data: {len(original_data)} cells")
    print(f"Optimized data: {len(optimized_data)} cells")
    
    # Bit distribution
    orig_zeros = len(original_data[original_data['bit'] == 0.0])
    orig_ones = len(original_data[original_data['bit'] == 1.0])
    opt_zeros = len(optimized_data[optimized_data['bit'] == 0])
    opt_ones = len(optimized_data[optimized_data['bit'] == 1])
    
    print(f"\nBit distribution:")
    print(f"Original: {orig_zeros} zeros ({orig_zeros/len(original_data)*100:.1f}%), {orig_ones} ones ({orig_ones/len(original_data)*100:.1f}%)")
    print(f"Optimized: {opt_zeros} zeros ({opt_zeros/len(optimized_data)*100:.1f}%), {opt_ones} ones ({opt_ones/len(optimized_data)*100:.1f}%)")
    
    # Improvement metrics
    improvement_factor = opt_zeros / orig_zeros if orig_zeros > 0 else float('inf')
    print(f"\nImprovement factor: {improvement_factor:.1f}x more zeros")
    
    # Coverage improvement
    coverage_improvement = len(optimized_data) / len(original_data)
    print(f"Coverage improvement: {coverage_improvement:.1f}x more cells")
    
    # Check for rationality
    print(f"\n=== RATIONALITY CHECKS ===")
    
    # Check 1: Did we get more zeros?
    if opt_zeros > orig_zeros:
        print("✅ PASS: More zeros extracted")
    else:
        print("❌ FAIL: No improvement in zero extraction")
        
    # Check 2: Is the distribution more balanced?
    orig_ratio = orig_ones / orig_zeros if orig_zeros > 0 else float('inf')
    opt_ratio = opt_ones / opt_zeros if opt_zeros > 0 else float('inf')
    
    if opt_ratio < orig_ratio:
        print("✅ PASS: Distribution is more balanced")
    else:
        print("❌ FAIL: Distribution is not more balanced")
        
    # Check 3: Did we increase coverage?
    if len(optimized_data) > len(original_data):
        print("✅ PASS: Increased coverage")
    else:
        print("❌ FAIL: No coverage increase")
        
    # Check 4: Are the numbers reasonable?
    if opt_zeros > 0 and opt_ones > 0:
        print("✅ PASS: Both zeros and ones present")
    else:
        print("❌ FAIL: Missing one bit type")
        
    # Check 5: Is the improvement dramatic enough?
    if improvement_factor > 10:
        print("✅ PASS: Dramatic improvement (10x+)")
    elif improvement_factor > 5:
        print("✅ PASS: Significant improvement (5x+)")
    elif improvement_factor > 2:
        print("⚠️ MODERATE: Some improvement (2x+)")
    else:
        print("❌ FAIL: Minimal improvement")
        
    # Analyze threshold
    print(f"\n=== THRESHOLD ANALYSIS ===")
    if 'blue_mean' in optimized_data.columns and 'threshold' in optimized_data.columns:
        blue_means = optimized_data['blue_mean'].dropna()
        threshold = optimized_data['threshold'].iloc[0]
        
        print(f"Applied threshold: {threshold}")
        print(f"Blue channel range: {blue_means.min():.1f} - {blue_means.max():.1f}")
        print(f"Blue channel mean: {blue_means.mean():.1f}")
        
        # Check if threshold makes sense
        zeros_above = len(blue_means[blue_means >= threshold])
        ones_below = len(blue_means[blue_means < threshold])
        
        print(f"Cells above threshold: {zeros_above}")
        print(f"Cells below threshold: {ones_below}")
        
        if zeros_above > ones_below:
            print("✅ PASS: Threshold logic appears correct")
        else:
            print("❌ FAIL: Threshold logic may be inverted")
    else:
        print("⚠️ No threshold data available")
        
    # Overall assessment
    print(f"\n=== OVERALL ASSESSMENT ===")
    
    passes = 0
    total_checks = 5
    
    if opt_zeros > orig_zeros: passes += 1
    if opt_ratio < orig_ratio: passes += 1
    if len(optimized_data) > len(original_data): passes += 1
    if opt_zeros > 0 and opt_ones > 0: passes += 1
    if improvement_factor > 2: passes += 1
    
    print(f"Passed {passes}/{total_checks} rationality checks")
    
    if passes >= 4:
        print("✅ HIGH CONFIDENCE: Results appear rational")
    elif passes >= 3:
        print("✅ MODERATE CONFIDENCE: Results mostly rational")
    else:
        print("❌ LOW CONFIDENCE: Results may not be rational")
        
    print(f"\nValidation complete. Review the data and visualizations for final confirmation.")

if __name__ == "__main__":
    main() 