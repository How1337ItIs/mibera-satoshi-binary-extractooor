#!/usr/bin/env python3
"""
Alternative grid alignment analysis using raw bit dump data.
Test different grid positions and extraction methods to address verification failures.
"""

import cv2
import numpy as np
import pandas as pd
import hashlib
import json
from scipy import signal
from collections import Counter

def load_raw_bit_dump():
    """Load the raw bit dump CSV for reanalysis."""
    
    print("=== LOADING RAW BIT DUMP ===")
    
    try:
        df = pd.read_csv('raw_bit_dump.csv')
        print(f"Loaded {len(df)} bit records")
        
        # Verify checksum
        with open('raw_bit_dump.csv', 'r') as f:
            content = f.read()
        checksum = hashlib.sha256(content.encode()).hexdigest()
        print(f"Checksum: {checksum[:16]}...")
        
        return df
    except Exception as e:
        print(f"Error loading bit dump: {e}")
        return None

def test_grid_offset_variations():
    """Test systematic grid offset variations."""
    
    print("\n=== GRID OFFSET VARIATION ANALYSIS ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Base configuration from verification
    base_row, base_col = 101, 53
    row_pitch, col_pitch = 31, 53
    threshold = 72
    
    # Test offset variations
    offset_range = range(-5, 6)  # ±5 pixel offsets
    
    results = []
    
    for row_offset in offset_range:
        for col_offset in offset_range:
            config = {
                'row_start': base_row + row_offset,
                'col_start': base_col + col_offset,
                'row_pitch': row_pitch,
                'col_pitch': col_pitch,
                'threshold': threshold
            }
            
            # Extract 128 bits for quick analysis
            bits = []
            for i in range(128):
                bit_row = i // 8
                bit_col = i % 8
                
                y = config['row_start'] + bit_row * config['row_pitch']
                x = config['col_start'] + bit_col * config['col_pitch']
                
                if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                    pixel_val = img[y, x]
                    bit = 1 if pixel_val > config['threshold'] else 0
                    bits.append(bit)
            
            if len(bits) >= 64:
                ones_ratio = sum(bits) / len(bits)
                
                # Calculate entropy
                if sum(bits) > 0 and sum(bits) < len(bits):
                    p1 = sum(bits) / len(bits)
                    p0 = 1 - p1
                    entropy = -p1 * np.log2(p1) - p0 * np.log2(p0)
                else:
                    entropy = 0
                
                results.append({
                    'row_offset': row_offset,
                    'col_offset': col_offset,
                    'ones_ratio': ones_ratio,
                    'entropy': entropy,
                    'bits_extracted': len(bits)
                })
    
    # Find best configurations
    print(f"Tested {len(results)} grid configurations")
    
    # Sort by entropy (want high entropy)
    results_by_entropy = sorted(results, key=lambda x: x['entropy'], reverse=True)
    
    print("\nTop 5 configurations by entropy:")
    for i, result in enumerate(results_by_entropy[:5]):
        print(f"  {i+1}. Offset ({result['row_offset']:+2d},{result['col_offset']:+2d}): "
              f"entropy={result['entropy']:.3f}, ones={result['ones_ratio']:.1%}")
    
    # Sort by balanced ones ratio (closer to 50%)
    results_by_balance = sorted(results, key=lambda x: abs(x['ones_ratio'] - 0.5))
    
    print("\nTop 5 configurations by ones ratio balance:")
    for i, result in enumerate(results_by_balance[:5]):
        print(f"  {i+1}. Offset ({result['row_offset']:+2d},{result['col_offset']:+2d}): "
              f"ones={result['ones_ratio']:.1%}, entropy={result['entropy']:.3f}")
    
    return results

def test_threshold_variations():
    """Test different threshold values for bit extraction."""
    
    print("\n=== THRESHOLD VARIATION ANALYSIS ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Fixed position from verification
    row0, col0 = 101, 53
    row_pitch, col_pitch = 31, 53
    
    # Test threshold range
    thresholds = range(40, 120, 5)
    
    results = []
    
    for threshold in thresholds:
        bits = []
        values = []
        
        # Extract 128 bits
        for i in range(128):
            bit_row = i // 8
            bit_col = i % 8
            
            y = row0 + bit_row * row_pitch
            x = col0 + bit_col * col_pitch
            
            if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                pixel_val = img[y, x]
                bit = 1 if pixel_val > threshold else 0
                bits.append(bit)
                values.append(pixel_val)
        
        if len(bits) >= 64:
            ones_ratio = sum(bits) / len(bits)
            
            # Calculate entropy
            if sum(bits) > 0 and sum(bits) < len(bits):
                p1 = sum(bits) / len(bits)
                p0 = 1 - p1
                entropy = -p1 * np.log2(p1) - p0 * np.log2(p0)
            else:
                entropy = 0
            
            # Value statistics
            val_mean = np.mean(values)
            val_std = np.std(values)
            
            results.append({
                'threshold': threshold,
                'ones_ratio': ones_ratio,
                'entropy': entropy,
                'value_mean': val_mean,
                'value_std': val_std,
                'separation': abs(val_mean - threshold) / val_std if val_std > 0 else 0
            })
    
    print(f"Tested {len(thresholds)} threshold values")
    
    # Find optimal threshold
    print(f"\nThreshold analysis:")
    print(f"{'Thresh':>6} {'Ones%':>6} {'Entropy':>8} {'Mean':>6} {'Std':>6} {'Sep':>6}")
    
    for result in results:
        print(f"{result['threshold']:6d} {result['ones_ratio']*100:5.1f}% "
              f"{result['entropy']:7.3f} {result['value_mean']:5.1f} "
              f"{result['value_std']:5.1f} {result['separation']:5.2f}")
    
    # Best by entropy
    best_entropy = max(results, key=lambda x: x['entropy'])
    print(f"\nBest entropy: threshold {best_entropy['threshold']} "
          f"(entropy={best_entropy['entropy']:.3f})")
    
    # Best by balance
    best_balance = min(results, key=lambda x: abs(x['ones_ratio'] - 0.5))
    print(f"Best balance: threshold {best_balance['threshold']} "
          f"(ones={best_balance['ones_ratio']:.1%})")
    
    return results

def analyze_pixel_value_distribution():
    """Analyze the distribution of pixel values at grid positions."""
    
    print("\n=== PIXEL VALUE DISTRIBUTION ANALYSIS ===")
    
    df = load_raw_bit_dump()
    if df is None:
        return
    
    # Analyze pixel value distribution
    pixel_values = df['pixel_value'].values
    
    print(f"Pixel value statistics:")
    print(f"  Count: {len(pixel_values)}")
    print(f"  Range: {pixel_values.min()} - {pixel_values.max()}")
    print(f"  Mean: {pixel_values.mean():.1f}")
    print(f"  Std: {pixel_values.std():.1f}")
    print(f"  Median: {np.median(pixel_values):.1f}")
    
    # Histogram analysis
    hist, bin_edges = np.histogram(pixel_values, bins=20)
    
    print(f"\nPixel value distribution:")
    for i in range(len(hist)):
        if hist[i] > 0:
            print(f"  {bin_edges[i]:5.1f}-{bin_edges[i+1]:5.1f}: {hist[i]:3d} pixels")
    
    # Find optimal threshold using various methods
    print(f"\nOptimal threshold candidates:")
    
    # Otsu's method approximation
    total_pixels = len(pixel_values)
    best_threshold = 0
    best_variance = 0
    
    for threshold in range(int(pixel_values.min()), int(pixel_values.max())):
        below = pixel_values[pixel_values <= threshold]
        above = pixel_values[pixel_values > threshold]
        
        if len(below) > 0 and len(above) > 0:
            w0 = len(below) / total_pixels
            w1 = len(above) / total_pixels
            
            if w0 > 0 and w1 > 0:
                var_between = w0 * w1 * (below.mean() - above.mean()) ** 2
                
                if var_between > best_variance:
                    best_variance = var_between
                    best_threshold = threshold
    
    print(f"  Otsu-like optimal: {best_threshold}")
    print(f"  Mean-based: {pixel_values.mean():.0f}")
    print(f"  Median-based: {np.median(pixel_values):.0f}")
    
    # Test current threshold
    current_threshold = 72
    below_current = len(pixel_values[pixel_values <= current_threshold])
    above_current = len(pixel_values[pixel_values > current_threshold])
    
    print(f"  Current (72): {below_current} below, {above_current} above "
          f"({above_current/total_pixels:.1%} ones)")
    
    return pixel_values

def test_alternative_extraction_methods():
    """Test alternative bit extraction methods."""
    
    print("\n=== ALTERNATIVE EXTRACTION METHODS ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Base configuration
    row0, col0 = 101, 53
    row_pitch, col_pitch = 31, 53
    
    methods = [
        ("Single pixel", lambda img, y, x: img[y, x]),
        ("3x3 mean", lambda img, y, x: np.mean(img[max(0,y-1):y+2, max(0,x-1):x+2])),
        ("3x3 median", lambda img, y, x: np.median(img[max(0,y-1):y+2, max(0,x-1):x+2])),
        ("5x5 mean", lambda img, y, x: np.mean(img[max(0,y-2):y+3, max(0,x-2):x+3])),
        ("Cross pattern", lambda img, y, x: np.mean([img[y,x], img[max(0,y-1),x], 
                                                    img[min(img.shape[0]-1,y+1),x],
                                                    img[y,max(0,x-1)], 
                                                    img[y,min(img.shape[1]-1,x+1)]])),
    ]
    
    results = {}
    
    for method_name, extract_func in methods:
        print(f"\n--- {method_name} ---")
        
        # Test multiple thresholds
        for threshold in [60, 72, 85, 100]:
            bits = []
            
            # Extract 128 bits
            for i in range(128):
                bit_row = i // 8
                bit_col = i % 8
                
                y = row0 + bit_row * row_pitch
                x = col0 + bit_col * col_pitch
                
                if (0 <= y < img.shape[0] and 0 <= x < img.shape[1] and
                    y >= 2 and x >= 2 and y < img.shape[0]-2 and x < img.shape[1]-2):
                    
                    try:
                        pixel_val = extract_func(img, y, x)
                        bit = 1 if pixel_val > threshold else 0
                        bits.append(bit)
                    except:
                        continue
            
            if len(bits) >= 64:
                ones_ratio = sum(bits) / len(bits)
                
                # Calculate entropy
                if sum(bits) > 0 and sum(bits) < len(bits):
                    p1 = sum(bits) / len(bits)
                    p0 = 1 - p1
                    entropy = -p1 * np.log2(p1) - p0 * np.log2(p0)
                else:
                    entropy = 0
                
                print(f"  Threshold {threshold:3d}: ones={ones_ratio:.1%}, entropy={entropy:.3f}")
                
                results[f"{method_name}_t{threshold}"] = {
                    'method': method_name,
                    'threshold': threshold,
                    'ones_ratio': ones_ratio,
                    'entropy': entropy,
                    'bits': bits
                }
    
    # Find best methods
    if results:
        print(f"\nBest methods by entropy:")
        sorted_results = sorted(results.items(), key=lambda x: x[1]['entropy'], reverse=True)
        
        for i, (key, result) in enumerate(sorted_results[:5]):
            print(f"  {i+1}. {result['method']} (t={result['threshold']}): "
                  f"entropy={result['entropy']:.3f}, ones={result['ones_ratio']:.1%}")
    
    return results

def comprehensive_grid_search():
    """Comprehensive grid search with statistical validation."""
    
    print("\n=== COMPREHENSIVE GRID SEARCH ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Search ranges
    row_range = range(90, 120, 2)  # Around 101
    col_range = range(40, 70, 2)   # Around 53
    pitch_range = [(31, 53), (30, 52), (32, 54), (25, 50)]
    
    best_configs = []
    
    for row_pitch, col_pitch in pitch_range:
        print(f"\nTesting pitch {row_pitch}x{col_pitch}")
        
        for row0 in row_range:
            for col0 in col_range:
                # Quick test with 64 bits
                bits = []
                
                for i in range(64):
                    bit_row = i // 8
                    bit_col = i % 8
                    
                    y = row0 + bit_row * row_pitch
                    x = col0 + bit_col * col_pitch
                    
                    if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                        pixel_val = img[y, x]
                        bit = 1 if pixel_val > 72 else 0
                        bits.append(bit)
                
                if len(bits) == 64:
                    ones_ratio = sum(bits) / len(bits)
                    
                    # Only consider balanced extractions
                    if 0.3 <= ones_ratio <= 0.7:
                        # Calculate entropy
                        p1 = ones_ratio
                        p0 = 1 - p1
                        entropy = -p1 * np.log2(p1) - p0 * np.log2(p0)
                        
                        best_configs.append({
                            'row0': row0,
                            'col0': col0,
                            'row_pitch': row_pitch,
                            'col_pitch': col_pitch,
                            'ones_ratio': ones_ratio,
                            'entropy': entropy
                        })
    
    # Sort by entropy
    best_configs.sort(key=lambda x: x['entropy'], reverse=True)
    
    print(f"\nFound {len(best_configs)} balanced configurations")
    print(f"Top 10 by entropy:")
    
    for i, config in enumerate(best_configs[:10]):
        print(f"  {i+1:2d}. pos=({config['row0']:3d},{config['col0']:2d}) "
              f"pitch=({config['row_pitch']:2d},{config['col_pitch']:2d}) "
              f"entropy={config['entropy']:.3f} ones={config['ones_ratio']:.1%}")
    
    return best_configs

def save_alternative_analysis_results():
    """Save all alternative analysis results."""
    
    print("\n=== SAVING ALTERNATIVE ANALYSIS RESULTS ===")
    
    # Run all analyses
    grid_offsets = test_grid_offset_variations()
    threshold_analysis = test_threshold_variations()
    pixel_distribution = analyze_pixel_value_distribution()
    extraction_methods = test_alternative_extraction_methods()
    comprehensive_search = comprehensive_grid_search()
    
    results = {
        "timestamp": "2024-01-17",
        "analysis_type": "alternative_grid_analysis",
        "verification_failures_addressed": [
            "28.9% ones ratio bias",
            "Pattern artifacts on shuffled data", 
            "Cannot extract 9 leading zeros hash",
            "Low entropy suggests systematic bias"
        ],
        "grid_offset_variations": {
            "total_tested": len(grid_offsets),
            "best_by_entropy": max(grid_offsets, key=lambda x: x['entropy']) if grid_offsets else None,
            "best_by_balance": min(grid_offsets, key=lambda x: abs(x['ones_ratio'] - 0.5)) if grid_offsets else None
        },
        "threshold_optimization": {
            "range_tested": "40-115",
            "best_entropy_threshold": max(threshold_analysis, key=lambda x: x['entropy'])['threshold'] if threshold_analysis else None,
            "best_balance_threshold": min(threshold_analysis, key=lambda x: abs(x['ones_ratio'] - 0.5))['threshold'] if threshold_analysis else None
        },
        "extraction_methods": {
            "methods_tested": len(extraction_methods),
            "best_method": max(extraction_methods.items(), key=lambda x: x[1]['entropy'])[0] if extraction_methods else None
        },
        "comprehensive_search": {
            "balanced_configs_found": len(comprehensive_search),
            "best_config": comprehensive_search[0] if comprehensive_search else None
        },
        "conclusion": "Multiple alternative approaches tested to address verification failures"
    }
    
    with open('alternative_grid_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Alternative analysis results saved to alternative_grid_analysis_results.json")
    
    return results

if __name__ == "__main__":
    print("Alternative Grid Alignment Analysis")
    print("Addressing verification failures through systematic testing")
    print("="*60)
    
    # Load raw bit dump
    df = load_raw_bit_dump()
    
    # Test grid offset variations
    grid_results = test_grid_offset_variations()
    
    # Test threshold variations
    threshold_results = test_threshold_variations()
    
    # Analyze pixel distribution
    pixel_analysis = analyze_pixel_value_distribution()
    
    # Test alternative extraction methods
    method_results = test_alternative_extraction_methods()
    
    # Comprehensive grid search
    comprehensive_results = comprehensive_grid_search()
    
    # Save results
    analysis_summary = save_alternative_analysis_results()
    
    print("\n" + "="*60)
    print("ALTERNATIVE ANALYSIS COMPLETE")
    
    print(f"\nKey findings:")
    if comprehensive_results:
        best = comprehensive_results[0]
        print(f"Best configuration: pos=({best['row0']},{best['col0']}) "
              f"pitch=({best['row_pitch']},{best['col_pitch']}) entropy={best['entropy']:.3f}")
    
    print(f"Analysis addresses the verification failures identified in independent audit")