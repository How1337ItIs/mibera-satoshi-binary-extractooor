#!/usr/bin/env python3
"""
Recalibrate extraction method to eliminate systematic bias.

Created by Claude Code - July 16, 2025
Purpose: Address the extreme bias toward 1s and find truly accurate data
"""
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json

def analyze_bias_sources():
    """Analyze sources of bias in the current extraction method"""
    
    print("=== BIAS SOURCE ANALYSIS ===")
    
    # Load current results
    df = pd.read_csv('optimized_extraction_binary_only.csv')
    
    # Analyze confidence scores by bit value
    ones_confidence = df[df['bit'] == 1]['confidence']
    zeros_confidence = df[df['bit'] == 0]['confidence']
    
    print(f"Current extraction statistics:")
    print(f"  Total bits: {len(df)}")
    print(f"  Ones: {len(ones_confidence)} ({len(ones_confidence)/len(df)*100:.1f}%)")
    print(f"  Zeros: {len(zeros_confidence)} ({len(zeros_confidence)/len(df)*100:.1f}%)")
    
    print(f"\nConfidence analysis:")
    print(f"  Ones - Mean: {ones_confidence.mean():.1f}, Std: {ones_confidence.std():.1f}")
    print(f"  Zeros - Mean: {zeros_confidence.mean():.1f}, Std: {zeros_confidence.std():.1f}")
    
    # Check threshold ranges
    print(f"\nThreshold analysis (current: >150=0, <100=1):")
    print(f"  Ones confidence range: {ones_confidence.min():.1f} to {ones_confidence.max():.1f}")
    print(f"  Zeros confidence range: {zeros_confidence.min():.1f} to {zeros_confidence.max():.1f}")
    
    # The issue: if most pixels are dark (low blue), they get classified as 1s
    # We need more balanced thresholds
    
    return ones_confidence, zeros_confidence

def find_balanced_thresholds():
    """Find thresholds that produce more balanced 0/1 distribution"""
    
    print(f"\n=== BALANCED THRESHOLD SEARCH ===")
    
    # Load all cell data with confidence scores
    img = cv2.imread('satoshi (1).png')
    
    with open('digit_regions.json', 'r') as f:
        regions = json.load(f)
    
    # Sample confidence values from all regions
    all_confidences = []
    
    for region_id, region in enumerate(regions[:10]):  # Sample first 10 regions
        max_rows = region['h'] // 15
        max_cols = region['w'] // 12
        
        if max_rows < 3 or max_cols < 5:
            continue
            
        for r in range(0, max_rows, 2):  # Sample every 2nd cell
            for c in range(0, max_cols, 2):
                global_y = region['y'] + r * 15
                global_x = region['x'] + c * 12
                
                if global_y >= img.shape[0] or global_x >= img.shape[1]:
                    continue
                
                cell = img[max(0, global_y-3):min(img.shape[0], global_y+4), 
                          max(0, global_x-3):min(img.shape[1], global_x+4)]
                
                if cell.size > 0:
                    blue_channel = cell[:, :, 0]
                    avg_blue = np.mean(blue_channel)
                    all_confidences.append(avg_blue)
    
    all_confidences = np.array(all_confidences)
    
    print(f"Sampled {len(all_confidences)} confidence values")
    print(f"Distribution: min={all_confidences.min():.1f}, max={all_confidences.max():.1f}, mean={all_confidences.mean():.1f}")
    
    # Find percentiles for balanced thresholds
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    threshold_values = np.percentile(all_confidences, percentiles)
    
    print(f"\nPercentile-based thresholds:")
    for p, t in zip(percentiles, threshold_values):
        print(f"  {p}th percentile: {t:.1f}")
    
    # Test different threshold combinations for balance
    print(f"\nTesting threshold combinations for balance:")
    
    test_thresholds = [
        (all_confidences.mean() + all_confidences.std(), all_confidences.mean() - all_confidences.std()),
        (threshold_values[7], threshold_values[2]),  # 80th/20th percentiles
        (threshold_values[6], threshold_values[3]),  # 70th/30th percentiles
        (threshold_values[5], threshold_values[4]),  # 60th/40th percentiles
    ]
    
    for i, (hi_thresh, lo_thresh) in enumerate(test_thresholds):
        zeros = np.sum(all_confidences > hi_thresh)
        ones = np.sum(all_confidences < lo_thresh)
        ambiguous = len(all_confidences) - zeros - ones
        
        total = zeros + ones
        if total > 0:
            zero_pct = zeros / total * 100
            one_pct = ones / total * 100
            balance_score = abs(50 - zero_pct)  # How far from 50/50
            
            print(f"  Threshold {i+1}: hi={hi_thresh:.1f}, lo={lo_thresh:.1f}")
            print(f"    Zeros: {zeros} ({zero_pct:.1f}%), Ones: {ones} ({one_pct:.1f}%)")
            print(f"    Ambiguous: {ambiguous}, Balance score: {balance_score:.1f}")
    
    # Return most balanced thresholds
    best_thresholds = test_thresholds[3]  # 60th/40th percentiles
    print(f"\nRecommended balanced thresholds: {best_thresholds}")
    
    return best_thresholds

def extract_with_balanced_thresholds(hi_thresh, lo_thresh):
    """Extract data using balanced thresholds"""
    
    print(f"\n=== BALANCED EXTRACTION ===")
    print(f"Using thresholds: hi={hi_thresh:.1f}, lo={lo_thresh:.1f}")
    
    img = cv2.imread('satoshi (1).png')
    
    with open('digit_regions.json', 'r') as f:
        regions = json.load(f)
    
    # Parameters
    row_pitch = 15
    col_pitch = 12
    
    all_cells = []
    region_stats = []
    
    for region_id, region in enumerate(regions):
        if region_id >= 5:  # Limit to first 5 regions for testing
            break
            
        max_rows = region['h'] // row_pitch
        max_cols = region['w'] // col_pitch
        
        if max_rows < 3 or max_cols < 5:
            continue
        
        region_cells = []
        zeros = ones = ambiguous = 0
        
        for r in range(max_rows):
            for c in range(max_cols):
                global_y = region['y'] + r * row_pitch
                global_x = region['x'] + c * col_pitch
                
                if global_y >= img.shape[0] or global_x >= img.shape[1]:
                    continue
                
                cell = img[max(0, global_y-3):min(img.shape[0], global_y+4), 
                          max(0, global_x-3):min(img.shape[1], global_x+4)]
                
                if cell.size == 0:
                    continue
                
                # Blue channel classification with balanced thresholds
                blue_channel = cell[:, :, 0]
                avg_blue = np.mean(blue_channel)
                
                if avg_blue > hi_thresh:
                    bit = '0'
                    zeros += 1
                elif avg_blue < lo_thresh:
                    bit = '1'
                    ones += 1
                else:
                    bit = 'ambiguous'
                    ambiguous += 1
                
                cell_data = {
                    'region_id': region_id,
                    'local_row': r,
                    'local_col': c,
                    'global_x': global_x,
                    'global_y': global_y,
                    'bit': bit,
                    'confidence': avg_blue
                }
                
                region_cells.append(cell_data)
                all_cells.append(cell_data)
        
        # Calculate region statistics
        total_cells = len(region_cells)
        clear_cells = zeros + ones
        clarity_rate = clear_cells / total_cells * 100 if total_cells > 0 else 0
        
        region_stat = {
            'region_id': region_id,
            'total_cells': total_cells,
            'zeros': zeros,
            'ones': ones,
            'ambiguous': ambiguous,
            'clear_cells': clear_cells,
            'clarity_rate': clarity_rate
        }
        
        region_stats.append(region_stat)
        
        print(f"Region {region_id}: {zeros} zeros, {ones} ones, {ambiguous} ambiguous ({clarity_rate:.1f}% clear)")
    
    # Overall statistics
    total_cells = len(all_cells)
    total_zeros = sum(1 for cell in all_cells if cell['bit'] == '0')
    total_ones = sum(1 for cell in all_cells if cell['bit'] == '1')
    total_ambiguous = sum(1 for cell in all_cells if cell['bit'] == 'ambiguous')
    total_clear = total_zeros + total_ones
    overall_clarity = total_clear / total_cells * 100 if total_cells > 0 else 0
    
    print(f"\nBalanced extraction results:")
    print(f"  Total cells: {total_cells}")
    print(f"  Clear binary digits: {total_clear} ({overall_clarity:.1f}%)")
    print(f"  Zeros: {total_zeros} ({total_zeros/total_clear*100:.1f}% of clear)")
    print(f"  Ones: {total_ones} ({total_ones/total_clear*100:.1f}% of clear)")
    print(f"  Ambiguous: {total_ambiguous}")
    
    return all_cells, region_stats

def test_multiple_extraction_methods():
    """Test multiple extraction methods for comparison"""
    
    print(f"\n=== MULTIPLE METHOD COMPARISON ===")
    
    img = cv2.imread('satoshi (1).png')
    
    # Test single region for detailed comparison
    test_region = {
        'x': 132, 'y': 1072, 'w': 164, 'h': 120  # Subset of region 0
    }
    
    methods = {
        'Blue Channel (Current)': lambda cell: np.mean(cell[:, :, 0]),
        'Green Channel': lambda cell: np.mean(cell[:, :, 1]),
        'Red Channel': lambda cell: np.mean(cell[:, :, 2]),
        'Grayscale': lambda cell: np.mean(cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)),
        'HSV Saturation': lambda cell: np.mean(cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)[:, :, 1]),
        'HSV Value': lambda cell: np.mean(cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)[:, :, 2]),
    }
    
    # Extract test data with each method
    for method_name, extract_func in methods.items():
        print(f"\nTesting {method_name}:")
        
        confidences = []
        
        # Sample 100 cells from test region
        for r in range(0, 8):  # 8 rows
            for c in range(0, 8):  # 8 cols
                global_y = test_region['y'] + r * 15
                global_x = test_region['x'] + c * 12
                
                if global_y >= img.shape[0] or global_x >= img.shape[1]:
                    continue
                
                cell = img[max(0, global_y-3):min(img.shape[0], global_y+4), 
                          max(0, global_x-3):min(img.shape[1], global_x+4)]
                
                if cell.size > 0:
                    try:
                        confidence = extract_func(cell)
                        confidences.append(confidence)
                    except:
                        continue
        
        if confidences:
            confidences = np.array(confidences)
            print(f"  Range: {confidences.min():.1f} to {confidences.max():.1f}")
            print(f"  Mean±Std: {confidences.mean():.1f}±{confidences.std():.1f}")
            print(f"  Dynamic range: {(confidences.max() - confidences.min()):.1f}")

def analyze_ground_truth_samples():
    """Manually analyze some cells to establish ground truth"""
    
    print(f"\n=== GROUND TRUTH ANALYSIS ===")
    
    img = cv2.imread('satoshi (1).png')
    
    # Select specific cells for manual inspection
    test_coordinates = [
        (132, 1072, "Expected 1 - dark area"),
        (148, 1072, "Expected 1 - dark area"),
        (350, 1580, "Expected 0 - cyan area"),
        (370, 1580, "Expected 0 - cyan area"),
        (200, 400, "Mixed area"),
        (250, 400, "Mixed area"),
    ]
    
    print("Manual inspection of specific cells:")
    
    for x, y, description in test_coordinates:
        cell = img[max(0, y-3):min(img.shape[0], y+4), 
                  max(0, x-3):min(img.shape[1], x+4)]
        
        if cell.size > 0:
            # Analyze all channels
            b_mean = np.mean(cell[:, :, 0])
            g_mean = np.mean(cell[:, :, 1])
            r_mean = np.mean(cell[:, :, 2])
            
            # HSV
            hsv_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
            h_mean = np.mean(hsv_cell[:, :, 0])
            s_mean = np.mean(hsv_cell[:, :, 1])
            v_mean = np.mean(hsv_cell[:, :, 2])
            
            print(f"\nCell at ({x}, {y}) - {description}:")
            print(f"  RGB: R={r_mean:.1f}, G={g_mean:.1f}, B={b_mean:.1f}")
            print(f"  HSV: H={h_mean:.1f}, S={s_mean:.1f}, V={v_mean:.1f}")
            
            # Visual inspection: save cell image
            cell_rgb = cv2.cvtColor(cell, cv2.COLOR_BGR2RGB)
            plt.imsave(f'cell_sample_{x}_{y}.png', cell_rgb)
            print(f"  Saved as cell_sample_{x}_{y}.png for visual inspection")

if __name__ == "__main__":
    print("=== EXTRACTION METHOD RECALIBRATION ===")
    
    # Analyze current bias sources
    ones_conf, zeros_conf = analyze_bias_sources()
    
    # Find balanced thresholds
    balanced_thresholds = find_balanced_thresholds()
    
    # Test balanced extraction
    balanced_cells, balanced_stats = extract_with_balanced_thresholds(*balanced_thresholds)
    
    # Test multiple methods
    test_multiple_extraction_methods()
    
    # Ground truth analysis
    analyze_ground_truth_samples()
    
    # Save recalibration results
    recalibration_results = {
        'original_bias': {
            'ones_percentage': len(ones_conf) / (len(ones_conf) + len(zeros_conf)) * 100,
            'zeros_percentage': len(zeros_conf) / (len(ones_conf) + len(zeros_conf)) * 100
        },
        'balanced_thresholds': {
            'high_threshold': balanced_thresholds[0],
            'low_threshold': balanced_thresholds[1]
        },
        'balanced_results': balanced_stats,
        'recommendation': 'Use balanced thresholds and validate with manual inspection'
    }
    
    with open('recalibration_results.json', 'w') as f:
        json.dump(recalibration_results, f, indent=2)
    
    print(f"\nRecalibration complete!")
    print(f"Recommended approach: Manual validation of ground truth samples")
    print(f"Next step: Visual inspection of saved cell samples")
    print(f"Results saved to recalibration_results.json")