#!/usr/bin/env python3
"""
Re-extract using optimized thresholds for 52.2% clarity rate.

Created by Claude Code - July 16, 2025
Purpose: Apply optimized parameters (150, 100) to achieve better extraction accuracy
"""
import cv2
import numpy as np
import json
import csv
from pathlib import Path

def optimized_extraction():
    """Re-extract using optimized thresholds"""
    
    # Load image and regions
    img = cv2.imread('satoshi (1).png')
    
    with open('digit_regions.json', 'r') as f:
        regions = json.load(f)
    
    print(f"Re-extracting with optimized thresholds (150, 100)...")
    print(f"Expected improvement: 36.4% -> 52.2% clarity")
    
    # Optimized parameters from analysis
    row_pitch = 15
    col_pitch = 12
    hi_threshold = 150  # More permissive (was 160)
    lo_threshold = 100  # More permissive (was 90)
    
    all_cells = []
    region_stats = []
    
    for region_id, region in enumerate(regions):
        print(f"Processing region {region_id}: {region['w']}x{region['h']} at ({region['x']}, {region['y']})")
        
        # Calculate grid dimensions
        max_rows = region['h'] // row_pitch
        max_cols = region['w'] // col_pitch
        
        if max_rows < 3 or max_cols < 5:
            print(f"  Skipping - too small ({max_rows}x{max_cols})")
            continue
        
        region_cells = []
        zeros = ones = ambiguous = 0
        
        for r in range(max_rows):
            for c in range(max_cols):
                # Global coordinates
                global_y = region['y'] + r * row_pitch
                global_x = region['x'] + c * col_pitch
                
                # Check bounds
                if global_y >= img.shape[0] or global_x >= img.shape[1]:
                    continue
                
                # Extract cell
                cell = img[max(0, global_y-3):min(img.shape[0], global_y+4), 
                          max(0, global_x-3):min(img.shape[1], global_x+4)]
                
                if cell.size == 0:
                    continue
                
                # Blue channel classification with optimized thresholds
                blue_channel = cell[:, :, 0]
                avg_blue = np.mean(blue_channel)
                
                # Optimized classification
                if avg_blue > hi_threshold:  # High blue = cyan = "0"
                    bit = '0'
                    zeros += 1
                elif avg_blue < lo_threshold:  # Low blue = dark = "1"
                    bit = '1'
                    ones += 1
                else:
                    bit = 'ambiguous'
                    ambiguous += 1
                
                # Store with global coordinates and region info
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
            'region': region,
            'total_cells': total_cells,
            'zeros': zeros,
            'ones': ones,
            'ambiguous': ambiguous,
            'clear_cells': clear_cells,
            'clarity_rate': clarity_rate
        }
        
        region_stats.append(region_stat)
        
        print(f"  Extracted {total_cells} cells: {zeros} zeros, {ones} ones, {ambiguous} ambiguous ({clarity_rate:.1f}% clear)")
    
    # Overall statistics
    total_cells = len(all_cells)
    total_zeros = sum(1 for cell in all_cells if cell['bit'] == '0')
    total_ones = sum(1 for cell in all_cells if cell['bit'] == '1')
    total_ambiguous = sum(1 for cell in all_cells if cell['bit'] == 'ambiguous')
    total_clear = total_zeros + total_ones
    overall_clarity = total_clear / total_cells * 100 if total_cells > 0 else 0
    
    print(f"\n" + "="*60)
    print(f"OPTIMIZED EXTRACTION RESULTS")
    print(f"="*60)
    print(f"Total regions processed: {len(region_stats)}")
    print(f"Total cells extracted: {total_cells}")
    print(f"Clear binary digits: {total_clear} ({overall_clarity:.1f}%)")
    print(f"  - Zeros: {total_zeros}")
    print(f"  - Ones: {total_ones}")
    print(f"Ambiguous cells: {total_ambiguous}")
    
    # Compare with original
    print(f"\nCOMPARISON WITH ORIGINAL:")
    print(f"Original clarity: 36.4% (524 clear digits)")
    print(f"Optimized clarity: {overall_clarity:.1f}% ({total_clear} clear digits)")
    improvement = (overall_clarity - 36.4) / 36.4 * 100
    print(f"Improvement: {improvement:+.1f}%")
    
    # Save results
    with open('optimized_extraction_detailed.json', 'w') as f:
        json.dump({
            'cells': all_cells,
            'region_stats': region_stats,
            'total_regions_processed': len(region_stats),
            'extraction_method': 'optimized_15x12_grid_blue_channel_150_100',
            'thresholds': {'high': hi_threshold, 'low': lo_threshold},
            'improvement_over_original': improvement
        }, f, indent=2)
    
    # Create binary-only CSV
    binary_only_path = 'optimized_extraction_binary_only.csv'
    with open(binary_only_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['region_id', 'local_row', 'local_col', 'global_x', 'global_y', 'bit', 'confidence'])
        
        binary_cells = [cell for cell in all_cells if cell['bit'] in ['0', '1']]
        for cell in binary_cells:
            writer.writerow([
                cell['region_id'], cell['local_row'], cell['local_col'],
                cell['global_x'], cell['global_y'], cell['bit'], cell['confidence']
            ])
    
    print(f"\nFiles created:")
    print(f"  - {binary_only_path} (binary digits only)")
    print(f"  - optimized_extraction_detailed.json (full data)")
    
    return all_cells, region_stats, overall_clarity

def analyze_optimized_vs_original():
    """Compare optimized extraction with original"""
    
    print(f"\n" + "="*60)
    print(f"DETAILED COMPARISON ANALYSIS")
    print(f"="*60)
    
    # Load both datasets
    try:
        import pandas as pd
        original_df = pd.read_csv('complete_extraction_binary_only.csv')
        optimized_df = pd.read_csv('optimized_extraction_binary_only.csv')
        
        print(f"Original dataset: {len(original_df)} binary digits")
        print(f"Optimized dataset: {len(optimized_df)} binary digits")
        
        # Analyze bit distribution changes
        orig_ones = len(original_df[original_df['bit'] == 1])
        orig_zeros = len(original_df[original_df['bit'] == 0])
        opt_ones = len(optimized_df[optimized_df['bit'] == 1])
        opt_zeros = len(optimized_df[optimized_df['bit'] == 0])
        
        print(f"\nBit distribution comparison:")
        print(f"Original - Ones: {orig_ones} ({orig_ones/len(original_df)*100:.1f}%), Zeros: {orig_zeros} ({orig_zeros/len(original_df)*100:.1f}%)")
        print(f"Optimized - Ones: {opt_ones} ({opt_ones/len(optimized_df)*100:.1f}%), Zeros: {opt_zeros} ({opt_zeros/len(optimized_df)*100:.1f}%)")
        
        # Bias analysis
        orig_bias = abs(0.5 - orig_ones/len(original_df))
        opt_bias = abs(0.5 - opt_ones/len(optimized_df))
        print(f"\nBias from 50/50:")
        print(f"Original: {orig_bias*100:.1f}%")
        print(f"Optimized: {opt_bias*100:.1f}%")
        
        if opt_bias < orig_bias:
            print("IMPROVEMENT: Optimized extraction shows less bias")
        else:
            print("NOTE: Optimized extraction shows similar/higher bias")
            
    except Exception as e:
        print(f"Error in comparison: {e}")

if __name__ == "__main__":
    print("=== OPTIMIZED EXTRACTION WITH IMPROVED THRESHOLDS ===")
    
    # Run optimized extraction
    cells, stats, clarity = optimized_extraction()
    
    # Compare with original
    analyze_optimized_vs_original()
    
    print(f"\nOptimized extraction complete!")
    print(f"Achieved {clarity:.1f}% clarity rate")
    print(f"Ready for enhanced cryptographic analysis")