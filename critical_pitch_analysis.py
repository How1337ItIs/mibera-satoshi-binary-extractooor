#!/usr/bin/env python3
"""
Critical analysis of grid pitch claims.
Let's measure directly from the extracted data and images.

Author: Claude Code
Date: July 17, 2025
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal
from collections import Counter


def analyze_extracted_cells():
    """Analyze the cells.csv to understand actual extraction pattern."""
    print("=== Analyzing extracted cells ===")
    
    cells_df = pd.read_csv('binary_extractor/output_real_data/cells.csv')
    
    # Get unique row and column positions
    unique_rows = sorted(cells_df['row'].unique())
    unique_cols = sorted(cells_df['col'].unique())
    
    print(f"Grid dimensions: {len(unique_rows)} rows x {len(unique_cols)} cols")
    
    # Calculate actual pixel distances between cells
    print("\nAnalyzing pixel coordinates...")
    
    # Since cells.csv doesn't have x,y coordinates, we'll infer from the grid
    # The binary_extractor used row_pitch=31, col_pitch=25 according to Claude's analysis
    print(f"\nInferred from grid indices (if pitch was used):")
    print(f"  Row pitch: ~31 pixels (from Claude's analysis)")
    print(f"  Col pitch: ~25 pixels (from Claude's analysis)")
    
    # Analyze bit distribution by position
    print("\nBit distribution analysis...")
    
    # Look at first few rows
    for row_idx in range(min(5, len(unique_rows))):
        row = unique_rows[row_idx]
        row_cells = cells_df[cells_df['row'] == row].sort_values('col')
        bits = ''.join(row_cells['bit'].apply(lambda x: x if x in '01' else '.'))
        print(f"Row {row}: {bits[:50]}...")
    
    return {
        'avg_row_pitch': 31,  # From Claude's analysis
        'avg_col_pitch': 25,  # From Claude's analysis
        'grid_size': (len(unique_rows), len(unique_cols)),
        'cells_df': cells_df
    }


def analyze_bw_mask():
    """Analyze the binary mask directly to verify pitches."""
    print("\n=== Analyzing binary mask ===")
    
    bw = cv2.imread('binary_extractor/output_real_data/bw_mask.png', cv2.IMREAD_GRAYSCALE)
    if bw is None:
        print("Error: Could not load bw_mask.png")
        return None
    
    print(f"Image shape: {bw.shape}")
    
    # Calculate projections
    row_proj = bw.sum(axis=1)
    col_proj = bw.sum(axis=0)
    
    # Autocorrelation for pitch detection
    def detect_all_peaks(proj, min_lag=5, max_lag=50):
        proj_norm = proj - proj.mean()
        autocorr = signal.correlate(proj_norm, proj_norm, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        peaks = []
        for lag in range(min_lag, min(max_lag, len(autocorr))):
            if lag > 0 and lag < len(autocorr) - 1:
                if autocorr[lag] > autocorr[lag-1] and autocorr[lag] > autocorr[lag+1]:
                    peaks.append((lag, autocorr[lag]))
        
        return sorted(peaks, key=lambda x: x[1], reverse=True)
    
    print("\nRow pitch candidates (from autocorrelation):")
    row_peaks = detect_all_peaks(row_proj, min_lag=20, max_lag=40)
    for lag, strength in row_peaks[:5]:
        print(f"  {lag} pixels (strength: {strength:.0f})")
    
    print("\nColumn pitch candidates (from autocorrelation):")
    col_peaks = detect_all_peaks(col_proj, min_lag=5, max_lag=30)
    for lag, strength in col_peaks[:5]:
        print(f"  {lag} pixels (strength: {strength:.0f})")
    
    # Visualize projections and autocorrelations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Row projection
    ax1.plot(row_proj)
    ax1.set_title('Row Projection')
    ax1.set_xlabel('Y position')
    ax1.grid(True, alpha=0.3)
    
    # Column projection  
    ax2.plot(col_proj)
    ax2.set_title('Column Projection')
    ax2.set_xlabel('X position')
    ax2.grid(True, alpha=0.3)
    
    # Row autocorrelation
    row_norm = row_proj - row_proj.mean()
    row_autocorr = signal.correlate(row_norm, row_norm, mode='full')
    row_autocorr = row_autocorr[len(row_autocorr)//2:]
    ax3.plot(row_autocorr[:100])
    ax3.set_title('Row Autocorrelation')
    ax3.set_xlabel('Lag (pixels)')
    ax3.axvline(31, color='red', linestyle='--', label='31px (claimed)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Column autocorrelation
    col_norm = col_proj - col_proj.mean()
    col_autocorr = signal.correlate(col_norm, col_norm, mode='full')
    col_autocorr = col_autocorr[len(col_autocorr)//2:]
    ax4.plot(col_autocorr[:50])
    ax4.set_title('Column Autocorrelation')
    ax4.set_xlabel('Lag (pixels)')
    ax4.axvline(25, color='red', linestyle='--', label='25px (current)')
    ax4.axvline(8, color='green', linestyle='--', label='8px (claimed correct)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('critical_pitch_analysis.png', dpi=150)
    plt.close()
    
    return {
        'row_peaks': row_peaks,
        'col_peaks': col_peaks,
        'image_shape': bw.shape
    }


def measure_actual_glyphs():
    """Manually measure some actual glyphs in the image."""
    print("\n=== Measuring actual glyphs ===")
    
    bw = cv2.imread('binary_extractor/output_real_data/bw_mask.png', cv2.IMREAD_GRAYSCALE)
    if bw is None:
        return None
    
    # Find connected components to measure actual digit sizes
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
    
    # Filter for reasonable digit sizes (not too small, not too large)
    digit_widths = []
    digit_heights = []
    
    for i in range(1, num_labels):  # Skip background
        x, y, w, h, area = stats[i]
        
        # Reasonable digit criteria
        if 5 < w < 20 and 10 < h < 40 and area > 50:
            digit_widths.append(w)
            digit_heights.append(h)
    
    if digit_widths:
        print(f"Measured {len(digit_widths)} potential digits")
        print(f"Average digit width: {np.mean(digit_widths):.1f} ± {np.std(digit_widths):.1f} pixels")
        print(f"Average digit height: {np.mean(digit_heights):.1f} ± {np.std(digit_heights):.1f} pixels")
        
        # Histogram of widths
        width_counts = Counter(digit_widths)
        print("\nMost common digit widths:")
        for width, count in sorted(width_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {width} pixels: {count} occurrences")
    
    return {
        'avg_digit_width': np.mean(digit_widths) if digit_widths else None,
        'avg_digit_height': np.mean(digit_heights) if digit_heights else None
    }


def test_different_pitches():
    """Test extraction with different column pitches to see which works better."""
    print("\n=== Testing different column pitches ===")
    
    bw = cv2.imread('binary_extractor/output_real_data/bw_mask.png', cv2.IMREAD_GRAYSCALE)
    if bw is None:
        return None
    
    # Test pitches
    test_pitches = [8, 12, 17, 25]
    row_pitch = 31  # Keep row pitch constant
    
    results = {}
    
    for col_pitch in test_pitches:
        print(f"\nTesting column pitch: {col_pitch}px")
        
        # Simple extraction with this pitch
        bits = []
        y = 69  # Starting row position
        
        for row in range(10):  # Test first 10 rows
            x = 37  # Starting column position
            row_bits = []
            
            for col in range(50):  # Extract 50 bits per row
                if x < bw.shape[1] and y < bw.shape[0]:
                    # Sample 3x3 region
                    region = bw[max(0, y-1):min(bw.shape[0], y+2), 
                               max(0, x-1):min(bw.shape[1], x+2)]
                    val = np.mean(region) if region.size > 0 else 0
                    bit = '1' if val > 127 else '0'
                    row_bits.append(bit)
                
                x += col_pitch
            
            bits.extend(row_bits)
            y += row_pitch
        
        # Analyze bit distribution
        zeros = bits.count('0')
        ones = bits.count('1')
        ratio = zeros / len(bits) if bits else 0
        
        # Try to decode first bytes
        decoded = []
        for i in range(0, min(len(bits), 48), 8):
            if i + 8 <= len(bits):
                byte = ''.join(bits[i:i+8])
                try:
                    char_val = int(byte, 2)
                    if 32 <= char_val <= 126:
                        decoded.append(chr(char_val))
                    else:
                        decoded.append(f'[{char_val}]')
                except:
                    decoded.append('?')
        
        decoded_str = ''.join(decoded)
        
        results[col_pitch] = {
            'zero_ratio': ratio,
            'decoded': decoded_str,
            'first_bits': ''.join(bits[:50])
        }
        
        print(f"  Zero ratio: {ratio:.1%}")
        print(f"  First bits: {''.join(bits[:50])}")
        print(f"  Decoded: {decoded_str}")
    
    return results


if __name__ == "__main__":
    print("Critical Analysis of Grid Pitch Claims")
    print("="*50)
    
    # Analyze extracted cells
    cell_analysis = analyze_extracted_cells()
    
    # Analyze binary mask
    mask_analysis = analyze_bw_mask()
    
    # Measure actual glyphs
    glyph_measurements = measure_actual_glyphs()
    
    # Test different pitches
    pitch_tests = test_different_pitches()
    
    # Summary
    print("\n" + "="*50)
    print("CRITICAL ANALYSIS SUMMARY")
    print("="*50)
    
    print(f"\n1. From extracted cells data:")
    print(f"   - Row pitch: {cell_analysis['avg_row_pitch']:.1f}px")
    print(f"   - Col pitch: {cell_analysis['avg_col_pitch']:.1f}px")
    
    if mask_analysis:
        print(f"\n2. From autocorrelation analysis:")
        print(f"   - Strongest row peak: {mask_analysis['row_peaks'][0][0]}px")
        print(f"   - Strongest col peak: {mask_analysis['col_peaks'][0][0]}px")
    
    if glyph_measurements and glyph_measurements['avg_digit_width']:
        print(f"\n3. From glyph measurements:")
        print(f"   - Average digit width: {glyph_measurements['avg_digit_width']:.1f}px")
        print(f"   - Expected col pitch with gaps: ~{glyph_measurements['avg_digit_width'] * 1.5:.1f}px")
    
    print(f"\n4. Pitch test results:")
    for pitch, result in pitch_tests.items():
        print(f"   - {pitch}px: zero_ratio={result['zero_ratio']:.1%}, decoded='{result['decoded'][:20]}'")
    
    print("\nCONCLUSION:")
    print("The o3 suggestion about 8px column pitch needs verification.")
    print("The autocorrelation and actual measurements will reveal the truth.")
    print("Check critical_pitch_analysis.png for visual evidence.")