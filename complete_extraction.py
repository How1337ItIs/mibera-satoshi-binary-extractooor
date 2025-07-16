#!/usr/bin/env python3
"""
Complete extraction of all binary digits from the Satoshi poster.

Created by Claude Code - July 16, 2025
Purpose: Extract every possible 1 and 0 from all identified digit regions
"""
import cv2
import numpy as np
import json
import csv
from pathlib import Path

def extract_all_regions():
    """Extract binary digits from all identified regions"""
    
    # Load image and regions
    img = cv2.imread('satoshi (1).png')
    
    with open('digit_regions.json', 'r') as f:
        regions = json.load(f)
    
    print(f"Extracting from {len(regions)} regions...")
    
    # Working parameters from our corrected method
    row_pitch = 15
    col_pitch = 12
    
    all_cells = []
    region_stats = []
    
    for region_id, region in enumerate(regions):
        print(f"\nExtracting from region {region_id}: {region['w']}x{region['h']} at ({region['x']}, {region['y']})")
        
        # Calculate grid dimensions for this region
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
                
                # Blue channel classification
                blue_channel = cell[:, :, 0]
                avg_blue = np.mean(blue_channel)
                
                # Conservative classification
                if avg_blue > 160:  # High blue = cyan = "0"
                    bit = '0'
                    zeros += 1
                elif avg_blue < 90:  # Low blue = dark = "1"
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
    
    # Save detailed results
    with open('complete_extraction_detailed.json', 'w') as f:
        json.dump({
            'cells': all_cells,
            'region_stats': region_stats,
            'total_regions_processed': len(region_stats),
            'extraction_method': 'corrected_15x12_grid_blue_channel'
        }, f, indent=2)
    
    # Create CSV of just the bits for analysis
    csv_path = 'complete_extraction_bits.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['region_id', 'local_row', 'local_col', 'global_x', 'global_y', 'bit', 'confidence'])
        
        for cell in all_cells:
            writer.writerow([
                cell['region_id'], cell['local_row'], cell['local_col'],
                cell['global_x'], cell['global_y'], cell['bit'], cell['confidence']
            ])
    
    # Create binary-only CSV (no ambiguous)
    binary_only_path = 'complete_extraction_binary_only.csv'
    with open(binary_only_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['region_id', 'local_row', 'local_col', 'global_x', 'global_y', 'bit', 'confidence'])
        
        binary_cells = [cell for cell in all_cells if cell['bit'] in ['0', '1']]
        for cell in binary_cells:
            writer.writerow([
                cell['region_id'], cell['local_row'], cell['local_col'],
                cell['global_x'], cell['global_y'], cell['bit'], cell['confidence']
            ])
    
    # Overall statistics
    total_cells = len(all_cells)
    total_zeros = sum(1 for cell in all_cells if cell['bit'] == '0')
    total_ones = sum(1 for cell in all_cells if cell['bit'] == '1')
    total_ambiguous = sum(1 for cell in all_cells if cell['bit'] == 'ambiguous')
    total_clear = total_zeros + total_ones
    overall_clarity = total_clear / total_cells * 100 if total_cells > 0 else 0
    
    print(f"\n" + "="*60)
    print(f"COMPLETE EXTRACTION SUMMARY")
    print(f"="*60)
    print(f"Total regions processed: {len(region_stats)}")
    print(f"Total cells extracted: {total_cells}")
    print(f"Clear binary digits: {total_clear} ({overall_clarity:.1f}%)")
    print(f"  - Zeros: {total_zeros}")
    print(f"  - Ones: {total_ones}")
    print(f"Ambiguous cells: {total_ambiguous}")
    print(f"\nFiles created:")
    print(f"  - {csv_path} (all cells)")
    print(f"  - {binary_only_path} (binary digits only)")
    print(f"  - complete_extraction_detailed.json (full data)")
    
    # Show best regions
    region_stats.sort(key=lambda r: r['clarity_rate'], reverse=True)
    print(f"\nTop 5 regions by clarity:")
    for i, stat in enumerate(region_stats[:5]):
        print(f"  {i+1}. Region {stat['region_id']}: {stat['clarity_rate']:.1f}% ({stat['clear_cells']}/{stat['total_cells']})")
    
    return all_cells, region_stats

def create_bit_matrix_visualization(all_cells):
    """Create a visual representation of the extracted bit matrix"""
    
    # Group cells by region for visualization
    regions = {}
    for cell in all_cells:
        region_id = cell['region_id']
        if region_id not in regions:
            regions[region_id] = []
        regions[region_id].append(cell)
    
    print(f"\nCreating visualizations for {len(regions)} regions...")
    
    for region_id, cells in regions.items():
        if len(cells) < 10:  # Skip tiny regions
            continue
        
        # Get region bounds
        min_row = min(cell['local_row'] for cell in cells)
        max_row = max(cell['local_row'] for cell in cells)
        min_col = min(cell['local_col'] for cell in cells)
        max_col = max(cell['local_col'] for cell in cells)
        
        rows = max_row - min_row + 1
        cols = max_col - min_col + 1
        
        if rows < 3 or cols < 5:  # Skip too small
            continue
        
        # Create bit matrix for this region
        bit_matrix = [['.' for _ in range(cols)] for _ in range(rows)]
        
        for cell in cells:
            r = cell['local_row'] - min_row
            c = cell['local_col'] - min_col
            if 0 <= r < rows and 0 <= c < cols:
                if cell['bit'] in ['0', '1']:
                    bit_matrix[r][c] = cell['bit']
                else:
                    bit_matrix[r][c] = '?'
        
        # Save as text file
        with open(f'region_{region_id}_bits.txt', 'w') as f:
            f.write(f"Region {region_id} Binary Matrix ({rows}x{cols})\n")
            f.write("="*50 + "\n\n")
            for row in bit_matrix:
                f.write(''.join(row) + '\n')
        
        print(f"  Region {region_id}: {rows}x{cols} matrix saved")

if __name__ == "__main__":
    print("=== COMPLETE SATOSHI POSTER EXTRACTION ===")
    cells, stats = extract_all_regions()
    
    print("\n=== CREATING BIT MATRIX VISUALIZATIONS ===")
    create_bit_matrix_visualization(cells)
    
    print(f"\n✅ EXTRACTION COMPLETE!")
    print(f"📊 Total binary digits extracted: {sum(1 for cell in cells if cell['bit'] in ['0', '1'])}")
    print(f"🎯 Ready for cryptographic analysis!")