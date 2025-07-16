#!/usr/bin/env python3
"""
Corrected extraction with proper grid parameters.

Created by Claude Code - July 16, 2025  
Purpose: Fix grid alignment to extract actual digits instead of background
"""
import cv2
import numpy as np
import csv

def extract_with_corrected_grid():
    """Extract using corrected grid parameters based on analysis"""
    
    # Load image
    img = cv2.imread('satoshi (1).png')
    
    # CORRECTED grid parameters based on analysis
    row_pitch = 15  # Much smaller than 31
    col_pitch = 12  # Much smaller than 25  
    row0 = 890      # Start at visible digit region
    col0 = 185
    
    print(f"Using corrected grid: pitch=({row_pitch}, {col_pitch}), origin=({row0}, {col0})")
    
    # Extract a test region (10 rows x 20 cols)
    rows = 10
    cols = 20
    
    cells = []
    
    for r in range(rows):
        for c in range(cols):
            y = row0 + r * row_pitch
            x = col0 + c * col_pitch
            
            # Check bounds
            if y >= img.shape[0] or x >= img.shape[1]:
                continue
                
            # Extract cell (smaller size for tighter characters)
            cell = img[max(0, y-3):min(img.shape[0], y+4), 
                      max(0, x-3):min(img.shape[1], x+4)]
            
            if cell.size == 0:
                continue
            
            # Simple classification using blue channel
            # Cyan digits (0) will have high blue, dark digits (1) will have low blue
            blue_channel = cell[:, :, 0]  # Blue channel in BGR
            avg_blue = np.mean(blue_channel)
            
            # Threshold based on observation
            if avg_blue > 150:  # High blue = cyan = "0"
                bit = '0'
            elif avg_blue < 100:  # Low blue = dark = "1"  
                bit = '1'
            else:
                bit = 'ambiguous'
            
            cells.append((r, c, bit, avg_blue))
            
            # Save cell for verification
            if r < 3 and c < 10:  # Save first few rows for inspection
                enlarged = cv2.resize(cell, (30, 30), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(f'corrected_cell_{r}_{c}_{bit}.png', enlarged)
    
    # Analyze results
    print(f"\nExtracted {len(cells)} cells")
    
    zeros = sum(1 for _, _, bit, _ in cells if bit == '0')
    ones = sum(1 for _, _, bit, _ in cells if bit == '1')
    ambiguous = sum(1 for _, _, bit, _ in cells if bit == 'ambiguous')
    
    print(f"Zeros: {zeros}")
    print(f"Ones: {ones}")  
    print(f"Ambiguous: {ambiguous}")
    print(f"Clear classifications: {zeros + ones}/{len(cells)} ({(zeros + ones)/len(cells)*100:.1f}%)")
    
    # Show first few rows
    print("\nFirst 3 rows extracted:")
    for r in range(min(3, rows)):
        row_bits = []
        for c in range(min(20, cols)):
            for cell_r, cell_c, bit, avg_blue in cells:
                if cell_r == r and cell_c == c:
                    row_bits.append(bit)
                    break
        print(f"Row {r}: {''.join(row_bits)}")
    
    # Save results
    with open('corrected_extraction.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['row', 'col', 'bit', 'avg_blue'])
        for r, c, bit, avg_blue in cells:
            writer.writerow([r, c, bit, avg_blue])
    
    print(f"\nResults saved to corrected_extraction.csv")
    
    # Create visualization
    result_img = np.zeros((rows * 20, cols * 20, 3), dtype=np.uint8)
    for r, c, bit, avg_blue in cells:
        y_start = r * 20
        x_start = c * 20
        
        if bit == '0':
            color = (255, 255, 0)  # Cyan for 0
        elif bit == '1':
            color = (0, 0, 0)      # Black for 1
        else:
            color = (128, 128, 128)  # Gray for ambiguous
            
        result_img[y_start:y_start+20, x_start:x_start+20] = color
    
    cv2.imwrite('corrected_extraction_visualization.png', result_img)

if __name__ == "__main__":
    extract_with_corrected_grid()