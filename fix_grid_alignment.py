#!/usr/bin/env python3
"""
Fix grid alignment to actually hit the binary digits.

Created by Claude Code - July 16, 2025
Purpose: Correct grid parameters to extract actual digits, not background
"""
import cv2
import numpy as np

def analyze_digit_spacing():
    """Analyze the actual spacing of binary digits in the poster"""
    
    # Load original image
    img = cv2.imread('satoshi (1).png')
    
    # Look at the clear binary region we found
    # From overview: digits are visible around (930, 230) area
    y_start, x_start = 880, 180
    digit_region = img[y_start:y_start+100, x_start:x_start+200]
    
    # Save the region for analysis
    cv2.imwrite('digit_region.png', digit_region)
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(digit_region, cv2.COLOR_BGR2GRAY)
    
    # Get horizontal projection (sum along rows)
    h_projection = np.sum(gray, axis=0)
    
    # Get vertical projection (sum along columns)  
    v_projection = np.sum(gray, axis=1)
    
    # Find peaks in projections (where digits are)
    print("Analyzing horizontal spacing (character width):")
    print("Projection values (first 50 pixels):")
    for i in range(min(50, len(h_projection))):
        print(f"x={i:2d}: {h_projection[i]:6.0f}")
    
    # Try to detect character spacing
    # Look for repeating patterns in projection
    print("\nTrying to detect character spacing...")
    
    # Simple peak detection
    threshold = np.mean(h_projection) * 1.1
    peaks = []
    for i in range(1, len(h_projection)-1):
        if h_projection[i] > threshold and h_projection[i] > h_projection[i-1] and h_projection[i] > h_projection[i+1]:
            peaks.append(i)
    
    print(f"Found {len(peaks)} potential character centers: {peaks}")
    
    if len(peaks) > 1:
        spacings = [peaks[i+1] - peaks[i] for i in range(len(peaks)-1)]
        avg_spacing = np.mean(spacings)
        print(f"Average character spacing: {avg_spacing:.1f} pixels")
    
    # Do the same for vertical spacing
    print(f"\nAnalyzing vertical spacing (line height):")
    threshold_v = np.mean(v_projection) * 1.1
    v_peaks = []
    for i in range(1, len(v_projection)-1):
        if v_projection[i] > threshold_v and v_projection[i] > v_projection[i-1] and v_projection[i] > v_projection[i+1]:
            v_peaks.append(i)
    
    print(f"Found {len(v_peaks)} potential line centers: {v_peaks}")
    
    if len(v_peaks) > 1:
        v_spacings = [v_peaks[i+1] - v_peaks[i] for i in range(len(v_peaks)-1)]
        avg_v_spacing = np.mean(v_spacings)
        print(f"Average line spacing: {avg_v_spacing:.1f} pixels")

def test_corrected_grid():
    """Test different grid parameters to find digits"""
    
    img = cv2.imread('satoshi (1).png')
    
    # Test different grid parameters based on visual inspection
    test_params = [
        {"row_pitch": 15, "col_pitch": 12, "row0": 890, "col0": 185, "name": "tight_grid"},
        {"row_pitch": 20, "col_pitch": 15, "row0": 890, "col0": 185, "name": "medium_grid"},
        {"row_pitch": 25, "col_pitch": 18, "row0": 890, "col0": 185, "name": "loose_grid"},
        {"row_pitch": 30, "col_pitch": 20, "row0": 890, "col0": 185, "name": "current_like"},
    ]
    
    for params in test_params:
        print(f"\nTesting {params['name']}: pitch=({params['row_pitch']}, {params['col_pitch']})")
        
        # Extract 5x10 region
        cells = []
        for r in range(5):
            for c in range(10):
                y = params['row0'] + r * params['row_pitch']
                x = params['col0'] + c * params['col_pitch']
                
                # Extract cell
                cell = img[max(0, y-3):min(img.shape[0], y+4), 
                          max(0, x-3):min(img.shape[1], x+4)]
                
                if cell.size > 0:
                    # Enlarge and save
                    enlarged = cv2.resize(cell, (30, 30), interpolation=cv2.INTER_NEAREST)
                    cv2.imwrite(f'test_{params["name"]}_cell_{r}_{c}.png', enlarged)
        
        # Create grid overlay
        test_region = img[params['row0']-10:params['row0']+5*params['row_pitch']+10,
                         params['col0']-10:params['col0']+10*params['col_pitch']+10].copy()
        
        # Draw grid lines
        for r in range(6):
            y = 10 + r * params['row_pitch']
            cv2.line(test_region, (0, y), (test_region.shape[1], y), (0, 255, 0), 1)
        
        for c in range(11):
            x = 10 + c * params['col_pitch']
            cv2.line(test_region, (x, 0), (x, test_region.shape[0]), (0, 255, 0), 1)
        
        cv2.imwrite(f'test_{params["name"]}_grid.png', test_region)

if __name__ == "__main__":
    print("=== ANALYZING DIGIT SPACING ===")
    analyze_digit_spacing()
    
    print("\n=== TESTING CORRECTED GRIDS ===")
    test_corrected_grid()
    
    print("\nFiles created:")
    print("- digit_region.png (clear digit area)")
    print("- test_*_grid.png (grid overlays)")
    print("- test_*_cell_*.png (individual cells)")