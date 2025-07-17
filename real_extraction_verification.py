#!/usr/bin/env python3
"""
REAL Extraction Verification - Claude Code Agent
Actually extract from the real poster image for verification
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path

def load_real_poster():
    """Load the actual poster image"""
    poster_path = Path("satoshi (1).png")
    if poster_path.exists():
        img = cv2.imread(str(poster_path))
        print(f"Loaded real poster: {img.shape}")
        return img
    else:
        print("ERROR: Real poster not found")
        return None

def extract_real_top_line(img, start_x=100, start_y=100, num_cells=20):
    """Extract actual bits from the real poster top line"""
    if img is None:
        return []
        
    print(f"Extracting from real image at y={start_y}")
    
    real_cells = []
    cell_size = 6
    grid_spacing = 12
    
    for i in range(num_cells):
        x = start_x + i * grid_spacing
        y = start_y
        
        # Extract actual cell region from real image
        x_start = max(0, x - cell_size//2)
        x_end = min(img.shape[1], x + cell_size//2)
        y_start = max(0, y - cell_size//2) 
        y_end = min(img.shape[0], y + cell_size//2)
        
        cell_region = img[y_start:y_end, x_start:x_end]
        
        if cell_region.size > 0:
            # Get actual blue channel value
            blue_channel = cell_region[:, :, 0]  # BGR format
            blue_mean = np.mean(blue_channel)
            
            # Apply threshold (using validated threshold=80)
            bit = 1 if blue_mean > 80 else 0
            
            # Calculate confidence based on distance from threshold
            distance = abs(blue_mean - 80)
            confidence = min(100, 50 + distance * 0.8)
            
            real_cells.append({
                'index': i,
                'global_x': x,
                'global_y': y,
                'blue_mean': blue_mean,
                'bit': bit,
                'confidence': confidence,
                'cell_size': cell_region.shape
            })
            
    return real_cells

def test_multiple_locations(img):
    """Test extraction from multiple real locations"""
    
    test_locations = [
        (100, 100, "top-left"),
        (500, 200, "upper-middle"), 
        (200, 500, "middle-left"),
        (800, 300, "right-side"),
        (400, 600, "lower-middle")
    ]
    
    print("\n=== TESTING MULTIPLE REAL LOCATIONS ===")
    
    all_results = {}
    
    for x, y, description in test_locations:
        print(f"\nTesting {description} at ({x}, {y}):")
        
        cells = extract_real_top_line(img, x, y, 15)
        
        if cells:
            bits = ''.join(str(cell['bit']) for cell in cells)
            blue_values = [cell['blue_mean'] for cell in cells]
            
            print(f"  Bits: {bits}")
            print(f"  Blue values: {[f'{v:.1f}' for v in blue_values[:8]]}")
            print(f"  Avg blue: {np.mean(blue_values):.1f}")
            
            # Check for suspicious patterns
            ones_count = bits.count('1')
            zeros_count = bits.count('0') 
            ratio = ones_count / max(zeros_count, 1)
            
            print(f"  Ratio: {ratio:.2f}:1 ({'SUSPICIOUS' if ratio > 5 or ratio < 0.2 else 'OK'})")
            
            all_results[description] = {
                'bits': bits,
                'blue_avg': np.mean(blue_values),
                'ratio': ratio,
                'cells': cells
            }
            
    return all_results

def main():
    print("=== REAL POSTER EXTRACTION VERIFICATION ===")
    
    # Load real poster
    img = load_real_poster()
    if img is None:
        print("Cannot proceed without real poster image")
        return
        
    # Test multiple real locations
    results = test_multiple_locations(img)
    
    print(f"\n=== REAL EXTRACTION SUMMARY ===")
    print(f"Image size: {img.shape}")
    print(f"Locations tested: {len(results)}")
    
    # Analyze patterns across all locations
    all_ratios = [r['ratio'] for r in results.values()]
    avg_ratio = np.mean(all_ratios)
    
    print(f"Average bit ratio across all locations: {avg_ratio:.2f}:1")
    print(f"Ratio range: {min(all_ratios):.2f} to {max(all_ratios):.2f}")
    
    # Check if results look realistic
    realistic = all(0.1 <= ratio <= 10 for ratio in all_ratios)
    print(f"Results appear realistic: {realistic}")
    
    if not realistic:
        print("WARNING: Some ratios still look suspicious")
        
    # Show some actual blue channel values to verify we're reading real data
    print(f"\nSample real blue channel values from poster:")
    for name, result in list(results.items())[:2]:
        blue_vals = [cell['blue_mean'] for cell in result['cells'][:8]]
        print(f"  {name}: {[f'{v:.1f}' for v in blue_vals]}")
        
    return results

if __name__ == "__main__":
    results = main()