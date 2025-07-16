#!/usr/bin/env python3
"""
Map the full poster to identify all digit regions.

Created by Claude Code - July 16, 2025
Purpose: Find all areas of the poster containing binary digits for systematic extraction
"""
import cv2
import numpy as np
import json

def analyze_full_poster():
    """Analyze the full poster to find digit regions"""
    
    # Load image
    img = cv2.imread('satoshi (1).png')
    print(f"Full poster size: {img.shape}")
    
    # Convert to different color spaces for analysis
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Create overview map by downsampling
    scale_factor = 4
    small_img = cv2.resize(img, (img.shape[1]//scale_factor, img.shape[0]//scale_factor))
    
    # Look for regions with high contrast (likely text/digits)
    # Use gradient magnitude to find text regions
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Threshold to find high-contrast regions
    _, high_contrast = cv2.threshold(gradient_magnitude.astype(np.uint8), 30, 255, cv2.THRESH_BINARY)
    
    # Downsample gradient map
    contrast_map = cv2.resize(high_contrast, (img.shape[1]//scale_factor, img.shape[0]//scale_factor))
    
    # Find regions with cyan/blue colors (like our working region)
    hsv_small = cv2.resize(hsv, (img.shape[1]//scale_factor, img.shape[0]//scale_factor))
    
    # Create mask for cyan-like colors (H=90-120, S>50, V>50)
    cyan_mask = cv2.inRange(hsv_small, (90, 50, 50), (120, 255, 255))
    
    # Combine contrast and color information
    text_regions = cv2.bitwise_and(contrast_map, cyan_mask)
    
    # Find contours of potential digit regions
    contours, _ = cv2.findContours(text_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size
    min_area = 50  # Minimum area for a digit region
    digit_regions = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Scale back up to full resolution
            full_x = x * scale_factor
            full_y = y * scale_factor
            full_w = w * scale_factor
            full_h = h * scale_factor
            
            digit_regions.append({
                'x': full_x, 'y': full_y, 'w': full_w, 'h': full_h,
                'area': area * scale_factor * scale_factor,
                'center_x': full_x + full_w//2,
                'center_y': full_y + full_h//2
            })
    
    print(f"Found {len(digit_regions)} potential digit regions")
    
    # Sort by size (largest first)
    digit_regions.sort(key=lambda r: r['area'], reverse=True)
    
    # Create visualization
    vis_img = img.copy()
    for i, region in enumerate(digit_regions[:10]):  # Show top 10 regions
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(vis_img, str(i), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Save visualization
    cv2.imwrite('poster_digit_regions.png', vis_img)
    
    # Save regions data
    with open('digit_regions.json', 'w') as f:
        json.dump(digit_regions, f, indent=2)
    
    # Print top regions
    print("\nTop digit regions found:")
    for i, region in enumerate(digit_regions[:5]):
        print(f"Region {i}: x={region['x']}, y={region['y']}, size={region['w']}x{region['h']}, area={region['area']}")
    
    return digit_regions

def test_regions_with_corrected_method(digit_regions):
    """Test our corrected extraction method on found regions"""
    
    img = cv2.imread('satoshi (1).png')
    
    # Our known working parameters
    row_pitch = 15
    col_pitch = 12
    
    results = []
    
    for i, region in enumerate(digit_regions[:5]):  # Test top 5 regions
        print(f"\nTesting region {i}: {region['w']}x{region['h']} at ({region['x']}, {region['y']})")
        
        # Calculate how many rows/cols fit in this region
        max_rows = region['h'] // row_pitch
        max_cols = region['w'] // col_pitch
        
        if max_rows < 3 or max_cols < 5:
            print(f"  Skipping - too small ({max_rows}x{max_cols})")
            continue
        
        # Test extraction on this region
        cells_extracted = 0
        clear_classifications = 0
        
        for r in range(min(max_rows, 10)):  # Test up to 10 rows
            for c in range(min(max_cols, 15)):  # Test up to 15 cols
                y = region['y'] + r * row_pitch
                x = region['x'] + c * col_pitch
                
                # Extract cell
                cell = img[max(0, y-3):min(img.shape[0], y+4), 
                          max(0, x-3):min(img.shape[1], x+4)]
                
                if cell.size == 0:
                    continue
                
                cells_extracted += 1
                
                # Blue channel classification
                blue_channel = cell[:, :, 0]
                avg_blue = np.mean(blue_channel)
                
                if avg_blue > 150 or avg_blue < 100:  # Clear 0 or 1
                    clear_classifications += 1
        
        if cells_extracted > 0:
            clarity_rate = clear_classifications / cells_extracted * 100
            print(f"  Extracted {cells_extracted} cells, {clear_classifications} clear ({clarity_rate:.1f}%)")
            
            results.append({
                'region_id': i,
                'region': region,
                'cells_extracted': cells_extracted,
                'clear_classifications': clear_classifications,
                'clarity_rate': clarity_rate
            })
    
    # Sort by clarity rate
    results.sort(key=lambda r: r['clarity_rate'], reverse=True)
    
    print(f"\nBest regions for extraction:")
    for result in results[:3]:
        print(f"Region {result['region_id']}: {result['clarity_rate']:.1f}% clarity ({result['clear_classifications']}/{result['cells_extracted']})")
    
    return results

if __name__ == "__main__":
    print("=== MAPPING FULL POSTER ===")
    regions = analyze_full_poster()
    
    print("\n=== TESTING REGIONS ===") 
    results = test_regions_with_corrected_method(regions)
    
    print(f"\nFiles created:")
    print("- poster_digit_regions.png (visualization)")
    print("- digit_regions.json (region data)")
    print(f"\nFound {len(results)} testable regions with extraction potential")