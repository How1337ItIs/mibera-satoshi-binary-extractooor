#!/usr/bin/env python3
"""
Optimize extraction parameters to improve clarity rate.

Created by Claude Code - July 16, 2025
Purpose: Analyze failed regions and optimize thresholds for better accuracy
"""
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_failed_regions():
    """Analyze regions with 0% clarity to understand failure modes"""
    
    # Load extraction results
    with open('complete_extraction_detailed.json', 'r') as f:
        data = json.load(f)
    
    region_stats = data['region_stats']
    failed_regions = [r for r in region_stats if r['clarity_rate'] == 0.0]
    
    print(f"Analyzing {len(failed_regions)} failed regions...")
    
    img = cv2.imread('satoshi (1).png')
    
    for region_stat in failed_regions[:5]:  # Analyze first 5 failed regions
        region = region_stat['region']
        region_id = region_stat['region_id']
        
        print(f"\n--- Region {region_id} Analysis ---")
        print(f"Size: {region['w']}x{region['h']} at ({region['x']}, {region['y']})")
        
        # Extract region image
        roi = img[region['y']:region['y']+region['h'], 
                 region['x']:region['x']+region['w']]
        
        if roi.size == 0:
            continue
            
        # Analyze color characteristics
        blue_channel = roi[:, :, 0]
        green_channel = roi[:, :, 1]
        red_channel = roi[:, :, 2]
        
        print(f"Blue channel stats: mean={np.mean(blue_channel):.1f}, std={np.std(blue_channel):.1f}")
        print(f"Green channel stats: mean={np.mean(green_channel):.1f}, std={np.std(green_channel):.1f}")
        print(f"Red channel stats: mean={np.mean(red_channel):.1f}, std={np.std(red_channel):.1f}")
        
        # Analyze HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1] 
        val = hsv[:, :, 2]
        
        print(f"HSV stats: H={np.mean(hue):.1f}, S={np.mean(sat):.1f}, V={np.mean(val):.1f}")
        
        # Save region sample for visual inspection
        cv2.imwrite(f'failed_region_{region_id}_sample.png', roi)
        
    return failed_regions

def test_threshold_variations():
    """Test different classification thresholds on existing data"""
    
    # Load extraction data
    with open('complete_extraction_detailed.json', 'r') as f:
        data = json.load(f)
    
    all_cells = data['cells']
    
    # Test different blue channel thresholds
    thresholds = [
        (140, 110),  # More permissive
        (150, 100),  # Slightly more permissive
        (160, 90),   # Current
        (170, 80),   # More strict
        (180, 70)    # Very strict
    ]
    
    print("Testing threshold variations on extracted data...")
    print("Threshold (high, low) | Clear Rate | Zeros | Ones | Ambiguous")
    print("-" * 60)
    
    for hi_thresh, lo_thresh in thresholds:
        zeros = ones = ambiguous = 0
        
        for cell in all_cells:
            confidence = cell['confidence']
            
            if confidence > hi_thresh:
                zeros += 1
            elif confidence < lo_thresh:
                ones += 1
            else:
                ambiguous += 1
        
        total = len(all_cells)
        clear = zeros + ones
        clear_rate = clear / total * 100
        
        print(f"({hi_thresh}, {lo_thresh})           | {clear_rate:5.1f}%   | {zeros:4d} | {ones:4d} | {ambiguous:4d}")
    
    return thresholds

def analyze_gradient_regions():
    """Analyze regions with gradient transitions for better parameter tuning"""
    
    img = cv2.imread('satoshi (1).png')
    
    # Load regions 
    with open('digit_regions.json', 'r') as f:
        regions = json.load(f)
    
    # Focus on medium-performing regions for optimization
    with open('complete_extraction_detailed.json', 'r') as f:
        data = json.load(f)
    
    region_stats = data['region_stats']
    medium_regions = [r for r in region_stats if 10 < r['clarity_rate'] < 40]
    
    print(f"\nAnalyzing {len(medium_regions)} medium-performing regions for optimization...")
    
    for region_stat in medium_regions[:3]:
        region_id = region_stat['region_id']
        region = regions[region_id]
        
        roi = img[region['y']:region['y']+region['h'], 
                 region['x']:region['x']+region['w']]
        
        # Compute gradients to find digit boundaries
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        
        # Analyze gradient distribution
        high_grad_threshold = np.percentile(gradient, 80)
        digit_candidates = gradient > high_grad_threshold
        
        print(f"Region {region_id}: {np.sum(digit_candidates)} high-gradient pixels")
        
        # Save gradient visualization
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        plt.title(f'Region {region_id} Original')
        plt.axis('off')
        
        plt.subplot(1, 3, 2) 
        plt.imshow(gradient, cmap='gray')
        plt.title('Gradient Magnitude')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(digit_candidates, cmap='gray')
        plt.title('Digit Candidates')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'region_{region_id}_gradient_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

def suggest_optimized_parameters():
    """Suggest optimized parameters based on analysis"""
    
    print("\n" + "="*60)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("="*60)
    
    print("\n1. THRESHOLD OPTIMIZATION:")
    print("   - Current: blue > 160 = '0', blue < 90 = '1'")
    print("   - Suggested: blue > 150 = '0', blue < 100 = '1'")
    print("   - Rationale: Slightly more permissive for better coverage")
    
    print("\n2. REGION-SPECIFIC TUNING:")
    print("   - High-contrast regions: Use stricter thresholds (170/80)")
    print("   - Low-contrast regions: Use permissive thresholds (140/110)")
    print("   - Medium regions: Use gradient-based refinement")
    
    print("\n3. GRID REFINEMENT:")
    print("   - Current 15×12 pitch is correct")
    print("   - Consider offset variations ±2 pixels for failed regions")
    print("   - Use gradient analysis to fine-tune grid alignment")
    
    print("\n4. MULTI-METHOD APPROACH:")
    print("   - Primary: Blue channel classification")
    print("   - Secondary: HSV saturation for cyan detection")
    print("   - Tertiary: Gradient-based edge detection")
    
    return {
        'permissive_threshold': (150, 100),
        'strict_threshold': (170, 80),
        'default_threshold': (160, 90),
        'grid_pitch': (15, 12),
        'offset_variations': [-2, -1, 0, 1, 2]
    }

if __name__ == "__main__":
    print("=== EXTRACTION PARAMETER OPTIMIZATION ===")
    
    print("\n1. Analyzing failed regions...")
    failed_regions = analyze_failed_regions()
    
    print("\n2. Testing threshold variations...")
    test_threshold_variations()
    
    print("\n3. Analyzing gradient patterns...")
    analyze_gradient_regions()
    
    print("\n4. Generating recommendations...")
    params = suggest_optimized_parameters()
    
    # Save recommendations
    with open('optimization_recommendations.json', 'w') as f:
        json.dump(params, f, indent=2)
    
    print(f"\n✅ Optimization analysis complete!")
    print(f"📊 Recommendations saved to optimization_recommendations.json")
    print(f"🔍 Visual analysis saved as region_*_gradient_analysis.png")