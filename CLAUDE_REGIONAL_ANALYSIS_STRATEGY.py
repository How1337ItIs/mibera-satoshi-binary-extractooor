#!/usr/bin/env python3
"""
Claude's Regional Analysis Strategy
The user is absolutely right - we need to test different methods for different poster regions.
Different areas (light vs dark, different backgrounds) likely need different approaches.

Author: Claude Code Agent
Date: July 17, 2025
Insight: Test accuracy against regional variations, not just single pattern
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def claude_comprehensive_regional_analysis():
    """Analyze poster regions and test methods appropriate for each area."""
    
    print("Claude: Comprehensive Regional Analysis")
    print("Testing different methods for different poster regions (light vs dark, etc.)")
    
    # Load poster
    img = cv2.imread('satoshi.png')
    if img is None:
        img = cv2.imread('satoshi (1).png')
    if img is None:
        print("ERROR: Could not load poster")
        return
    
    print(f"Claude: Loaded poster {img.shape}")
    
    # Convert to different color spaces for analysis
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    
    # STEP 1: Analyze poster regions by brightness and characteristics
    print(f"\nClaude: STEP 1 - Poster Region Characterization")
    
    regions = claude_identify_poster_regions(gray, hsv)
    
    # STEP 2: Current accuracy testing scope
    print(f"\nClaude: STEP 2 - Current Testing Scope Analysis")
    
    current_test_analysis = claude_analyze_current_testing_scope()
    
    # STEP 3: Test different methods on different regions
    print(f"\nClaude: STEP 3 - Regional Method Testing")
    
    regional_results = claude_test_methods_by_region(img, regions)
    
    # STEP 4: Identify optimal method per region
    print(f"\nClaude: STEP 4 - Regional Optimization")
    
    optimization_results = claude_optimize_per_region(img, regions, regional_results)
    
    # STEP 5: Create comprehensive testing framework
    print(f"\nClaude: STEP 5 - Comprehensive Testing Framework")
    
    testing_framework = claude_create_regional_testing_framework(regions, optimization_results)
    
    # Save results
    claude_save_regional_analysis(regions, current_test_analysis, regional_results, optimization_results, testing_framework)
    
    return {
        'regions': regions,
        'current_scope': current_test_analysis,
        'regional_results': regional_results,
        'optimization': optimization_results,
        'framework': testing_framework
    }

def claude_identify_poster_regions(gray, hsv):
    """Identify different regions of the poster by characteristics."""
    
    print("  Analyzing poster brightness, contrast, and texture regions...")
    
    h, w = gray.shape
    
    # Divide poster into grid for analysis
    grid_size = 100  # 100x100 pixel regions
    regions = []
    
    for y in range(0, h - grid_size, grid_size):
        for x in range(0, w - grid_size, grid_size):
            region_gray = gray[y:y+grid_size, x:x+grid_size]
            region_hsv = hsv[y:y+grid_size, x:x+grid_size]
            
            # Calculate region characteristics
            brightness = np.mean(region_gray)
            contrast = np.std(region_gray)
            saturation = np.mean(region_hsv[:, :, 1])
            value = np.mean(region_hsv[:, :, 2])
            
            # Classify region type
            if brightness < 50:
                region_type = "very_dark"
            elif brightness < 100:
                region_type = "dark"
            elif brightness < 150:
                region_type = "medium"
            elif brightness < 200:
                region_type = "light"
            else:
                region_type = "very_light"
            
            # Add contrast classification
            if contrast < 20:
                contrast_type = "low_contrast"
            elif contrast < 40:
                contrast_type = "medium_contrast"
            else:
                contrast_type = "high_contrast"
            
            regions.append({
                'x': x,
                'y': y,
                'size': grid_size,
                'brightness': brightness,
                'contrast': contrast,
                'saturation': saturation,
                'value': value,
                'region_type': region_type,
                'contrast_type': contrast_type,
                'combined_type': f"{region_type}_{contrast_type}"
            })
    
    # Group by region types
    region_types = {}
    for region in regions:
        rtype = region['combined_type']
        if rtype not in region_types:
            region_types[rtype] = []
        region_types[rtype].append(region)
    
    print(f"  Found {len(regions)} total regions across {len(region_types)} types:")
    for rtype, rlist in region_types.items():
        print(f"    {rtype}: {len(rlist)} regions")
    
    return {
        'individual': regions,
        'by_type': region_types,
        'summary': {
            'total_regions': len(regions),
            'region_types': len(region_types),
            'type_distribution': {k: len(v) for k, v in region_types.items()}
        }
    }

def claude_analyze_current_testing_scope():
    """Analyze what we're currently testing accuracy against."""
    
    print("  Analyzing current testing methodology...")
    
    # Our current test
    current_test = {
        'pattern': "010011110110111000100000011101000110100001100101",
        'decoded': "On the ",
        'coordinates': {'y': 69.6, 'x': 37.0, 'spacing': 17.9},
        'length': 48,  # bits
        'region': 'single_location'
    }
    
    # Analyze the poster region at our test coordinates
    img = cv2.imread('satoshi.png')
    if img is None:
        img = cv2.imread('satoshi (1).png')
    
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Sample region around our test coordinates
        y, x = int(current_test['coordinates']['y']), int(current_test['coordinates']['x'])
        sample_size = 50
        
        if y + sample_size < gray.shape[0] and x + sample_size < gray.shape[1]:
            sample_region = gray[y-sample_size//2:y+sample_size//2, x-sample_size//2:x+sample_size//2]
            hsv_sample = hsv[y-sample_size//2:y+sample_size//2, x-sample_size//2:x+sample_size//2]
            
            current_test['region_characteristics'] = {
                'brightness': np.mean(sample_region),
                'contrast': np.std(sample_region),
                'saturation': np.mean(hsv_sample[:, :, 1]),
                'value': np.mean(hsv_sample[:, :, 2]),
                'location': f"y={y}, x={x}"
            }
    
    print(f"  Current test scope:")
    print(f"    Pattern: '{current_test['decoded']}' ({current_test['length']} bits)")
    print(f"    Location: {current_test['coordinates']}")
    if 'region_characteristics' in current_test:
        rc = current_test['region_characteristics']
        print(f"    Region: brightness={rc['brightness']:.1f}, contrast={rc['contrast']:.1f}")
    
    print(f"  LIMITATION: Testing only single region/pattern - not representative of full poster!")
    
    return current_test

def claude_test_methods_by_region(img, regions):
    """Test different extraction methods on different poster regions."""
    
    print("  Testing extraction methods across different poster regions...")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    
    # Define methods to test
    methods = {
        'hsv_v_simple': lambda img: hsv[:, :, 2],
        'hsv_v_clahe': lambda img: cv2.createCLAHE(clipLimit=3.0).apply(hsv[:, :, 2]),
        'lab_b_channel': lambda img: lab[:, :, 2],
        'gray_adaptive': lambda img: cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
        'fft_highpass': lambda img: claude_apply_fft_highpass_simple(gray, 30),
        'bilateral_filtered': lambda img: cv2.bilateralFilter(gray, 9, 75, 75)
    }
    
    regional_results = {}
    
    # Test each region type
    for region_type, region_list in regions['by_type'].items():
        if len(region_list) < 3:  # Skip region types with too few samples
            continue
            
        print(f"    Testing {region_type} regions ({len(region_list)} samples)...")
        
        regional_results[region_type] = {}
        
        # Sample a few regions of this type
        sample_regions = region_list[:min(5, len(region_list))]
        
        for method_name, method_func in methods.items():
            scores = []
            
            for region in sample_regions:
                # Apply method and test in this region
                processed_img = method_func(img)
                
                # Test extraction quality in this region
                quality_score = claude_assess_extraction_quality(processed_img, region)
                scores.append(quality_score)
            
            avg_score = np.mean(scores) if scores else 0
            regional_results[region_type][method_name] = {
                'avg_score': avg_score,
                'scores': scores,
                'sample_count': len(scores)
            }
            
            print(f"      {method_name}: {avg_score:.3f}")
    
    return regional_results

def claude_apply_fft_highpass_simple(image, filter_size):
    """Simple FFT high-pass filter for regional testing."""
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    
    mask = np.ones((rows, cols), np.uint8)
    mask[crow-filter_size:crow+filter_size, ccol-filter_size:ccol+filter_size] = 0
    
    f_shift_filtered = f_shift * mask
    f_ishift = np.fft.ifftshift(f_shift_filtered)
    img_filtered = np.fft.ifft2(f_ishift)
    img_filtered = np.abs(img_filtered)
    
    return ((img_filtered - img_filtered.min()) / (img_filtered.max() - img_filtered.min()) * 255).astype(np.uint8)

def claude_assess_extraction_quality(processed_img, region):
    """Assess extraction quality in a specific region."""
    
    # Extract region from processed image
    x, y, size = region['x'], region['y'], region['size']
    region_img = processed_img[y:y+size, x:x+size]
    
    # Calculate quality metrics
    edge_strength = np.std(cv2.Laplacian(region_img, cv2.CV_64F))
    contrast = np.std(region_img)
    
    # Look for binary-like patterns (bimodal distribution)
    hist = cv2.calcHist([region_img], [0], None, [256], [0, 256])
    hist_smooth = cv2.blur(hist.flatten(), (5, 1))
    
    # Find peaks in histogram (binary should have 2 peaks)
    peaks = []
    for i in range(1, len(hist_smooth) - 1):
        if hist_smooth[i] > hist_smooth[i-1] and hist_smooth[i] > hist_smooth[i+1]:
            if hist_smooth[i] > np.max(hist_smooth) * 0.1:  # Significant peaks only
                peaks.append(i)
    
    # Score based on having 2 clear peaks (binary pattern)
    binary_score = 1.0 if len(peaks) == 2 else max(0, 1.0 - abs(len(peaks) - 2) * 0.3)
    
    # Combined quality score
    quality = (edge_strength * 0.3 + contrast * 0.3 + binary_score * 0.4) / 100.0
    
    return min(1.0, quality)

def claude_optimize_per_region(img, regions, regional_results):
    """Identify optimal extraction method for each region type."""
    
    print("  Identifying optimal methods per region type...")
    
    optimization = {}
    
    for region_type, method_results in regional_results.items():
        # Find best method for this region type
        best_method = None
        best_score = 0
        
        for method_name, result in method_results.items():
            if result['avg_score'] > best_score:
                best_score = result['avg_score']
                best_method = method_name
        
        optimization[region_type] = {
            'best_method': best_method,
            'best_score': best_score,
            'all_scores': {k: v['avg_score'] for k, v in method_results.items()}
        }
        
        print(f"    {region_type}: {best_method} ({best_score:.3f})")
    
    return optimization

def claude_create_regional_testing_framework(regions, optimization_results):
    """Create framework for comprehensive regional testing."""
    
    print("  Creating comprehensive regional testing framework...")
    
    framework = {
        'region_types': list(regions['by_type'].keys()),
        'methods_per_region': {},
        'testing_protocol': {},
        'validation_approach': {}
    }
    
    # Define testing protocol
    framework['testing_protocol'] = {
        'sample_regions_per_type': 5,
        'test_patterns_per_region': 3,
        'accuracy_metrics': ['bit_accuracy', 'ascii_readability', 'pattern_consistency'],
        'validation_methods': ['manual_verification', 'cross_region_consistency', 'method_agreement']
    }
    
    # Define methods per region type based on optimization
    for region_type, opt_result in optimization_results.items():
        framework['methods_per_region'][region_type] = {
            'primary': opt_result['best_method'],
            'alternatives': sorted(opt_result['all_scores'].items(), key=lambda x: x[1], reverse=True)[:3],
            'confidence': opt_result['best_score']
        }
    
    # Define validation approach
    framework['validation_approach'] = {
        'current_limitation': 'Testing only single pattern "On the " at one location',
        'recommended_approach': 'Test multiple patterns across different region types',
        'expansion_needed': [
            'Sample light vs dark regions',
            'Test high vs low contrast areas',
            'Validate across poster quadrants',
            'Use region-specific optimal methods'
        ]
    }
    
    return framework

def claude_save_regional_analysis(regions, current_scope, regional_results, optimization, framework):
    """Save comprehensive regional analysis results."""
    
    with open('CLAUDE_REGIONAL_ANALYSIS_RESULTS.txt', 'w') as f:
        f.write("=== CLAUDE'S COMPREHENSIVE REGIONAL ANALYSIS ===\n")
        f.write("Author: Claude Code Agent\n")
        f.write("Insight: Different poster regions need different extraction methods\n\n")
        
        f.write("CURRENT TESTING SCOPE ANALYSIS:\n")
        f.write(f"  Pattern tested: '{current_scope['decoded']}'\n")
        f.write(f"  Location: {current_scope['coordinates']}\n")
        f.write(f"  Bits tested: {current_scope['length']}\n")
        if 'region_characteristics' in current_scope:
            rc = current_scope['region_characteristics']
            f.write(f"  Region characteristics: brightness={rc['brightness']:.1f}, contrast={rc['contrast']:.1f}\n")
        f.write(f"  LIMITATION: Single region/pattern - not representative!\n\n")
        
        f.write("POSTER REGION ANALYSIS:\n")
        f.write(f"  Total regions analyzed: {regions['summary']['total_regions']}\n")
        f.write(f"  Region types found: {regions['summary']['region_types']}\n")
        f.write("  Type distribution:\n")
        for rtype, count in regions['summary']['type_distribution'].items():
            f.write(f"    {rtype}: {count} regions\n")
        f.write("\n")
        
        f.write("REGIONAL METHOD OPTIMIZATION:\n")
        for region_type, opt_result in optimization.items():
            f.write(f"  {region_type}:\n")
            f.write(f"    Best method: {opt_result['best_method']} ({opt_result['best_score']:.3f})\n")
            f.write(f"    All methods: {opt_result['all_scores']}\n")
        f.write("\n")
        
        f.write("RECOMMENDED TESTING FRAMEWORK:\n")
        f.write("  Current issue: Testing only single pattern at one location\n")
        f.write("  Solution: Region-specific testing with appropriate methods\n")
        f.write("  Implementation:\n")
        for region_type, methods in framework['methods_per_region'].items():
            f.write(f"    {region_type}: Use {methods['primary']} (confidence: {methods['confidence']:.3f})\n")
        f.write("\n")
        
        f.write("NEXT STEPS:\n")
        f.write("1. Expand testing beyond single 'On the ' pattern\n")
        f.write("2. Test extraction in light vs dark poster regions\n")
        f.write("3. Use region-appropriate methods for each area\n")
        f.write("4. Validate across multiple poster locations\n")
        f.write("5. Create comprehensive accuracy assessment\n")

if __name__ == "__main__":
    print("Claude Code Agent: Comprehensive Regional Analysis")
    print("User insight: Test different methods for different poster regions")
    
    results = claude_comprehensive_regional_analysis()
    
    print(f"\nClaude: Regional analysis complete!")
    print(f"Claude: Found {results['regions']['summary']['region_types']} region types")
    print(f"Claude: Identified region-specific optimal methods")
    print(f"Claude: Created framework for comprehensive testing")
    print(f"Claude: Results saved to CLAUDE_REGIONAL_ANALYSIS_RESULTS.txt")