#!/usr/bin/env python3
"""
Comprehensive color space testing for Satoshi poster binary extraction.
Tests all available color spaces to find optimal extraction parameters.
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path

# Add the binary_extractor to path
sys.path.append(str(Path(__file__).parent / "binary_extractor"))

from extractor.pipeline import run

def test_color_spaces():
    """Test all available color spaces for extraction quality."""
    
    color_spaces = [
        'RGB_R', 'RGB_G', 'RGB_B',
        'HSV_H', 'HSV_S', 'HSV_V', 
        'LAB_L', 'LAB_A', 'LAB_B',
        'YUV_Y', 'YUV_U', 'YUV_V',
        'HLS_H', 'HLS_L', 'HLS_S',
        'LUV_L', 'LUV_U', 'LUV_V'
    ]
    
    image_path = "satoshi (1).png"
    results = []
    
    print("Testing all color spaces for optimal extraction...")
    print("=" * 60)
    
    for color_space in color_spaces:
        print(f"\n📊 Testing {color_space}...")
        
        # Create output directory
        output_dir = f"test_results/color_space_{color_space.lower()}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create custom config for this test
        config = {
            'use_color_space': color_space,
            'blur_sigma': 25,
            'threshold': {
                'method': 'otsu',
                'adaptive_C': 4,
                'sauvola_window_size': 15,
                'sauvola_k': 0.2
            },
            'morph_k': 3,
            'morph_iterations': 2,
            'use_mahotas_thin': False,
            'row_pitch': None,
            'col_pitch': None,
            'row0': 50,
            'col0': 20,
            'bit_hi': 0.7,
            'bit_lo': 0.3,
            'overlay': {
                'saturation_threshold': 40,
                'value_threshold': 180,
                'cell_coverage_threshold': 0.2,
                'dilate_pixels': 2
            },
            'template_match': False,
            'tm_thresh': 0.45,
            'save_debug': True,
            'debug_artifacts': ['bw_mask.png', 'grid_overlay.png'],
            'output': {'csv_encoding': 'utf-8'}
        }
        
        try:
            # Run extraction
            from extractor.pipeline import run_with_config
            result = run_with_config(image_path, output_dir, config)
            
            # Count results
            cells_file = os.path.join(output_dir, "cells.csv")
            if os.path.exists(cells_file):
                with open(cells_file, 'r') as f:
                    lines = f.readlines()
                    
                total_cells = len(lines) - 1  # subtract header
                zeros = sum(1 for line in lines[1:] if ',0' in line and line.strip().endswith(',0'))
                ones = sum(1 for line in lines[1:] if ',1' in line and line.strip().endswith(',1'))
                blanks = sum(1 for line in lines[1:] if ',blank' in line)
                overlays = sum(1 for line in lines[1:] if ',overlay' in line)
                
                success_rate = (zeros + ones) / total_cells * 100 if total_cells > 0 else 0
                
                results.append({
                    'color_space': color_space,
                    'total_cells': total_cells,
                    'zeros': zeros,
                    'ones': ones,
                    'blanks': blanks,
                    'overlays': overlays,
                    'success_rate': success_rate,
                    'binary_ratio': ones / zeros if zeros > 0 else 0
                })
                
                print(f"✅ {color_space}: {success_rate:.1f}% success, {zeros} zeros, {ones} ones")
            else:
                print(f"❌ {color_space}: Failed to generate results")
                
        except Exception as e:
            print(f"❌ {color_space}: Error - {str(e)}")
            
    return results

def create_color_space_config(color_space):
    """Create a temporary config file for testing."""
    config_content = f"""# Test configuration for {color_space}
use_color_space: {color_space}
blur_sigma: 25
threshold:
  method: otsu
  adaptive_C: 4
  sauvola_window_size: 15
  sauvola_k: 0.2
morph_k: 3
morph_iterations: 2
use_mahotas_thin: false
row_pitch: null
col_pitch: null
row0: 50
col0: 20
bit_hi: 0.7
bit_lo: 0.3
overlay:
  saturation_threshold: 40
  value_threshold: 180
  cell_coverage_threshold: 0.2
  dilate_pixels: 2
template_match: false
tm_thresh: 0.45
save_debug: true
debug_artifacts:
  - bw_mask.png
  - grid_overlay.png
output:
  csv_encoding: 'utf-8'
"""
    
    config_path = f"binary_extractor/test_cfg_{color_space.lower()}.yaml"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    return config_path

if __name__ == "__main__":
    # Create test results directory
    os.makedirs("test_results", exist_ok=True)
    
    # Test each color space
    color_spaces = [
        'RGB_R', 'RGB_G', 'RGB_B',
        'HSV_H', 'HSV_S', 'HSV_V', 
        'LAB_L', 'LAB_A', 'LAB_B',
        'YUV_Y', 'YUV_U', 'YUV_V',
        'HLS_H', 'HLS_L', 'HLS_S'
    ]
    
    results = []
    
    for color_space in color_spaces:
        print(f"\n🔬 Testing {color_space}")
        
        # Create config file
        config_path = create_color_space_config(color_space)
        
        # Create output directory
        output_dir = f"test_results/color_{color_space.lower()}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Run extraction
        os.system(f'cd binary_extractor && python -c "from extractor.pipeline import run; import yaml; cfg = yaml.safe_load(open(\'{config_path}\', \'r\')); run(\'../satoshi (1).png\', \'../{output_dir}\', cfg)"')
        
        # Analyze results
        cells_file = os.path.join(output_dir, "cells.csv")
        if os.path.exists(cells_file):
            with open(cells_file, 'r') as f:
                lines = f.readlines()
                
            total_cells = len(lines) - 1
            zeros = sum(1 for line in lines[1:] if line.strip().endswith(',0'))
            ones = sum(1 for line in lines[1:] if line.strip().endswith(',1'))
            blanks = sum(1 for line in lines[1:] if ',blank' in line)
            overlays = sum(1 for line in lines[1:] if ',overlay' in line)
            
            success_rate = (zeros + ones) / total_cells * 100 if total_cells > 0 else 0
            
            results.append({
                'color_space': color_space,
                'total_cells': total_cells,
                'zeros': zeros,
                'ones': ones,
                'blanks': blanks,
                'overlays': overlays,
                'success_rate': success_rate
            })
            
            print(f"✅ {color_space}: {success_rate:.1f}% success, {total_cells} total cells")
        else:
            print(f"❌ {color_space}: No results file generated")
        
        # Cleanup
        if os.path.exists(config_path):
            os.remove(config_path)
    
    # Generate summary report
    with open("test_results/COLOR_SPACE_RESULTS.md", 'w') as f:
        f.write("# Color Space Testing Results\n\n")
        f.write("| Color Space | Total Cells | Zeros | Ones | Blanks | Overlays | Success Rate |\n")
        f.write("|-------------|-------------|-------|------|--------|----------|-------------|\n")
        
        # Sort by success rate
        results.sort(key=lambda x: x['success_rate'], reverse=True)
        
        for result in results:
            f.write(f"| {result['color_space']} | {result['total_cells']} | {result['zeros']} | {result['ones']} | {result['blanks']} | {result['overlays']} | {result['success_rate']:.1f}% |\n")
    
    print(f"\n📊 Results saved to test_results/COLOR_SPACE_RESULTS.md")
    print(f"🏆 Best color space: {results[0]['color_space']} ({results[0]['success_rate']:.1f}% success)")