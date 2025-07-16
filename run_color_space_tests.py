#!/usr/bin/env python3
"""
Automated color space testing for Satoshi poster extraction.
"""

import os
import shutil
import yaml
from pathlib import Path

def test_color_space(color_space, output_suffix=""):
    """Test a specific color space configuration."""
    
    # Backup original config
    config_path = "binary_extractor/extractor/cfg.yaml"
    backup_path = "binary_extractor/extractor/cfg.yaml.backup"
    
    if not os.path.exists(backup_path):
        shutil.copy2(config_path, backup_path)
    
    # Read current config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify color space
    config['use_color_space'] = color_space
    config['template_match'] = False  # Disable for consistency
    
    # Write modified config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Create output directory
    output_dir = f"test_results/color_{color_space.lower()}{output_suffix}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run extraction
    print(f"Testing {color_space}...")
    result = os.system(f'cd binary_extractor && python scripts/extract.py "../satoshi (1).png" "../{output_dir}/" > nul 2>&1')
    
    # Analyze results
    cells_file = f"{output_dir}/cells.csv"
    if os.path.exists(cells_file):
        with open(cells_file, 'r') as f:
            lines = f.readlines()
        
        total_cells = len(lines) - 1
        zeros = sum(1 for line in lines[1:] if line.strip().endswith(',0'))
        ones = sum(1 for line in lines[1:] if line.strip().endswith(',1'))
        blanks = sum(1 for line in lines[1:] if ',blank' in line)
        overlays = sum(1 for line in lines[1:] if ',overlay' in line)
        
        success_rate = (zeros + ones) / total_cells * 100 if total_cells > 0 else 0
        
        print(f"SUCCESS {color_space}: {success_rate:.1f}% success ({zeros} zeros, {ones} ones, {blanks} blanks, {overlays} overlays)")
        
        return {
            'color_space': color_space,
            'total_cells': total_cells,
            'zeros': zeros,
            'ones': ones,
            'blanks': blanks,
            'overlays': overlays,
            'success_rate': success_rate,
            'binary_ratio': ones / zeros if zeros > 0 else 0
        }
    else:
        print(f"FAILED {color_space}: No results generated")
        return None

def restore_config():
    """Restore original config."""
    config_path = "binary_extractor/extractor/cfg.yaml"
    backup_path = "binary_extractor/extractor/cfg.yaml.backup"
    
    if os.path.exists(backup_path):
        shutil.copy2(backup_path, config_path)
        print("Original config restored")

if __name__ == "__main__":
    # Create test results directory
    os.makedirs("test_results", exist_ok=True)
    
    # Test different color spaces
    color_spaces = [
        'RGB_R', 'RGB_G', 'RGB_B',
        'HSV_H', 'HSV_S', 'HSV_V', 
        'LAB_L', 'LAB_A', 'LAB_B',
        'YUV_Y', 'YUV_U', 'YUV_V',
        'HLS_H', 'HLS_L', 'HLS_S'
    ]
    
    results = []
    
    print("Testing Color Spaces for Optimal Extraction")
    print("=" * 50)
    
    for color_space in color_spaces:
        result = test_color_space(color_space)
        if result:
            results.append(result)
    
    # Restore original config
    restore_config()
    
    # Sort results by success rate
    results.sort(key=lambda x: x['success_rate'], reverse=True)
    
    # Generate detailed report
    report_path = "test_results/COLOR_SPACE_ANALYSIS.md"
    with open(report_path, 'w') as f:
        f.write("# Color Space Analysis for Satoshi Poster Extraction\n\n")
        f.write("## Summary\n")
        f.write(f"Tested {len(results)} color spaces to find optimal extraction parameters.\n\n")
        
        f.write("## Results (Sorted by Success Rate)\n\n")
        f.write("| Rank | Color Space | Total Cells | Zeros | Ones | Blanks | Overlays | Success Rate | Binary Ratio |\n")
        f.write("|------|-------------|-------------|-------|------|--------|----------|--------------|-------------|\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"| {i} | {result['color_space']} | {result['total_cells']} | {result['zeros']} | {result['ones']} | {result['blanks']} | {result['overlays']} | {result['success_rate']:.1f}% | {result['binary_ratio']:.3f} |\n")
        
        f.write("\n## Analysis\n\n")
        
        if results:
            best = results[0]
            f.write(f"**Best Color Space**: {best['color_space']} with {best['success_rate']:.1f}% success rate\n")
            f.write(f"**Total Extractable Bits**: {best['zeros'] + best['ones']}\n")
            f.write(f"**Binary Distribution**: {best['zeros']} zeros ({best['zeros']/(best['zeros']+best['ones'])*100:.1f}%), {best['ones']} ones ({best['ones']/(best['zeros']+best['ones'])*100:.1f}%)\n\n")
            
            # Find color spaces with different characteristics
            high_success = [r for r in results if r['success_rate'] > 90]
            high_ones = [r for r in results if r['ones'] > 1000]
            low_blanks = [r for r in results if r['blanks'] < 100]
            
            f.write("### High Success Rate (>90%)\n")
            for r in high_success:
                f.write(f"- {r['color_space']}: {r['success_rate']:.1f}%\n")
            
            f.write("\n### High Ones Count (>1000)\n")
            for r in high_ones:
                f.write(f"- {r['color_space']}: {r['ones']} ones\n")
            
            f.write("\n### Low Blanks Count (<100)\n")
            for r in low_blanks:
                f.write(f"- {r['color_space']}: {r['blanks']} blanks\n")
    
    print(f"\nDetailed analysis saved to: {report_path}")
    print(f"Best color space: {results[0]['color_space']} ({results[0]['success_rate']:.1f}% success)")
    print(f"Best extractable bits: {results[0]['zeros'] + results[0]['ones']}")