#!/usr/bin/env python3
"""
Systematic testing of all extraction methods for maximum bit recovery.
"""

import os
import shutil
import yaml
import csv
from pathlib import Path

def backup_config():
    """Backup the original config."""
    config_path = "binary_extractor/extractor/cfg.yaml"
    backup_path = "binary_extractor/extractor/cfg.yaml.original"
    
    if not os.path.exists(backup_path):
        shutil.copy2(config_path, backup_path)
        print("Config backed up")

def restore_config():
    """Restore the original config."""
    config_path = "binary_extractor/extractor/cfg.yaml"
    backup_path = "binary_extractor/extractor/cfg.yaml.original"
    
    if os.path.exists(backup_path):
        shutil.copy2(backup_path, config_path)
        print("Config restored")

def run_test(test_name, config_changes, output_dir):
    """Run a single test with given config changes."""
    print(f"\n=== Testing {test_name} ===")
    
    # Read base config
    config_path = "binary_extractor/extractor/cfg.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply changes
    for key, value in config_changes.items():
        if '.' in key:
            # Handle nested keys like threshold.method
            parts = key.split('.')
            current = config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            config[key] = value
    
    # Write modified config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run extraction
    result = os.system(f'cd binary_extractor && python scripts/extract.py "../satoshi (1).png" "../{output_dir}/" >nul 2>&1')
    
    # Analyze results
    cells_file = f"{output_dir}/cells.csv"
    if os.path.exists(cells_file):
        with open(cells_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        total_cells = len(rows)
        zeros = sum(1 for row in rows if row['bit'] == '0')
        ones = sum(1 for row in rows if row['bit'] == '1')
        blanks = sum(1 for row in rows if row['bit'] == 'blank')
        overlays = sum(1 for row in rows if row['bit'] == 'overlay')
        
        success_rate = (zeros + ones) / total_cells * 100 if total_cells > 0 else 0
        
        print(f"SUCCESS: {success_rate:.1f}% ({zeros} zeros, {ones} ones, {blanks} blanks, {overlays} overlays)")
        
        return {
            'test_name': test_name,
            'total_cells': total_cells,
            'zeros': zeros,
            'ones': ones,
            'blanks': blanks,
            'overlays': overlays,
            'success_rate': success_rate,
            'config_changes': config_changes
        }
    else:
        print(f"FAILED: No results generated")
        return None

def main():
    """Main testing function."""
    backup_config()
    
    # Test configurations
    tests = [
        # Color space tests
        ("RGB_R", {"use_color_space": "RGB_R"}),
        ("RGB_G", {"use_color_space": "RGB_G"}),
        ("RGB_B", {"use_color_space": "RGB_B"}),
        ("HSV_H", {"use_color_space": "HSV_H"}),
        ("HSV_S", {"use_color_space": "HSV_S"}),
        ("HSV_V", {"use_color_space": "HSV_V"}),
        ("LAB_L", {"use_color_space": "LAB_L"}),
        ("LAB_A", {"use_color_space": "LAB_A"}),
        ("LAB_B", {"use_color_space": "LAB_B"}),
        ("YUV_Y", {"use_color_space": "YUV_Y"}),
        ("YUV_U", {"use_color_space": "YUV_U"}),
        ("YUV_V", {"use_color_space": "YUV_V"}),
        ("HLS_H", {"use_color_space": "HLS_H"}),
        ("HLS_L", {"use_color_space": "HLS_L"}),
        ("HLS_S", {"use_color_space": "HLS_S"}),
        
        # Threshold method tests
        ("HSV_S_Otsu", {"use_color_space": "HSV_S", "threshold.method": "otsu"}),
        ("HSV_S_Adaptive", {"use_color_space": "HSV_S", "threshold.method": "adaptive"}),
        ("HSV_S_Sauvola", {"use_color_space": "HSV_S", "threshold.method": "sauvola"}),
        
        # Morphology tests
        ("HSV_S_NoMorph", {"use_color_space": "HSV_S", "morph_k": 1, "morph_iterations": 0}),
        ("HSV_S_HeavyMorph", {"use_color_space": "HSV_S", "morph_k": 5, "morph_iterations": 3}),
        
        # Bit threshold tests
        ("HSV_S_Sensitive", {"use_color_space": "HSV_S", "bit_hi": 0.6, "bit_lo": 0.4}),
        ("HSV_S_Conservative", {"use_color_space": "HSV_S", "bit_hi": 0.8, "bit_lo": 0.2}),
        
        # Template matching tests
        ("HSV_S_NoTemplate", {"use_color_space": "HSV_S", "template_match": False}),
        ("HSV_S_LowTemplate", {"use_color_space": "HSV_S", "template_match": True, "tm_thresh": 0.3}),
        ("HSV_S_HighTemplate", {"use_color_space": "HSV_S", "template_match": True, "tm_thresh": 0.6}),
        
        # Combined optimal tests
        ("LAB_B_Adaptive", {"use_color_space": "LAB_B", "threshold.method": "adaptive", "template_match": True}),
        ("YUV_Y_Otsu", {"use_color_space": "YUV_Y", "threshold.method": "otsu", "template_match": True}),
        ("RGB_G_Sauvola", {"use_color_space": "RGB_G", "threshold.method": "sauvola", "template_match": True}),
    ]
    
    results = []
    
    for test_name, config_changes in tests:
        output_dir = f"test_results/method_{test_name.lower()}"
        result = run_test(test_name, config_changes, output_dir)
        if result:
            results.append(result)
    
    # Restore original config
    restore_config()
    
    # Sort results by success rate
    results.sort(key=lambda x: x['success_rate'], reverse=True)
    
    # Generate comprehensive report
    report_path = "test_results/COMPREHENSIVE_METHOD_ANALYSIS.md"
    with open(report_path, 'w') as f:
        f.write("# Comprehensive Method Analysis for Satoshi Poster Extraction\n\n")
        f.write("## Overview\n")
        f.write(f"Tested {len(results)} different extraction methods to maximize bit recovery.\n\n")
        
        f.write("## Results Summary (Top 10)\n\n")
        f.write("| Rank | Method | Success Rate | Zeros | Ones | Blanks | Overlays | Total Bits |\n")
        f.write("|------|--------|--------------|-------|------|--------|----------|------------|\n")
        
        for i, result in enumerate(results[:10], 1):
            f.write(f"| {i} | {result['test_name']} | {result['success_rate']:.1f}% | {result['zeros']} | {result['ones']} | {result['blanks']} | {result['overlays']} | {result['zeros'] + result['ones']} |\n")
        
        f.write("\n## Detailed Analysis\n\n")
        
        if results:
            best = results[0]
            f.write(f"### Best Method: {best['test_name']}\n")
            f.write(f"- **Success Rate**: {best['success_rate']:.1f}%\n")
            f.write(f"- **Total Extractable Bits**: {best['zeros'] + best['ones']}\n")
            f.write(f"- **Configuration**: {best['config_changes']}\n\n")
            
            # Category analysis
            f.write("### Analysis by Category\n\n")
            
            # Color space analysis
            color_space_results = [r for r in results if r['test_name'] in ['RGB_R', 'RGB_G', 'RGB_B', 'HSV_H', 'HSV_S', 'HSV_V', 'LAB_L', 'LAB_A', 'LAB_B', 'YUV_Y', 'YUV_U', 'YUV_V', 'HLS_H', 'HLS_L', 'HLS_S']]
            if color_space_results:
                color_space_results.sort(key=lambda x: x['success_rate'], reverse=True)
                f.write("#### Best Color Spaces\n")
                for r in color_space_results[:5]:
                    f.write(f"- {r['test_name']}: {r['success_rate']:.1f}% ({r['zeros'] + r['ones']} bits)\n")
                f.write("\n")
            
            # Threshold method analysis
            threshold_results = [r for r in results if 'Otsu' in r['test_name'] or 'Adaptive' in r['test_name'] or 'Sauvola' in r['test_name']]
            if threshold_results:
                threshold_results.sort(key=lambda x: x['success_rate'], reverse=True)
                f.write("#### Best Threshold Methods\n")
                for r in threshold_results:
                    f.write(f"- {r['test_name']}: {r['success_rate']:.1f}% ({r['zeros'] + r['ones']} bits)\n")
                f.write("\n")
        
        f.write("## Complete Results\n\n")
        f.write("| Method | Success Rate | Zeros | Ones | Blanks | Overlays | Configuration |\n")
        f.write("|--------|--------------|-------|------|--------|----------|---------------|\n")
        
        for result in results:
            config_str = str(result['config_changes']).replace('|', '\\|')
            f.write(f"| {result['test_name']} | {result['success_rate']:.1f}% | {result['zeros']} | {result['ones']} | {result['blanks']} | {result['overlays']} | {config_str} |\n")
    
    print(f"\n=== TESTING COMPLETE ===")
    print(f"Tested {len(results)} methods")
    print(f"Best method: {results[0]['test_name']} ({results[0]['success_rate']:.1f}% success)")
    print(f"Best bit count: {results[0]['zeros'] + results[0]['ones']} bits")
    print(f"Report saved to: {report_path}")

if __name__ == "__main__":
    main()