#!/usr/bin/env python3
"""
CURSOR AGENT: Optimization Validation Tool
=========================================

Validates the optimized extraction results and compares them with the original
biased results to ensure we've improved the bit distribution.

Created by: Cursor Agent
Date: July 16, 2025
"""

import json
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

class OptimizationValidator:
    def __init__(self):
        self.poster_image = self.load_poster_image()
        self.original_data = pd.read_csv("complete_extraction_binary_only.csv")
        self.optimized_data = pd.read_csv("cursor_optimized_extraction.csv")
        
    def load_poster_image(self):
        """Load the original poster image"""
        try:
            return Image.open("satoshi (1).png")
        except:
            print("Warning: Could not load poster image")
            return None
            
    def compare_distributions(self):
        """Compare original vs optimized bit distributions"""
        print("=== DISTRIBUTION COMPARISON ===")
        
        # Original distribution
        orig_counts = self.original_data['bit'].value_counts()
        orig_total = len(self.original_data)
        orig_zeros = orig_counts.get(0.0, 0)
        orig_ones = orig_counts.get(1.0, 0)
        
        # Optimized distribution
        opt_counts = self.optimized_data['bit'].value_counts()
        opt_total = len(self.optimized_data)
        opt_zeros = opt_counts.get(0, 0)
        opt_ones = opt_counts.get(1, 0)
        
        print("Metric          | Original | Optimized | Improvement")
        print("-" * 55)
        print(f"Total cells     | {orig_total:8d} | {opt_total:9d} | {opt_total-orig_total:+d}")
        print(f"Zeros           | {orig_zeros:8d} | {opt_zeros:9d} | {opt_zeros-orig_zeros:+d}")
        print(f"Ones            | {orig_ones:8d} | {opt_ones:9d} | {opt_ones-orig_ones:+d}")
        print(f"% Zeros         | {orig_zeros/orig_total*100:7.1f}% | {opt_zeros/opt_total*100:8.1f}% | {(opt_zeros/opt_total-orig_zeros/orig_total)*100:+.1f}%")
        print(f"% Ones          | {orig_ones/orig_total*100:7.1f}% | {opt_ones/opt_total*100:8.1f}% | {(opt_ones/opt_total-orig_ones/orig_total)*100:+.1f}%")
        
        if orig_zeros > 0:
            orig_ratio = orig_ones / orig_zeros
        else:
            orig_ratio = float('inf')
            
        if opt_zeros > 0:
            opt_ratio = opt_ones / opt_zeros
        else:
            opt_ratio = float('inf')
            
        print(f"Ratio (1s:0s)   | {orig_ratio:8.1f} | {opt_ratio:9.1f} | {opt_ratio-orig_ratio:+.1f}")
        
        # Improvement assessment
        print(f"\n=== IMPROVEMENT ASSESSMENT ===")
        
        if opt_zeros/orig_zeros > 10:
            print("✅ MAJOR IMPROVEMENT: 10x+ more zeros extracted")
        elif opt_zeros/orig_zeros > 5:
            print("✅ SIGNIFICANT IMPROVEMENT: 5x+ more zeros extracted")
        elif opt_zeros/orig_zeros > 2:
            print("✅ MODERATE IMPROVEMENT: 2x+ more zeros extracted")
        else:
            print("⚠️ MINIMAL IMPROVEMENT: Less than 2x improvement")
            
        if abs(opt_zeros/opt_total - 0.5) < 0.1:
            print("✅ EXCELLENT BALANCE: Close to 50/50 distribution")
        elif abs(opt_zeros/opt_total - 0.5) < 0.2:
            print("✅ GOOD BALANCE: Reasonable distribution")
        else:
            print("⚠️ STILL BIASED: Distribution needs further tuning")
            
    def create_comparison_visualization(self, region_id=0, max_cells=20):
        """Create visual comparison of original vs optimized extraction"""
        print(f"\n=== VISUAL COMPARISON FOR REGION {region_id} ===")
        
        if not self.poster_image:
            print("Cannot create visualization - no poster image")
            return
            
        # Get cells for this region
        orig_region = self.original_data[self.original_data['region_id'] == region_id][:max_cells]
        opt_region = self.optimized_data[self.optimized_data['region_id'] == region_id][:max_cells]
        
        if len(orig_region) == 0 or len(opt_region) == 0:
            print(f"No data for region {region_id}")
            return
            
        # Create comparison grid
        cell_size = 80
        cols = 10
        rows = 2  # Original vs Optimized
        
        grid_width = cols * cell_size
        grid_height = rows * cell_size
        
        grid_image = Image.new('RGB', (grid_width, grid_height), 'white')
        draw = ImageDraw.Draw(grid_image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()
            
        # Add headers
        draw.text((10, 10), "ORIGINAL EXTRACTION", fill='red', font=font)
        draw.text((10, cell_size + 10), "OPTIMIZED EXTRACTION", fill='green', font=font)
        
        # Original extraction row
        for i, (_, row) in enumerate(orig_region.iterrows()):
            if i >= cols:
                break
                
            x = i * cell_size
            y = 0
            
            try:
                cell_img = self.poster_image.crop((
                    int(row['global_x']) - 5, int(row['global_y']) - 5,
                    int(row['global_x']) + 6, int(row['global_y']) + 6
                ))
                cell_img = cell_img.resize((cell_size-4, cell_size-4))
                grid_image.paste(cell_img, (x+2, y+2))
                
                # Add bit label
                bit = int(row['bit'])
                color = 'green' if bit == 1 else 'red'
                draw.text((x+5, y+5), str(bit), fill=color, font=font)
                
            except Exception as e:
                draw.rectangle([x, y, x+cell_size, y+cell_size], outline='red')
                draw.text((x+5, y+5), "ERR", fill='red', font=font)
                
        # Optimized extraction row
        for i, (_, row) in enumerate(opt_region.iterrows()):
            if i >= cols:
                break
                
            x = i * cell_size
            y = cell_size
            
            try:
                cell_img = self.poster_image.crop((
                    int(row['global_x']) - 5, int(row['global_y']) - 5,
                    int(row['global_x']) + 6, int(row['global_y']) + 6
                ))
                cell_img = cell_img.resize((cell_size-4, cell_size-4))
                grid_image.paste(cell_img, (x+2, y+2))
                
                # Add bit label
                bit = int(row['bit'])
                color = 'green' if bit == 1 else 'red'
                draw.text((x+5, y+5), str(bit), fill=color, font=font)
                
            except Exception as e:
                draw.rectangle([x, y, x+cell_size, y+cell_size], outline='red')
                draw.text((x+5, y+5), "ERR", fill='red', font=font)
                
        # Save comparison
        filename = f"cursor_optimization_comparison_region_{region_id}.png"
        grid_image.save(filename)
        print(f"Comparison visualization saved: {filename}")
        
    def analyze_region_improvements(self):
        """Analyze improvements by region"""
        print("\n=== REGION IMPROVEMENT ANALYSIS ===")
        
        # Group by region
        orig_by_region = self.original_data.groupby('region_id')['bit'].value_counts().unstack(fill_value=0)
        opt_by_region = self.optimized_data.groupby('region_id')['bit'].value_counts().unstack(fill_value=0)
        
        print("Region | Orig Zeros | Orig Ones | Opt Zeros | Opt Ones | Improvement")
        print("-" * 70)
        
        for region_id in sorted(set(orig_by_region.index) | set(opt_by_region.index)):
            orig_zeros = orig_by_region.loc[region_id, 0.0] if region_id in orig_by_region.index and 0.0 in orig_by_region.columns else 0
            orig_ones = orig_by_region.loc[region_id, 1.0] if region_id in orig_by_region.index and 1.0 in orig_by_region.columns else 0
            opt_zeros = opt_by_region.loc[region_id, 0] if region_id in opt_by_region.index and 0 in opt_by_region.columns else 0
            opt_ones = opt_by_region.loc[region_id, 1] if region_id in opt_by_region.index and 1 in opt_by_region.columns else 0
            
            improvement = opt_zeros - orig_zeros
            print(f"{region_id:6d} | {orig_zeros:10d} | {orig_ones:9d} | {opt_zeros:9d} | {opt_ones:8d} | {improvement:+d}")
            
    def create_summary_report(self):
        """Create a summary report of the optimization"""
        print("\n=== OPTIMIZATION SUMMARY REPORT ===")
        
        # Calculate key metrics
        orig_zeros = len(self.original_data[self.original_data['bit'] == 0.0])
        orig_ones = len(self.original_data[self.original_data['bit'] == 1.0])
        opt_zeros = len(self.optimized_data[self.optimized_data['bit'] == 0])
        opt_ones = len(self.optimized_data[self.optimized_data['bit'] == 1])
        
        improvement_factor = opt_zeros / orig_zeros if orig_zeros > 0 else float('inf')
        
        print(f"Original extraction: {orig_zeros} zeros, {orig_ones} ones")
        print(f"Optimized extraction: {opt_zeros} zeros, {opt_ones} ones")
        print(f"Improvement factor: {improvement_factor:.1f}x more zeros")
        
        if improvement_factor > 10:
            print("🎉 EXCELLENT: Major improvement achieved!")
        elif improvement_factor > 5:
            print("✅ GOOD: Significant improvement achieved!")
        elif improvement_factor > 2:
            print("👍 MODERATE: Some improvement achieved")
        else:
            print("⚠️ MINIMAL: Little improvement achieved")
            
        # Save summary
        summary = {
            'original': {
                'zeros': int(orig_zeros),
                'ones': int(orig_ones),
                'total': len(self.original_data),
                'zero_percentage': float(orig_zeros / len(self.original_data) * 100)
            },
            'optimized': {
                'zeros': int(opt_zeros),
                'ones': int(opt_ones),
                'total': len(self.optimized_data),
                'zero_percentage': float(opt_zeros / len(self.optimized_data) * 100)
            },
            'improvement': {
                'factor': float(improvement_factor),
                'additional_zeros': int(opt_zeros - orig_zeros),
                'percentage_change': float((opt_zeros / len(self.optimized_data) - orig_zeros / len(self.original_data)) * 100)
            }
        }
        
        with open('cursor_optimization_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"\nSummary saved: cursor_optimization_summary.json")
        
    def run_validation(self):
        """Run the complete validation"""
        print("="*60)
        print("CURSOR AGENT: OPTIMIZATION VALIDATION")
        print("="*60)
        
        # Run all validations
        self.compare_distributions()
        self.analyze_region_improvements()
        
        # Create visualizations for key regions
        for region_id in [0, 2, 5]:  # Good, good, bad regions
            self.create_comparison_visualization(region_id)
            
        self.create_summary_report()
        
        print(f"\n=== VALIDATION COMPLETE ===")
        print("Check the generated files for detailed analysis:")

def main():
    """Main validation function"""
    validator = OptimizationValidator()
    validator.run_validation()

if __name__ == "__main__":
    main() 