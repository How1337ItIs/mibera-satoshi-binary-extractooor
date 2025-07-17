#!/usr/bin/env python3
"""
CURSOR AGENT: Visual Validation Check
====================================

Validates that the optimized extraction results are rational and hold up
to visual inspection by comparing original vs optimized classifications.

Created by: Cursor Agent
Date: July 16, 2025
"""

import json
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

class VisualValidator:
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
            
    def create_detailed_comparison_grid(self, region_id=0, max_cells=25):
        """Create a detailed comparison grid showing original vs optimized"""
        print(f"\n=== DETAILED VISUAL COMPARISON FOR REGION {region_id} ===")
        
        if not self.poster_image:
            print("Cannot create comparison - no poster image")
            return
            
        # Get cells for this region
        orig_region = self.original_data[self.original_data['region_id'] == region_id][:max_cells]
        opt_region = self.optimized_data[self.optimized_data['region_id'] == region_id][:max_cells]
        
        if len(orig_region) == 0 or len(opt_region) == 0:
            print(f"No data for region {region_id}")
            return
            
        # Create comparison grid
        cell_size = 100
        cols = 5
        rows = 2  # Original vs Optimized
        
        grid_width = cols * cell_size
        grid_height = rows * cell_size
        
        grid_image = Image.new('RGB', (grid_width, grid_height), 'white')
        draw = ImageDraw.Draw(grid_image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 16)
            small_font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
            
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
                    int(row['global_x']) - 8, int(row['global_y']) - 8,
                    int(row['global_x']) + 9, int(row['global_y']) + 9
                ))
                cell_img = cell_img.resize((cell_size-4, cell_size-4))
                grid_image.paste(cell_img, (x+2, y+2))
                
                # Add bit label
                bit = int(row['bit'])
                color = 'green' if bit == 1 else 'red'
                draw.text((x+5, y+5), str(bit), fill=color, font=font)
                
                # Add confidence
                conf = row['confidence']
                draw.text((x+5, y+cell_size-25), f"conf:{conf:.0f}", fill='blue', font=small_font)
                
                # Add coordinates
                draw.text((x+5, y+cell_size-40), f"{int(row['global_x'])},{int(row['global_y'])}", 
                         fill='purple', font=small_font)
                
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
                    int(row['global_x']) - 8, int(row['global_y']) - 8,
                    int(row['global_x']) + 9, int(row['global_y']) + 9
                ))
                cell_img = cell_img.resize((cell_size-4, cell_size-4))
                grid_image.paste(cell_img, (x+2, y+2))
                
                # Add bit label
                bit = int(row['bit'])
                color = 'green' if bit == 1 else 'red'
                draw.text((x+5, y+5), str(bit), fill=color, font=font)
                
                # Add blue mean and threshold
                blue_mean = row['blue_mean']
                threshold = row['threshold']
                draw.text((x+5, y+cell_size-25), f"B:{blue_mean:.0f}", fill='blue', font=small_font)
                draw.text((x+5, y+cell_size-40), f"T:{threshold}", fill='purple', font=small_font)
                
            except Exception as e:
                draw.rectangle([x, y, x+cell_size, y+cell_size], outline='red')
                draw.text((x+5, y+5), "ERR", fill='red', font=font)
                
        # Save comparison
        filename = f"cursor_detailed_comparison_region_{region_id}.png"
        grid_image.save(filename)
        print(f"Detailed comparison saved: {filename}")
        
    def analyze_classification_changes(self):
        """Analyze how classifications changed between original and optimized"""
        print("\n=== CLASSIFICATION CHANGE ANALYSIS ===")
        
        # Find cells that exist in both datasets
        changes = []
        
        for _, opt_row in self.optimized_data.iterrows():
            # Find matching cell in original data
            matching = self.original_data[
                (self.original_data['region_id'] == opt_row['region_id']) &
                (self.original_data['local_row'] == opt_row['local_row']) &
                (self.original_data['local_col'] == opt_row['local_col'])
            ]
            
            if len(matching) > 0:
                orig_row = matching.iloc[0]
                orig_bit = int(orig_row['bit'])
                opt_bit = int(opt_row['bit'])
                
                if orig_bit != opt_bit:
                    changes.append({
                        'region_id': opt_row['region_id'],
                        'local_row': opt_row['local_row'],
                        'local_col': opt_row['local_col'],
                        'global_x': opt_row['global_x'],
                        'global_y': opt_row['global_y'],
                        'original_bit': orig_bit,
                        'optimized_bit': opt_bit,
                        'blue_mean': opt_row['blue_mean'],
                        'threshold': opt_row['threshold'],
                        'original_confidence': orig_row['confidence']
                    })
                    
        print(f"Total cells analyzed: {len(self.optimized_data)}")
        print(f"Cells that changed classification: {len(changes)}")
        print(f"Change rate: {len(changes)/len(self.optimized_data)*100:.1f}%")
        
        if changes:
            # Analyze changes
            changes_df = pd.DataFrame(changes)
            
            print(f"\nChange analysis:")
            print(f"  0→1 changes: {len(changes_df[changes_df['original_bit']==0])}")
            print(f"  1→0 changes: {len(changes_df[changes_df['original_bit']==1])}")
            
            # Show some examples of changes
            print(f"\nExample changes (first 10):")
            print("Region | Pos | Orig→Opt | Blue | Thresh | Orig Conf")
            print("-" * 55)
            
            for i, change in enumerate(changes[:10]):
                print(f"{change['region_id']:6d} | {change['local_row']:2d},{change['local_col']:2d} | "
                      f"{change['original_bit']}→{change['optimized_bit']} | {change['blue_mean']:5.0f} | "
                      f"{change['threshold']:6.0f} | {change['original_confidence']:8.1f}")
                      
        return changes
        
    def create_change_visualization(self, changes, max_examples=20):
        """Create visualization of cells that changed classification"""
        if not changes or not self.poster_image:
            return
            
        print(f"\n=== CREATING CHANGE VISUALIZATION ===")
        
        # Create grid of changed cells
        cell_size = 120
        cols = 5
        rows = (min(len(changes), max_examples) + cols - 1) // cols
        
        grid_width = cols * cell_size
        grid_height = rows * cell_size
        
        grid_image = Image.new('RGB', (grid_width, grid_height), 'white')
        draw = ImageDraw.Draw(grid_image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 14)
            small_font = ImageFont.truetype("arial.ttf", 10)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
            
        for i, change in enumerate(changes[:max_examples]):
            row = i // cols
            col = i % cols
            
            x = col * cell_size
            y = row * cell_size
            
            try:
                cell_img = self.poster_image.crop((
                    int(change['global_x']) - 8, int(change['global_y']) - 8,
                    int(change['global_x']) + 9, int(change['global_y']) + 9
                ))
                cell_img = cell_img.resize((cell_size-4, cell_size-4))
                grid_image.paste(cell_img, (x+2, y+2))
                
                # Add change label
                change_text = f"{change['original_bit']}→{change['optimized_bit']}"
                color = 'green' if change['optimized_bit'] == 0 else 'red'
                draw.text((x+5, y+5), change_text, fill=color, font=font)
                
                # Add blue mean
                draw.text((x+5, y+cell_size-35), f"B:{change['blue_mean']:.0f}", fill='blue', font=small_font)
                
                # Add threshold
                draw.text((x+5, y+cell_size-25), f"T:{change['threshold']}", fill='purple', font=small_font)
                
                # Add coordinates
                draw.text((x+5, y+cell_size-15), f"{int(change['global_x'])},{int(change['global_y'])}", 
                         fill='gray', font=small_font)
                
            except Exception as e:
                draw.rectangle([x, y, x+cell_size, y+cell_size], outline='red')
                draw.text((x+5, y+5), "ERR", fill='red', font=font)
                
        filename = "cursor_classification_changes.png"
        grid_image.save(filename)
        print(f"Change visualization saved: {filename}")
        
    def analyze_threshold_rationality(self):
        """Analyze if the threshold changes make sense"""
        print("\n=== THRESHOLD RATIONALITY ANALYSIS ===")
        
        # Analyze blue channel values
        blue_means = self.optimized_data['blue_mean'].dropna()
        
        print(f"Blue channel statistics:")
        print(f"  Mean: {blue_means.mean():.1f}")
        print(f"  Median: {blue_means.median():.1f}")
        print(f"  Std: {blue_means.std():.1f}")
        print(f"  Min: {blue_means.min():.1f}")
        print(f"  Max: {blue_means.max():.1f}")
        
        # Analyze by bit value
        zeros = self.optimized_data[self.optimized_data['bit'] == 0]['blue_mean']
        ones = self.optimized_data[self.optimized_data['bit'] == 1]['blue_mean']
        
        print(f"\nBlue channel by bit value:")
        print(f"  Zeros (blue mean): {zeros.mean():.1f} ± {zeros.std():.1f}")
        print(f"  Ones (blue mean): {ones.mean():.1f} ± {ones.std():.1f}")
        print(f"  Separation: {abs(zeros.mean() - ones.mean()):.1f}")
        
        # Check if threshold makes sense
        threshold = self.optimized_data['threshold'].iloc[0]  # Should be same for all
        print(f"\nThreshold analysis:")
        print(f"  Applied threshold: {threshold}")
        print(f"  Zeros above threshold: {len(zeros[zeros >= threshold])} ({len(zeros[zeros >= threshold])/len(zeros)*100:.1f}%)")
        print(f"  Ones below threshold: {len(ones[ones < threshold])} ({len(ones[ones < threshold])/len(ones)*100:.1f}%)")
        
        # Assess rationality
        if abs(zeros.mean() - ones.mean()) > 20:
            print("✅ GOOD SEPARATION: Blue channel clearly distinguishes bits")
        else:
            print("⚠️ POOR SEPARATION: Blue channel may not be discriminating well")
            
        if len(zeros[zeros >= threshold])/len(zeros) > 0.8:
            print("✅ THRESHOLD WORKS: Most zeros are above threshold")
        else:
            print("⚠️ THRESHOLD ISSUE: Many zeros below threshold")
            
        if len(ones[ones < threshold])/len(ones) > 0.8:
            print("✅ THRESHOLD WORKS: Most ones are below threshold")
        else:
            print("⚠️ THRESHOLD ISSUE: Many ones above threshold")
            
    def run_validation_check(self):
        """Run the complete visual validation check"""
        print("="*60)
        print("CURSOR AGENT: VISUAL VALIDATION CHECK")
        print("="*60)
        
        # Check basic statistics
        print("=== BASIC STATISTICS CHECK ===")
        print(f"Original data: {len(self.original_data)} cells")
        print(f"Optimized data: {len(self.optimized_data)} cells")
        
        orig_zeros = len(self.original_data[self.original_data['bit'] == 0.0])
        orig_ones = len(self.original_data[self.original_data['bit'] == 1.0])
        opt_zeros = len(self.optimized_data[self.optimized_data['bit'] == 0])
        opt_ones = len(self.optimized_data[self.optimized_data['bit'] == 1])
        
        print(f"Original: {orig_zeros} zeros, {orig_ones} ones")
        print(f"Optimized: {opt_zeros} zeros, {opt_ones} ones")
        print(f"Improvement: {opt_zeros - orig_zeros} more zeros")
        
        # Analyze changes
        changes = self.analyze_classification_changes()
        
        # Analyze threshold rationality
        self.analyze_threshold_rationality()
        
        # Create visual comparisons
        for region_id in [0, 2, 5]:  # Key regions
            self.create_detailed_comparison_grid(region_id)
            
        # Create change visualization
        if changes:
            self.create_change_visualization(changes)
            
        # Summary assessment
        print(f"\n=== VALIDATION ASSESSMENT ===")
        
        if opt_zeros > orig_zeros * 10:
            print("✅ MAJOR IMPROVEMENT: 10x+ more zeros extracted")
        elif opt_zeros > orig_zeros * 5:
            print("✅ SIGNIFICANT IMPROVEMENT: 5x+ more zeros extracted")
        else:
            print("⚠️ MINIMAL IMPROVEMENT: Less than 5x improvement")
            
        if len(changes) < len(self.optimized_data) * 0.5:
            print("✅ REASONABLE CHANGES: Less than 50% of cells changed")
        else:
            print("⚠️ MANY CHANGES: More than 50% of cells changed")
            
        print(f"\nValidation check complete. Review the generated visualizations:")
        print("  - cursor_detailed_comparison_region_*.png")
        print("  - cursor_classification_changes.png")

def main():
    """Main validation function"""
    validator = VisualValidator()
    validator.run_validation_check()

if __name__ == "__main__":
    main() 