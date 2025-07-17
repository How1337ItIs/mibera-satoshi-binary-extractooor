#!/usr/bin/env python3
"""
CURSOR AGENT: Bit Bias Investigation Tool
========================================

Investigates the severe bit bias (93.5% ones vs 6.5% zeros) to determine
if we're extracting actual digits or background artifacts.

Created by: Cursor Agent
Date: July 16, 2025
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
from collections import defaultdict

class BitBiasInvestigator:
    def __init__(self):
        self.poster_image = self.load_poster_image()
        self.binary_data = pd.read_csv("complete_extraction_binary_only.csv")
        self.detailed_data = self.load_detailed_data()
        
    def load_poster_image(self):
        """Load the original poster image"""
        try:
            return Image.open("satoshi (1).png")
        except:
            print("Warning: Could not load poster image")
            return None
            
    def load_detailed_data(self):
        """Load detailed extraction data"""
        with open("complete_extraction_detailed.json", 'r') as f:
            return json.load(f)
            
    def analyze_bit_distribution_by_region(self):
        """Analyze bit distribution across different regions"""
        print("=== BIT DISTRIBUTION BY REGION ===")
        
        region_bits = defaultdict(lambda: {'0': 0, '1': 0})
        
        for _, row in self.binary_data.iterrows():
            region_id = row['region_id']
            bit = str(int(row['bit']))  # Convert float to int to string
            region_bits[region_id][bit] += 1
            
        print("Region | Zeros | Ones | Ratio (1s:0s)")
        print("-" * 40)
        
        for region_id in sorted(region_bits.keys()):
            zeros = region_bits[region_id]['0']
            ones = region_bits[region_id]['1']
            total = zeros + ones
            
            if total > 0:
                ratio = ones / zeros if zeros > 0 else float('inf')
                print(f"{int(region_id):6d} | {zeros:5d} | {ones:4d} | {ratio:8.1f}")
                
        return region_bits
        
    def analyze_bit_distribution_by_confidence(self):
        """Analyze bit distribution across confidence levels"""
        print("\n=== BIT DISTRIBUTION BY CONFIDENCE ===")
        
        confidence_ranges = [
            (0, 60, "Low"),
            (60, 80, "Medium"),
            (80, 100, "High"),
            (100, 200, "Very High")
        ]
        
        for min_conf, max_conf, label in confidence_ranges:
            mask = (self.binary_data['confidence'] >= min_conf) & (self.binary_data['confidence'] < max_conf)
            subset = self.binary_data[mask]
            
            if len(subset) > 0:
                zeros = len(subset[subset['bit'] == 0.0])
                ones = len(subset[subset['bit'] == 1.0])
                total = zeros + ones
                
                print(f"{label:10} confidence ({min_conf}-{max_conf}): {zeros} zeros, {ones} ones, {ones/total*100:.1f}% ones")
                
    def create_bit_validation_grid(self, region_id=0, max_cells=50):
        """Create a detailed validation grid showing actual extracted cells"""
        print(f"\n=== BIT VALIDATION GRID FOR REGION {region_id} ===")
        
        if not self.poster_image:
            print("Cannot create validation grid - no poster image")
            return
            
        # Get cells for this region
        region_cells = [cell for cell in self.detailed_data['cells'] 
                       if cell['region_id'] == region_id and cell['bit'] != 'ambiguous'][:max_cells]
        
        if not region_cells:
            print(f"No clear cells found for region {region_id}")
            return
            
        # Create grid
        cols = 10
        rows = (len(region_cells) + cols - 1) // cols
        
        cell_size = 60
        grid_width = cols * cell_size
        grid_height = rows * cell_size
        
        grid_image = Image.new('RGB', (grid_width, grid_height), 'white')
        draw = ImageDraw.Draw(grid_image)
        
        # Add header
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
            
        draw.text((10, 10), f"Region {region_id} - Bit Validation Grid", fill='black', font=font)
        
        for i, cell in enumerate(region_cells):
            row = (i // cols) + 1  # Skip header row
            col = i % cols
            
            x = col * cell_size
            y = row * cell_size
            
            # Extract cell from poster
            try:
                cell_img = self.poster_image.crop((
                    cell['global_x'] - 5, cell['global_y'] - 5,
                    cell['global_x'] + 6, cell['global_y'] + 6
                ))
                cell_img = cell_img.resize((cell_size-4, cell_size-4))
                
                grid_image.paste(cell_img, (x+2, y+2))
                
                # Add bit label with color coding
                bit = cell['bit']
                color = 'green' if bit == '1' else 'red' if bit == '0' else 'gray'
                draw.text((x+5, y+5), str(bit), fill=color, font=font)
                
                # Add confidence
                conf = cell['confidence']
                draw.text((x+5, y+cell_size-20), f"{conf:.0f}", fill='blue', font=font)
                
                # Add coordinates
                draw.text((x+5, y+cell_size-35), f"{cell['global_x']},{cell['global_y']}", 
                         fill='purple', font=font)
                
            except Exception as e:
                draw.rectangle([x, y, x+cell_size, y+cell_size], outline='red')
                draw.text((x+5, y+5), "ERR", fill='red', font=font)
                
        # Save grid
        filename = f"cursor_bit_validation_region_{region_id}.png"
        grid_image.save(filename)
        print(f"Bit validation grid saved: {filename}")
        
    def analyze_color_channels(self, sample_cells=20):
        """Analyze color channel values for extracted cells"""
        print(f"\n=== COLOR CHANNEL ANALYSIS (Sample: {sample_cells} cells) ===")
        
        if not self.poster_image:
            print("Cannot analyze color channels - no poster image")
            return
            
        # Convert to numpy array for analysis
        poster_array = np.array(self.poster_image)
        
        # Sample cells from different regions
        sample_data = []
        
        for region_id in [0, 2, 5, 10, 25]:  # Mix of good and bad regions
            region_cells = [cell for cell in self.detailed_data['cells'] 
                           if cell['region_id'] == region_id and cell['bit'] != 'ambiguous'][:4]
            
            for cell in region_cells:
                x, y = cell['global_x'], cell['global_y']
                
                try:
                    # Extract 7x7 region around the cell
                    region = poster_array[y-3:y+4, x-3:x+4]
                    
                    # Calculate channel statistics
                    blue_mean = np.mean(region[:, :, 2])  # Blue channel
                    green_mean = np.mean(region[:, :, 1])  # Green channel
                    red_mean = np.mean(region[:, :, 0])    # Red channel
                    
                    sample_data.append({
                        'region_id': region_id,
                        'bit': cell['bit'],
                        'confidence': cell['confidence'],
                        'blue_mean': blue_mean,
                        'green_mean': green_mean,
                        'red_mean': red_mean,
                        'x': x,
                        'y': y
                    })
                    
                except Exception as e:
                    continue
                    
        # Analyze by bit value
        if sample_data:
            df = pd.DataFrame(sample_data)
            
            print("\nColor channel statistics by bit value:")
            print("Bit | Count | Blue Mean | Green Mean | Red Mean")
            print("-" * 50)
            
            for bit in [0.0, 1.0]:
                subset = df[df['bit'] == bit]
                if len(subset) > 0:
                    print(f"{int(bit):3d} | {len(subset):5d} | {subset['blue_mean'].mean():9.1f} | "
                          f"{subset['green_mean'].mean():10.1f} | {subset['red_mean'].mean():8.1f}")
                          
            # Check if blue channel is actually discriminating
            zeros = df[df['bit'] == 0.0]['blue_mean']
            ones = df[df['bit'] == 1.0]['blue_mean']
            
            if len(zeros) > 0 and len(ones) > 0:
                print(f"\nBlue channel discrimination:")
                print(f"  Zeros (blue mean): {zeros.mean():.1f} ± {zeros.std():.1f}")
                print(f"  Ones (blue mean): {ones.mean():.1f} ± {ones.std():.1f}")
                print(f"  Separation: {abs(ones.mean() - zeros.mean()):.1f}")
                
        return sample_data
        
    def test_alternative_thresholds(self):
        """Test alternative classification thresholds"""
        print("\n=== ALTERNATIVE THRESHOLD TESTING ===")
        
        if not self.poster_image:
            print("Cannot test thresholds - no poster image")
            return
            
        poster_array = np.array(self.poster_image)
        
        # Test different blue channel thresholds
        thresholds = [100, 120, 140, 160, 180, 200]
        
        for threshold in thresholds:
            zeros = 0
            ones = 0
            
            # Test on a sample of cells
            for cell in self.detailed_data['cells'][:100]:  # Sample first 100 cells
                if cell['bit'] != 'ambiguous':
                    x, y = cell['global_x'], cell['global_y']
                    
                    try:
                        region = poster_array[y-3:y+4, x-3:x+4]
                        blue_mean = np.mean(region[:, :, 2])
                        
                        # Apply threshold
                        predicted_bit = 1 if blue_mean < threshold else 0
                        
                        if predicted_bit == 0:
                            zeros += 1
                        else:
                            ones += 1
                            
                    except:
                        continue
                        
            total = zeros + ones
            if total > 0:
                ratio = ones / zeros if zeros > 0 else float('inf')
                print(f"Threshold {threshold:3d}: {zeros:3d} zeros, {ones:3d} ones, "
                      f"ratio {ratio:5.1f}, {ones/total*100:5.1f}% ones")
                      
    def create_comparison_visualization(self):
        """Create visualization comparing zeros vs ones"""
        print("\n=== CREATING ZERO vs ONE COMPARISON ===")
        
        if not self.poster_image:
            print("Cannot create comparison - no poster image")
            return
            
        # Find some zeros and ones
        zeros = [cell for cell in self.detailed_data['cells'] 
                if cell['bit'] == 0.0 and cell['bit'] != 'ambiguous'][:10]
        ones = [cell for cell in self.detailed_data['cells'] 
               if cell['bit'] == 1.0 and cell['bit'] != 'ambiguous'][:10]
        
        if not zeros or not ones:
            print("Not enough zeros and ones for comparison")
            return
            
        # Create comparison grid
        cell_size = 80
        grid_width = 20 * cell_size
        grid_height = cell_size
        
        grid_image = Image.new('RGB', (grid_width, grid_height), 'white')
        draw = ImageDraw.Draw(grid_image)
        
        # Add zeros
        for i, cell in enumerate(zeros):
            x = i * cell_size
            try:
                cell_img = self.poster_image.crop((
                    cell['global_x'] - 5, cell['global_y'] - 5,
                    cell['global_x'] + 6, cell['global_y'] + 6
                ))
                cell_img = cell_img.resize((cell_size-4, cell_size-4))
                grid_image.paste(cell_img, (x+2, 2))
                draw.text((x+5, 5), "0", fill='red')
            except:
                draw.rectangle([x, 0, x+cell_size, cell_size], outline='red')
                
        # Add ones
        for i, cell in enumerate(ones):
            x = (i + 10) * cell_size
            try:
                cell_img = self.poster_image.crop((
                    cell['global_x'] - 5, cell['global_y'] - 5,
                    cell['global_x'] + 6, cell['global_y'] + 6
                ))
                cell_img = cell_img.resize((cell_size-4, cell_size-4))
                grid_image.paste(cell_img, (x+2, 2))
                draw.text((x+5, 5), "1", fill='green')
            except:
                draw.rectangle([x, 0, x+cell_size, cell_size], outline='red')
                
        # Add labels
        draw.text((10, cell_size-20), "ZEROS", fill='red')
        draw.text((10*cell_size+10, cell_size-20), "ONES", fill='green')
        
        filename = "cursor_zero_vs_one_comparison.png"
        grid_image.save(filename)
        print(f"Zero vs One comparison saved: {filename}")
        
    def run_complete_investigation(self):
        """Run the complete bit bias investigation"""
        print("="*60)
        print("CURSOR AGENT: BIT BIAS INVESTIGATION")
        print("="*60)
        
        # Run all analyses
        region_bits = self.analyze_bit_distribution_by_region()
        self.analyze_bit_distribution_by_confidence()
        color_data = self.analyze_color_channels()
        self.test_alternative_thresholds()
        
        # Create visualizations
        for region_id in [0, 2, 5]:  # Good, good, bad regions
            self.create_bit_validation_grid(region_id)
            
        self.create_comparison_visualization()
        
        # Summary
        print(f"\n=== INVESTIGATION SUMMARY ===")
        total_zeros = sum(r['0'] for r in region_bits.values())
        total_ones = sum(r['1'] for r in region_bits.values())
        total_bits = total_zeros + total_ones
        
        print(f"Total zeros: {total_zeros}")
        print(f"Total ones: {total_ones}")
        print(f"Zero percentage: {total_zeros/total_bits*100:.1f}%")
        print(f"One percentage: {total_ones/total_bits*100:.1f}%")
        print(f"Ratio (1s:0s): {total_ones/total_zeros:.1f}")
        
        if total_zeros/total_bits < 0.1:
            print("\n🚨 CRITICAL ISSUE: Severe bit bias detected!")
            print("   - Less than 10% zeros suggests classification error")
            print("   - May be extracting background instead of digits")
            print("   - Need to investigate color channel thresholds")
        else:
            print("\n✅ Bit distribution appears reasonable")

def main():
    """Main investigation function"""
    investigator = BitBiasInvestigator()
    investigator.run_complete_investigation()

if __name__ == "__main__":
    main() 