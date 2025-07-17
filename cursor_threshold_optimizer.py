#!/usr/bin/env python3
"""
CURSOR AGENT: Threshold Optimization Tool
========================================

Optimizes color channel thresholds to fix the severe bit bias (93.5% ones vs 6.5% zeros).
Tests different thresholds and finds optimal balance.

Created by: Cursor Agent
Date: July 16, 2025
"""

import json
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict

class ThresholdOptimizer:
    def __init__(self):
        self.poster_image = self.load_poster_image()
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
            
    def test_blue_channel_thresholds(self, sample_size=200):
        """Test different blue channel thresholds"""
        print("=== BLUE CHANNEL THRESHOLD TESTING ===")
        
        if not self.poster_image:
            print("Cannot test thresholds - no poster image")
            return
            
        poster_array = np.array(self.poster_image)
        
        # Get sample cells
        sample_cells = [cell for cell in self.detailed_data['cells'] 
                       if cell['bit'] != 'ambiguous'][:sample_size]
        
        if not sample_cells:
            print("No cells to test")
            return
            
        # Test different thresholds
        thresholds = list(range(80, 220, 10))  # 80 to 210 in steps of 10
        results = []
        
        print("Threshold | Zeros | Ones | Total | % Zeros | % Ones | Ratio")
        print("-" * 65)
        
        for threshold in thresholds:
            zeros = 0
            ones = 0
            
            for cell in sample_cells:
                x, y = cell['global_x'], cell['global_y']
                
                try:
                    # Extract 7x7 region around the cell
                    region = poster_array[y-3:y+4, x-3:x+4]
                    blue_mean = np.mean(region[:, :, 2])  # Blue channel
                    
                    # Apply threshold
                    predicted_bit = 1 if blue_mean < threshold else 0
                    
                    if predicted_bit == 0:
                        zeros += 1
                    else:
                        ones += 1
                        
                except Exception as e:
                    continue
                    
            total = zeros + ones
            if total > 0:
                pct_zeros = zeros / total * 100
                pct_ones = ones / total * 100
                ratio = ones / zeros if zeros > 0 else float('inf')
                
                results.append({
                    'threshold': threshold,
                    'zeros': zeros,
                    'ones': ones,
                    'total': total,
                    'pct_zeros': pct_zeros,
                    'pct_ones': pct_ones,
                    'ratio': ratio
                })
                
                print(f"{threshold:9d} | {zeros:5d} | {ones:4d} | {total:5d} | "
                      f"{pct_zeros:7.1f}% | {pct_ones:6.1f}% | {ratio:5.1f}")
                      
        return results
        
    def test_color_ratios(self, sample_size=200):
        """Test different color ratios (B/G, B/R, etc.)"""
        print("\n=== COLOR RATIO TESTING ===")
        
        if not self.poster_image:
            print("Cannot test color ratios - no poster image")
            return
            
        poster_array = np.array(self.poster_image)
        
        # Get sample cells
        sample_cells = [cell for cell in self.detailed_data['cells'] 
                       if cell['bit'] != 'ambiguous'][:sample_size]
        
        if not sample_cells:
            print("No cells to test")
            return
            
        # Test different ratios
        ratios = ['B/G', 'B/R', 'G/R', 'B/(G+R)']
        thresholds = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        
        for ratio_type in ratios:
            print(f"\n{ratio_type} Ratio Testing:")
            print("Threshold | Zeros | Ones | Total | % Zeros | % Ones | Ratio")
            print("-" * 65)
            
            for threshold in thresholds:
                zeros = 0
                ones = 0
                
                for cell in sample_cells:
                    x, y = cell['global_x'], cell['global_y']
                    
                    try:
                        region = poster_array[y-3:y+4, x-3:x+4]
                        blue_mean = np.mean(region[:, :, 2])
                        green_mean = np.mean(region[:, :, 1])
                        red_mean = np.mean(region[:, :, 0])
                        
                        # Calculate ratio
                        if ratio_type == 'B/G':
                            ratio = blue_mean / green_mean if green_mean > 0 else 0
                        elif ratio_type == 'B/R':
                            ratio = blue_mean / red_mean if red_mean > 0 else 0
                        elif ratio_type == 'G/R':
                            ratio = green_mean / red_mean if red_mean > 0 else 0
                        elif ratio_type == 'B/(G+R)':
                            ratio = blue_mean / (green_mean + red_mean) if (green_mean + red_mean) > 0 else 0
                            
                        # Apply threshold
                        predicted_bit = 1 if ratio < threshold else 0
                        
                        if predicted_bit == 0:
                            zeros += 1
                        else:
                            ones += 1
                            
                    except Exception as e:
                        continue
                        
                total = zeros + ones
                if total > 0:
                    pct_zeros = zeros / total * 100
                    pct_ones = ones / total * 100
                    ratio_val = ones / zeros if zeros > 0 else float('inf')
                    
                    print(f"{threshold:9.1f} | {zeros:5d} | {ones:4d} | {total:5d} | "
                          f"{pct_zeros:7.1f}% | {pct_ones:6.1f}% | {ratio_val:5.1f}")
                          
    def find_optimal_threshold(self, results):
        """Find the optimal threshold that gives closest to 50/50 distribution"""
        if not results:
            return None
            
        # Find threshold closest to 50% zeros
        best_threshold = None
        best_diff = float('inf')
        
        for result in results:
            diff = abs(result['pct_zeros'] - 50.0)
            if diff < best_diff:
                best_diff = diff
                best_threshold = result
                
        return best_threshold
        
    def create_optimized_extraction(self, optimal_threshold):
        """Create optimized extraction using the best threshold"""
        print(f"\n=== CREATING OPTIMIZED EXTRACTION ===")
        print(f"Using threshold: {optimal_threshold['threshold']}")
        print(f"Expected distribution: {optimal_threshold['pct_zeros']:.1f}% zeros, {optimal_threshold['pct_ones']:.1f}% ones")
        
        if not self.poster_image:
            print("Cannot create optimized extraction - no poster image")
            return
            
        poster_array = np.array(self.poster_image)
        threshold = optimal_threshold['threshold']
        
        # Process all cells
        optimized_cells = []
        
        for cell in self.detailed_data['cells']:
            x, y = cell['global_x'], cell['global_y']
            
            try:
                region = poster_array[y-3:y+4, x-3:x+4]
                blue_mean = np.mean(region[:, :, 2])
                
                # Apply optimized threshold
                predicted_bit = 1 if blue_mean < threshold else 0
                
                # Calculate confidence based on distance from threshold
                distance = abs(blue_mean - threshold)
                confidence = max(50, 100 - distance)  # Higher confidence for clearer separation
                
                optimized_cells.append({
                    'region_id': cell['region_id'],
                    'local_row': cell['local_row'],
                    'local_col': cell['local_col'],
                    'global_x': x,
                    'global_y': y,
                    'bit': predicted_bit,
                    'confidence': confidence,
                    'blue_mean': blue_mean,
                    'threshold': threshold
                })
                
            except Exception as e:
                # Keep original classification for cells that can't be processed
                optimized_cells.append({
                    'region_id': cell['region_id'],
                    'local_row': cell['local_row'],
                    'local_col': cell['local_col'],
                    'global_x': x,
                    'global_y': y,
                    'bit': cell['bit'],
                    'confidence': cell['confidence'],
                    'blue_mean': None,
                    'threshold': threshold
                })
                
        # Save optimized results
        optimized_data = {'cells': optimized_cells}
        
        with open('cursor_optimized_extraction.json', 'w') as f:
            json.dump(optimized_data, f, indent=2)
            
        # Create CSV for easy analysis
        clear_cells = [cell for cell in optimized_cells if cell['bit'] != 'ambiguous']
        df = pd.DataFrame(clear_cells)
        df.to_csv('cursor_optimized_extraction.csv', index=False)
        
        # Analyze results
        bit_counts = df['bit'].value_counts()
        total_clear = len(clear_cells)
        
        print(f"\nOptimized extraction results:")
        print(f"Total clear cells: {total_clear}")
        print(f"Bit distribution: {bit_counts.to_dict()}")
        
        if 0 in bit_counts and 1 in bit_counts:
            zeros = bit_counts[0]
            ones = bit_counts[1]
            pct_zeros = zeros / total_clear * 100
            pct_ones = ones / total_clear * 100
            
            print(f"Zeros: {zeros} ({pct_zeros:.1f}%)")
            print(f"Ones: {ones} ({pct_ones:.1f}%)")
            print(f"Ratio (1s:0s): {ones/zeros:.1f}")
            
            if abs(pct_zeros - 50) < 10:
                print("✅ Good balance achieved!")
            else:
                print("⚠️ Still some bias, but improved")
                
        print(f"\nFiles saved:")
        print(f"  cursor_optimized_extraction.json")
        print(f"  cursor_optimized_extraction.csv")
        
    def run_optimization(self):
        """Run the complete threshold optimization"""
        print("="*60)
        print("CURSOR AGENT: THRESHOLD OPTIMIZATION")
        print("="*60)
        
        # Test blue channel thresholds
        blue_results = self.test_blue_channel_thresholds()
        
        # Test color ratios
        self.test_color_ratios()
        
        # Find optimal threshold
        if blue_results:
            optimal = self.find_optimal_threshold(blue_results)
            
            if optimal:
                print(f"\n=== OPTIMAL THRESHOLD FOUND ===")
                print(f"Threshold: {optimal['threshold']}")
                print(f"Distribution: {optimal['pct_zeros']:.1f}% zeros, {optimal['pct_ones']:.1f}% ones")
                print(f"Ratio: {optimal['ratio']:.1f}")
                
                # Create optimized extraction
                self.create_optimized_extraction(optimal)
            else:
                print("No optimal threshold found")
        else:
            print("No threshold results available")

def main():
    """Main optimization function"""
    optimizer = ThresholdOptimizer()
    optimizer.run_optimization()

if __name__ == "__main__":
    main() 