#!/usr/bin/env python3
"""
CURSOR AGENT: Complete Extraction Analysis & Optimization
========================================================

Visual validation and analysis of the complete poster extraction results.
Identifies patterns, validates accuracy, and suggests optimizations.

Created by: Cursor Agent
Date: July 16, 2025
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import os
from collections import defaultdict, Counter
import cv2

class ExtractionAnalyzer:
    def __init__(self, detailed_json_path="complete_extraction_detailed.json", 
                 binary_csv_path="complete_extraction_binary_only.csv"):
        self.detailed_data = self.load_detailed_data(detailed_json_path)
        self.binary_data = self.load_binary_data(binary_csv_path)
        self.poster_image = self.load_poster_image()
        
    def load_detailed_data(self, path):
        """Load the detailed extraction JSON data"""
        with open(path, 'r') as f:
            return json.load(f)
            
    def load_binary_data(self, path):
        """Load the binary-only CSV data"""
        return pd.read_csv(path)
        
    def load_poster_image(self):
        """Load the original poster image"""
        try:
            return Image.open("satoshi (1).png")
        except:
            print("Warning: Could not load poster image")
            return None
            
    def analyze_extraction_statistics(self):
        """Analyze overall extraction statistics"""
        print("=== EXTRACTION STATISTICS ANALYSIS ===")
        
        # Overall stats
        total_cells = len(self.detailed_data['cells'])
        clear_cells = len(self.binary_data)
        ambiguous_cells = total_cells - clear_cells
        
        print(f"Total cells processed: {total_cells}")
        print(f"Clear classifications: {clear_cells} ({clear_cells/total_cells*100:.1f}%)")
        print(f"Ambiguous cells: {ambiguous_cells} ({ambiguous_cells/total_cells*100:.1f}%)")
        
        # Bit distribution
        bit_counts = self.binary_data['bit'].value_counts()
        print(f"\nBit distribution:")
        print(f"  Zeros: {bit_counts.get('0', 0)}")
        print(f"  Ones: {bit_counts.get('1', 0)}")
        
        # Confidence analysis
        confidences = self.binary_data['confidence']
        print(f"\nConfidence statistics:")
        print(f"  Mean: {confidences.mean():.1f}")
        print(f"  Median: {confidences.median():.1f}")
        print(f"  Std: {confidences.std():.1f}")
        print(f"  Min: {confidences.min():.1f}")
        print(f"  Max: {confidences.max():.1f}")
        
        return {
            'total_cells': total_cells,
            'clear_cells': clear_cells,
            'ambiguous_cells': ambiguous_cells,
            'clarity_rate': clear_cells/total_cells*100,
            'bit_distribution': bit_counts.to_dict(),
            'confidence_stats': {
                'mean': confidences.mean(),
                'median': confidences.median(),
                'std': confidences.std(),
                'min': confidences.min(),
                'max': confidences.max()
            }
        }
        
    def analyze_region_performance(self):
        """Analyze performance by region"""
        print("\n=== REGION PERFORMANCE ANALYSIS ===")
        
        region_stats = defaultdict(lambda: {'clear': 0, 'ambiguous': 0, 'total': 0})
        
        for cell in self.detailed_data['cells']:
            region_id = cell['region_id']
            region_stats[region_id]['total'] += 1
            if cell['bit'] != 'ambiguous':
                region_stats[region_id]['clear'] += 1
            else:
                region_stats[region_id]['ambiguous'] += 1
                
        # Calculate clarity rates
        region_clarity = {}
        for region_id, stats in region_stats.items():
            if stats['total'] > 0:
                clarity = stats['clear'] / stats['total'] * 100
                region_clarity[region_id] = clarity
                print(f"Region {region_id}: {stats['clear']}/{stats['total']} = {clarity:.1f}%")
                
        # Find best and worst regions
        if region_clarity:
            best_region = max(region_clarity.items(), key=lambda x: x[1])
            worst_region = min(region_clarity.items(), key=lambda x: x[1])
            print(f"\nBest region: {best_region[0]} ({best_region[1]:.1f}%)")
            print(f"Worst region: {worst_region[0]} ({worst_region[1]:.1f}%)")
            
        return region_clarity
        
    def analyze_confidence_patterns(self):
        """Analyze confidence patterns and thresholds"""
        print("\n=== CONFIDENCE PATTERN ANALYSIS ===")
        
        # Confidence distribution
        confidences = self.binary_data['confidence']
        
        # Analyze confidence ranges
        ranges = [
            (0, 60, "Low confidence"),
            (60, 80, "Medium confidence"), 
            (80, 100, "High confidence"),
            (100, 120, "Very high confidence")
        ]
        
        for min_conf, max_conf, label in ranges:
            count = len(confidences[(confidences >= min_conf) & (confidences < max_conf)])
            print(f"{label} ({min_conf}-{max_conf}): {count} cells")
            
        # Find optimal confidence threshold
        print(f"\nConfidence threshold analysis:")
        thresholds = [50, 60, 70, 80, 90, 100]
        for threshold in thresholds:
            high_conf_cells = len(confidences[confidences >= threshold])
            print(f"  Threshold {threshold}: {high_conf_cells} cells")
            
    def create_visual_validation_grid(self, region_id=0, max_cells=25):
        """Create a visual grid showing extracted cells for manual validation"""
        print(f"\n=== VISUAL VALIDATION FOR REGION {region_id} ===")
        
        if not self.poster_image:
            print("Cannot create visual validation - no poster image")
            return
            
        # Get cells for this region
        region_cells = [cell for cell in self.detailed_data['cells'] 
                       if cell['region_id'] == region_id][:max_cells]
        
        if not region_cells:
            print(f"No cells found for region {region_id}")
            return
            
        # Create grid
        cols = 5
        rows = (len(region_cells) + cols - 1) // cols
        
        cell_size = 50
        grid_width = cols * cell_size
        grid_height = rows * cell_size
        
        grid_image = Image.new('RGB', (grid_width, grid_height), 'white')
        draw = ImageDraw.Draw(grid_image)
        
        for i, cell in enumerate(region_cells):
            row = i // cols
            col = i % cols
            
            x = col * cell_size
            y = row * cell_size
            
            # Extract cell from poster
            try:
                cell_img = self.poster_image.crop((
                    cell['global_x'] - 3, cell['global_y'] - 3,
                    cell['global_x'] + 4, cell['global_y'] + 4
                ))
                cell_img = cell_img.resize((cell_size-2, cell_size-2))
                
                grid_image.paste(cell_img, (x+1, y+1))
                
                # Add bit label
                bit = cell['bit']
                color = 'green' if bit == '1' else 'red' if bit == '0' else 'gray'
                draw.text((x+5, y+5), str(bit), fill=color)
                
                # Add confidence
                conf = cell['confidence']
                draw.text((x+5, y+cell_size-15), f"{conf:.0f}", fill='blue')
                
            except Exception as e:
                draw.rectangle([x, y, x+cell_size, y+cell_size], outline='red')
                draw.text((x+5, y+5), "ERR", fill='red')
                
        # Save grid
        filename = f"cursor_validation_region_{region_id}.png"
        grid_image.save(filename)
        print(f"Visual validation grid saved: {filename}")
        
    def analyze_bit_patterns(self):
        """Analyze patterns in the extracted binary data"""
        print("\n=== BIT PATTERN ANALYSIS ===")
        
        # Convert to binary string
        binary_string = ''.join(str(bit) for bit in self.binary_data['bit'].tolist())
        
        print(f"Total binary digits: {len(binary_string)}")
        print(f"Binary string (first 100 chars): {binary_string[:100]}")
        
        # Analyze patterns
        patterns = {
            'consecutive_ones': 0,
            'consecutive_zeros': 0,
            'alternating': 0
        }
        
        for i in range(len(binary_string) - 1):
            if binary_string[i] == binary_string[i+1]:
                if binary_string[i] == '1':
                    patterns['consecutive_ones'] += 1
                else:
                    patterns['consecutive_zeros'] += 1
            else:
                patterns['alternating'] += 1
                
        print(f"\nPattern analysis:")
        print(f"  Consecutive ones: {patterns['consecutive_ones']}")
        print(f"  Consecutive zeros: {patterns['consecutive_zeros']}")
        print(f"  Alternating: {patterns['alternating']}")
        
        # Check for cryptographic patterns
        print(f"\nCryptographic pattern checks:")
        
        # Entropy calculation
        ones = binary_string.count('1')
        zeros = binary_string.count('0')
        total = len(binary_string)
        
        if total > 0:
            p1 = ones / total
            p0 = zeros / total
            
            if p1 > 0 and p0 > 0:
                entropy = -p1 * np.log2(p1) - p0 * np.log2(p0)
                print(f"  Shannon entropy: {entropy:.3f} bits")
                print(f"  Expected for random: 1.0 bits")
                print(f"  Entropy ratio: {entropy:.1%}")
                
        return patterns
        
    def suggest_optimizations(self):
        """Suggest optimization strategies based on analysis"""
        print("\n=== OPTIMIZATION SUGGESTIONS ===")
        
        suggestions = []
        
        # Analyze confidence distribution
        confidences = self.binary_data['confidence']
        low_conf_count = len(confidences[confidences < 70])
        
        if low_conf_count > len(confidences) * 0.3:
            suggestions.append({
                'issue': 'Many low-confidence classifications',
                'suggestion': 'Adjust blue channel thresholds or try different color channels',
                'priority': 'high'
            })
            
        # Analyze region performance
        region_clarity = self.analyze_region_performance()
        if region_clarity:
            worst_regions = [r for r, c in region_clarity.items() if c < 20]
            if worst_regions:
                suggestions.append({
                    'issue': f'Poor performance in regions: {worst_regions}',
                    'suggestion': 'Investigate lighting/contrast issues in these regions',
                    'priority': 'medium'
                })
                
        # Check bit distribution
        bit_counts = self.binary_data['bit'].value_counts()
        if '0' in bit_counts and '1' in bit_counts:
            ratio = bit_counts['1'] / bit_counts['0']
            if ratio > 10 or ratio < 0.1:
                suggestions.append({
                    'issue': f'Extreme bit ratio: {ratio:.1f} (1s:0s)',
                    'suggestion': 'Check if classification is biased toward one bit value',
                    'priority': 'medium'
                })
                
        # Print suggestions
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. [{suggestion['priority'].upper()}] {suggestion['issue']}")
            print(f"   → {suggestion['suggestion']}")
            
        return suggestions
        
    def create_comprehensive_report(self):
        """Create a comprehensive analysis report"""
        print("\n" + "="*60)
        print("CURSOR AGENT: COMPLETE EXTRACTION ANALYSIS REPORT")
        print("="*60)
        
        # Run all analyses
        stats = self.analyze_extraction_statistics()
        region_clarity = self.analyze_region_performance()
        self.analyze_confidence_patterns()
        patterns = self.analyze_bit_patterns()
        suggestions = self.suggest_optimizations()
        
        # Create visual validation for first few regions
        for region_id in range(min(3, len(region_clarity))):
            self.create_visual_validation_grid(region_id)
            
        # Summary
        print(f"\n=== SUMMARY ===")
        print(f"Overall clarity rate: {stats['clarity_rate']:.1f}%")
        print(f"Total binary digits extracted: {len(self.binary_data)}")
        print(f"Best region clarity: {max(region_clarity.values()):.1f}%")
        print(f"Optimization suggestions: {len(suggestions)}")
        
        return {
            'statistics': stats,
            'region_performance': region_clarity,
            'patterns': patterns,
            'suggestions': suggestions
        }

def main():
    """Main analysis function"""
    analyzer = ExtractionAnalyzer()
    report = analyzer.create_comprehensive_report()
    
    # Save report
    with open('cursor_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nAnalysis report saved: cursor_analysis_report.json")

if __name__ == "__main__":
    main() 