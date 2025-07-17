#!/usr/bin/env python3
"""
Visual Ground Truth Validator
Created by: Claude Code Agent
Date: July 17, 2025

Purpose: Establish actual ground truth by visual inspection to determine
which extraction method is correct - Claude Code's or Cursor's.
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

class VisualValidator:
    def __init__(self):
        self.poster = self.load_poster()
        self.claude_data = self.load_claude_data()
        self.cursor_data = self.load_cursor_data()
        
    def load_poster(self):
        """Load poster image"""
        poster_path = Path("satoshi (1).png")
        if poster_path.exists():
            return cv2.imread(str(poster_path))
        else:
            print("ERROR: Poster image not found")
            return None
            
    def load_claude_data(self):
        """Load Claude Code's original extraction"""
        try:
            return pd.read_csv("complete_extraction_binary_only.csv")
        except:
            print("WARNING: Claude Code data not found")
            return None
            
    def load_cursor_data(self):
        """Load Cursor's optimized extraction"""
        try:
            return pd.read_csv("cursor_optimized_extraction.csv")
        except:
            print("WARNING: Cursor data not found")
            return None

    def extract_cell_region(self, x, y, size=6):
        """Extract cell region from poster for visual inspection"""
        if self.poster is None:
            return None
            
        # Extract region around coordinates
        x_start = max(0, x - size)
        x_end = min(self.poster.shape[1], x + size)
        y_start = max(0, y - size)
        y_end = min(self.poster.shape[0], y + size)
        
        return self.poster[y_start:y_end, x_start:x_end]

    def create_comparison_sample(self, n_samples=20):
        """Create side-by-side comparison of Claude vs Cursor for same coordinates"""
        if self.claude_data is None or self.cursor_data is None:
            print("ERROR: Missing data for comparison")
            return
            
        # Find overlapping coordinates
        claude_coords = set(zip(self.claude_data['global_x'], self.claude_data['global_y']))
        cursor_coords = set(zip(self.cursor_data['global_x'], self.cursor_data['global_y']))
        overlap = list(claude_coords.intersection(cursor_coords))
        
        print(f"Found {len(overlap)} overlapping coordinates")
        
        # Sample random coordinates from overlap
        import random
        sample_coords = random.sample(overlap, min(n_samples, len(overlap)))
        
        results = []
        
        for i, (x, y) in enumerate(sample_coords):
            # Get Claude's classification
            claude_row = self.claude_data[
                (self.claude_data['global_x'] == x) & 
                (self.claude_data['global_y'] == y)
            ].iloc[0]
            
            # Get Cursor's classification  
            cursor_row = self.cursor_data[
                (self.cursor_data['global_x'] == x) & 
                (self.cursor_data['global_y'] == y)
            ].iloc[0]
            
            # Extract visual cell
            cell_image = self.extract_cell_region(x, y)
            
            results.append({
                'coord': (x, y),
                'claude_bit': claude_row['bit'],
                'claude_conf': claude_row['confidence'],
                'cursor_bit': cursor_row['bit'], 
                'cursor_conf': cursor_row['confidence'],
                'cell_image': cell_image,
                'disagreement': claude_row['bit'] != cursor_row['bit']
            })
            
        return results

    def save_visual_comparison(self, results, output_dir="visual_validation"):
        """Save visual comparison for manual inspection"""
        Path(output_dir).mkdir(exist_ok=True)
        
        disagreements = [r for r in results if r['disagreement']]
        agreements = [r for r in results if not r['disagreement']]
        
        print(f"Disagreements: {len(disagreements)}")
        print(f"Agreements: {len(agreements)}")
        
        # Save disagreement cases for manual review
        for i, result in enumerate(disagreements):
            if result['cell_image'] is not None:
                x, y = result['coord']
                filename = f"disagreement_{i:02d}_x{x}_y{y}_claude{result['claude_bit']}_cursor{result['cursor_bit']}.png"
                cv2.imwrite(str(Path(output_dir) / filename), result['cell_image'])
                
        # Create summary
        summary = {
            'total_samples': len(results),
            'disagreements': len(disagreements),
            'agreements': len(agreements),
            'disagreement_rate': len(disagreements) / len(results) if results else 0,
            'claude_stats': {
                'ones': sum(1 for r in results if r['claude_bit'] == 1),
                'zeros': sum(1 for r in results if r['claude_bit'] == 0),
                'avg_confidence': np.mean([r['claude_conf'] for r in results])
            },
            'cursor_stats': {
                'ones': sum(1 for r in results if r['cursor_bit'] == 1),
                'zeros': sum(1 for r in results if r['cursor_bit'] == 0),
                'avg_confidence': np.mean([r['cursor_conf'] for r in results])
            }
        }
        
        with open(Path(output_dir) / "validation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
            
        return summary

    def run_validation(self):
        """Run complete validation"""
        print("=== VISUAL GROUND TRUTH VALIDATION ===")
        print("Comparing Claude Code vs Cursor Agent extractions")
        
        if self.poster is None:
            print("ERROR: Cannot validate without poster image")
            return
            
        # Create comparison sample
        results = self.create_comparison_sample(30)
        if not results:
            print("ERROR: No comparison data available")
            return
            
        # Save visual comparison
        summary = self.save_visual_comparison(results)
        
        print("\n=== VALIDATION SUMMARY ===")
        print(f"Total samples: {summary['total_samples']}")
        print(f"Disagreements: {summary['disagreements']}")
        print(f"Disagreement rate: {summary['disagreement_rate']:.1%}")
        
        print(f"\nClaude Code: {summary['claude_stats']['ones']} ones, {summary['claude_stats']['zeros']} zeros")
        print(f"Cursor Agent: {summary['cursor_stats']['ones']} ones, {summary['cursor_stats']['zeros']} zeros")
        
        print(f"\nDISAGREEMENT IMAGES saved to visual_validation/ directory")
        print("MANUALLY INSPECT these images to determine which agent is correct")
        
        return summary

if __name__ == "__main__":
    validator = VisualValidator()
    validator.run_validation()