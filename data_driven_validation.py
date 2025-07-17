#!/usr/bin/env python3
"""
Data-Driven Validation - Claude Code Agent
Created: July 17, 2025

Purpose: Determine which extraction method is more reliable using
statistical analysis and logical consistency checks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

class DataValidator:
    def __init__(self):
        self.claude_data = self.load_claude_data()
        self.cursor_data = self.load_cursor_data()
        
    def load_claude_data(self):
        """Load Claude Code's extraction"""
        try:
            return pd.read_csv("complete_extraction_binary_only.csv")
        except:
            print("ERROR: Claude Code data not found")
            return None
            
    def load_cursor_data(self):
        """Load Cursor's extraction"""
        try:
            return pd.read_csv("cursor_optimized_extraction.csv")
        except:
            print("ERROR: Cursor data not found")
            return None

    def analyze_spatial_consistency(self, data, name):
        """Check if bits show spatial patterns vs random distribution"""
        print(f"\n=== SPATIAL ANALYSIS: {name} ===")
        
        if data is None or len(data) == 0:
            print("No data available")
            return {}
            
        # Group by region
        region_stats = []
        for region_id in sorted(data['region_id'].unique()):
            region_data = data[data['region_id'] == region_id]
            
            ones = (region_data['bit'] == 1).sum()
            zeros = (region_data['bit'] == 0).sum()
            total = len(region_data)
            
            if total > 0:
                ones_pct = ones / total * 100
                zeros_pct = zeros / total * 100
                
                region_stats.append({
                    'region': region_id,
                    'total': total,
                    'ones': ones,
                    'zeros': zeros,
                    'ones_pct': ones_pct,
                    'zeros_pct': zeros_pct
                })
                
                print(f"Region {region_id}: {total} cells, {ones_pct:.1f}% ones, {zeros_pct:.1f}% zeros")
        
        return region_stats

    def analyze_confidence_distribution(self, data, name):
        """Analyze confidence score patterns"""
        print(f"\n=== CONFIDENCE ANALYSIS: {name} ===")
        
        if data is None or len(data) == 0:
            return {}
            
        conf_stats = {
            'mean': data['confidence'].mean(),
            'std': data['confidence'].std(),
            'min': data['confidence'].min(),
            'max': data['confidence'].max(),
            'ones_conf': data[data['bit'] == 1]['confidence'].mean(),
            'zeros_conf': data[data['bit'] == 0]['confidence'].mean()
        }
        
        print(f"Mean confidence: {conf_stats['mean']:.1f}")
        print(f"Std confidence: {conf_stats['std']:.1f}")
        print(f"Range: {conf_stats['min']:.1f} - {conf_stats['max']:.1f}")
        print(f"Ones confidence: {conf_stats['ones_conf']:.1f}")
        print(f"Zeros confidence: {conf_stats['zeros_conf']:.1f}")
        
        return conf_stats

    def compare_overlapping_cells(self):
        """Compare predictions for same coordinates"""
        print(f"\n=== OVERLAP COMPARISON ===")
        
        if self.claude_data is None or self.cursor_data is None:
            print("Missing data for comparison")
            return {}
            
        # Find overlapping coordinates
        claude_coords = self.claude_data.set_index(['global_x', 'global_y'])
        cursor_coords = self.cursor_data.set_index(['global_x', 'global_y'])
        
        overlap_coords = claude_coords.index.intersection(cursor_coords.index)
        print(f"Overlapping coordinates: {len(overlap_coords)}")
        
        if len(overlap_coords) == 0:
            print("No overlapping coordinates found")
            return {}
            
        agreements = 0
        disagreements = 0
        
        for coord in overlap_coords:
            claude_bit = claude_coords.loc[coord, 'bit']
            cursor_bit = cursor_coords.loc[coord, 'bit']
            
            if claude_bit == cursor_bit:
                agreements += 1
            else:
                disagreements += 1
                
        agreement_rate = agreements / len(overlap_coords)
        
        print(f"Agreements: {agreements}")
        print(f"Disagreements: {disagreements}")
        print(f"Agreement rate: {agreement_rate:.1%}")
        
        return {
            'overlapping_cells': len(overlap_coords),
            'agreements': agreements,
            'disagreements': disagreements,
            'agreement_rate': agreement_rate
        }

    def analyze_bit_balance_plausibility(self):
        """Determine which bit distribution is more plausible"""
        print(f"\n=== BIT BALANCE PLAUSIBILITY ===")
        
        claude_ones = (self.claude_data['bit'] == 1).sum() if self.claude_data is not None else 0
        claude_zeros = (self.claude_data['bit'] == 0).sum() if self.claude_data is not None else 0
        claude_total = len(self.claude_data) if self.claude_data is not None else 0
        
        cursor_ones = (self.cursor_data['bit'] == 1).sum() if self.cursor_data is not None else 0
        cursor_zeros = (self.cursor_data['bit'] == 0).sum() if self.cursor_data is not None else 0
        cursor_total = len(self.cursor_data) if self.cursor_data is not None else 0
        
        print("Claude Code:")
        print(f"  {claude_ones} ones ({claude_ones/claude_total*100:.1f}%)")
        print(f"  {claude_zeros} zeros ({claude_zeros/claude_total*100:.1f}%)")
        print(f"  Ratio ones:zeros = {claude_ones/max(claude_zeros,1):.1f}:1")
        
        print("\nCursor Agent:")
        print(f"  {cursor_ones} ones ({cursor_ones/cursor_total*100:.1f}%)")
        print(f"  {cursor_zeros} zeros ({cursor_zeros/cursor_total*100:.1f}%)")
        print(f"  Ratio ones:zeros = {cursor_ones/max(cursor_zeros,1):.1f}:1")
        
        # For binary data in images, extreme ratios (>10:1) are suspicious
        claude_ratio = claude_ones / max(claude_zeros, 1)
        cursor_ratio = cursor_ones / max(cursor_zeros, 1)
        
        print(f"\nPlausibility Assessment:")
        print(f"Claude ratio {claude_ratio:.1f}:1 - {'SUSPICIOUS' if claude_ratio > 10 else 'REASONABLE'}")
        print(f"Cursor ratio {cursor_ratio:.1f}:1 - {'SUSPICIOUS' if cursor_ratio > 10 else 'REASONABLE'}")
        
        return {
            'claude_ratio': claude_ratio,
            'cursor_ratio': cursor_ratio,
            'claude_suspicious': claude_ratio > 10,
            'cursor_suspicious': cursor_ratio > 10
        }

    def run_complete_analysis(self):
        """Run all validation analyses"""
        print("=== DATA-DRIVEN VALIDATION ANALYSIS ===")
        
        results = {}
        
        # Spatial consistency
        if self.claude_data is not None:
            results['claude_spatial'] = self.analyze_spatial_consistency(self.claude_data, "Claude Code")
            results['claude_confidence'] = self.analyze_confidence_distribution(self.claude_data, "Claude Code")
            
        if self.cursor_data is not None:
            results['cursor_spatial'] = self.analyze_spatial_consistency(self.cursor_data, "Cursor Agent")
            results['cursor_confidence'] = self.analyze_confidence_distribution(self.cursor_data, "Cursor Agent")
            
        # Overlap comparison
        results['overlap'] = self.compare_overlapping_cells()
        
        # Bit balance plausibility
        results['balance'] = self.analyze_bit_balance_plausibility()
        
        # Save results
        with open("validation_analysis_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        print(f"\n=== SUMMARY RECOMMENDATION ===")
        self.make_recommendation(results)
        
        return results

    def make_recommendation(self, results):
        """Make data-driven recommendation"""
        claude_suspicious = results.get('balance', {}).get('claude_suspicious', False)
        cursor_suspicious = results.get('balance', {}).get('cursor_suspicious', False)
        
        agreement_rate = results.get('overlap', {}).get('agreement_rate', 0)
        
        print("Based on statistical analysis:")
        
        if claude_suspicious and not cursor_suspicious:
            print("RECOMMENDATION: Cursor's data appears more balanced and plausible")
        elif cursor_suspicious and not claude_suspicious:
            print("RECOMMENDATION: Claude Code's data appears more plausible")
        elif agreement_rate > 0.8:
            print("RECOMMENDATION: Both methods largely agree - use either")
        else:
            print("RECOMMENDATION: Manual visual validation required")
            print("Both methods show concerning patterns that need verification")

if __name__ == "__main__":
    validator = DataValidator()
    validator.run_complete_analysis()