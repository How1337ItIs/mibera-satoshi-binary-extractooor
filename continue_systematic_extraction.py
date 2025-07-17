#!/usr/bin/env python3
"""
Continue Systematic Extraction - Claude Code Agent
Created: July 17, 2025

Purpose: Continue extraction using Cursor's validated methodology
across all remaining poster regions.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

class SystematicExtractor:
    def __init__(self):
        self.validated_data = self.load_validated_data()
        self.grid_params = self.get_validated_grid_params()
        
    def load_validated_data(self):
        """Load Cursor's validated extraction results"""
        try:
            return pd.read_csv("cursor_optimized_extraction.csv")
        except:
            print("ERROR: Cannot load validated extraction data")
            return None
            
    def get_validated_grid_params(self):
        """Get the validated grid parameters from Cursor's work"""
        return {
            'pitch_row': 15,
            'pitch_col': 12, 
            'origin_row': 890,
            'origin_col': 185,
            'threshold': 80,
            'method': 'blue_channel'
        }

    def analyze_current_coverage(self):
        """Analyze what regions are already covered"""
        if self.validated_data is None:
            return {}
            
        coverage = {}
        regions_covered = sorted(self.validated_data['region_id'].unique())
        
        print("=== CURRENT EXTRACTION COVERAGE ===")
        print(f"Regions covered: {len(regions_covered)}")
        print(f"Region IDs: {regions_covered}")
        
        total_cells = len(self.validated_data)
        ones = (self.validated_data['bit'] == 1).sum()
        zeros = (self.validated_data['bit'] == 0).sum()
        
        print(f"Total cells: {total_cells}")
        print(f"Ones: {ones} ({ones/total_cells*100:.1f}%)")
        print(f"Zeros: {zeros} ({zeros/total_cells*100:.1f}%)")
        print(f"Ratio ones:zeros = {ones/zeros:.2f}:1")
        
        # Analyze per-region statistics
        print(f"\n=== PER-REGION STATISTICS ===")
        for region_id in regions_covered:
            region_data = self.validated_data[self.validated_data['region_id'] == region_id]
            r_total = len(region_data)
            r_ones = (region_data['bit'] == 1).sum()
            r_zeros = (region_data['bit'] == 0).sum()
            r_conf = region_data['confidence'].mean()
            
            print(f"Region {region_id:2d}: {r_total:3d} cells, {r_ones/r_total*100:5.1f}% ones, conf={r_conf:.1f}")
            
        return {
            'regions_covered': regions_covered,
            'total_cells': total_cells,
            'ones': ones,
            'zeros': zeros,
            'bit_ratio': ones / max(zeros, 1)
        }

    def identify_extraction_targets(self):
        """Identify remaining regions for extraction"""
        print(f"\n=== IDENTIFYING REMAINING TARGETS ===")
        
        # From previous analysis, we know there were 61+ potential regions
        # Current coverage shows 16 regions extracted
        
        covered_regions = set(self.validated_data['region_id'].unique()) if self.validated_data is not None else set()
        
        # Based on previous poster analysis, high-potential regions were identified
        # Let's target the most promising ones that aren't covered yet
        high_potential_regions = list(range(0, 50))  # Regions 0-49 identified previously
        
        remaining_targets = [r for r in high_potential_regions if r not in covered_regions]
        
        print(f"Covered regions: {len(covered_regions)}")
        print(f"Remaining high-potential targets: {len(remaining_targets)}")
        print(f"Next priority regions: {remaining_targets[:10]}")
        
        return remaining_targets[:20]  # Focus on top 20 remaining

    def estimate_extraction_parameters(self, region_id):
        """Estimate extraction parameters for a region based on validated approach"""
        
        # Use the validated parameters from Cursor's work
        base_params = {
            'region_id': region_id,
            'grid_pitch_row': self.grid_params['pitch_row'],
            'grid_pitch_col': self.grid_params['pitch_col'],
            'threshold': self.grid_params['threshold'],
            'method': self.grid_params['method'],
            'confidence_threshold': 50.0  # Minimum confidence for inclusion
        }
        
        # Estimate grid origin based on region position
        # This would need the poster image to be precise, but we can estimate
        estimated_origin_y = self.grid_params['origin_row'] + (region_id // 8) * 200
        estimated_origin_x = self.grid_params['origin_col'] + (region_id % 8) * 150
        
        base_params.update({
            'estimated_origin_x': estimated_origin_x,
            'estimated_origin_y': estimated_origin_y,
            'grid_rows': 30,  # Estimated grid size
            'grid_cols': 30   # Estimated grid size
        })
        
        return base_params

    def create_extraction_plan(self):
        """Create systematic extraction plan"""
        print(f"\n=== CREATING EXTRACTION PLAN ===")
        
        current_coverage = self.analyze_current_coverage()
        remaining_targets = self.identify_extraction_targets()
        
        extraction_plan = {
            'current_status': current_coverage,
            'remaining_regions': remaining_targets,
            'total_regions_planned': len(remaining_targets),
            'extraction_parameters': self.grid_params,
            'region_plans': []
        }
        
        for region_id in remaining_targets:
            region_params = self.estimate_extraction_parameters(region_id)
            extraction_plan['region_plans'].append(region_params)
            
        # Save extraction plan
        with open('systematic_extraction_plan.json', 'w') as f:
            json.dump(extraction_plan, f, indent=2)
            
        print(f"Created extraction plan for {len(remaining_targets)} regions")
        print(f"Plan saved to: systematic_extraction_plan.json")
        
        return extraction_plan

    def generate_progress_summary(self):
        """Generate overall progress summary"""
        print(f"\n=== PROJECT PROGRESS SUMMARY ===")
        
        coverage = self.analyze_current_coverage()
        
        # Estimate total project scope
        estimated_total_regions = 60  # Based on previous full poster analysis
        estimated_total_cells = 3000  # Conservative estimate
        
        current_regions = len(coverage.get('regions_covered', []))
        current_cells = coverage.get('total_cells', 0)
        
        region_progress = current_regions / estimated_total_regions * 100
        cell_progress = current_cells / estimated_total_cells * 100
        
        print(f"Regions completed: {current_regions}/{estimated_total_regions} ({region_progress:.1f}%)")
        print(f"Cells extracted: {current_cells}/{estimated_total_cells} ({cell_progress:.1f}%)")
        
        print(f"\nCurrent extraction quality:")
        print(f"  Bit balance: {coverage.get('bit_ratio', 0):.2f}:1 ones:zeros")
        print(f"  Statistical validity: {'GOOD' if 0.1 <= coverage.get('bit_ratio', 0) <= 4.0 else 'SUSPICIOUS'}")
        
        return {
            'region_progress_pct': region_progress,
            'cell_progress_pct': cell_progress,
            'quality_status': 'GOOD' if 0.1 <= coverage.get('bit_ratio', 0) <= 4.0 else 'SUSPICIOUS'
        }

if __name__ == "__main__":
    extractor = SystematicExtractor()
    
    # Analyze current state
    extractor.analyze_current_coverage()
    
    # Create extraction plan
    plan = extractor.create_extraction_plan()
    
    # Generate progress summary
    progress = extractor.generate_progress_summary()
    
    print(f"\n=== NEXT STEPS ===")
    print("1. Apply validated extraction method to remaining regions")
    print("2. Monitor bit ratios for quality control (target: 0.1-4.0 range)")
    print("3. Build complete validated bit matrix")
    print("4. Continue toward goal of extracting every poster digit")