#!/usr/bin/env python3
"""
Region-Based Analysis Tool for Satoshi Poster Binary Extraction
Agent: Cursor Agent
Purpose: Visual validation and manual parameter tuning for region-based extraction
Date: 2025-07-16
"""

import cv2
import numpy as np
import yaml
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
from datetime import datetime

@dataclass
class RegionAnalysis:
    """Data structure for region analysis results"""
    region_id: str
    x1: int
    y1: int
    x2: int
    y2: int
    contrast_level: str  # 'high', 'medium', 'low'
    estimated_cells: int
    overlay_density: float  # 0-1 scale
    lighting_consistency: float  # 0-1 scale
    extraction_difficulty: str  # 'easy', 'moderate', 'hard'
    recommended_params: Dict
    notes: str

class RegionAnalyzer:
    """Systematic region analysis for poster extraction optimization"""
    
    def __init__(self, image_path: str, config_path: str = "binary_extractor/cfg.yaml"):
        self.image_path = image_path
        self.config_path = config_path
        self.image = None
        self.config = None
        self.regions = []
        self.analysis_results = []
        
        self.load_image()
        self.load_config()
        
    def load_image(self):
        """Load and validate poster image"""
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise ValueError(f"Failed to load image: {self.image_path}")
        
        print(f"Loaded image: {self.image.shape[1]}x{self.image.shape[0]} pixels")
        
    def load_config(self):
        """Load current extraction configuration"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print(f"Loaded config with {len(self.config)} parameters")
        
    def divide_poster_regions(self, grid_size: int = 3) -> List[Tuple[int, int, int, int]]:
        """Divide poster into grid regions for systematic analysis"""
        height, width = self.image.shape[:2]
        
        region_width = width // grid_size
        region_height = height // grid_size
        
        regions = []
        for row in range(grid_size):
            for col in range(grid_size):
                x1 = col * region_width
                y1 = row * region_height
                x2 = x1 + region_width if col < grid_size - 1 else width
                y2 = y1 + region_height if row < grid_size - 1 else height
                
                regions.append((x1, y1, x2, y2))
        
        return regions
    
    def analyze_region_contrast(self, region: Tuple[int, int, int, int]) -> Dict:
        """Analyze contrast characteristics of a region"""
        x1, y1, x2, y2 = region
        region_img = self.image[y1:y2, x1:x2]
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(region_img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(region_img, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
        
        # Calculate contrast metrics
        contrast_metrics = {
            'hsv_saturation_std': np.std(hsv[:, :, 1]),
            'lab_b_std': np.std(lab[:, :, 2]),
            'gray_std': np.std(gray),
            'gray_mean': np.mean(gray),
            'gray_min': np.min(gray),
            'gray_max': np.max(gray),
            'contrast_ratio': (np.max(gray) - np.min(gray)) / (np.max(gray) + np.min(gray) + 1e-6)
        }
        
        return contrast_metrics
    
    def estimate_overlay_density(self, region: Tuple[int, int, int, int]) -> float:
        """Estimate overlay interference density in region"""
        x1, y1, x2, y2 = region
        region_img = self.image[y1:y2, x1:x2]
        
        # Convert to HSV for overlay detection
        hsv = cv2.cvtColor(region_img, cv2.COLOR_BGR2HSV)
        
        # High saturation areas likely indicate overlay
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        
        # Count pixels that match overlay criteria
        overlay_pixels = np.sum(
            (saturation > self.config.get('overlay', {}).get('saturation_threshold', 25)) &
            (value > self.config.get('overlay', {}).get('value_threshold', 190))
        )
        
        total_pixels = saturation.size
        overlay_density = overlay_pixels / total_pixels
        
        return overlay_density
    
    def assess_lighting_consistency(self, region: Tuple[int, int, int, int]) -> float:
        """Assess lighting consistency across region"""
        x1, y1, x2, y2 = region
        region_img = self.image[y1:y2, x1:x2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
        
        # Calculate local standard deviation (lower = more consistent)
        kernel_size = min(15, min(gray.shape) // 4)
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        local_std = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        lighting_variation = np.std(local_std)
        
        # Normalize to 0-1 scale (lower = more consistent)
        max_variation = 50  # Empirical threshold
        consistency = max(0, 1 - (lighting_variation / max_variation))
        
        return consistency
    
    def estimate_cell_count(self, region: Tuple[int, int, int, int]) -> int:
        """Estimate number of binary cells in region"""
        x1, y1, x2, y2 = region
        region_width = x2 - x1
        region_height = y2 - y1
        
        # Use current grid parameters as baseline
        row_pitch = self.config.get('row_pitch', 40)  # Default if not set
        col_pitch = self.config.get('col_pitch', 40)  # Default if not set
        
        if row_pitch is None or col_pitch is None:
            # Estimate based on typical cell size
            estimated_cell_size = 30  # pixels
            rows = region_height // estimated_cell_size
            cols = region_width // estimated_cell_size
        else:
            rows = region_height // row_pitch
            cols = region_width // col_pitch
        
        return max(1, rows * cols)
    
    def classify_region_difficulty(self, contrast_metrics: Dict, overlay_density: float, 
                                 lighting_consistency: float) -> str:
        """Classify extraction difficulty based on region characteristics"""
        
        # Contrast assessment
        gray_contrast = contrast_metrics['contrast_ratio']
        hsv_contrast = contrast_metrics['hsv_saturation_std'] / 255.0
        
        # Difficulty scoring
        difficulty_score = 0
        
        # Contrast factor (0-3 points)
        if gray_contrast > 0.6:
            difficulty_score += 0  # Easy
        elif gray_contrast > 0.4:
            difficulty_score += 1  # Moderate
        else:
            difficulty_score += 2  # Hard
            
        # Overlay factor (0-2 points)
        if overlay_density < 0.1:
            difficulty_score += 0  # Low interference
        elif overlay_density < 0.3:
            difficulty_score += 1  # Moderate interference
        else:
            difficulty_score += 2  # High interference
            
        # Lighting factor (0-2 points)
        if lighting_consistency > 0.8:
            difficulty_score += 0  # Consistent
        elif lighting_consistency > 0.5:
            difficulty_score += 1  # Variable
        else:
            difficulty_score += 2  # Inconsistent
        
        # Classify based on total score
        if difficulty_score <= 2:
            return 'easy'
        elif difficulty_score <= 4:
            return 'moderate'
        else:
            return 'hard'
    
    def generate_region_params(self, difficulty: str, contrast_metrics: Dict) -> Dict:
        """Generate recommended parameters for region based on difficulty"""
        
        base_params = self.config.copy()
        
        if difficulty == 'easy':
            # High-contrast regions: aggressive thresholds
            base_params.update({
                'use_color_space': 'HSV_S',
                'bit_hi': 0.75,
                'bit_lo': 0.30,
                'blur_sigma': 10,
                'overlay': {
                    'saturation_threshold': 20,
                    'value_threshold': 200,
                    'cell_coverage_threshold': 0.15
                }
            })
            
        elif difficulty == 'moderate':
            # Medium-contrast regions: balanced approach
            base_params.update({
                'use_color_space': 'Lab_b',
                'bit_hi': 0.65,
                'bit_lo': 0.40,
                'blur_sigma': 15,
                'template_match': True,
                'tm_thresh': 0.50,
                'overlay': {
                    'saturation_threshold': 30,
                    'value_threshold': 180,
                    'cell_coverage_threshold': 0.25
                }
            })
            
        else:  # hard
            # Low-contrast regions: conservative with template matching
            base_params.update({
                'use_color_space': 'RGB_G',
                'bit_hi': 0.60,
                'bit_lo': 0.45,
                'blur_sigma': 20,
                'template_match': True,
                'tm_thresh': 0.40,
                'overlay': {
                    'saturation_threshold': 40,
                    'value_threshold': 160,
                    'cell_coverage_threshold': 0.35
                }
            })
        
        return base_params
    
    def analyze_all_regions(self, grid_size: int = 3) -> List[RegionAnalysis]:
        """Perform comprehensive region analysis"""
        print(f"Analyzing poster with {grid_size}x{grid_size} grid...")
        
        regions = self.divide_poster_regions(grid_size)
        results = []
        
        for i, region in enumerate(regions):
            region_id = f"R{i+1}"
            x1, y1, x2, y2 = region
            
            print(f"Analyzing {region_id} ({x1},{y1}) to ({x2},{y2})...")
            
            # Perform analysis
            contrast_metrics = self.analyze_region_contrast(region)
            overlay_density = self.estimate_overlay_density(region)
            lighting_consistency = self.assess_lighting_consistency(region)
            estimated_cells = self.estimate_cell_count(region)
            
            # Classify difficulty
            difficulty = self.classify_region_difficulty(
                contrast_metrics, overlay_density, lighting_consistency
            )
            
            # Determine contrast level
            if difficulty == 'easy':
                contrast_level = 'high'
            elif difficulty == 'moderate':
                contrast_level = 'medium'
            else:
                contrast_level = 'low'
            
            # Generate recommended parameters
            recommended_params = self.generate_region_params(difficulty, contrast_metrics)
            
            # Create analysis result
            analysis = RegionAnalysis(
                region_id=region_id,
                x1=x1, y1=y1, x2=x2, y2=y2,
                contrast_level=contrast_level,
                estimated_cells=estimated_cells,
                overlay_density=overlay_density,
                lighting_consistency=lighting_consistency,
                extraction_difficulty=difficulty,
                recommended_params=recommended_params,
                notes=f"Auto-analyzed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            results.append(analysis)
            
            print(f"  Difficulty: {difficulty}, Cells: ~{estimated_cells}, "
                  f"Overlay: {overlay_density:.2f}, Lighting: {lighting_consistency:.2f}")
        
        self.analysis_results = results
        return results
    
    def create_region_map(self, output_path: str = "region_analysis_map.png"):
        """Create visual map of region analysis results"""
        if not self.analysis_results:
            print("No analysis results available. Run analyze_all_regions() first.")
            return
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Display original image
        img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        
        # Draw region boundaries and labels
        colors = {'easy': 'green', 'moderate': 'yellow', 'hard': 'red'}
        
        for analysis in self.analysis_results:
            color = colors[analysis.extraction_difficulty]
            
            # Draw rectangle
            rect = plt.Rectangle(
                (analysis.x1, analysis.y1), 
                analysis.x2 - analysis.x1, 
                analysis.y2 - analysis.y1,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            ax.text(
                analysis.x1 + 10, analysis.y1 + 30,
                f"{analysis.region_id}\n{analysis.extraction_difficulty.upper()}\n~{analysis.estimated_cells} cells",
                color=color, fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
            )
        
        ax.set_title("Satoshi Poster Region Analysis Map\nGreen=Easy, Yellow=Moderate, Red=Hard")
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Region map saved to: {output_path}")
    
    def save_analysis_report(self, output_path: str = "region_analysis_report.json"):
        """Save comprehensive analysis report"""
        if not self.analysis_results:
            print("No analysis results available. Run analyze_all_regions() first.")
            return
        
        # Convert to serializable format
        report = {
            'timestamp': datetime.now().isoformat(),
            'image_path': self.image_path,
            'image_shape': self.image.shape,
            'analysis_summary': {
                'total_regions': len(self.analysis_results),
                'easy_regions': len([r for r in self.analysis_results if r.extraction_difficulty == 'easy']),
                'moderate_regions': len([r for r in self.analysis_results if r.extraction_difficulty == 'moderate']),
                'hard_regions': len([r for r in self.analysis_results if r.extraction_difficulty == 'hard']),
                'total_estimated_cells': sum(r.estimated_cells for r in self.analysis_results)
            },
            'regions': []
        }
        
        for analysis in self.analysis_results:
            region_data = {
                'region_id': analysis.region_id,
                'coordinates': {'x1': analysis.x1, 'y1': analysis.y1, 'x2': analysis.x2, 'y2': analysis.y2},
                'characteristics': {
                    'contrast_level': analysis.contrast_level,
                    'extraction_difficulty': analysis.extraction_difficulty,
                    'estimated_cells': analysis.estimated_cells,
                    'overlay_density': analysis.overlay_density,
                    'lighting_consistency': analysis.lighting_consistency
                },
                'recommended_params': analysis.recommended_params,
                'notes': analysis.notes
            }
            report['regions'].append(region_data)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Analysis report saved to: {output_path}")
        
        # Print summary
        summary = report['analysis_summary']
        print(f"\n=== ANALYSIS SUMMARY ===")
        print(f"Total regions: {summary['total_regions']}")
        print(f"Easy regions: {summary['easy_regions']} (start here)")
        print(f"Moderate regions: {summary['moderate_regions']}")
        print(f"Hard regions: {summary['hard_regions']}")
        print(f"Total estimated cells: {summary['total_estimated_cells']}")
        
        # Recommend next steps
        easy_regions = [r for r in self.analysis_results if r.extraction_difficulty == 'easy']
        if easy_regions:
            print(f"\n=== RECOMMENDED NEXT STEPS ===")
            print(f"1. Start with region {easy_regions[0].region_id} for initial optimization")
            print(f"2. Use parameters from analysis report for each region")
            print(f"3. Validate extraction accuracy on easy regions first")
            print(f"4. Coordinate with Cursor agent for visual validation")

def main():
    """Main execution function"""
    print("=== Satoshi Poster Region Analysis Tool ===")
    print("Agent: Cursor Agent")
    print("Purpose: Visual validation and manual parameter tuning")
    print()
    
    # Configuration
    image_path = "satoshi_poster.png"
    config_path = "binary_extractor/cfg.yaml"
    
    try:
        # Initialize analyzer
        analyzer = RegionAnalyzer(image_path, config_path)
        
        # Perform analysis
        results = analyzer.analyze_all_regions(grid_size=3)
        
        # Create outputs
        analyzer.create_region_map("region_analysis_map.png")
        analyzer.save_analysis_report("region_analysis_report.json")
        
        print("\n=== VISUAL ANALYSIS COMPLETE ===")
        print("Next steps for Cursor Agent:")
        print("1. Review region_analysis_map.png for visual overview")
        print("2. Check region_analysis_report.json for detailed parameters")
        print("3. Begin visual grid calibration with current parameters")
        print("4. Start extraction optimization with region R1")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 