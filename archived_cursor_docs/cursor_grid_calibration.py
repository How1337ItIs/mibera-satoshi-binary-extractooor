#!/usr/bin/env python3
"""
Visual Grid Calibration Tool for Satoshi Poster Binary Extraction
Agent: Cursor Agent
Purpose: Visual validation and manual parameter tuning for grid alignment
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
from datetime import datetime

class CursorGridCalibrator:
    """
    Visual grid calibration tool for Cursor Agent
    Purpose: Visual validation and manual parameter tuning
    """
    
    def __init__(self, image_path: str, config_path: str = "binary_extractor/cfg.yaml"):
        self.image_path = image_path
        self.config_path = config_path
        self.image = None
        self.config = None
        self.calibration_results = []
        
        self.load_image()
        self.load_config()
        
    def load_image(self):
        """Load and validate poster image"""
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise ValueError(f"Failed to load image: {self.image_path}")
        
        print(f"Cursor Agent: Loaded image: {self.image.shape[1]}x{self.image.shape[0]} pixels")
        
    def load_config(self):
        """Load current extraction configuration"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print(f"Cursor Agent: Loaded config with {len(self.config)} parameters")
        
    def generate_grid_overlay(self, row_pitch: int = 40, col_pitch: int = 40, 
                            row0: int = 40, col0: int = 10) -> np.ndarray:
        """
        Generate grid overlay for visual calibration
        
        Agent: Cursor Agent
        Purpose: Visual validation and manual parameter tuning
        """
        height, width = self.image.shape[:2]
        
        # Create overlay image
        overlay = self.image.copy()
        
        # Draw horizontal grid lines
        for row in range(0, height, row_pitch):
            y = row0 + row
            if y < height:
                cv2.line(overlay, (0, y), (width, y), (0, 255, 0), 2)
        
        # Draw vertical grid lines
        for col in range(0, width, col_pitch):
            x = col0 + col
            if x < width:
                cv2.line(overlay, (x, 0), (x, height), (0, 255, 0), 2)
        
        return overlay
    
    def save_grid_overlay(self, output_path: str = "cursor_grid_overlay.png", 
                         row_pitch: int = 40, col_pitch: int = 40,
                         row0: int = 40, col0: int = 10):
        """
        Save grid overlay for visual inspection
        
        Agent: Cursor Agent
        Purpose: Visual validation and manual parameter tuning
        """
        overlay = self.generate_grid_overlay(row_pitch, col_pitch, row0, col0)
        
        # Convert BGR to RGB for matplotlib
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
        # Create figure with grid overlay
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.imshow(overlay_rgb)
        ax.set_title(f"Cursor Agent: Grid Overlay Calibration\n"
                    f"row_pitch={row_pitch}, col_pitch={col_pitch}, "
                    f"row0={row0}, col0={col0}")
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Cursor Agent: Grid overlay saved to: {output_path}")
        
        # Log calibration attempt
        self.log_calibration_attempt(row_pitch, col_pitch, row0, col0, output_path)
        
    def log_calibration_attempt(self, row_pitch: int, col_pitch: int, 
                              row0: int, col0: int, output_path: str):
        """
        Log calibration attempt for visual tracking
        
        Agent: Cursor Agent
        Purpose: Visual validation and manual parameter tuning
        """
        calibration_data = {
            "timestamp": datetime.now().isoformat(),
            "agent": "cursor",
            "parameters": {
                "row_pitch": row_pitch,
                "col_pitch": col_pitch,
                "row0": row0,
                "col0": col0
            },
            "output_file": output_path,
            "visual_notes": "Generated for manual inspection",
            "next_action": "Manual visual verification required"
        }
        
        self.calibration_results.append(calibration_data)
        
        # Save to JSON log
        with open("cursor_calibration_log.json", "w") as f:
            json.dump(self.calibration_results, f, indent=2)
            
    def test_parameter_combinations(self):
        """
        Test various parameter combinations for visual calibration
        
        Agent: Cursor Agent
        Purpose: Visual validation and manual parameter tuning
        """
        print("Cursor Agent: Testing parameter combinations for visual calibration...")
        
        # Test different pitch values
        pitch_values = [35, 40, 45, 50]
        origin_values = [(30, 5), (40, 10), (50, 15)]
        
        for row_pitch in pitch_values:
            for col_pitch in pitch_values:
                for row0, col0 in origin_values:
                    output_path = f"cursor_grid_overlay_r{row_pitch}_c{col_pitch}_r0{row0}_c0{col0}.png"
                    self.save_grid_overlay(output_path, row_pitch, col_pitch, row0, col0)
        
        print(f"Cursor Agent: Generated {len(pitch_values)**2 * len(origin_values)} grid overlays for visual inspection")
        
    def create_visual_comparison(self, best_params: Dict):
        """
        Create visual comparison of current vs optimized parameters
        
        Agent: Cursor Agent
        Purpose: Visual validation and manual parameter tuning
        """
        # Current parameters
        current_overlay = self.generate_grid_overlay(
            self.config.get('row_pitch', 40),
            self.config.get('col_pitch', 40),
            self.config.get('row0', 40),
            self.config.get('col0', 10)
        )
        
        # Optimized parameters
        optimized_overlay = self.generate_grid_overlay(
            best_params.get('row_pitch', 40),
            best_params.get('col_pitch', 40),
            best_params.get('row0', 40),
            best_params.get('col0', 10)
        )
        
        # Create comparison figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Current parameters
        current_rgb = cv2.cvtColor(current_overlay, cv2.COLOR_BGR2RGB)
        ax1.imshow(current_rgb)
        ax1.set_title("Cursor Agent: Current Parameters\n"
                     f"row_pitch={self.config.get('row_pitch', 'null')}, "
                     f"col_pitch={self.config.get('col_pitch', 'null')}, "
                     f"row0={self.config.get('row0', 40)}, "
                     f"col0={self.config.get('col0', 10)}")
        ax1.axis('off')
        
        # Optimized parameters
        optimized_rgb = cv2.cvtColor(optimized_overlay, cv2.COLOR_BGR2RGB)
        ax2.imshow(optimized_rgb)
        ax2.set_title("Cursor Agent: Optimized Parameters\n"
                     f"row_pitch={best_params.get('row_pitch', 40)}, "
                     f"col_pitch={best_params.get('col_pitch', 40)}, "
                     f"row0={best_params.get('row0', 40)}, "
                     f"col0={best_params.get('col0', 10)}")
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig("cursor_parameter_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print("Cursor Agent: Parameter comparison saved to: cursor_parameter_comparison.png")
        
    def update_config_with_optimized_params(self, optimized_params: Dict):
        """
        Update config file with optimized parameters
        
        Agent: Cursor Agent
        Purpose: Visual validation and manual parameter tuning
        """
        # Update config with optimized parameters
        self.config.update(optimized_params)
        
        # Save updated config
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        print(f"Cursor Agent: Updated config with optimized parameters: {optimized_params}")
        
        # Log the update
        update_log = {
            "timestamp": datetime.now().isoformat(),
            "agent": "cursor",
            "action": "config_update",
            "parameters": optimized_params,
            "rationale": "Visual calibration optimization",
            "visual_evidence": "cursor_parameter_comparison.png"
        }
        
        with open("cursor_config_updates.json", "w") as f:
            json.dump([update_log], f, indent=2)

def main():
    """Main execution function for Cursor Agent grid calibration"""
    print("=== Cursor Agent Grid Calibration Tool ===")
    print("Agent: Cursor Agent")
    print("Purpose: Visual validation and manual parameter tuning")
    print()
    
    # Configuration
    image_path = "satoshi_poster.png"
    config_path = "binary_extractor/cfg.yaml"
    
    try:
        # Initialize calibrator
        calibrator = CursorGridCalibrator(image_path, config_path)
        
        # Generate initial grid overlay with current parameters
        print("Cursor Agent: Generating initial grid overlay...")
        calibrator.save_grid_overlay("cursor_initial_grid_overlay.png")
        
        # Test parameter combinations
        print("Cursor Agent: Testing parameter combinations...")
        calibrator.test_parameter_combinations()
        
        print("\n=== CURSOR AGENT CALIBRATION COMPLETE ===")
        print("Next steps for visual validation:")
        print("1. Review cursor_initial_grid_overlay.png")
        print("2. Inspect all generated grid overlays")
        print("3. Identify best parameter combination visually")
        print("4. Update config with optimized parameters")
        print("5. Coordinate with Claude and Codex agents")
        
    except Exception as e:
        print(f"Cursor Agent: Error during calibration: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 