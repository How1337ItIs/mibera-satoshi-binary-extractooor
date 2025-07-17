#!/usr/bin/env python3
"""
Visual Validation Tool for Corrected Extraction Method
Agent: Cursor Agent
Purpose: Visual validation and manual parameter tuning for the corrected 15×12 grid method
Date: 2025-07-16
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

class CursorVisualValidator:
    """
    Visual validation tool for Cursor Agent
    Purpose: Visual validation and manual parameter tuning
    """
    
    def __init__(self, image_path: str = "satoshi (1).png"):
        self.image_path = image_path
        self.image = None
        self.validation_results = []
        
        self.load_image()
        
    def load_image(self):
        """Load and validate poster image"""
        if not Path(self.image_path).exists():
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise ValueError(f"Failed to load image: {self.image_path}")
        
        print(f"Cursor Agent: Loaded image: {self.image.shape[1]}x{self.image.shape[0]} pixels")
        
    def generate_corrected_grid_overlay(self, row_pitch: int = 15, col_pitch: int = 12, 
                                      row0: int = 890, col0: int = 185) -> np.ndarray:
        """
        Generate grid overlay using Claude's corrected parameters
        
        Agent: Cursor Agent
        Purpose: Visual validation and manual parameter tuning
        """
        height, width = self.image.shape[:2]
        
        # Create overlay image
        overlay = self.image.copy()
        
        # Draw horizontal grid lines using corrected parameters
        for row in range(0, height, row_pitch):
            y = row0 + row
            if 0 <= y < height:
                cv2.line(overlay, (0, y), (width, y), (0, 255, 0), 2)
        
        # Draw vertical grid lines using corrected parameters
        for col in range(0, width, col_pitch):
            x = col0 + col
            if 0 <= x < width:
                cv2.line(overlay, (x, 0), (x, height), (0, 255, 0), 2)
        
        return overlay
    
    def save_corrected_grid_overlay(self, output_path: str = "cursor_corrected_grid_overlay.png"):
        """
        Save corrected grid overlay for visual inspection
        
        Agent: Cursor Agent
        Purpose: Visual validation and manual parameter tuning
        """
        overlay = self.generate_corrected_grid_overlay()
        
        # Convert BGR to RGB for matplotlib
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
        # Create figure with corrected grid overlay
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.imshow(overlay_rgb)
        ax.set_title("Cursor Agent: Corrected Grid Overlay (15×12 pitch)\n"
                    f"Origin: (890, 185) - Claude's Working Parameters")
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Cursor Agent: Corrected grid overlay saved to: {output_path}")
        
    def test_parameter_variations(self):
        """
        Test slight variations around Claude's working parameters
        
        Agent: Cursor Agent
        Purpose: Visual validation and manual parameter tuning
        """
        print("Cursor Agent: Testing parameter variations around Claude's working solution...")
        
        # Claude's working parameters
        base_row_pitch = 15
        base_col_pitch = 12
        base_row0 = 890
        base_col0 = 185
        
        # Test variations
        pitch_variations = [14, 15, 16]  # Small variations around 15
        origin_variations = [(885, 180), (890, 185), (895, 190)]  # Small variations
        
        for row_pitch in pitch_variations:
            for col_pitch in pitch_variations:
                for row0, col0 in origin_variations:
                    output_path = f"cursor_variation_r{row_pitch}_c{col_pitch}_r0{row0}_c0{col0}.png"
                    overlay = self.generate_corrected_grid_overlay(row_pitch, col_pitch, row0, col0)
                    
                    # Convert and save
                    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
                    ax.imshow(overlay_rgb)
                    ax.set_title(f"Cursor Agent: Parameter Variation\n"
                               f"Pitch: ({row_pitch}, {col_pitch}), Origin: ({row0}, {col0})")
                    ax.axis('off')
                    plt.tight_layout()
                    plt.savefig(output_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    # Log variation
                    self.log_parameter_variation(row_pitch, col_pitch, row0, col0, output_path)
        
        print(f"Cursor Agent: Generated parameter variation overlays for visual inspection")
        
    def log_parameter_variation(self, row_pitch: int, col_pitch: int, row0: int, col0: int, output_path: str):
        """
        Log parameter variation for tracking
        
        Agent: Cursor Agent
        Purpose: Visual validation and manual parameter tuning
        """
        variation_data = {
            "timestamp": datetime.now().isoformat(),
            "agent": "cursor",
            "parameters": {
                "row_pitch": row_pitch,
                "col_pitch": col_pitch,
                "row0": row0,
                "col0": col0
            },
            "output_file": output_path,
            "base_comparison": {
                "claude_row_pitch": 15,
                "claude_col_pitch": 12,
                "claude_row0": 890,
                "claude_col0": 185
            },
            "visual_notes": "Generated for manual inspection and comparison",
            "next_action": "Manual visual verification required"
        }
        
        self.validation_results.append(variation_data)
        
        # Save to JSON log
        with open("cursor_validation_log.json", "w") as f:
            json.dump(self.validation_results, f, indent=2)
            
    def create_visual_comparison(self):
        """
        Create visual comparison of Claude's working method vs variations
        
        Agent: Cursor Agent
        Purpose: Visual validation and manual parameter tuning
        """
        # Claude's working parameters
        claude_overlay = self.generate_corrected_grid_overlay(15, 12, 890, 185)
        
        # Test variation (slightly different)
        variation_overlay = self.generate_corrected_grid_overlay(14, 12, 885, 180)
        
        # Create comparison figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Claude's method
        claude_rgb = cv2.cvtColor(claude_overlay, cv2.COLOR_BGR2RGB)
        ax1.imshow(claude_rgb)
        ax1.set_title("Cursor Agent: Claude's Working Method\n"
                     f"Pitch: (15, 12), Origin: (890, 185)")
        ax1.axis('off')
        
        # Variation
        variation_rgb = cv2.cvtColor(variation_overlay, cv2.COLOR_BGR2RGB)
        ax2.imshow(variation_rgb)
        ax2.set_title("Cursor Agent: Parameter Variation\n"
                     f"Pitch: (14, 12), Origin: (885, 180)")
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig("cursor_parameter_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print("Cursor Agent: Parameter comparison saved to: cursor_parameter_comparison.png")
        
    def validate_corrected_extraction_region(self):
        """
        Validate the specific region where Claude's method works
        
        Agent: Cursor Agent
        Purpose: Visual validation and manual parameter tuning
        """
        print("Cursor Agent: Validating Claude's corrected extraction region...")
        
        # Extract the region where Claude's method works
        row0, col0 = 890, 185
        row_pitch, col_pitch = 15, 12
        
        # Extract a 5x5 test region
        test_cells = []
        for r in range(5):
            for c in range(5):
                y = row0 + r * row_pitch
                x = col0 + c * col_pitch
                
                # Check bounds
                if y >= self.image.shape[0] or x >= self.image.shape[1]:
                    continue
                    
                # Extract cell
                cell = self.image[max(0, y-3):min(self.image.shape[0], y+4), 
                                max(0, x-3):min(self.image.shape[1], x+4)]
                
                if cell.size == 0:
                    continue
                
                # Save cell for visual inspection
                enlarged = cv2.resize(cell, (60, 60), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(f'cursor_validation_cell_{r}_{c}.png', enlarged)
                
                # Simple classification (same as Claude's method)
                blue_channel = cell[:, :, 0]
                avg_blue = np.mean(blue_channel)
                
                if avg_blue > 150:
                    bit = '0'
                elif avg_blue < 100:
                    bit = '1'
                else:
                    bit = 'ambiguous'
                
                test_cells.append((r, c, bit, avg_blue))
        
        # Analyze validation results
        zeros = sum(1 for _, _, bit, _ in test_cells if bit == '0')
        ones = sum(1 for _, _, bit, _ in test_cells if bit == '1')
        ambiguous = sum(1 for _, _, bit, _ in test_cells if bit == 'ambiguous')
        
        print(f"Cursor Agent: Validation Results (5x5 region):")
        print(f"  Zeros: {zeros}")
        print(f"  Ones: {ones}")
        print(f"  Ambiguous: {ambiguous}")
        print(f"  Clear classifications: {zeros + ones}/{len(test_cells)} ({(zeros + ones)/len(test_cells)*100:.1f}%)")
        
        # Save validation results
        validation_data = {
            "timestamp": datetime.now().isoformat(),
            "agent": "cursor",
            "validation_type": "corrected_extraction_region",
            "parameters": {
                "row_pitch": row_pitch,
                "col_pitch": col_pitch,
                "row0": row0,
                "col0": col0
            },
            "results": {
                "total_cells": len(test_cells),
                "zeros": zeros,
                "ones": ones,
                "ambiguous": ambiguous,
                "clear_classifications": zeros + ones,
                "clear_percentage": (zeros + ones)/len(test_cells)*100 if test_cells else 0
            },
            "visual_evidence": [f"cursor_validation_cell_{r}_{c}.png" for r in range(5) for c in range(5)]
        }
        
        with open("cursor_validation_results.json", "w") as f:
            json.dump(validation_data, f, indent=2)
        
        print(f"Cursor Agent: Validation results saved to cursor_validation_results.json")

def main():
    """Main execution function for Cursor Agent visual validation"""
    print("=== Cursor Agent Visual Validation Tool ===")
    print("Agent: Cursor Agent")
    print("Purpose: Visual validation and manual parameter tuning")
    print("Focus: Validate Claude's corrected extraction method")
    print()
    
    try:
        # Initialize validator
        validator = CursorVisualValidator()
        
        # Generate corrected grid overlay
        print("Cursor Agent: Generating corrected grid overlay...")
        validator.save_corrected_grid_overlay()
        
        # Test parameter variations
        print("Cursor Agent: Testing parameter variations...")
        validator.test_parameter_variations()
        
        # Create visual comparison
        print("Cursor Agent: Creating parameter comparison...")
        validator.create_visual_comparison()
        
        # Validate corrected extraction region
        print("Cursor Agent: Validating corrected extraction region...")
        validator.validate_corrected_extraction_region()
        
        print("\n=== CURSOR AGENT VALIDATION COMPLETE ===")
        print("Next steps for visual validation:")
        print("1. Review cursor_corrected_grid_overlay.png")
        print("2. Inspect parameter variation overlays")
        print("3. Compare cursor_parameter_comparison.png")
        print("4. Validate cursor_validation_cell_*.png images")
        print("5. Coordinate findings with Claude Code Agent")
        
    except Exception as e:
        print(f"Cursor Agent: Error during validation: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 