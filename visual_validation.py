#!/usr/bin/env python3
"""
Visual validation of extraction accuracy by comparing extracted bits 
with actual visual inspection of the poster image.
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

class VisualValidator:
    """Validate extraction accuracy through visual inspection."""
    
    def __init__(self, image_path, cells_csv_path):
        self.image_path = image_path
        self.cells_csv_path = cells_csv_path
        
        # Load image
        self.original = cv2.imread(image_path)
        self.rgb = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)
        self.gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        
        # Load extraction results
        self.cells_df = pd.read_csv(cells_csv_path)
        
        # Grid parameters from extraction
        self.row_pitch = 31
        self.col_pitch = 25
        self.row0 = 1
        self.col0 = 5
        
        print(f"Image shape: {self.original.shape}")
        print(f"Grid parameters: pitch=({self.row_pitch}, {self.col_pitch}), origin=({self.row0}, {self.col0})")
        
    def extract_cell_visual(self, row, col):
        """Extract a cell region from the image and classify it visually."""
        
        # Calculate cell center
        center_y = self.row0 + row * self.row_pitch
        center_x = self.col0 + col * self.col_pitch
        
        # Define cell boundaries (smaller region for precise analysis)
        cell_size = min(self.row_pitch, self.col_pitch) // 3
        y1 = max(0, center_y - cell_size // 2)
        y2 = min(self.gray.shape[0], center_y + cell_size // 2)
        x1 = max(0, center_x - cell_size // 2)
        x2 = min(self.gray.shape[1], center_x + cell_size // 2)
        
        if y2 <= y1 or x2 <= x1:
            return None, None
        
        # Extract cell region
        cell_region = self.gray[y1:y2, x1:x2]
        rgb_region = self.rgb[y1:y2, x1:x2]
        
        # Simple visual classification
        mean_intensity = np.mean(cell_region)
        
        # Use a simple threshold - you can adjust this based on visual inspection
        visual_bit = '1' if mean_intensity > 127 else '0'
        
        return cell_region, visual_bit
    
    def compare_sample_regions(self, num_samples=20):
        """Compare extracted bits with visual inspection for sample regions."""
        
        print(f"Comparing {num_samples} sample regions...")
        
        # Get random sample of extracted cells
        binary_cells = self.cells_df[self.cells_df['bit'].isin(['0', '1'])]
        
        if len(binary_cells) < num_samples:
            sample_cells = binary_cells
        else:
            sample_cells = binary_cells.sample(n=num_samples, random_state=42)
        
        comparisons = []
        
        for _, cell in sample_cells.iterrows():
            row, col = cell['row'], cell['col']
            extracted_bit = cell['bit']
            
            # Get visual classification
            cell_region, visual_bit = self.extract_cell_visual(row, col)
            
            if cell_region is not None and visual_bit is not None:
                agreement = (extracted_bit == visual_bit)
                
                comparisons.append({
                    'row': row,
                    'col': col,
                    'extracted': extracted_bit,
                    'visual': visual_bit,
                    'agreement': agreement,
                    'mean_intensity': np.mean(cell_region)
                })
        
        return comparisons
    
    def create_detailed_inspection_grid(self, start_row=0, start_col=0, grid_size=10):
        """Create a detailed visual inspection grid for manual verification."""
        
        os.makedirs("test_results/visual_validation", exist_ok=True)
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
        fig.suptitle(f'Visual Inspection Grid (starting at row {start_row}, col {start_col})')
        
        for i in range(grid_size):
            for j in range(grid_size):
                row = start_row + i
                col = start_col + j
                
                if row < 54 and col < 50:
                    # Extract cell region
                    cell_region, visual_bit = self.extract_cell_visual(row, col)
                    
                    if cell_region is not None:
                        # Show the cell region
                        axes[i, j].imshow(cell_region, cmap='gray')
                        
                        # Get extracted bit from CSV
                        cell_data = self.cells_df[(self.cells_df['row'] == row) & 
                                                (self.cells_df['col'] == col)]
                        
                        if not cell_data.empty:
                            extracted_bit = cell_data.iloc[0]['bit']
                            
                            # Color code the title based on agreement
                            if extracted_bit == visual_bit:
                                title_color = 'green'
                            elif extracted_bit in ['0', '1']:
                                title_color = 'red'
                            else:
                                title_color = 'orange'
                            
                            axes[i, j].set_title(f'({row},{col})\\nCSV:{extracted_bit} Visual:{visual_bit}', 
                                                color=title_color, fontsize=8)
                        else:
                            axes[i, j].set_title(f'({row},{col})\\nCSV:missing Visual:{visual_bit}', 
                                                color='purple', fontsize=8)
                    else:
                        axes[i, j].set_title(f'({row},{col})\\nOut of bounds', fontsize=8)
                        axes[i, j].set_facecolor('lightgray')
                else:
                    axes[i, j].set_title(f'({row},{col})\\nOut of grid', fontsize=8)
                    axes[i, j].set_facecolor('lightgray')
                
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
        
        plt.tight_layout()
        plt.savefig(f"test_results/visual_validation/inspection_grid_{start_row}_{start_col}.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Inspection grid saved: test_results/visual_validation/inspection_grid_{start_row}_{start_col}.png")
    
    def check_grid_alignment(self):
        """Check if the grid parameters are correctly aligned with the image."""
        
        print("Checking grid alignment...")
        
        # Create overlay image showing grid positions
        overlay = self.rgb.copy()
        
        # Draw grid points
        for row in range(0, 54, 5):  # Every 5th row
            for col in range(0, 50, 5):  # Every 5th column
                center_y = self.row0 + row * self.row_pitch
                center_x = self.col0 + col * self.col_pitch
                
                if 0 <= center_y < overlay.shape[0] and 0 <= center_x < overlay.shape[1]:
                    # Draw cross at grid position
                    cv2.drawMarker(overlay, (center_x, center_y), (255, 0, 0), 
                                 markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
        
        # Save overlay
        os.makedirs("test_results/visual_validation", exist_ok=True)
        
        plt.figure(figsize=(15, 12))
        plt.imshow(overlay)
        plt.title('Grid Alignment Check (Red crosses show grid positions)')
        plt.axis('off')
        plt.savefig("test_results/visual_validation/grid_alignment.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print("Grid alignment saved: test_results/visual_validation/grid_alignment.png")
    
    def manual_inspection_report(self, comparisons):
        """Generate a manual inspection report."""
        
        if not comparisons:
            return "No valid comparisons available."
        
        # Calculate accuracy
        total_comparisons = len(comparisons)
        agreements = sum(1 for c in comparisons if c['agreement'])
        accuracy = (agreements / total_comparisons) * 100
        
        report = []
        report.append("# Visual Validation Report")
        report.append("## Manual Inspection Results")
        report.append("")
        report.append(f"**Total Comparisons**: {total_comparisons}")
        report.append(f"**Agreements**: {agreements}")
        report.append(f"**Accuracy**: {accuracy:.1f}%")
        report.append("")
        
        # Show disagreements
        disagreements = [c for c in comparisons if not c['agreement']]
        
        if disagreements:
            report.append("## Disagreements (CSV vs Visual)")
            report.append("")
            report.append("| Row | Col | CSV | Visual | Mean Intensity |")
            report.append("|-----|-----|-----|--------|----------------|")
            
            for d in disagreements:
                report.append(f"| {d['row']} | {d['col']} | {d['extracted']} | {d['visual']} | {d['mean_intensity']:.1f} |")
            
            report.append("")
        
        # Analysis
        report.append("## Analysis")
        report.append("")
        
        if accuracy < 70:
            report.append("❌ **CRITICAL**: Extraction accuracy is very low. Grid parameters or extraction method needs major revision.")
        elif accuracy < 85:
            report.append("⚠️ **WARNING**: Extraction accuracy is moderate. Some parameters may need adjustment.")
        else:
            report.append("✅ **GOOD**: Extraction accuracy is acceptable.")
        
        report.append("")
        report.append("## Recommendations")
        report.append("")
        
        if accuracy < 85:
            report.append("1. **Check grid parameters**: Verify row_pitch, col_pitch, row0, col0")
            report.append("2. **Adjust thresholds**: Review bit classification thresholds")
            report.append("3. **Validate grid alignment**: Check if grid overlay matches actual patterns")
            report.append("4. **Manual inspection**: Review the inspection grid images")
        
        return "\\n".join(report)
    
    def comprehensive_validation(self):
        """Run comprehensive visual validation."""
        
        print("=== Visual Validation ===")
        
        # Step 1: Check grid alignment
        print("\\n1. Checking grid alignment...")
        self.check_grid_alignment()
        
        # Step 2: Create inspection grids for different regions
        print("\\n2. Creating inspection grids...")
        
        # Top-left region
        self.create_detailed_inspection_grid(0, 0, 10)
        
        # Middle region
        self.create_detailed_inspection_grid(20, 20, 10)
        
        # Bottom-right region
        self.create_detailed_inspection_grid(40, 35, 10)
        
        # Step 3: Compare sample regions
        print("\\n3. Comparing sample regions...")
        comparisons = self.compare_sample_regions(50)
        
        # Step 4: Generate report
        print("\\n4. Generating validation report...")
        report = self.manual_inspection_report(comparisons)
        
        # Save report
        with open("test_results/visual_validation/VISUAL_VALIDATION_REPORT.md", 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\\n=== Visual Validation Complete ===")
        print(f"Check the inspection grid images to manually verify extraction accuracy.")
        print(f"Report saved to: test_results/visual_validation/VISUAL_VALIDATION_REPORT.md")
        
        return comparisons

def main():
    """Main validation function."""
    
    image_path = "satoshi (1).png"
    cells_csv = "output_final/cells.csv"
    
    if not os.path.exists(image_path) or not os.path.exists(cells_csv):
        print("Error: Required files not found")
        return
    
    # Initialize validator
    validator = VisualValidator(image_path, cells_csv)
    
    # Run validation
    comparisons = validator.comprehensive_validation()
    
    print("\\nIMPORTANT: Please manually inspect the generated grid images to verify extraction accuracy.")
    print("The grid alignment image will show if our grid parameters are correct.")
    print("The inspection grids will show individual cell comparisons.")

if __name__ == "__main__":
    main()