#!/usr/bin/env python3
"""
Verify the binary reconstruction process - check if we're correctly 
interpreting the spatial grid data vs linear binary string.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import os

class BinaryReconstructionVerifier:
    """Verify how we're reconstructing the binary data."""
    
    def __init__(self, cells_csv_path, image_path):
        self.cells_csv_path = cells_csv_path
        self.image_path = image_path
        self.cells_df = pd.read_csv(cells_csv_path)
        
        # Load original image for visual comparison
        self.original_image = cv2.imread(image_path)
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        print(f"Loaded {len(self.cells_df)} total cells")
        
    def analyze_spatial_distribution(self):
        """Analyze the actual spatial distribution in the grid."""
        
        # Get binary cells
        binary_cells = self.cells_df[self.cells_df['bit'].isin(['0', '1'])]
        
        print(f"Binary cells: {len(binary_cells)}")
        print(f"Zeros: {len(binary_cells[binary_cells['bit'] == '0'])}")
        print(f"Ones: {len(binary_cells[binary_cells['bit'] == '1'])}")
        
        # Create 2D grid
        grid = np.full((54, 50), -1, dtype=int)  # -1 for missing
        
        for _, row in binary_cells.iterrows():
            r, c = row['row'], row['col']
            if 0 <= r < 54 and 0 <= c < 50:
                grid[r][c] = int(row['bit'])
        
        # Analyze spatial patterns
        print("\\nSpatial Analysis:")
        print(f"Grid shape: {grid.shape}")
        print(f"Total cells: {54 * 50}")
        print(f"Filled cells: {np.sum(grid != -1)}")
        print(f"Empty cells: {np.sum(grid == -1)}")
        print(f"Zero cells: {np.sum(grid == 0)}")
        print(f"One cells: {np.sum(grid == 1)}")
        
        # Check distribution by row
        print("\\nRow-by-row distribution:")
        for r in range(min(10, 54)):  # First 10 rows
            row_data = grid[r]
            zeros = np.sum(row_data == 0)
            ones = np.sum(row_data == 1)
            empty = np.sum(row_data == -1)
            print(f"Row {r:2d}: {zeros:2d} zeros, {ones:2d} ones, {empty:2d} empty")
        
        # Check distribution by column
        print("\\nColumn-by-column distribution (first 10):")
        for c in range(min(10, 50)):  # First 10 columns
            col_data = grid[:, c]
            zeros = np.sum(col_data == 0)
            ones = np.sum(col_data == 1)
            empty = np.sum(col_data == -1)
            print(f"Col {c:2d}: {zeros:2d} zeros, {ones:2d} ones, {empty:2d} empty")
        
        return grid
    
    def compare_reconstruction_methods(self):
        """Compare different ways of reconstructing the binary string."""
        
        binary_cells = self.cells_df[self.cells_df['bit'].isin(['0', '1'])].copy()
        
        # Method 1: Row-major order (row by row, left to right)
        binary_cells_row_major = binary_cells.sort_values(['row', 'col'])
        binary_string_row_major = ''.join(binary_cells_row_major['bit'].values)
        
        # Method 2: Column-major order (column by column, top to bottom)
        binary_cells_col_major = binary_cells.sort_values(['col', 'row'])
        binary_string_col_major = ''.join(binary_cells_col_major['bit'].values)
        
        # Method 3: Sequential order (as they appear in the CSV)
        binary_string_sequential = ''.join(binary_cells['bit'].values)
        
        print("\\nReconstruction Method Comparison:")
        print(f"Row-major order: {len(binary_string_row_major)} bits")
        print(f"  First 50 bits: {binary_string_row_major[:50]}")
        print(f"  Zeros: {binary_string_row_major.count('0')} ({binary_string_row_major.count('0')/len(binary_string_row_major)*100:.1f}%)")
        print(f"  Ones: {binary_string_row_major.count('1')} ({binary_string_row_major.count('1')/len(binary_string_row_major)*100:.1f}%)")
        
        print(f"\\nColumn-major order: {len(binary_string_col_major)} bits")
        print(f"  First 50 bits: {binary_string_col_major[:50]}")
        print(f"  Zeros: {binary_string_col_major.count('0')} ({binary_string_col_major.count('0')/len(binary_string_col_major)*100:.1f}%)")
        print(f"  Ones: {binary_string_col_major.count('1')} ({binary_string_col_major.count('1')/len(binary_string_col_major)*100:.1f}%)")
        
        print(f"\\nSequential order: {len(binary_string_sequential)} bits")
        print(f"  First 50 bits: {binary_string_sequential[:50]}")
        print(f"  Zeros: {binary_string_sequential.count('0')} ({binary_string_sequential.count('0')/len(binary_string_sequential)*100:.1f}%)")
        print(f"  Ones: {binary_string_sequential.count('1')} ({binary_string_sequential.count('1')/len(binary_string_sequential)*100:.1f}%)")
        
        # Check for long runs in each method
        print("\\nLong runs analysis:")
        for method_name, binary_string in [
            ("Row-major", binary_string_row_major),
            ("Column-major", binary_string_col_major),
            ("Sequential", binary_string_sequential)
        ]:
            max_zeros = self.find_longest_run(binary_string, '0')
            max_ones = self.find_longest_run(binary_string, '1')
            print(f"  {method_name}: max zeros = {max_zeros}, max ones = {max_ones}")
        
        return {
            'row_major': binary_string_row_major,
            'col_major': binary_string_col_major,
            'sequential': binary_string_sequential
        }
    
    def find_longest_run(self, binary_string, char):
        """Find longest consecutive run of character."""
        max_run = 0
        current_run = 0
        
        for c in binary_string:
            if c == char:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        
        return max_run
    
    def visualize_grid_patterns(self, grid):
        """Create visualization of the grid patterns."""
        
        # Create output directory
        os.makedirs("test_results/verification", exist_ok=True)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Raw grid data
        display_grid = grid.copy().astype(float)
        display_grid[display_grid == -1] = 0.5  # Gray for missing
        
        im1 = axes[0, 0].imshow(display_grid, cmap='RdBu', vmin=0, vmax=1)
        axes[0, 0].set_title('Binary Grid (Blue=0, Red=1, Gray=Missing)')
        axes[0, 0].set_xlabel('Column')
        axes[0, 0].set_ylabel('Row')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Plot 2: Density heatmap by region
        region_size = 5
        density_grid = np.zeros((54//region_size, 50//region_size))
        
        for r in range(0, 54, region_size):
            for c in range(0, 50, region_size):
                region = grid[r:r+region_size, c:c+region_size]
                binary_cells = region[region != -1]
                if len(binary_cells) > 0:
                    density_grid[r//region_size, c//region_size] = np.mean(binary_cells)
        
        im2 = axes[0, 1].imshow(density_grid, cmap='RdBu', vmin=0, vmax=1)
        axes[0, 1].set_title('Regional Density (5x5 blocks)')
        axes[0, 1].set_xlabel('Column Block')
        axes[0, 1].set_ylabel('Row Block')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Plot 3: Row sums
        row_sums = []
        for r in range(54):
            row_data = grid[r]
            binary_cells = row_data[row_data != -1]
            if len(binary_cells) > 0:
                row_sums.append(np.mean(binary_cells))
            else:
                row_sums.append(0)
        
        axes[1, 0].plot(row_sums)
        axes[1, 0].set_title('Average Bit Value by Row')
        axes[1, 0].set_xlabel('Row')
        axes[1, 0].set_ylabel('Average Bit Value')
        axes[1, 0].grid(True)
        
        # Plot 4: Column sums
        col_sums = []
        for c in range(50):
            col_data = grid[:, c]
            binary_cells = col_data[col_data != -1]
            if len(binary_cells) > 0:
                col_sums.append(np.mean(binary_cells))
            else:
                col_sums.append(0)
        
        axes[1, 1].plot(col_sums)
        axes[1, 1].set_title('Average Bit Value by Column')
        axes[1, 1].set_xlabel('Column')
        axes[1, 1].set_ylabel('Average Bit Value')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig("test_results/verification/grid_visualization.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print("\\nVisualization saved to: test_results/verification/grid_visualization.png")
    
    def check_original_image_regions(self):
        """Check specific regions of the original image."""
        
        # Grid parameters
        row_pitch = 31
        col_pitch = 25
        row0 = 1
        col0 = 5
        
        print("\\nSampling original image at grid positions:")
        
        # Sample first few rows
        for r in range(3):
            row_bits = []
            for c in range(10):  # First 10 columns
                center_y = row0 + r * row_pitch
                center_x = col0 + c * col_pitch
                
                # Extract small region around center
                y1 = max(0, center_y - row_pitch//4)
                y2 = min(self.gray_image.shape[0], center_y + row_pitch//4)
                x1 = max(0, center_x - col_pitch//4)
                x2 = min(self.gray_image.shape[1], center_x + col_pitch//4)
                
                if y2 > y1 and x2 > x1:
                    cell_region = self.gray_image[y1:y2, x1:x2]
                    mean_intensity = np.mean(cell_region)
                    predicted_bit = '1' if mean_intensity > 127 else '0'
                    row_bits.append(predicted_bit)
                else:
                    row_bits.append('?')
            
            print(f"Row {r}: {''.join(row_bits)}")
        
        # Compare with CSV data
        print("\\nFrom CSV data:")
        for r in range(3):
            row_data = self.cells_df[self.cells_df['row'] == r].sort_values('col')
            csv_bits = []
            for c in range(10):
                cell_data = row_data[row_data['col'] == c]
                if not cell_data.empty:
                    bit = cell_data.iloc[0]['bit']
                    csv_bits.append(bit if bit in ['0', '1'] else '?')
                else:
                    csv_bits.append('?')
            print(f"Row {r}: {''.join(csv_bits)}")
    
    def comprehensive_verification(self):
        """Run comprehensive verification."""
        
        print("=== Binary Reconstruction Verification ===")
        
        # Step 1: Analyze spatial distribution
        print("\\n1. Spatial Distribution Analysis")
        grid = self.analyze_spatial_distribution()
        
        # Step 2: Compare reconstruction methods
        print("\\n2. Reconstruction Method Comparison")
        reconstructions = self.compare_reconstruction_methods()
        
        # Step 3: Visualize patterns
        print("\\n3. Creating Visualizations")
        self.visualize_grid_patterns(grid)
        
        # Step 4: Check original image
        print("\\n4. Original Image Verification")
        self.check_original_image_regions()
        
        # Step 5: Summary
        print("\\n5. Summary and Recommendations")
        print("\\nKey Findings:")
        print(f"- Total extracted bits: {len(self.cells_df[self.cells_df['bit'].isin(['0', '1'])])}")
        print(f"- Grid completeness: {(len(self.cells_df[self.cells_df['bit'].isin(['0', '1'])]) / (54*50)) * 100:.1f}%")
        
        # Check if the massive zero run is real
        for method_name, binary_string in reconstructions.items():
            max_zeros = self.find_longest_run(binary_string, '0')
            max_ones = self.find_longest_run(binary_string, '1')
            print(f"- {method_name}: longest zero run = {max_zeros}, longest one run = {max_ones}")
        
        return grid, reconstructions

def main():
    """Main verification function."""
    
    cells_csv = "output_final/cells.csv"
    image_path = "satoshi (1).png"
    
    if not os.path.exists(cells_csv) or not os.path.exists(image_path):
        print("Error: Required files not found")
        return
    
    # Initialize verifier
    verifier = BinaryReconstructionVerifier(cells_csv, image_path)
    
    # Run verification
    grid, reconstructions = verifier.comprehensive_verification()
    
    print("\\n=== Verification Complete ===")
    print("Check the visualization to see the actual spatial distribution.")
    print("The massive zero run might be an artifact of the reconstruction method.")

if __name__ == "__main__":
    main()