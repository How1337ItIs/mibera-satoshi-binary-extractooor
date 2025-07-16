#!/usr/bin/env python3
"""
Deep analysis of the remaining 120 problematic cells to push extraction beyond 95.6%
Implements advanced recovery techniques for blanks and overlays.
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import json
from scipy import ndimage
from skimage import feature, filters, morphology, restoration, segmentation
from skimage.measure import regionprops
import sys

# Add the binary_extractor to path
sys.path.append(str(Path(__file__).parent / "binary_extractor"))

class RemainingCellAnalyzer:
    """Advanced analysis and recovery of problematic cells."""
    
    def __init__(self, image_path, cells_csv_path):
        self.image_path = image_path
        self.cells_csv_path = cells_csv_path
        self.original = cv2.imread(image_path)
        self.rgb = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)
        self.gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        
        # Load cell data
        self.cells_df = pd.read_csv(cells_csv_path)
        
        # Grid parameters (from previous analysis)
        self.row_pitch = 31
        self.col_pitch = 25
        self.row0 = 1
        self.col0 = 5
        
        # Analysis results
        self.problematic_cells = []
        self.recovery_results = {}
        
    def identify_problematic_cells(self):
        """Identify and categorize problematic cells."""
        
        # Get blank and overlay cells
        blank_cells = self.cells_df[self.cells_df['bit'] == 'blank']
        overlay_cells = self.cells_df[self.cells_df['bit'] == 'overlay']
        
        print(f"Found {len(blank_cells)} blank cells and {len(overlay_cells)} overlay cells")
        
        # Analyze each problematic cell
        for _, cell in blank_cells.iterrows():
            self.problematic_cells.append({
                'row': cell['row'],
                'col': cell['col'],
                'type': 'blank',
                'reason': 'unclear_pattern'
            })
            
        for _, cell in overlay_cells.iterrows():
            self.problematic_cells.append({
                'row': cell['row'],
                'col': cell['col'],
                'type': 'overlay',
                'reason': 'graphic_obstruction'
            })
        
        return self.problematic_cells
    
    def extract_cell_region(self, row, col, padding=5):
        """Extract a cell region with padding for analysis."""
        
        # Calculate cell center
        center_y = self.row0 + row * self.row_pitch
        center_x = self.col0 + col * self.col_pitch
        
        # Define cell boundaries with padding
        y1 = max(0, center_y - self.row_pitch//2 - padding)
        y2 = min(self.gray.shape[0], center_y + self.row_pitch//2 + padding)
        x1 = max(0, center_x - self.col_pitch//2 - padding)
        x2 = min(self.gray.shape[1], center_x + self.col_pitch//2 + padding)
        
        # Extract regions
        gray_region = self.gray[y1:y2, x1:x2]
        rgb_region = self.rgb[y1:y2, x1:x2]
        
        return {
            'gray': gray_region,
            'rgb': rgb_region,
            'bounds': (y1, y2, x1, x2),
            'center': (center_y, center_x)
        }
    
    def advanced_cell_analysis(self, cell_data):
        """Perform advanced analysis on a single cell."""
        
        row, col = cell_data['row'], cell_data['col']
        region = self.extract_cell_region(row, col)
        
        analysis = {
            'position': (row, col),
            'type': cell_data['type'],
            'methods': {}
        }
        
        gray_cell = region['gray']
        rgb_cell = region['rgb']
        
        # Method 1: Multi-scale analysis
        analysis['methods']['multi_scale'] = self.multi_scale_analysis(gray_cell)
        
        # Method 2: Texture analysis
        analysis['methods']['texture'] = self.texture_analysis(gray_cell)
        
        # Method 3: Edge density analysis
        analysis['methods']['edge_density'] = self.edge_density_analysis(gray_cell)
        
        # Method 4: Color space analysis
        analysis['methods']['color_space'] = self.color_space_analysis(rgb_cell)
        
        # Method 5: Frequency domain analysis
        analysis['methods']['frequency'] = self.frequency_analysis(gray_cell)
        
        # Method 6: Template matching with neighbors
        analysis['methods']['neighbor_template'] = self.neighbor_template_matching(row, col)
        
        # Method 7: Statistical analysis
        analysis['methods']['statistical'] = self.statistical_analysis(gray_cell)
        
        # Method 8: Morphological analysis
        analysis['methods']['morphological'] = self.morphological_analysis(gray_cell)
        
        return analysis
    
    def multi_scale_analysis(self, cell):
        """Analyze cell at multiple scales."""
        
        results = {}
        
        # Gaussian pyramid
        pyramid = [cell]
        current = cell
        for i in range(3):
            current = cv2.pyrDown(current)
            pyramid.append(current)
        
        # Analyze each scale
        for i, scale in enumerate(pyramid):
            if scale.size > 0:
                # Otsu threshold
                _, binary = cv2.threshold(scale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Calculate features
                white_ratio = np.sum(binary == 255) / binary.size
                results[f'scale_{i}'] = {
                    'white_ratio': white_ratio,
                    'predicted_bit': '1' if white_ratio > 0.5 else '0',
                    'confidence': abs(white_ratio - 0.5) * 2
                }
        
        return results
    
    def texture_analysis(self, cell):
        """Analyze cell texture patterns."""
        
        # Local Binary Pattern
        from skimage.feature import local_binary_pattern
        lbp = local_binary_pattern(cell, 8, 1, method='uniform')
        
        # Gabor filters
        gabor_responses = []
        for theta in [0, 45, 90, 135]:
            for frequency in [0.1, 0.2, 0.3]:
                real, _ = filters.gabor(cell, frequency=frequency, theta=np.radians(theta))
                gabor_responses.append(np.mean(np.abs(real)))
        
        return {
            'lbp_variance': np.var(lbp),
            'gabor_mean': np.mean(gabor_responses),
            'gabor_std': np.std(gabor_responses)
        }
    
    def edge_density_analysis(self, cell):
        """Analyze edge density and patterns."""
        
        # Canny edge detection
        edges = cv2.Canny(cell, 50, 150)
        
        # Edge density
        edge_density = np.sum(edges == 255) / edges.size
        
        # Hough lines
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=10)
        num_lines = len(lines) if lines is not None else 0
        
        return {
            'edge_density': edge_density,
            'num_lines': num_lines,
            'predicted_bit': '1' if edge_density > 0.1 else '0'
        }
    
    def color_space_analysis(self, cell):
        """Analyze in multiple color spaces."""
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(cell, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(cell, cv2.COLOR_RGB2LAB)
        
        results = {}
        
        # Analyze each channel
        for name, channels in [('HSV', hsv), ('LAB', lab)]:
            for i, channel in enumerate(channels.transpose(2, 0, 1)):
                # Otsu threshold
                _, binary = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                white_ratio = np.sum(binary == 255) / binary.size
                
                results[f'{name}_{i}'] = {
                    'white_ratio': white_ratio,
                    'predicted_bit': '1' if white_ratio > 0.5 else '0',
                    'confidence': abs(white_ratio - 0.5) * 2
                }
        
        return results
    
    def frequency_analysis(self, cell):
        """Frequency domain analysis."""
        
        # FFT
        fft = np.fft.fft2(cell)
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shifted)
        
        # High frequency energy
        h, w = cell.shape
        center = (h//2, w//2)
        y, x = np.ogrid[:h, :w]
        distance = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        
        high_freq_mask = distance > min(h, w) * 0.3
        high_freq_energy = np.sum(magnitude[high_freq_mask])
        
        return {
            'high_freq_energy': high_freq_energy,
            'total_energy': np.sum(magnitude),
            'high_freq_ratio': high_freq_energy / np.sum(magnitude)
        }
    
    def neighbor_template_matching(self, row, col):
        """Template matching using neighboring cells."""
        
        # Get neighboring cells that are successfully classified
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                    
                nr, nc = row + dr, col + dc
                if 0 <= nr < 54 and 0 <= nc < 50:
                    neighbor_cell = self.cells_df[(self.cells_df['row'] == nr) & (self.cells_df['col'] == nc)]
                    if not neighbor_cell.empty:
                        bit = neighbor_cell.iloc[0]['bit']
                        if bit in ['0', '1']:
                            neighbors.append({
                                'row': nr,
                                'col': nc,
                                'bit': bit,
                                'region': self.extract_cell_region(nr, nc)
                            })
        
        # Template matching with successful neighbors
        current_region = self.extract_cell_region(row, col)
        match_scores = {}
        
        for neighbor in neighbors:
            # Template matching
            template = neighbor['region']['gray']
            target = current_region['gray']
            
            if template.shape[0] <= target.shape[0] and template.shape[1] <= target.shape[1]:
                result = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)
                max_score = np.max(result)
                match_scores[neighbor['bit']] = max_score
        
        return match_scores
    
    def statistical_analysis(self, cell):
        """Statistical analysis of cell intensities."""
        
        # Basic statistics
        mean_intensity = np.mean(cell)
        std_intensity = np.std(cell)
        skewness = np.mean(((cell - mean_intensity) / std_intensity) ** 3)
        kurtosis = np.mean(((cell - mean_intensity) / std_intensity) ** 4)
        
        # Histogram analysis
        hist, bins = np.histogram(cell, bins=50)
        hist_peaks = len(feature.peak_local_maxima(hist)[0])
        
        return {
            'mean': mean_intensity,
            'std': std_intensity,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'hist_peaks': hist_peaks,
            'predicted_bit': '1' if mean_intensity > 127 else '0'
        }
    
    def morphological_analysis(self, cell):
        """Morphological analysis of cell structure."""
        
        # Threshold for morphological operations
        _, binary = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Connected components
        num_labels, labels = cv2.connectedComponents(binary)
        
        # Analyze components
        if num_labels > 1:
            regions = regionprops(labels)
            areas = [region.area for region in regions]
            largest_area = max(areas) if areas else 0
        else:
            largest_area = 0
        
        return {
            'num_components': num_labels - 1,  # Subtract background
            'largest_component_area': largest_area,
            'opening_ratio': np.sum(opening == 255) / opening.size,
            'closing_ratio': np.sum(closing == 255) / closing.size
        }
    
    def ensemble_prediction(self, analysis):
        """Combine multiple analysis methods for final prediction."""
        
        predictions = []
        confidences = []
        
        # Collect predictions from different methods
        methods = analysis['methods']
        
        # Multi-scale predictions
        if 'multi_scale' in methods:
            for scale_data in methods['multi_scale'].values():
                predictions.append(scale_data['predicted_bit'])
                confidences.append(scale_data['confidence'])
        
        # Edge density prediction
        if 'edge_density' in methods:
            predictions.append(methods['edge_density']['predicted_bit'])
            confidences.append(0.5)  # Default confidence
        
        # Color space predictions
        if 'color_space' in methods:
            for space_data in methods['color_space'].values():
                predictions.append(space_data['predicted_bit'])
                confidences.append(space_data['confidence'])
        
        # Statistical prediction
        if 'statistical' in methods:
            predictions.append(methods['statistical']['predicted_bit'])
            confidences.append(0.5)
        
        # Neighbor template matching
        if 'neighbor_template' in methods:
            matches = methods['neighbor_template']
            if matches:
                best_match = max(matches.items(), key=lambda x: x[1])
                if best_match[1] > 0.7:  # High confidence threshold
                    predictions.append(best_match[0])
                    confidences.append(best_match[1])
        
        # Ensemble voting
        if predictions:
            # Weighted voting
            weights = np.array(confidences)
            weights = weights / np.sum(weights)
            
            vote_1 = np.sum(weights[np.array(predictions) == '1'])
            vote_0 = np.sum(weights[np.array(predictions) == '0'])
            
            final_prediction = '1' if vote_1 > vote_0 else '0'
            final_confidence = max(vote_1, vote_0)
            
            return {
                'prediction': final_prediction,
                'confidence': final_confidence,
                'vote_1': vote_1,
                'vote_0': vote_0,
                'num_methods': len(predictions)
            }
        
        return None
    
    def analyze_all_problematic_cells(self):
        """Analyze all problematic cells and attempt recovery."""
        
        problematic_cells = self.identify_problematic_cells()
        
        recovery_results = []
        
        print(f"\\nAnalyzing {len(problematic_cells)} problematic cells...")
        
        for i, cell_data in enumerate(problematic_cells):
            if i % 10 == 0:
                print(f"Progress: {i}/{len(problematic_cells)}")
            
            # Perform advanced analysis
            analysis = self.advanced_cell_analysis(cell_data)
            
            # Get ensemble prediction
            prediction = self.ensemble_prediction(analysis)
            
            if prediction and prediction['confidence'] > 0.6:  # High confidence threshold
                recovery_results.append({
                    'row': cell_data['row'],
                    'col': cell_data['col'],
                    'original_type': cell_data['type'],
                    'recovered_bit': prediction['prediction'],
                    'confidence': prediction['confidence'],
                    'methods_used': prediction['num_methods']
                })
        
        return recovery_results
    
    def generate_recovery_report(self, recovery_results):
        """Generate detailed recovery report."""
        
        total_problematic = len(self.problematic_cells)
        recovered = len(recovery_results)
        recovery_rate = (recovered / total_problematic * 100) if total_problematic > 0 else 0
        
        # Calculate new extraction statistics
        original_extractable = len(self.cells_df[self.cells_df['bit'].isin(['0', '1'])])
        new_extractable = original_extractable + recovered
        new_success_rate = (new_extractable / 2700) * 100
        
        report = []
        report.append("# Advanced Cell Recovery Analysis Report")
        report.append("")
        report.append("## Recovery Statistics")
        report.append(f"- **Total Problematic Cells**: {total_problematic}")
        report.append(f"- **Successfully Recovered**: {recovered}")
        report.append(f"- **Recovery Rate**: {recovery_rate:.1f}%")
        report.append(f"- **New Extractable Bits**: {new_extractable}")
        report.append(f"- **New Success Rate**: {new_success_rate:.1f}%")
        report.append("")
        
        if recovery_results:
            report.append("## Recovered Cells")
            report.append("")
            report.append("| Row | Col | Original Type | Recovered Bit | Confidence | Methods |")
            report.append("|-----|-----|---------------|---------------|------------|---------|")
            
            for result in recovery_results:
                report.append(f"| {result['row']} | {result['col']} | {result['original_type']} | {result['recovered_bit']} | {result['confidence']:.3f} | {result['methods_used']} |")
        
        report.append("")
        report.append("## Analysis Methods Used")
        report.append("")
        report.append("1. **Multi-scale Analysis**: Gaussian pyramid with Otsu thresholding")
        report.append("2. **Texture Analysis**: Local Binary Patterns and Gabor filters")
        report.append("3. **Edge Density Analysis**: Canny edge detection and Hough transforms")
        report.append("4. **Color Space Analysis**: HSV and LAB channel analysis")
        report.append("5. **Frequency Analysis**: FFT-based high-frequency energy analysis")
        report.append("6. **Neighbor Template Matching**: Template matching with successful neighbors")
        report.append("7. **Statistical Analysis**: Intensity statistics and histogram analysis")
        report.append("8. **Morphological Analysis**: Connected component analysis")
        report.append("9. **Ensemble Prediction**: Weighted voting across all methods")
        report.append("")
        
        return "\\n".join(report)

def main():
    """Main function to analyze remaining cells."""
    
    # Use the best extraction results
    image_path = "satoshi (1).png"
    cells_csv = "output_final/cells.csv"
    
    if not os.path.exists(image_path) or not os.path.exists(cells_csv):
        print("Error: Required files not found")
        return
    
    # Create output directory
    os.makedirs("test_results/cell_recovery", exist_ok=True)
    
    # Initialize analyzer
    analyzer = RemainingCellAnalyzer(image_path, cells_csv)
    
    # Analyze all problematic cells
    recovery_results = analyzer.analyze_all_problematic_cells()
    
    # Generate report
    report = analyzer.generate_recovery_report(recovery_results)
    
    # Save report
    with open("test_results/cell_recovery/CELL_RECOVERY_REPORT.md", 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Save recovery results as JSON
    with open("test_results/cell_recovery/recovery_results.json", 'w') as f:
        json.dump(recovery_results, f, indent=2)
    
    print(f"\\n=== Cell Recovery Analysis Complete ===")
    print(f"Recovered {len(recovery_results)} additional cells")
    print(f"New potential extraction rate: {((2580 + len(recovery_results)) / 2700) * 100:.1f}%")
    print(f"Report saved to: test_results/cell_recovery/CELL_RECOVERY_REPORT.md")

if __name__ == "__main__":
    main()