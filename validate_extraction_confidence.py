#!/usr/bin/env python3
"""
Comprehensive validation system for bit extraction confidence.
Re-evaluates ALL extracted bits with confidence scoring and cross-validation.
"""

import cv2
import numpy as np
import pandas as pd
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import sys

# Add the binary_extractor to path
sys.path.append(str(Path(__file__).parent / "binary_extractor"))

class ExtractionValidator:
    """Comprehensive validation of all extracted bits."""
    
    def __init__(self, image_path, cells_csv_path):
        self.image_path = image_path
        self.cells_csv_path = cells_csv_path
        self.original = cv2.imread(image_path)
        self.rgb = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)
        self.gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        
        # Load cell data
        self.cells_df = pd.read_csv(cells_csv_path)
        
        # Grid parameters
        self.row_pitch = 31
        self.col_pitch = 25
        self.row0 = 1
        self.col0 = 5
        
        # Validation results
        self.validation_results = []
        self.confidence_scores = {}
        
    def extract_cell_region(self, row, col):
        """Extract cell region for analysis."""
        center_y = self.row0 + row * self.row_pitch
        center_x = self.col0 + col * self.col_pitch
        
        # Cell boundaries
        y1 = max(0, center_y - self.row_pitch//2)
        y2 = min(self.gray.shape[0], center_y + self.row_pitch//2)
        x1 = max(0, center_x - self.col_pitch//2)
        x2 = min(self.gray.shape[1], center_x + self.col_pitch//2)
        
        return self.gray[y1:y2, x1:x2]
    
    def calculate_multiple_thresholds(self, cell_image):
        """Calculate bit classification using multiple threshold methods."""
        
        methods = {}
        
        # Otsu thresholding
        try:
            _, binary = cv2.threshold(cell_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            white_ratio = np.sum(binary == 255) / binary.size
            methods['otsu'] = {
                'white_ratio': white_ratio,
                'predicted_bit': '1' if white_ratio > 0.5 else '0',
                'confidence': abs(white_ratio - 0.5) * 2
            }
        except:
            methods['otsu'] = {'predicted_bit': 'error', 'confidence': 0}
        
        # Adaptive thresholding
        try:
            binary = cv2.adaptiveThreshold(cell_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            white_ratio = np.sum(binary == 255) / binary.size
            methods['adaptive'] = {
                'white_ratio': white_ratio,
                'predicted_bit': '1' if white_ratio > 0.5 else '0',
                'confidence': abs(white_ratio - 0.5) * 2
            }
        except:
            methods['adaptive'] = {'predicted_bit': 'error', 'confidence': 0}
        
        # Mean thresholding
        mean_val = np.mean(cell_image)
        methods['mean'] = {
            'threshold': mean_val,
            'predicted_bit': '1' if mean_val > 127 else '0',
            'confidence': abs(mean_val - 127) / 127
        }
        
        # Median thresholding
        median_val = np.median(cell_image)
        methods['median'] = {
            'threshold': median_val,
            'predicted_bit': '1' if median_val > 127 else '0',
            'confidence': abs(median_val - 127) / 127
        }
        
        # Percentile thresholding
        p25 = np.percentile(cell_image, 25)
        p75 = np.percentile(cell_image, 75)
        methods['percentile'] = {
            'p25': p25,
            'p75': p75,
            'predicted_bit': '1' if p75 > 127 else '0',
            'confidence': abs(p75 - 127) / 127
        }
        
        return methods
    
    def analyze_cell_features(self, cell_image):
        """Extract detailed features from cell for confidence analysis."""
        
        features = {}
        
        # Basic statistics
        features['mean'] = np.mean(cell_image)
        features['std'] = np.std(cell_image)
        features['min'] = np.min(cell_image)
        features['max'] = np.max(cell_image)
        features['range'] = features['max'] - features['min']
        
        # Histogram features
        hist, bins = np.histogram(cell_image, bins=32, range=(0, 256))
        features['hist_entropy'] = -np.sum(hist * np.log(hist + 1e-10)) / len(hist)
        features['hist_peaks'] = len(np.where(np.diff(np.sign(np.diff(hist))) < 0)[0])
        
        # Texture features
        # Variance of Laplacian (blur detection)
        laplacian = cv2.Laplacian(cell_image, cv2.CV_64F)
        features['laplacian_var'] = np.var(laplacian)
        
        # Gradient magnitude
        grad_x = cv2.Sobel(cell_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(cell_image, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        features['grad_mean'] = np.mean(grad_magnitude)
        features['grad_std'] = np.std(grad_magnitude)
        
        # Edge density
        edges = cv2.Canny(cell_image, 50, 150)
        features['edge_density'] = np.sum(edges == 255) / edges.size
        
        return features
    
    def calculate_neighbor_consistency(self, row, col, predicted_bit):
        """Calculate consistency with neighboring cells."""
        
        consistent_neighbors = 0
        total_neighbors = 0
        
        # Check 8-connected neighbors
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                    
                nr, nc = row + dr, col + dc
                if 0 <= nr < 54 and 0 <= nc < 50:
                    neighbor = self.cells_df[(self.cells_df['row'] == nr) & 
                                           (self.cells_df['col'] == nc)]
                    if not neighbor.empty:
                        neighbor_bit = neighbor.iloc[0]['bit']
                        if neighbor_bit in ['0', '1']:
                            total_neighbors += 1
                            if neighbor_bit == predicted_bit:
                                consistent_neighbors += 1
        
        return consistent_neighbors / total_neighbors if total_neighbors > 0 else 0
    
    def validate_single_cell(self, row, col, original_bit):
        """Comprehensive validation of a single cell."""
        
        # Extract cell region
        cell_image = self.extract_cell_region(row, col)
        
        if cell_image.size == 0:
            return {
                'row': row,
                'col': col,
                'original_bit': original_bit,
                'validation_result': 'error',
                'confidence': 0,
                'issues': ['empty_cell']
            }
        
        # Calculate multiple thresholds
        threshold_methods = self.calculate_multiple_thresholds(cell_image)
        
        # Extract features
        features = self.analyze_cell_features(cell_image)
        
        # Collect predictions
        predictions = []
        confidences = []
        
        for method_name, method_data in threshold_methods.items():
            if method_data['predicted_bit'] not in ['error']:
                predictions.append(method_data['predicted_bit'])
                confidences.append(method_data.get('confidence', 0.5))
        
        # Consensus analysis
        if predictions:
            unique_predictions = list(set(predictions))
            
            if len(unique_predictions) == 1:
                # All methods agree
                consensus_bit = unique_predictions[0]
                agreement_score = 1.0
            else:
                # Methods disagree - use weighted voting
                vote_counts = {'0': 0, '1': 0}
                for pred, conf in zip(predictions, confidences):
                    vote_counts[pred] += conf
                
                consensus_bit = max(vote_counts, key=vote_counts.get)
                agreement_score = vote_counts[consensus_bit] / sum(vote_counts.values())
        else:
            consensus_bit = 'error'
            agreement_score = 0
        
        # Calculate neighbor consistency
        neighbor_consistency = self.calculate_neighbor_consistency(row, col, consensus_bit)
        
        # Overall confidence calculation
        base_confidence = np.mean(confidences) if confidences else 0
        
        # Confidence factors
        confidence_factors = {
            'method_agreement': agreement_score,
            'neighbor_consistency': neighbor_consistency,
            'image_quality': min(1.0, features['laplacian_var'] / 100),  # Normalize
            'contrast': min(1.0, features['range'] / 255),
            'edge_strength': min(1.0, features['edge_density'] * 10)
        }
        
        # Weighted confidence score
        weights = {
            'method_agreement': 0.3,
            'neighbor_consistency': 0.25,
            'image_quality': 0.15,
            'contrast': 0.15,
            'edge_strength': 0.15
        }
        
        final_confidence = sum(confidence_factors[k] * weights[k] for k in weights)
        
        # Identify issues
        issues = []
        if agreement_score < 0.7:
            issues.append('method_disagreement')
        if neighbor_consistency < 0.3:
            issues.append('neighbor_inconsistency')
        if features['laplacian_var'] < 10:
            issues.append('low_image_quality')
        if features['range'] < 50:
            issues.append('low_contrast')
        
        # Final validation result
        if consensus_bit == original_bit:
            validation_result = 'confirmed'
        elif consensus_bit in ['0', '1'] and original_bit in ['0', '1']:
            validation_result = 'contradicted'
        elif original_bit in ['blank', 'overlay'] and consensus_bit in ['0', '1']:
            validation_result = 'recovered'
        else:
            validation_result = 'uncertain'
        
        return {
            'row': row,
            'col': col,
            'original_bit': original_bit,
            'consensus_bit': consensus_bit,
            'validation_result': validation_result,
            'confidence': final_confidence,
            'method_agreement': agreement_score,
            'neighbor_consistency': neighbor_consistency,
            'features': features,
            'threshold_methods': threshold_methods,
            'issues': issues,
            'num_methods': len(predictions)
        }
    
    def validate_all_cells(self):
        """Validate all cells in the dataset."""
        
        print("Validating all 2,700 cells...")
        
        validation_results = []
        
        for index, row in self.cells_df.iterrows():
            if index % 100 == 0:
                print(f"Progress: {index}/2700")
            
            validation = self.validate_single_cell(row['row'], row['col'], row['bit'])
            validation_results.append(validation)
        
        self.validation_results = validation_results
        return validation_results
    
    def analyze_validation_results(self):
        """Analyze validation results and generate statistics."""
        
        if not self.validation_results:
            return None
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(self.validation_results)
        
        # Overall statistics
        stats = {}
        
        # Validation result counts
        result_counts = df['validation_result'].value_counts()
        stats['validation_results'] = result_counts.to_dict()
        
        # Confidence statistics
        stats['confidence'] = {
            'mean': df['confidence'].mean(),
            'std': df['confidence'].std(),
            'min': df['confidence'].min(),
            'max': df['confidence'].max(),
            'median': df['confidence'].median()
        }
        
        # High confidence cells (>0.8)
        high_conf = df[df['confidence'] > 0.8]
        stats['high_confidence'] = {
            'count': len(high_conf),
            'percentage': len(high_conf) / len(df) * 100
        }
        
        # Low confidence cells (<0.5)
        low_conf = df[df['confidence'] < 0.5]
        stats['low_confidence'] = {
            'count': len(low_conf),
            'percentage': len(low_conf) / len(df) * 100
        }
        
        # Contradicted cells
        contradicted = df[df['validation_result'] == 'contradicted']
        stats['contradicted'] = {
            'count': len(contradicted),
            'percentage': len(contradicted) / len(df) * 100
        }
        
        # Recovered cells
        recovered = df[df['validation_result'] == 'recovered']
        stats['recovered'] = {
            'count': len(recovered),
            'percentage': len(recovered) / len(df) * 100
        }
        
        # Issue analysis
        all_issues = []
        for result in self.validation_results:
            all_issues.extend(result['issues'])
        
        issue_counts = pd.Series(all_issues).value_counts()
        stats['common_issues'] = issue_counts.to_dict()
        
        return stats
    
    def generate_validation_report(self, stats):
        """Generate comprehensive validation report."""
        
        report = []
        report.append("# Comprehensive Bit Extraction Validation Report")
        report.append("")
        report.append("## Validation Overview")
        report.append(f"This report validates the confidence of all 2,700 extracted bits using multiple")
        report.append(f"threshold methods, feature analysis, and neighbor consistency checking.")
        report.append("")
        
        # Validation results
        report.append("## Validation Results")
        report.append("")
        report.append("| Result Type | Count | Percentage |")
        report.append("|-------------|-------|------------|")
        
        for result_type, count in stats['validation_results'].items():
            percentage = (count / 2700) * 100
            report.append(f"| {result_type} | {count} | {percentage:.1f}% |")
        
        report.append("")
        
        # Confidence analysis
        report.append("## Confidence Analysis")
        report.append("")
        report.append(f"- **Mean Confidence**: {stats['confidence']['mean']:.3f}")
        report.append(f"- **Median Confidence**: {stats['confidence']['median']:.3f}")
        report.append(f"- **Standard Deviation**: {stats['confidence']['std']:.3f}")
        report.append(f"- **Range**: {stats['confidence']['min']:.3f} - {stats['confidence']['max']:.3f}")
        report.append("")
        
        # High/Low confidence breakdown
        report.append("### Confidence Breakdown")
        report.append("")
        report.append(f"- **High Confidence (>0.8)**: {stats['high_confidence']['count']} cells ({stats['high_confidence']['percentage']:.1f}%)")
        report.append(f"- **Low Confidence (<0.5)**: {stats['low_confidence']['count']} cells ({stats['low_confidence']['percentage']:.1f}%)")
        report.append("")
        
        # Issues analysis
        if stats['common_issues']:
            report.append("## Common Issues Identified")
            report.append("")
            report.append("| Issue Type | Count | Impact |")
            report.append("|------------|-------|---------|")
            
            for issue, count in stats['common_issues'].items():
                impact = "High" if count > 100 else "Medium" if count > 50 else "Low"
                report.append(f"| {issue} | {count} | {impact} |")
        
        report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("")
        
        if stats['contradicted']['count'] > 0:
            report.append(f"1. **Review {stats['contradicted']['count']} contradicted cells** - These require manual inspection")
        
        if stats['recovered']['count'] > 0:
            report.append(f"2. **Validate {stats['recovered']['count']} recovered cells** - Previously blank/overlay cells now classified")
        
        if stats['low_confidence']['count'] > 0:
            report.append(f"3. **Re-examine {stats['low_confidence']['count']} low-confidence cells** - May need enhanced processing")
        
        report.append("")
        report.append("## Methodology")
        report.append("")
        report.append("This validation used:")
        report.append("- **5 threshold methods**: Otsu, Adaptive, Mean, Median, Percentile")
        report.append("- **Feature analysis**: Texture, contrast, edge density, image quality")
        report.append("- **Neighbor consistency**: Agreement with surrounding cells")
        report.append("- **Confidence scoring**: Weighted combination of multiple factors")
        report.append("")
        
        return "\\n".join(report)
    
    def get_questionable_cells(self, min_confidence=0.5):
        """Get cells that should be re-examined."""
        
        questionable = []
        
        for result in self.validation_results:
            if (result['confidence'] < min_confidence or 
                result['validation_result'] in ['contradicted', 'uncertain'] or
                len(result['issues']) > 2):
                questionable.append(result)
        
        return questionable

def main():
    """Main validation function."""
    
    # Use the final extraction results
    image_path = "satoshi (1).png"
    cells_csv = "output_final/cells.csv"
    
    if not os.path.exists(image_path) or not os.path.exists(cells_csv):
        print("Error: Required files not found")
        return
    
    # Create output directory
    os.makedirs("test_results/validation", exist_ok=True)
    
    # Initialize validator
    validator = ExtractionValidator(image_path, cells_csv)
    
    # Validate all cells
    validation_results = validator.validate_all_cells()
    
    # Analyze results
    stats = validator.analyze_validation_results()
    
    # Generate report
    report = validator.generate_validation_report(stats)
    
    # Save validation results
    with open("test_results/validation/validation_results.json", 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    # Save report
    with open("test_results/validation/VALIDATION_REPORT.md", 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Get questionable cells
    questionable = validator.get_questionable_cells()
    
    with open("test_results/validation/questionable_cells.json", 'w') as f:
        json.dump(questionable, f, indent=2)
    
    print(f"\\n=== Validation Complete ===")
    print(f"Validated all 2,700 cells")
    print(f"High confidence cells: {stats['high_confidence']['count']} ({stats['high_confidence']['percentage']:.1f}%)")
    print(f"Low confidence cells: {stats['low_confidence']['count']} ({stats['low_confidence']['percentage']:.1f}%)")
    print(f"Contradicted cells: {stats['contradicted']['count']} ({stats['contradicted']['percentage']:.1f}%)")
    print(f"Recovered cells: {stats['recovered']['count']} ({stats['recovered']['percentage']:.1f}%)")
    print(f"Questionable cells requiring review: {len(questionable)}")
    print(f"Report saved to: test_results/validation/VALIDATION_REPORT.md")

if __name__ == "__main__":
    main()