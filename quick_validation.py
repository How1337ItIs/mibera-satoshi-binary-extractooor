#!/usr/bin/env python3
"""
Quick validation to assess extraction confidence and identify questionable bits.
"""

import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path
import sys

# Add the binary_extractor to path
sys.path.append(str(Path(__file__).parent / "binary_extractor"))

class QuickValidator:
    """Fast validation of extraction quality."""
    
    def __init__(self, image_path, cells_csv_path):
        self.image_path = image_path
        self.cells_csv_path = cells_csv_path
        self.original = cv2.imread(image_path)
        self.gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        self.cells_df = pd.read_csv(cells_csv_path)
        
        # Grid parameters
        self.row_pitch = 31
        self.col_pitch = 25
        self.row0 = 1
        self.col0 = 5
        
    def extract_cell_region(self, row, col):
        """Extract cell region."""
        center_y = self.row0 + row * self.row_pitch
        center_x = self.col0 + col * self.col_pitch
        
        y1 = max(0, center_y - self.row_pitch//2)
        y2 = min(self.gray.shape[0], center_y + self.row_pitch//2)
        x1 = max(0, center_x - self.col_pitch//2)
        x2 = min(self.gray.shape[1], center_x + self.col_pitch//2)
        
        return self.gray[y1:y2, x1:x2]
    
    def validate_cell_confidence(self, row, col, original_bit):
        """Quick confidence validation of a single cell."""
        
        if original_bit not in ['0', '1']:
            return {'confidence': 0, 'issues': ['not_binary']}
        
        cell_image = self.extract_cell_region(row, col)
        
        if cell_image.size == 0:
            return {'confidence': 0, 'issues': ['empty_cell']}
        
        issues = []
        confidence_factors = []
        
        # Method 1: Otsu threshold consistency
        try:
            _, binary = cv2.threshold(cell_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            white_ratio = np.sum(binary == 255) / binary.size
            otsu_prediction = '1' if white_ratio > 0.5 else '0'
            
            if otsu_prediction == original_bit:
                confidence_factors.append(abs(white_ratio - 0.5) * 2)
            else:
                confidence_factors.append(0)
                issues.append('otsu_disagrees')
        except:
            confidence_factors.append(0)
            issues.append('otsu_failed')
        
        # Method 2: Mean intensity consistency
        mean_intensity = np.mean(cell_image)
        mean_prediction = '1' if mean_intensity > 127 else '0'
        
        if mean_prediction == original_bit:
            confidence_factors.append(abs(mean_intensity - 127) / 127)
        else:
            confidence_factors.append(0)
            issues.append('mean_disagrees')
        
        # Method 3: Contrast quality
        contrast = np.max(cell_image) - np.min(cell_image)
        if contrast < 30:
            issues.append('low_contrast')
        confidence_factors.append(min(1.0, contrast / 100))
        
        # Method 4: Edge strength
        laplacian = cv2.Laplacian(cell_image, cv2.CV_64F)
        laplacian_var = np.var(laplacian)
        if laplacian_var < 10:
            issues.append('blurry')
        confidence_factors.append(min(1.0, laplacian_var / 100))
        
        # Method 5: Neighbor consistency
        neighbor_consistency = self.check_neighbor_consistency(row, col, original_bit)
        confidence_factors.append(neighbor_consistency)
        if neighbor_consistency < 0.3:
            issues.append('neighbor_inconsistent')
        
        # Calculate overall confidence
        overall_confidence = np.mean(confidence_factors)
        
        return {
            'confidence': float(overall_confidence),
            'issues': issues,
            'num_issues': len(issues)
        }
    
    def check_neighbor_consistency(self, row, col, bit):
        """Check consistency with neighboring cells."""
        
        consistent = 0
        total = 0
        
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
                            total += 1
                            if neighbor_bit == bit:
                                consistent += 1
        
        return consistent / total if total > 0 else 0.5
    
    def validate_all_binary_cells(self):
        """Validate all cells marked as 0 or 1."""
        
        binary_cells = self.cells_df[self.cells_df['bit'].isin(['0', '1'])]
        
        print(f"Validating {len(binary_cells)} binary cells...")
        
        validation_results = []
        
        for index, row in binary_cells.iterrows():
            if index % 200 == 0:
                print(f"Progress: {index}/{len(binary_cells)}")
            
            validation = self.validate_cell_confidence(row['row'], row['col'], row['bit'])
            validation_results.append({
                'row': int(row['row']),
                'col': int(row['col']),
                'bit': row['bit'],
                'confidence': validation['confidence'],
                'num_issues': validation['num_issues'],
                'issues': validation['issues']
            })
        
        return validation_results
    
    def analyze_validation_results(self, validation_results):
        """Analyze validation results."""
        
        df = pd.DataFrame(validation_results)
        
        # Overall statistics
        stats = {
            'total_binary_cells': len(df),
            'mean_confidence': float(df['confidence'].mean()),
            'std_confidence': float(df['confidence'].std()),
            'min_confidence': float(df['confidence'].min()),
            'max_confidence': float(df['confidence'].max()),
            'median_confidence': float(df['confidence'].median())
        }
        
        # Confidence categories
        high_conf = df[df['confidence'] > 0.8]
        medium_conf = df[(df['confidence'] >= 0.5) & (df['confidence'] <= 0.8)]
        low_conf = df[df['confidence'] < 0.5]
        
        stats['high_confidence'] = {
            'count': len(high_conf),
            'percentage': len(high_conf) / len(df) * 100
        }
        
        stats['medium_confidence'] = {
            'count': len(medium_conf),
            'percentage': len(medium_conf) / len(df) * 100
        }
        
        stats['low_confidence'] = {
            'count': len(low_conf),
            'percentage': len(low_conf) / len(df) * 100
        }
        
        # Issue analysis
        all_issues = []
        for result in validation_results:
            all_issues.extend(result['issues'])
        
        issue_counts = pd.Series(all_issues).value_counts()
        stats['common_issues'] = issue_counts.to_dict()
        
        # Questionable cells
        questionable = df[(df['confidence'] < 0.5) | (df['num_issues'] > 2)]
        stats['questionable_cells'] = len(questionable)
        
        return stats, questionable
    
    def generate_confidence_report(self, stats, questionable):
        """Generate confidence assessment report."""
        
        report = []
        report.append("# Extraction Confidence Assessment Report")
        report.append("")
        report.append("## Summary")
        report.append(f"Validated {stats['total_binary_cells']} binary cells (0s and 1s) for extraction confidence.")
        report.append("")
        
        # Confidence statistics
        report.append("## Confidence Statistics")
        report.append("")
        report.append(f"- **Mean Confidence**: {stats['mean_confidence']:.3f}")
        report.append(f"- **Median Confidence**: {stats['median_confidence']:.3f}")
        report.append(f"- **Standard Deviation**: {stats['std_confidence']:.3f}")
        report.append(f"- **Range**: {stats['min_confidence']:.3f} - {stats['max_confidence']:.3f}")
        report.append("")
        
        # Confidence breakdown
        report.append("## Confidence Breakdown")
        report.append("")
        report.append("| Confidence Level | Count | Percentage |")
        report.append("|------------------|-------|------------|")
        report.append(f"| High (>0.8) | {stats['high_confidence']['count']} | {stats['high_confidence']['percentage']:.1f}% |")
        report.append(f"| Medium (0.5-0.8) | {stats['medium_confidence']['count']} | {stats['medium_confidence']['percentage']:.1f}% |")
        report.append(f"| Low (<0.5) | {stats['low_confidence']['count']} | {stats['low_confidence']['percentage']:.1f}% |")
        report.append("")
        
        # Issues identified
        if stats['common_issues']:
            report.append("## Common Issues Identified")
            report.append("")
            report.append("| Issue Type | Count | Description |")
            report.append("|------------|-------|-------------|")
            
            issue_descriptions = {
                'otsu_disagrees': 'Otsu threshold gives different result',
                'mean_disagrees': 'Mean intensity suggests different bit',
                'low_contrast': 'Cell has low contrast (<30)',
                'blurry': 'Cell appears blurry or unclear',
                'neighbor_inconsistent': 'Disagrees with neighboring cells',
                'otsu_failed': 'Otsu thresholding failed'
            }
            
            for issue, count in stats['common_issues'].items():
                desc = issue_descriptions.get(issue, 'Unknown issue')
                report.append(f"| {issue} | {count} | {desc} |")
        
        report.append("")
        
        # Questionable cells
        report.append("## Questionable Cells")
        report.append("")
        report.append(f"**{stats['questionable_cells']} cells** require further review due to low confidence or multiple issues.")
        report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("")
        
        if stats['low_confidence']['percentage'] > 10:
            report.append("⚠️ **HIGH PRIORITY**: Over 10% of cells have low confidence - consider re-extraction")
        elif stats['low_confidence']['percentage'] > 5:
            report.append("⚠️ **MEDIUM PRIORITY**: 5-10% of cells have low confidence - spot check recommended")
        else:
            report.append("✅ **GOOD QUALITY**: Less than 5% of cells have confidence issues")
        
        report.append("")
        report.append(f"**Action Items:**")
        report.append(f"1. Review {stats['questionable_cells']} questionable cells manually")
        report.append(f"2. Consider enhanced processing for low-confidence regions")
        report.append(f"3. Validate extraction parameters based on issue patterns")
        
        return "\\n".join(report)

def main():
    """Main validation function."""
    
    image_path = "satoshi (1).png"
    cells_csv = "output_final/cells.csv"
    
    if not os.path.exists(image_path) or not os.path.exists(cells_csv):
        print("Error: Required files not found")
        return
    
    # Create output directory
    os.makedirs("test_results/confidence_check", exist_ok=True)
    
    # Initialize validator
    validator = QuickValidator(image_path, cells_csv)
    
    # Validate binary cells
    validation_results = validator.validate_all_binary_cells()
    
    # Analyze results
    stats, questionable = validator.analyze_validation_results(validation_results)
    
    # Generate report
    report = validator.generate_confidence_report(stats, questionable)
    
    # Save report
    with open("test_results/confidence_check/CONFIDENCE_REPORT.md", 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Save questionable cells for manual review
    questionable_list = questionable[['row', 'col', 'bit', 'confidence', 'num_issues', 'issues']].to_dict('records')
    
    # Convert to JSON-serializable format
    import json
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)
    
    with open("test_results/confidence_check/questionable_cells.json", 'w') as f:
        json.dump(questionable_list, f, indent=2, cls=NumpyEncoder)
    
    print(f"\\n=== Confidence Assessment Complete ===")
    print(f"Validated {stats['total_binary_cells']} binary cells")
    print(f"Mean confidence: {stats['mean_confidence']:.3f}")
    print(f"High confidence: {stats['high_confidence']['count']} ({stats['high_confidence']['percentage']:.1f}%)")
    print(f"Low confidence: {stats['low_confidence']['count']} ({stats['low_confidence']['percentage']:.1f}%)")
    print(f"Questionable cells: {stats['questionable_cells']}")
    
    # Assessment
    if stats['low_confidence']['percentage'] < 5:
        print("\\n✅ ASSESSMENT: Extraction quality is EXCELLENT")
    elif stats['low_confidence']['percentage'] < 10:
        print("\\n⚠️ ASSESSMENT: Extraction quality is GOOD but could be improved")
    else:
        print("\\n❌ ASSESSMENT: Extraction quality needs IMPROVEMENT")
    
    print(f"Report saved to: test_results/confidence_check/CONFIDENCE_REPORT.md")

if __name__ == "__main__":
    main()