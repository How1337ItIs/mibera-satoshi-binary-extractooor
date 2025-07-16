#!/usr/bin/env python3
"""
Regional Extraction System
Implements region-specific extraction with optimized thresholds.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from systematic_extraction_research import SystematicExtractor

class RegionalExtractor:
    """Extract binary matrix using region-specific optimized thresholds."""
    
    def __init__(self, image_path):
        self.image_path = image_path
        self.extractor = SystematicExtractor(image_path)
        
        # Load optimized grid parameters
        self.extractor.grid_params = {
            'row_pitch': 33,
            'col_pitch': 26,
            'row0': 5,
            'col0': 3,
            'confidence': 1.350
        }
        
        # Regional thresholds (will be loaded from optimization results)
        self.regional_thresholds = {
            'top_sparse': 120,
            'middle_dense': 127,
            'bottom_mixed': 130,
            'left_margin': 115,
            'right_margin': 115,
            'center': 127,
            'default': 127
        }
        
        self.extraction_results = {}
        
    def load_threshold_optimization_results(self, threshold_file=None):
        """Load optimized thresholds from threshold optimization."""
        if threshold_file is None:
            threshold_file = "documentation/threshold_optimization.json"
        
        try:
            with open(threshold_file, 'r') as f:
                data = json.load(f)
            
            optimization_results = data.get('optimization_results', {})
            
            # Update regional thresholds with optimized values
            for region_name, results in optimization_results.items():
                if region_name in self.regional_thresholds:
                    self.regional_thresholds[region_name] = results['best_threshold']
            
            print(f"Loaded optimized thresholds: {self.regional_thresholds}")
            return True
            
        except FileNotFoundError:
            print(f"Threshold optimization file not found: {threshold_file}")
            print("Using default thresholds")
            return False
        except Exception as e:
            print(f"Error loading threshold optimization: {e}")
            return False
    
    def classify_cell_region(self, row, col):
        """Classify a cell into its appropriate region."""
        # Primary classification by row
        if row < 15:
            primary_region = 'top_sparse'
        elif row < 35:
            primary_region = 'middle_dense'
        else:
            primary_region = 'bottom_mixed'
        
        # Secondary classification by column
        if col < 10:
            secondary_region = 'left_margin'
        elif col > 40:
            secondary_region = 'right_margin'
        else:
            secondary_region = 'center'
        
        # Priority: margin regions override row-based regions
        if secondary_region in ['left_margin', 'right_margin']:
            return secondary_region
        else:
            return primary_region
    
    def extract_cell_with_regional_threshold(self, row, col):
        """Extract a single cell using its region-specific threshold."""
        region = self.classify_cell_region(row, col)
        threshold = self.regional_thresholds.get(region, self.regional_thresholds['default'])
        
        # Extract using the SystematicExtractor with specific threshold
        cell_result = self.extractor.extract_single_cell(row, col, threshold)
        
        if cell_result:
            cell_result['region'] = region
            cell_result['threshold_used'] = threshold
        
        return cell_result
    
    def extract_complete_matrix(self):
        """Extract the complete 54×50 binary matrix."""
        print("=== Regional Binary Matrix Extraction ===")
        print(f"Extracting {54 * 50} cells with region-specific thresholds...")
        
        # Load optimized thresholds if available
        self.load_threshold_optimization_results()
        
        matrix_results = []
        binary_matrix = np.full((54, 50), -1, dtype=int)  # -1 for unextracted
        confidence_matrix = np.zeros((54, 50))
        region_matrix = np.full((54, 50), '', dtype=object)
        
        region_stats = {}
        
        for row in range(54):
            if row % 10 == 0:
                print(f"Processing row {row}/54...")
            
            for col in range(50):
                cell_result = self.extract_cell_with_regional_threshold(row, col)
                
                if cell_result:
                    # Store detailed result
                    matrix_results.append(cell_result)
                    
                    # Update matrices
                    if cell_result['bit'] == '0':
                        binary_matrix[row, col] = 0
                    elif cell_result['bit'] == '1':
                        binary_matrix[row, col] = 1
                    else:  # uncertain
                        binary_matrix[row, col] = -1
                    
                    confidence_matrix[row, col] = cell_result['confidence']
                    region_matrix[row, col] = cell_result['region']
                    
                    # Update region statistics
                    region = cell_result['region']
                    if region not in region_stats:
                        region_stats[region] = {'total': 0, 'zeros': 0, 'ones': 0, 'uncertain': 0}
                    
                    region_stats[region]['total'] += 1
                    if cell_result['bit'] == '0':
                        region_stats[region]['zeros'] += 1
                    elif cell_result['bit'] == '1':
                        region_stats[region]['ones'] += 1
                    else:
                        region_stats[region]['uncertain'] += 1
        
        self.extraction_results = {
            'matrix_results': matrix_results,
            'binary_matrix': binary_matrix,
            'confidence_matrix': confidence_matrix,
            'region_matrix': region_matrix,
            'region_stats': region_stats,
            'regional_thresholds': self.regional_thresholds,
            'extraction_timestamp': datetime.now().isoformat()
        }
        
        print(f"Extraction complete! Processed {len(matrix_results)} cells")
        return self.extraction_results
    
    def analyze_extraction_quality(self):
        """Analyze the quality of the regional extraction."""
        if not self.extraction_results:
            return {}
        
        matrix_results = self.extraction_results['matrix_results']
        region_stats = self.extraction_results['region_stats']
        
        # Overall statistics
        total_cells = len(matrix_results)
        zeros = sum(1 for r in matrix_results if r['bit'] == '0')
        ones = sum(1 for r in matrix_results if r['bit'] == '1')
        uncertain = sum(1 for r in matrix_results if r['bit'] == 'uncertain')
        
        # Confidence analysis
        confidences = [r['confidence'] for r in matrix_results]
        avg_confidence = np.mean(confidences) if confidences else 0
        high_confidence = sum(1 for c in confidences if c > 0.8)
        
        # Regional analysis
        regional_analysis = {}
        for region, stats in region_stats.items():
            if stats['total'] > 0:
                regional_analysis[region] = {
                    'total_cells': stats['total'],
                    'zero_ratio': stats['zeros'] / stats['total'],
                    'one_ratio': stats['ones'] / stats['total'],
                    'uncertain_ratio': stats['uncertain'] / stats['total'],
                    'threshold_used': self.regional_thresholds.get(region, 127)
                }
        
        quality_analysis = {
            'total_cells': total_cells,
            'bit_distribution': {
                'zeros': zeros,
                'ones': ones,
                'uncertain': uncertain
            },
            'bit_ratios': {
                'zero_ratio': zeros / total_cells if total_cells > 0 else 0,
                'one_ratio': ones / total_cells if total_cells > 0 else 0,
                'uncertain_ratio': uncertain / total_cells if total_cells > 0 else 0
            },
            'confidence_stats': {
                'average_confidence': avg_confidence,
                'high_confidence_count': high_confidence,
                'high_confidence_ratio': high_confidence / total_cells if total_cells > 0 else 0
            },
            'regional_analysis': regional_analysis
        }
        
        return quality_analysis
    
    def create_extraction_visualization(self):
        """Create comprehensive visualization of the extraction results."""
        if not self.extraction_results:
            return
        
        binary_matrix = self.extraction_results['binary_matrix']
        confidence_matrix = self.extraction_results['confidence_matrix']
        region_matrix = self.extraction_results['region_matrix']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Binary matrix visualization
        ax = axes[0, 0]
        # Convert -1 (uncertain) to 0.5 for visualization
        vis_matrix = binary_matrix.copy().astype(float)
        vis_matrix[vis_matrix == -1] = 0.5
        
        im1 = ax.imshow(vis_matrix, cmap='RdBu', vmin=0, vmax=1)
        ax.set_title('Extracted Binary Matrix\\n(Blue=0, Red=1, Gray=Uncertain)')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        plt.colorbar(im1, ax=ax)
        
        # 2. Confidence matrix
        ax = axes[0, 1]
        im2 = ax.imshow(confidence_matrix, cmap='viridis', vmin=0, vmax=1)
        ax.set_title('Confidence Matrix')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        plt.colorbar(im2, ax=ax)
        
        # 3. Regional classification
        ax = axes[1, 0]
        # Create numeric region map for visualization
        region_map = np.zeros((54, 50))
        region_names = list(set(region_matrix.flatten()))
        region_to_num = {name: i for i, name in enumerate(region_names) if name}
        
        for row in range(54):
            for col in range(50):
                if region_matrix[row, col] in region_to_num:
                    region_map[row, col] = region_to_num[region_matrix[row, col]]
        
        im3 = ax.imshow(region_map, cmap='Set3')
        ax.set_title('Regional Classification')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        
        # Add region legend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=plt.cm.Set3(region_to_num[name]/len(region_names)), 
                                       label=name) for name in region_names if name]
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        # 4. Quality summary
        ax = axes[1, 1]
        quality_analysis = self.analyze_extraction_quality()
        
        if quality_analysis:
            bit_dist = quality_analysis['bit_distribution']
            categories = ['Zeros', 'Ones', 'Uncertain']
            counts = [bit_dist['zeros'], bit_dist['ones'], bit_dist['uncertain']]
            colors = ['blue', 'red', 'gray']
            
            wedges, texts, autotexts = ax.pie(counts, labels=categories, colors=colors, 
                                             autopct='%1.1f%%', startangle=90)
            ax.set_title('Bit Distribution')
        
        plt.tight_layout()
        
        # Save visualization
        Path("documentation").mkdir(exist_ok=True)
        plt.savefig("documentation/regional_extraction_results.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print("Regional extraction visualization saved: documentation/regional_extraction_results.png")
    
    def export_binary_matrix(self, format='multiple'):
        """Export binary matrix in multiple formats."""
        if not self.extraction_results:
            print("No extraction results to export")
            return
        
        binary_matrix = self.extraction_results['binary_matrix']
        confidence_matrix = self.extraction_results['confidence_matrix']
        
        Path("documentation").mkdir(exist_ok=True)
        
        if format in ['multiple', 'csv']:
            # CSV format
            df = pd.DataFrame(binary_matrix)
            df.to_csv("documentation/binary_matrix.csv", index=False, header=False)
            
            # CSV with confidence
            df_conf = pd.DataFrame(confidence_matrix)
            df_conf.to_csv("documentation/confidence_matrix.csv", index=False, header=False)
        
        if format in ['multiple', 'json']:
            # JSON format
            json_data = {
                'binary_matrix': binary_matrix.tolist(),
                'confidence_matrix': confidence_matrix.tolist(),
                'extraction_metadata': {
                    'timestamp': self.extraction_results['extraction_timestamp'],
                    'grid_params': self.extractor.grid_params,
                    'regional_thresholds': self.regional_thresholds,
                    'matrix_dimensions': {'rows': 54, 'cols': 50}
                }
            }
            
            with open("documentation/binary_matrix.json", "w") as f:
                json.dump(json_data, f, indent=2)
        
        if format in ['multiple', 'numpy']:
            # NumPy format
            np.save("documentation/binary_matrix.npy", binary_matrix)
            np.save("documentation/confidence_matrix.npy", confidence_matrix)
        
        if format in ['multiple', 'text']:
            # Simple text format
            with open("documentation/binary_matrix.txt", "w") as f:
                for row in binary_matrix:
                    f.write("".join(str(bit) if bit != -1 else '?' for bit in row) + "\\n")
        
        print(f"Binary matrix exported in {format} format to documentation/")
    
    def generate_extraction_report(self):
        """Generate comprehensive extraction report."""
        if not self.extraction_results:
            return "No extraction results available"
        
        quality_analysis = self.analyze_extraction_quality()
        
        report = []
        report.append("# Regional Binary Matrix Extraction Report")
        report.append("## Complete 54×50 Matrix Extraction Results")
        report.append("")
        report.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Method**: Regional extraction with optimized thresholds")
        report.append(f"**Total cells**: {quality_analysis['total_cells']}")
        report.append("")
        
        # Grid parameters
        report.append("## Grid Parameters")
        report.append("")
        grid_params = self.extractor.grid_params
        report.append(f"- **Row pitch**: {grid_params['row_pitch']} pixels")
        report.append(f"- **Column pitch**: {grid_params['col_pitch']} pixels")
        report.append(f"- **Row origin**: {grid_params['row0']} pixels")
        report.append(f"- **Column origin**: {grid_params['col0']} pixels")
        report.append(f"- **Confidence score**: {grid_params['confidence']:.3f}")
        report.append("")
        
        # Regional thresholds
        report.append("## Regional Thresholds Used")
        report.append("")
        report.append("| Region | Threshold | Purpose |")
        report.append("|--------|-----------|---------|")
        for region, threshold in self.regional_thresholds.items():
            report.append(f"| {region} | {threshold} | Region-specific optimization |")
        report.append("")
        
        # Extraction results
        report.append("## Extraction Results")
        report.append("")
        bit_dist = quality_analysis['bit_distribution']
        bit_ratios = quality_analysis['bit_ratios']
        
        report.append(f"- **Total cells processed**: {quality_analysis['total_cells']}")
        report.append(f"- **Zeros extracted**: {bit_dist['zeros']} ({bit_ratios['zero_ratio']*100:.1f}%)")
        report.append(f"- **Ones extracted**: {bit_dist['ones']} ({bit_ratios['one_ratio']*100:.1f}%)")
        report.append(f"- **Uncertain cells**: {bit_dist['uncertain']} ({bit_ratios['uncertain_ratio']*100:.1f}%)")
        report.append("")
        
        # Confidence analysis
        conf_stats = quality_analysis['confidence_stats']
        report.append("## Confidence Analysis")
        report.append("")
        report.append(f"- **Average confidence**: {conf_stats['average_confidence']:.3f}")
        report.append(f"- **High confidence cells**: {conf_stats['high_confidence_count']} ({conf_stats['high_confidence_ratio']*100:.1f}%)")
        report.append("")
        
        # Regional analysis
        report.append("## Regional Analysis")
        report.append("")
        report.append("| Region | Cells | Zeros | Ones | Uncertain | Threshold |")
        report.append("|--------|-------|-------|------|-----------|-----------|")
        
        for region, analysis in quality_analysis['regional_analysis'].items():
            report.append(f"| {region} | {analysis['total_cells']} | {analysis['zero_ratio']*100:.1f}% | {analysis['one_ratio']*100:.1f}% | {analysis['uncertain_ratio']*100:.1f}% | {analysis['threshold_used']} |")
        
        report.append("")
        
        # Quality assessment
        report.append("## Quality Assessment")
        report.append("")
        
        uncertain_ratio = bit_ratios['uncertain_ratio']
        avg_confidence = conf_stats['average_confidence']
        
        if uncertain_ratio < 0.05 and avg_confidence > 0.8:
            report.append("### [EXCELLENT] High quality extraction")
            report.append("- Low uncertainty rate and high confidence")
            report.append("- Ready for cryptographic analysis")
        elif uncertain_ratio < 0.1 and avg_confidence > 0.7:
            report.append("### [GOOD] Good quality extraction")
            report.append("- Reasonable uncertainty and confidence levels")
            report.append("- Suitable for analysis with minor validation")
        elif uncertain_ratio < 0.2 and avg_confidence > 0.6:
            report.append("### [MODERATE] Moderate quality extraction")
            report.append("- Some uncertainty but generally reliable")
            report.append("- Consider additional validation")
        else:
            report.append("### [NEEDS REVIEW] Quality concerns")
            report.append("- High uncertainty or low confidence")
            report.append("- Review extraction methods and parameters")
        
        report.append("")
        
        # Output files
        report.append("## Output Files Generated")
        report.append("")
        report.append("- `binary_matrix.csv` - Matrix in CSV format")
        report.append("- `binary_matrix.json` - Matrix with metadata in JSON")
        report.append("- `binary_matrix.txt` - Simple text format")
        report.append("- `confidence_matrix.csv` - Confidence scores")
        report.append("- `regional_extraction_results.png` - Visualization")
        report.append("")
        
        # Next steps
        report.append("## Next Steps")
        report.append("")
        report.append("1. **Cryptographic Analysis**: Analyze binary patterns for hidden messages")
        report.append("2. **Pattern Recognition**: Look for structured data or encoding")
        report.append("3. **Validation**: Cross-reference with known cryptographic techniques")
        report.append("4. **Documentation**: Record findings for future research")
        
        return "\\n".join(report)
    
    def run_complete_extraction(self):
        """Run the complete regional extraction process."""
        print("=== Complete Regional Binary Matrix Extraction ===")
        
        # Extract complete matrix
        extraction_results = self.extract_complete_matrix()
        
        # Analyze quality
        print("\\nAnalyzing extraction quality...")
        quality_analysis = self.analyze_extraction_quality()
        
        # Create visualization
        print("\\nCreating extraction visualization...")
        self.create_extraction_visualization()
        
        # Export in multiple formats
        print("\\nExporting binary matrix...")
        self.export_binary_matrix()
        
        # Generate report
        print("\\nGenerating extraction report...")
        report = self.generate_extraction_report()
        
        # Save report
        Path("documentation").mkdir(exist_ok=True)
        with open("documentation/REGIONAL_EXTRACTION_REPORT.md", "w") as f:
            f.write(report)
        
        # Save detailed results
        with open("documentation/regional_extraction_results.json", "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {
                'binary_matrix': extraction_results['binary_matrix'].tolist(),
                'confidence_matrix': extraction_results['confidence_matrix'].tolist(),
                'region_stats': extraction_results['region_stats'],
                'regional_thresholds': extraction_results['regional_thresholds'],
                'extraction_timestamp': extraction_results['extraction_timestamp'],
                'quality_analysis': quality_analysis
            }
            json.dump(json_results, f, indent=2)
        
        print(f"\\n=== Regional Extraction Complete ===")
        print(f"Extracted {quality_analysis['total_cells']} cells")
        print(f"Quality: {quality_analysis['confidence_stats']['average_confidence']:.3f} average confidence")
        print(f"Results saved to: documentation/REGIONAL_EXTRACTION_REPORT.md")
        
        return extraction_results

def main():
    """Main regional extraction function."""
    
    image_path = "satoshi (1).png"
    
    if not Path(image_path).exists():
        print(f"Error: Image file not found: {image_path}")
        return
    
    # Initialize regional extractor
    extractor = RegionalExtractor(image_path)
    
    # Run complete extraction
    results = extractor.run_complete_extraction()
    
    print("\\nRegional extraction completed successfully!")
    print("Binary matrix ready for cryptographic analysis.")

if __name__ == "__main__":
    main()