#!/usr/bin/env python3
"""
Threshold Optimization System
Uses validation results to optimize thresholds for different poster regions.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from systematic_extraction_research import SystematicExtractor
from validation_framework import ValidationFramework

class ThresholdOptimizer:
    """Optimize thresholds based on validation results."""
    
    def __init__(self, image_path, ground_truth_path):
        self.image_path = image_path
        self.ground_truth_path = ground_truth_path
        self.extractor = SystematicExtractor(image_path)
        
        # Load optimized grid parameters
        self.extractor.grid_params = {
            'row_pitch': 33,
            'col_pitch': 26,
            'row0': 5,
            'col0': 3,
            'confidence': 1.350
        }
        
        self.ground_truth_data = None
        self.optimization_results = {}
        
    def load_ground_truth(self):
        """Load ground truth annotations."""
        try:
            with open(self.ground_truth_path, 'r') as f:
                data = json.load(f)
            
            # Filter annotated entries
            self.ground_truth_data = [
                item for item in data 
                if item.get('ground_truth_bit') in ['0', '1']
            ]
            
            return len(self.ground_truth_data) > 0
        except:
            return False
    
    def classify_regions(self):
        """Classify ground truth cells by poster region."""
        if not self.ground_truth_data:
            return {}
        
        regions = {
            'top_sparse': [],
            'middle_dense': [],
            'bottom_mixed': [],
            'left_margin': [],
            'right_margin': [],
            'center': []
        }
        
        for cell in self.ground_truth_data:
            row = cell['row']
            col = cell['col']
            
            # Classify by row
            if row < 15:
                regions['top_sparse'].append(cell)
            elif row < 35:
                regions['middle_dense'].append(cell)
            else:
                regions['bottom_mixed'].append(cell)
            
            # Classify by column
            if col < 10:
                regions['left_margin'].append(cell)
            elif col > 40:
                regions['right_margin'].append(cell)
            else:
                regions['center'].append(cell)
        
        # Filter empty regions
        return {k: v for k, v in regions.items() if v}
    
    def test_threshold_range(self, region_cells, threshold_range):
        """Test different thresholds on a region."""
        results = []
        
        for threshold in threshold_range:
            correct = 0
            total = 0
            
            for cell in region_cells:
                row = cell['row']
                col = cell['col']
                ground_truth = cell['ground_truth_bit']
                
                # Extract with specific threshold
                extracted = self.extractor.extract_single_cell(row, col, threshold)
                
                if extracted and extracted['bit'] in ['0', '1']:
                    total += 1
                    if extracted['bit'] == ground_truth:
                        correct += 1
            
            accuracy = correct / total if total > 0 else 0
            results.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'correct': correct,
                'total': total
            })
        
        return results
    
    def optimize_regional_thresholds(self):
        """Optimize thresholds for different regions."""
        if not self.load_ground_truth():
            print("Cannot load ground truth data")
            return {}
        
        regions = self.classify_regions()
        if not regions:
            print("No regional data available")
            return {}
        
        print("Optimizing thresholds for different regions...")
        
        # Test threshold ranges
        threshold_ranges = {
            'coarse': range(100, 160, 10),  # Quick initial test
            'fine': range(115, 140, 2)     # Detailed optimization
        }
        
        optimization_results = {}
        
        for region_name, region_cells in regions.items():
            if len(region_cells) < 3:  # Need minimum samples
                continue
            
            print(f"Optimizing {region_name} ({len(region_cells)} samples)...")
            
            # Coarse optimization
            coarse_results = self.test_threshold_range(region_cells, threshold_ranges['coarse'])
            best_coarse = max(coarse_results, key=lambda x: x['accuracy'])
            
            # Fine optimization around best coarse result
            fine_center = best_coarse['threshold']
            fine_range = range(max(100, fine_center - 15), min(160, fine_center + 15), 1)
            fine_results = self.test_threshold_range(region_cells, fine_range)
            
            best_fine = max(fine_results, key=lambda x: x['accuracy'])
            
            optimization_results[region_name] = {
                'best_threshold': best_fine['threshold'],
                'best_accuracy': best_fine['accuracy'],
                'sample_count': len(region_cells),
                'coarse_results': coarse_results,
                'fine_results': fine_results
            }
            
            print(f"  Best threshold: {best_fine['threshold']} (accuracy: {best_fine['accuracy']:.3f})")
        
        return optimization_results
    
    def create_optimization_visualization(self, optimization_results):
        """Create visualization of threshold optimization results."""
        if not optimization_results:
            return
        
        num_regions = len(optimization_results)
        if num_regions == 0:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        region_names = list(optimization_results.keys())
        
        # Plot threshold curves for each region
        for i, (region_name, results) in enumerate(optimization_results.items()):
            if i >= 6:  # Maximum 6 subplots
                break
            
            ax = axes[i]
            
            # Plot fine results
            fine_data = results['fine_results']
            thresholds = [r['threshold'] for r in fine_data]
            accuracies = [r['accuracy'] for r in fine_data]
            
            ax.plot(thresholds, accuracies, 'b-', linewidth=2, label='Fine optimization')
            
            # Mark best threshold
            best_threshold = results['best_threshold']
            best_accuracy = results['best_accuracy']
            ax.axvline(best_threshold, color='red', linestyle='--', 
                      label=f'Best: {best_threshold}')
            ax.scatter([best_threshold], [best_accuracy], 
                      color='red', s=100, zorder=5)
            
            ax.set_xlabel('Threshold')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'{region_name}\\n({results["sample_count"]} samples)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_ylim(0, 1)
        
        # Hide unused subplots
        for i in range(num_regions, 6):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('documentation/threshold_optimization.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("Threshold optimization visualization saved: documentation/threshold_optimization.png")
    
    def generate_optimization_report(self, optimization_results):
        """Generate threshold optimization report."""
        if not optimization_results:
            return "No optimization results available"
        
        report = []
        report.append("# Threshold Optimization Report")
        report.append("## Regional Threshold Optimization Results")
        report.append("")
        report.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Method**: Regional threshold optimization with ground truth validation")
        report.append("")
        
        # Summary table
        report.append("## Optimization Summary")
        report.append("")
        report.append("| Region | Best Threshold | Accuracy | Sample Count |")
        report.append("|--------|---------------|----------|-------------|")
        
        for region_name, results in optimization_results.items():
            report.append(f"| {region_name} | {results['best_threshold']} | {results['best_accuracy']:.3f} | {results['sample_count']} |")
        
        report.append("")
        
        # Detailed results
        report.append("## Detailed Results")
        report.append("")
        
        for region_name, results in optimization_results.items():
            report.append(f"### {region_name}")
            report.append(f"- **Best threshold**: {results['best_threshold']}")
            report.append(f"- **Best accuracy**: {results['best_accuracy']:.3f} ({results['best_accuracy']*100:.1f}%)")
            report.append(f"- **Sample count**: {results['sample_count']}")
            report.append("")
            
            # Show top 3 thresholds
            top_results = sorted(results['fine_results'], key=lambda x: x['accuracy'], reverse=True)[:3]
            report.append("**Top 3 thresholds:**")
            for i, result in enumerate(top_results, 1):
                report.append(f"{i}. Threshold {result['threshold']}: {result['accuracy']:.3f} accuracy")
            report.append("")
        
        # Implementation recommendations
        report.append("## Implementation Recommendations")
        report.append("")
        
        # Calculate average improvement
        accuracies = [r['best_accuracy'] for r in optimization_results.values()]
        avg_accuracy = sum(accuracies) / len(accuracies)
        
        if avg_accuracy > 0.85:
            report.append("### [EXCELLENT] Regional optimization successful")
            report.append("- Regional thresholds show significant improvement")
            report.append("- Ready for region-specific extraction")
        elif avg_accuracy > 0.75:
            report.append("### [GOOD] Regional optimization shows promise")
            report.append("- Moderate improvement with regional thresholds")
            report.append("- Consider further refinement")
        else:
            report.append("### [NEEDS WORK] Regional optimization limited")
            report.append("- Regional thresholds may not be sufficient")
            report.append("- Consider alternative approaches")
        
        report.append("")
        report.append("### Usage in Extraction")
        report.append("```python")
        report.append("# Regional threshold map")
        report.append("regional_thresholds = {")
        for region_name, results in optimization_results.items():
            report.append(f"    '{region_name}': {results['best_threshold']},")
        report.append("}")
        report.append("```")
        
        report.append("")
        report.append("## Next Steps")
        report.append("1. **Implement regional extraction**: Use optimized thresholds for each region")
        report.append("2. **Validate improvement**: Test on larger sample set")
        report.append("3. **Refine boundaries**: Adjust region boundaries if needed")
        report.append("4. **Full matrix extraction**: Apply to complete 54x50 grid")
        
        return "\\n".join(report)
    
    def run_optimization(self):
        """Run complete threshold optimization process."""
        print("=== Threshold Optimization ===")
        
        # Optimize regional thresholds
        optimization_results = self.optimize_regional_thresholds()
        
        if not optimization_results:
            print("No optimization results obtained")
            return None
        
        # Create visualization
        print("Creating optimization visualization...")
        self.create_optimization_visualization(optimization_results)
        
        # Generate report
        print("Generating optimization report...")
        report = self.generate_optimization_report(optimization_results)
        
        # Save results
        Path("documentation").mkdir(exist_ok=True)
        
        with open("documentation/THRESHOLD_OPTIMIZATION_REPORT.md", "w") as f:
            f.write(report)
        
        with open("documentation/threshold_optimization.json", "w") as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'optimization_results': optimization_results
            }, f, indent=2)
        
        print(f"\\n=== Threshold Optimization Complete ===")
        print(f"Results saved to: documentation/THRESHOLD_OPTIMIZATION_REPORT.md")
        
        return optimization_results

def main():
    """Main threshold optimization function."""
    
    image_path = "satoshi (1).png"
    ground_truth_path = "documentation/ground_truth/annotation_data.json"
    
    # Check if files exist
    if not Path(image_path).exists():
        print(f"Error: Image file not found: {image_path}")
        return
    
    if not Path(ground_truth_path).exists():
        print(f"Error: Ground truth file not found: {ground_truth_path}")
        print("Please complete the ground truth annotation first.")
        return
    
    # Initialize optimizer
    optimizer = ThresholdOptimizer(image_path, ground_truth_path)
    
    # Run optimization
    results = optimizer.run_optimization()
    
    if results:
        print("\\nThreshold optimization completed successfully!")
        print("Use the optimized thresholds for regional extraction.")
    else:
        print("\\nThreshold optimization failed. Check ground truth annotations.")

if __name__ == "__main__":
    main()