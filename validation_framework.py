#!/usr/bin/env python3
"""
Validation Framework for Binary Extraction Accuracy
Uses ground truth annotations to measure extraction performance.
"""

import cv2
import numpy as np
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from systematic_extraction_research import SystematicExtractor

class ValidationFramework:
    """Framework for validating binary extraction accuracy."""
    
    def __init__(self, image_path, ground_truth_path):
        self.image_path = image_path
        self.ground_truth_path = ground_truth_path
        self.extractor = SystematicExtractor(image_path)
        
        # Load optimized grid parameters from calibration
        self.extractor.grid_params = {
            'row_pitch': 33,
            'col_pitch': 26,
            'row0': 5,
            'col0': 3,
            'confidence': 1.350
        }
        
        self.ground_truth_data = None
        self.validation_results = {}
        
    def load_ground_truth(self):
        """Load ground truth annotations."""
        try:
            with open(self.ground_truth_path, 'r') as f:
                data = json.load(f)
            
            # Filter out unannotated entries
            annotated_data = [
                item for item in data 
                if item.get('ground_truth_bit') is not None
            ]
            
            if not annotated_data:
                print("No ground truth annotations found. Please complete annotation first.")
                return False
            
            self.ground_truth_data = annotated_data
            print(f"Loaded {len(annotated_data)} ground truth annotations")
            return True
            
        except FileNotFoundError:
            print(f"Ground truth file not found: {self.ground_truth_path}")
            return False
        except json.JSONDecodeError:
            print(f"Invalid JSON in ground truth file: {self.ground_truth_path}")
            return False
    
    def extract_validation_cells(self):
        """Extract bits for all ground truth cells using current method."""
        if not self.ground_truth_data:
            print("No ground truth data loaded")
            return []
        
        extraction_results = []
        
        for gt_cell in self.ground_truth_data:
            row = gt_cell['row']
            col = gt_cell['col']
            
            # Extract using systematic extractor
            cell_result = self.extractor.extract_single_cell(row, col)
            
            if cell_result:
                extraction_results.append({
                    'row': row,
                    'col': col,
                    'extracted_bit': cell_result['bit'],
                    'confidence': cell_result['confidence'],
                    'ground_truth_bit': gt_cell['ground_truth_bit'],
                    'mean_intensity': cell_result['mean_intensity'],
                    'contrast': cell_result['contrast'],
                    'variance': cell_result['variance'],
                    'methods': cell_result['methods']
                })
        
        return extraction_results
    
    def calculate_accuracy_metrics(self, extraction_results):
        """Calculate comprehensive accuracy metrics."""
        if not extraction_results:
            return {}
        
        # Convert to pandas for easier analysis
        df = pd.DataFrame(extraction_results)
        
        # Filter out uncertain ground truth
        certain_df = df[df['ground_truth_bit'].isin(['0', '1'])]
        
        if len(certain_df) == 0:
            return {'error': 'No certain ground truth annotations'}
        
        # Basic accuracy metrics
        correct_predictions = certain_df[
            certain_df['extracted_bit'] == certain_df['ground_truth_bit']
        ]
        
        total_certain = len(certain_df)
        correct_count = len(correct_predictions)
        accuracy = correct_count / total_certain if total_certain > 0 else 0
        
        # Precision and recall for each class
        tp_0 = len(certain_df[
            (certain_df['extracted_bit'] == '0') & 
            (certain_df['ground_truth_bit'] == '0')
        ])
        fp_0 = len(certain_df[
            (certain_df['extracted_bit'] == '0') & 
            (certain_df['ground_truth_bit'] == '1')
        ])
        fn_0 = len(certain_df[
            (certain_df['extracted_bit'] == '1') & 
            (certain_df['ground_truth_bit'] == '0')
        ])
        
        tp_1 = len(certain_df[
            (certain_df['extracted_bit'] == '1') & 
            (certain_df['ground_truth_bit'] == '1')
        ])
        fp_1 = len(certain_df[
            (certain_df['extracted_bit'] == '1') & 
            (certain_df['ground_truth_bit'] == '0')
        ])
        fn_1 = len(certain_df[
            (certain_df['extracted_bit'] == '0') & 
            (certain_df['ground_truth_bit'] == '1')
        ])
        
        # Calculate metrics
        precision_0 = tp_0 / (tp_0 + fp_0) if (tp_0 + fp_0) > 0 else 0
        recall_0 = tp_0 / (tp_0 + fn_0) if (tp_0 + fn_0) > 0 else 0
        f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0
        
        precision_1 = tp_1 / (tp_1 + fp_1) if (tp_1 + fp_1) > 0 else 0
        recall_1 = tp_1 / (tp_1 + fn_1) if (tp_1 + fn_1) > 0 else 0
        f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0
        
        # Overall F1 score
        f1_macro = (f1_0 + f1_1) / 2
        
        # Confidence analysis
        high_conf_correct = len(correct_predictions[correct_predictions['confidence'] > 0.8])
        high_conf_total = len(certain_df[certain_df['confidence'] > 0.8])
        high_conf_accuracy = high_conf_correct / high_conf_total if high_conf_total > 0 else 0
        
        # Uncertainty analysis
        uncertain_count = len(df[df['extracted_bit'] == 'uncertain'])
        uncertain_ratio = uncertain_count / len(df)
        
        return {
            'total_samples': len(df),
            'certain_ground_truth': total_certain,
            'uncertain_ground_truth': len(df) - total_certain,
            'accuracy': accuracy,
            'correct_predictions': correct_count,
            'class_0': {
                'precision': precision_0,
                'recall': recall_0,
                'f1': f1_0,
                'tp': tp_0,
                'fp': fp_0,
                'fn': fn_0
            },
            'class_1': {
                'precision': precision_1,
                'recall': recall_1,
                'f1': f1_1,
                'tp': tp_1,
                'fp': fp_1,
                'fn': fn_1
            },
            'f1_macro': f1_macro,
            'high_confidence_accuracy': high_conf_accuracy,
            'high_confidence_samples': high_conf_total,
            'uncertain_predictions': uncertain_count,
            'uncertain_ratio': uncertain_ratio
        }
    
    def analyze_error_patterns(self, extraction_results):
        """Analyze patterns in extraction errors."""
        if not extraction_results:
            return {}
        
        df = pd.DataFrame(extraction_results)
        certain_df = df[df['ground_truth_bit'].isin(['0', '1'])]
        
        if len(certain_df) == 0:
            return {}
        
        # Find errors
        errors = certain_df[certain_df['extracted_bit'] != certain_df['ground_truth_bit']]
        
        error_analysis = {
            'total_errors': len(errors),
            'error_rate': len(errors) / len(certain_df),
            'error_types': {}
        }
        
        if len(errors) > 0:
            # Error by type
            false_positives = errors[errors['ground_truth_bit'] == '0']  # predicted 1, actual 0
            false_negatives = errors[errors['ground_truth_bit'] == '1']  # predicted 0, actual 1
            
            error_analysis['error_types'] = {
                'false_positives': len(false_positives),
                'false_negatives': len(false_negatives)
            }
            
            # Error by confidence
            high_conf_errors = errors[errors['confidence'] > 0.8]
            error_analysis['high_confidence_errors'] = len(high_conf_errors)
            
            # Error by intensity characteristics
            if len(errors) > 0:
                error_analysis['error_intensity_stats'] = {
                    'mean_intensity': errors['mean_intensity'].mean(),
                    'mean_contrast': errors['contrast'].mean(),
                    'mean_variance': errors['variance'].mean()
                }
            
            # Correct predictions stats for comparison
            correct = certain_df[certain_df['extracted_bit'] == certain_df['ground_truth_bit']]
            if len(correct) > 0:
                error_analysis['correct_intensity_stats'] = {
                    'mean_intensity': correct['mean_intensity'].mean(),
                    'mean_contrast': correct['contrast'].mean(),
                    'mean_variance': correct['variance'].mean()
                }
        
        return error_analysis
    
    def create_validation_visualization(self, extraction_results, accuracy_metrics):
        """Create comprehensive validation visualization."""
        if not extraction_results:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        df = pd.DataFrame(extraction_results)
        certain_df = df[df['ground_truth_bit'].isin(['0', '1'])]
        
        # 1. Accuracy summary
        ax = axes[0, 0]
        metrics = ['Accuracy', 'Precision (0)', 'Recall (0)', 'Precision (1)', 'Recall (1)', 'F1 Macro']
        values = [
            accuracy_metrics['accuracy'],
            accuracy_metrics['class_0']['precision'],
            accuracy_metrics['class_0']['recall'],
            accuracy_metrics['class_1']['precision'],
            accuracy_metrics['class_1']['recall'],
            accuracy_metrics['f1_macro']
        ]
        
        bars = ax.bar(metrics, values, color=['skyblue', 'lightcoral', 'lightcoral', 'lightgreen', 'lightgreen', 'gold'])
        ax.set_ylim(0, 1)
        ax.set_ylabel('Score')
        ax.set_title('Validation Metrics')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 2. Confusion matrix
        ax = axes[0, 1]
        if len(certain_df) > 0:
            tp_0 = accuracy_metrics['class_0']['tp']
            fp_0 = accuracy_metrics['class_0']['fp']
            fn_0 = accuracy_metrics['class_0']['fn']
            tp_1 = accuracy_metrics['class_1']['tp']
            
            confusion_matrix = np.array([[tp_0, fp_0], [fn_0, tp_1]])
            
            im = ax.imshow(confusion_matrix, cmap='Blues')
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Pred 0', 'Pred 1'])
            ax.set_yticklabels(['True 0', 'True 1'])
            ax.set_title('Confusion Matrix')
            
            # Add text annotations
            for i in range(2):
                for j in range(2):
                    text = ax.text(j, i, confusion_matrix[i, j],
                                 ha="center", va="center", color="black", fontsize=12)
        
        # 3. Confidence distribution
        ax = axes[0, 2]
        if len(df) > 0:
            ax.hist(df['confidence'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_xlabel('Confidence Score')
            ax.set_ylabel('Frequency')
            ax.set_title('Confidence Distribution')
            ax.axvline(0.8, color='red', linestyle='--', label='High Confidence Threshold')
            ax.legend()
        
        # 4. Intensity vs Ground Truth
        ax = axes[1, 0]
        if len(certain_df) > 0:
            zeros = certain_df[certain_df['ground_truth_bit'] == '0']
            ones = certain_df[certain_df['ground_truth_bit'] == '1']
            
            if len(zeros) > 0:
                ax.scatter(zeros['mean_intensity'], [0] * len(zeros), 
                          c='blue', alpha=0.6, label='Ground Truth 0')
            if len(ones) > 0:
                ax.scatter(ones['mean_intensity'], [1] * len(ones), 
                          c='red', alpha=0.6, label='Ground Truth 1')
            
            ax.set_xlabel('Mean Intensity')
            ax.set_ylabel('Ground Truth Bit')
            ax.set_title('Intensity vs Ground Truth')
            ax.set_yticks([0, 1])
            ax.legend()
        
        # 5. Error analysis
        ax = axes[1, 1]
        if len(certain_df) > 0:
            correct = certain_df[certain_df['extracted_bit'] == certain_df['ground_truth_bit']]
            errors = certain_df[certain_df['extracted_bit'] != certain_df['ground_truth_bit']]
            
            categories = ['Correct', 'Errors']
            counts = [len(correct), len(errors)]
            colors = ['green', 'red']
            
            bars = ax.bar(categories, counts, color=colors, alpha=0.7)
            ax.set_ylabel('Count')
            ax.set_title('Correct vs Errors')
            
            # Add count labels
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{count}', ha='center', va='bottom', fontsize=12)
        
        # 6. Method comparison
        ax = axes[1, 2]
        if len(df) > 0:
            # Compare different extraction methods
            method_accuracy = {}
            for method in ['simple', 'otsu', 'adaptive']:
                method_correct = 0
                method_total = 0
                
                for _, row in certain_df.iterrows():
                    if method in row['methods']:
                        method_total += 1
                        if row['methods'][method] == row['ground_truth_bit']:
                            method_correct += 1
                
                if method_total > 0:
                    method_accuracy[method] = method_correct / method_total
            
            if method_accuracy:
                methods = list(method_accuracy.keys())
                accuracies = list(method_accuracy.values())
                
                bars = ax.bar(methods, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'])
                ax.set_ylim(0, 1)
                ax.set_ylabel('Accuracy')
                ax.set_title('Method Comparison')
                
                # Add accuracy labels
                for bar, acc in zip(bars, accuracies):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('documentation/validation_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("Validation visualization saved: documentation/validation_results.png")
    
    def generate_validation_report(self, extraction_results, accuracy_metrics, error_analysis):
        """Generate comprehensive validation report."""
        
        report = []
        report.append("# Binary Extraction Validation Report")
        report.append("## Accuracy Assessment with Ground Truth Data")
        report.append("")
        report.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Method**: Ground truth validation with {len(extraction_results)} samples")
        report.append("")
        
        # Overview
        report.append("## Validation Overview")
        report.append("")
        report.append(f"- **Total samples**: {accuracy_metrics['total_samples']}")
        report.append(f"- **Certain ground truth**: {accuracy_metrics['certain_ground_truth']}")
        report.append(f"- **Uncertain ground truth**: {accuracy_metrics['uncertain_ground_truth']}")
        report.append(f"- **Overall accuracy**: {accuracy_metrics['accuracy']:.3f} ({accuracy_metrics['accuracy']*100:.1f}%)")
        report.append("")
        
        # Detailed metrics
        report.append("## Detailed Accuracy Metrics")
        report.append("")
        report.append("### Class 0 (Dark/Zero)")
        report.append(f"- **Precision**: {accuracy_metrics['class_0']['precision']:.3f}")
        report.append(f"- **Recall**: {accuracy_metrics['class_0']['recall']:.3f}")
        report.append(f"- **F1 Score**: {accuracy_metrics['class_0']['f1']:.3f}")
        report.append("")
        
        report.append("### Class 1 (Light/One)")
        report.append(f"- **Precision**: {accuracy_metrics['class_1']['precision']:.3f}")
        report.append(f"- **Recall**: {accuracy_metrics['class_1']['recall']:.3f}")
        report.append(f"- **F1 Score**: {accuracy_metrics['class_1']['f1']:.3f}")
        report.append("")
        
        report.append("### Overall Performance")
        report.append(f"- **Macro F1 Score**: {accuracy_metrics['f1_macro']:.3f}")
        report.append(f"- **High Confidence Accuracy**: {accuracy_metrics['high_confidence_accuracy']:.3f}")
        report.append(f"- **High Confidence Samples**: {accuracy_metrics['high_confidence_samples']}")
        report.append("")
        
        # Error analysis
        report.append("## Error Analysis")
        report.append("")
        if error_analysis:
            report.append(f"- **Total errors**: {error_analysis['total_errors']}")
            report.append(f"- **Error rate**: {error_analysis['error_rate']:.3f} ({error_analysis['error_rate']*100:.1f}%)")
            
            if 'error_types' in error_analysis:
                report.append(f"- **False positives**: {error_analysis['error_types']['false_positives']}")
                report.append(f"- **False negatives**: {error_analysis['error_types']['false_negatives']}")
            
            if 'high_confidence_errors' in error_analysis:
                report.append(f"- **High confidence errors**: {error_analysis['high_confidence_errors']}")
        
        report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("")
        
        accuracy = accuracy_metrics['accuracy']
        if accuracy > 0.90:
            report.append("### [EXCELLENT] Accuracy > 90%")
            report.append("- Ready for full binary matrix extraction")
            report.append("- Consider expanding ground truth for final validation")
        elif accuracy > 0.80:
            report.append("### [GOOD] Accuracy 80-90%")
            report.append("- Good progress, minor improvements needed")
            report.append("- Focus on reducing specific error types")
        elif accuracy > 0.70:
            report.append("### [MODERATE] Accuracy 70-80%")
            report.append("- Significant improvement over previous 30-40%")
            report.append("- Continue threshold optimization")
        else:
            report.append("### [NEEDS WORK] Accuracy < 70%")
            report.append("- Further algorithm refinement required")
            report.append("- Consider alternative extraction methods")
        
        report.append("")
        report.append("### Specific Improvements")
        
        if error_analysis and 'error_types' in error_analysis:
            fp = error_analysis['error_types']['false_positives']
            fn = error_analysis['error_types']['false_negatives']
            
            if fp > fn:
                report.append("- **Reduce false positives**: Threshold too low, increase threshold")
            elif fn > fp:
                report.append("- **Reduce false negatives**: Threshold too high, decrease threshold")
            else:
                report.append("- **Balanced errors**: Consider regional threshold adaptation")
        
        if accuracy_metrics['uncertain_ratio'] > 0.2:
            report.append("- **High uncertainty**: Improve confidence scoring mechanism")
        
        report.append("")
        report.append("## Next Steps")
        report.append("")
        report.append("1. **Threshold Optimization**: Use validation results to tune thresholds")
        report.append("2. **Regional Processing**: Apply different thresholds to different poster regions")
        report.append("3. **Method Refinement**: Improve extraction algorithms based on error patterns")
        report.append("4. **Expanded Validation**: Create larger ground truth dataset if needed")
        report.append("5. **Full Extraction**: Extract complete 54x50 binary matrix")
        
        return "\\n".join(report)
    
    def run_validation(self):
        """Run complete validation process."""
        print("=== Binary Extraction Validation ===")
        
        # Load ground truth
        if not self.load_ground_truth():
            print("Cannot proceed without ground truth annotations")
            return None
        
        # Extract validation cells
        print("Extracting validation cells...")
        extraction_results = self.extract_validation_cells()
        
        if not extraction_results:
            print("No extraction results obtained")
            return None
        
        # Calculate accuracy metrics
        print("Calculating accuracy metrics...")
        accuracy_metrics = self.calculate_accuracy_metrics(extraction_results)
        
        if 'error' in accuracy_metrics:
            print(f"Error in accuracy calculation: {accuracy_metrics['error']}")
            return None
        
        # Analyze error patterns
        print("Analyzing error patterns...")
        error_analysis = self.analyze_error_patterns(extraction_results)
        
        # Create visualizations
        print("Creating validation visualization...")
        self.create_validation_visualization(extraction_results, accuracy_metrics)
        
        # Generate report
        print("Generating validation report...")
        report = self.generate_validation_report(extraction_results, accuracy_metrics, error_analysis)
        
        # Save results
        Path("documentation").mkdir(exist_ok=True)
        
        with open("documentation/VALIDATION_REPORT.md", "w") as f:
            f.write(report)
        
        validation_data = {
            'timestamp': datetime.now().isoformat(),
            'extraction_results': extraction_results,
            'accuracy_metrics': accuracy_metrics,
            'error_analysis': error_analysis
        }
        
        with open("documentation/validation_results.json", "w") as f:
            json.dump(validation_data, f, indent=2)
        
        print(f"\\n=== Validation Complete ===")
        print(f"Overall accuracy: {accuracy_metrics['accuracy']:.3f} ({accuracy_metrics['accuracy']*100:.1f}%)")
        print(f"Results saved to: documentation/VALIDATION_REPORT.md")
        
        return validation_data

def main():
    """Main validation function."""
    
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
    
    # Initialize validation framework
    validator = ValidationFramework(image_path, ground_truth_path)
    
    # Run validation
    results = validator.run_validation()
    
    if results:
        print("\\nValidation completed successfully!")
        print("Use the results to optimize extraction parameters.")
    else:
        print("\\nValidation failed. Check ground truth annotations.")

if __name__ == "__main__":
    main()