#!/usr/bin/env python3
"""
Systematic Extraction Research - Ground-up approach to get every 1 and 0
with rigorous verification and documentation for other developers.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import pandas as pd
from pathlib import Path

class SystematicExtractor:
    """Systematic approach to extract every possible bit with verification."""
    
    def __init__(self, image_path):
        self.image_path = image_path
        self.original = cv2.imread(image_path)
        self.rgb = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)
        self.gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        
        # Research log
        self.research_log = []
        self.ground_truth = {}
        self.extraction_results = {}
        
        # Grid parameters - to be calibrated
        self.grid_params = {
            'row_pitch': 31,
            'col_pitch': 25,
            'row0': 1,
            'col0': 5,
            'confidence': 0.0
        }
        
        self.log_research("Initialized SystematicExtractor", {
            'image_shape': self.original.shape,
            'initial_grid_params': self.grid_params
        })
        
    def log_research(self, action, data):
        """Log research steps for other developers."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'data': data
        }
        self.research_log.append(entry)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {action}")
        
    def manual_grid_calibration(self):
        """Manual grid calibration with systematic testing."""
        
        self.log_research("Starting manual grid calibration", {
            'method': 'systematic_parameter_sweep',
            'goal': 'find_optimal_grid_alignment'
        })
        
        # Test parameter ranges
        test_ranges = {
            'row_pitch': range(28, 35),
            'col_pitch': range(22, 28),
            'row0': range(0, 6),
            'col0': range(3, 8)
        }
        
        best_score = 0
        best_params = None
        test_results = []
        
        print("Testing grid parameter combinations...")
        total_tests = (len(test_ranges['row_pitch']) * len(test_ranges['col_pitch']) * 
                      len(test_ranges['row0']) * len(test_ranges['col0']))
        
        test_count = 0
        for row_pitch in test_ranges['row_pitch']:
            for col_pitch in test_ranges['col_pitch']:
                for row0 in test_ranges['row0']:
                    for col0 in test_ranges['col0']:
                        test_count += 1
                        if test_count % 20 == 0:
                            print(f"Progress: {test_count}/{total_tests}")
                        
                        score = self.evaluate_grid_alignment(row_pitch, col_pitch, row0, col0)
                        
                        result = {
                            'row_pitch': row_pitch,
                            'col_pitch': col_pitch,
                            'row0': row0,
                            'col0': col0,
                            'score': score
                        }
                        test_results.append(result)
                        
                        if score > best_score:
                            best_score = score
                            best_params = result
        
        # Update grid parameters
        if best_params:
            self.grid_params = {
                'row_pitch': best_params['row_pitch'],
                'col_pitch': best_params['col_pitch'],
                'row0': best_params['row0'],
                'col0': best_params['col0'],
                'confidence': best_score
            }
            
            self.log_research("Grid calibration complete", {
                'best_params': best_params,
                'best_score': best_score,
                'total_tests': len(test_results)
            })
        
        # Save test results
        self.save_calibration_results(test_results, best_params)
        
        return best_params, best_score
    
    def evaluate_grid_alignment(self, row_pitch, col_pitch, row0, col0):
        """Evaluate how well grid parameters align with actual image patterns."""
        
        # Sample multiple grid positions
        sample_positions = []
        for row in range(2, 52, 8):  # Every 8th row
            for col in range(2, 48, 8):  # Every 8th column
                center_y = row0 + row * row_pitch
                center_x = col0 + col * col_pitch
                
                # Check if position is within image bounds
                if (15 <= center_y < self.gray.shape[0] - 15 and 
                    15 <= center_x < self.gray.shape[1] - 15):
                    sample_positions.append((center_y, center_x))
        
        if not sample_positions:
            return 0.0
        
        # Evaluate alignment quality
        alignment_scores = []
        for center_y, center_x in sample_positions:
            # Extract region around center
            region_size = min(row_pitch, col_pitch) // 2
            y1 = center_y - region_size // 2
            y2 = center_y + region_size // 2
            x1 = center_x - region_size // 2
            x2 = center_x + region_size // 2
            
            if (y1 >= 0 and y2 < self.gray.shape[0] and 
                x1 >= 0 and x2 < self.gray.shape[1]):
                
                region = self.gray[y1:y2, x1:x2]
                
                # Calculate structure score
                # Higher variance = more structured content
                variance_score = np.var(region) / 1000.0
                
                # Edge density score
                edges = cv2.Canny(region, 30, 100)
                edge_score = np.sum(edges > 0) / edges.size
                
                # Gradient magnitude score
                grad_x = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
                grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                gradient_score = np.mean(grad_magnitude) / 100.0
                
                # Combined score
                combined_score = variance_score + edge_score + gradient_score
                alignment_scores.append(combined_score)
        
        return np.mean(alignment_scores) if alignment_scores else 0.0
    
    def save_calibration_results(self, test_results, best_params):
        """Save calibration results for analysis."""
        
        os.makedirs("documentation/calibration", exist_ok=True)
        
        # Save detailed results
        with open("documentation/calibration/grid_calibration_results.json", "w") as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'best_params': best_params,
                'all_results': test_results,
                'methodology': 'systematic_parameter_sweep'
            }, f, indent=2)
        
        # Create visualization
        self.visualize_calibration_results(test_results, best_params)
    
    def visualize_calibration_results(self, test_results, best_params):
        """Create visualization of calibration results."""
        
        if not test_results:
            return
        
        # Create heatmap of scores
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Group results by parameter pairs
        pitch_scores = {}
        origin_scores = {}
        
        for result in test_results:
            pitch_key = (result['row_pitch'], result['col_pitch'])
            origin_key = (result['row0'], result['col0'])
            
            if pitch_key not in pitch_scores:
                pitch_scores[pitch_key] = []
            if origin_key not in origin_scores:
                origin_scores[origin_key] = []
            
            pitch_scores[pitch_key].append(result['score'])
            origin_scores[origin_key].append(result['score'])
        
        # Average scores
        pitch_avg = {k: np.mean(v) for k, v in pitch_scores.items()}
        origin_avg = {k: np.mean(v) for k, v in origin_scores.items()}
        
        # Plot 1: Best parameters overlay
        overlay = self.rgb.copy()
        if best_params:
            # Draw grid with best parameters
            for row in range(0, 54, 3):
                y = best_params['row0'] + row * best_params['row_pitch']
                if 0 <= y < overlay.shape[0]:
                    cv2.line(overlay, (0, y), (overlay.shape[1], y), (255, 0, 0), 1)
            
            for col in range(0, 50, 3):
                x = best_params['col0'] + col * best_params['col_pitch']
                if 0 <= x < overlay.shape[1]:
                    cv2.line(overlay, (x, 0), (x, overlay.shape[0]), (255, 0, 0), 1)
        
        axes[0, 0].imshow(overlay)
        axes[0, 0].set_title(f'Best Grid Parameters\nScore: {best_params["score"]:.3f}' if best_params else 'No Best Params')
        axes[0, 0].axis('off')
        
        # Plot 2: Score distribution
        scores = [r['score'] for r in test_results]
        axes[0, 1].hist(scores, bins=20, alpha=0.7)
        axes[0, 1].set_title('Score Distribution')
        axes[0, 1].set_xlabel('Alignment Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(best_params['score'] if best_params else 0, color='red', linestyle='--', label='Best Score')
        axes[0, 1].legend()
        
        # Plot 3: Top 10 results
        top_results = sorted(test_results, key=lambda x: x['score'], reverse=True)[:10]
        param_labels = [f"({r['row_pitch']},{r['col_pitch']},{r['row0']},{r['col0']})" for r in top_results]
        param_scores = [r['score'] for r in top_results]
        
        axes[1, 0].barh(range(len(param_labels)), param_scores)
        axes[1, 0].set_yticks(range(len(param_labels)))
        axes[1, 0].set_yticklabels(param_labels, fontsize=8)
        axes[1, 0].set_title('Top 10 Parameter Combinations')
        axes[1, 0].set_xlabel('Score')
        
        # Plot 4: Parameter sensitivity
        row_pitches = sorted(set(r['row_pitch'] for r in test_results))
        pitch_avg_scores = [np.mean([r['score'] for r in test_results if r['row_pitch'] == rp]) for rp in row_pitches]
        
        axes[1, 1].plot(row_pitches, pitch_avg_scores, 'o-', label='Row Pitch')
        axes[1, 1].set_xlabel('Row Pitch')
        axes[1, 1].set_ylabel('Average Score')
        axes[1, 1].set_title('Parameter Sensitivity')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig("documentation/calibration/calibration_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log_research("Calibration visualization saved", {
            'file': 'documentation/calibration/calibration_analysis.png',
            'top_score': max(scores) if scores else 0
        })
    
    def create_ground_truth_dataset(self, num_samples=50):
        """Create ground truth dataset for validation."""
        
        self.log_research("Creating ground truth dataset", {
            'num_samples': num_samples,
            'method': 'manual_annotation'
        })
        
        # Select diverse sample positions
        sample_positions = []
        
        # Top region (sparse)
        for i in range(num_samples // 4):
            row = np.random.randint(0, 15)
            col = np.random.randint(0, 50)
            sample_positions.append((row, col, 'top_sparse'))
        
        # Middle region (dense)
        for i in range(num_samples // 4):
            row = np.random.randint(15, 35)
            col = np.random.randint(0, 50)
            sample_positions.append((row, col, 'middle_dense'))
        
        # Bottom region (mixed)
        for i in range(num_samples // 4):
            row = np.random.randint(35, 54)
            col = np.random.randint(0, 50)
            sample_positions.append((row, col, 'bottom_mixed'))
        
        # Random positions
        for i in range(num_samples - 3 * (num_samples // 4)):
            row = np.random.randint(0, 54)
            col = np.random.randint(0, 50)
            sample_positions.append((row, col, 'random'))
        
        # Create ground truth visualization
        ground_truth_data = []
        
        for row, col, region_type in sample_positions:
            cell_data = self.extract_cell_for_annotation(row, col)
            if cell_data:
                cell_data['region_type'] = region_type
                ground_truth_data.append(cell_data)
        
        # Create annotation interface
        self.create_annotation_interface(ground_truth_data)
        
        self.log_research("Ground truth dataset created", {
            'total_samples': len(ground_truth_data),
            'interface_created': True
        })
        
        return ground_truth_data
    
    def extract_cell_for_annotation(self, row, col):
        """Extract cell data for manual annotation."""
        
        # Calculate cell center
        center_y = self.grid_params['row0'] + row * self.grid_params['row_pitch']
        center_x = self.grid_params['col0'] + col * self.grid_params['col_pitch']
        
        # Check bounds
        if (center_y < 20 or center_y >= self.gray.shape[0] - 20 or
            center_x < 20 or center_x >= self.gray.shape[1] - 20):
            return None
        
        # Extract cell region
        cell_size = min(self.grid_params['row_pitch'], self.grid_params['col_pitch']) // 2
        y1 = center_y - cell_size // 2
        y2 = center_y + cell_size // 2
        x1 = center_x - cell_size // 2
        x2 = center_x + cell_size // 2
        
        if (y1 < 0 or y2 >= self.gray.shape[0] or 
            x1 < 0 or x2 >= self.gray.shape[1]):
            return None
        
        cell_region = self.gray[y1:y2, x1:x2]
        cell_rgb = self.rgb[y1:y2, x1:x2]
        
        # Calculate features
        mean_intensity = np.mean(cell_region)
        std_intensity = np.std(cell_region)
        min_intensity = np.min(cell_region)
        max_intensity = np.max(cell_region)
        
        # Simple classification suggestions
        otsu_threshold = cv2.threshold(cell_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        suggested_bit = '1' if mean_intensity > 127 else '0'
        
        return {
            'row': row,
            'col': col,
            'center_y': center_y,
            'center_x': center_x,
            'cell_region': cell_region,
            'cell_rgb': cell_rgb,
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'min_intensity': min_intensity,
            'max_intensity': max_intensity,
            'otsu_threshold': otsu_threshold,
            'suggested_bit': suggested_bit,
            'ground_truth_bit': None  # To be filled manually
        }
    
    def create_annotation_interface(self, ground_truth_data):
        """Create visual interface for manual annotation."""
        
        os.makedirs("documentation/ground_truth", exist_ok=True)
        
        # Create annotation sheets
        cells_per_sheet = 25
        num_sheets = (len(ground_truth_data) + cells_per_sheet - 1) // cells_per_sheet
        
        for sheet_idx in range(num_sheets):
            start_idx = sheet_idx * cells_per_sheet
            end_idx = min(start_idx + cells_per_sheet, len(ground_truth_data))
            sheet_data = ground_truth_data[start_idx:end_idx]
            
            # Create annotation sheet
            cols = 5
            rows = (len(sheet_data) + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(20, 4*rows))
            if rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, cell_data in enumerate(sheet_data):
                row_idx = i // cols
                col_idx = i % cols
                
                if row_idx < rows and col_idx < cols:
                    ax = axes[row_idx, col_idx]
                    
                    # Show cell region
                    ax.imshow(cell_data['cell_region'], cmap='gray')
                    
                    # Add information
                    title = f"({cell_data['row']},{cell_data['col']})\\n"
                    title += f"Mean: {cell_data['mean_intensity']:.1f}\\n"
                    title += f"Suggested: {cell_data['suggested_bit']}\\n"
                    title += f"Truth: ___"
                    
                    ax.set_title(title, fontsize=10)
                    ax.set_xticks([])
                    ax.set_yticks([])
            
            # Hide empty subplots
            for i in range(len(sheet_data), rows * cols):
                row_idx = i // cols
                col_idx = i % cols
                if row_idx < rows and col_idx < cols:
                    axes[row_idx, col_idx].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f"documentation/ground_truth/annotation_sheet_{sheet_idx+1}.png", 
                       dpi=150, bbox_inches='tight')
            plt.close()
        
        # Save annotation data
        annotation_data = []
        for cell_data in ground_truth_data:
            annotation_data.append({
                'row': cell_data['row'],
                'col': cell_data['col'],
                'mean_intensity': cell_data['mean_intensity'],
                'suggested_bit': cell_data['suggested_bit'],
                'ground_truth_bit': None,
                'region_type': cell_data['region_type']
            })
        
        with open("documentation/ground_truth/annotation_data.json", "w") as f:
            json.dump(annotation_data, f, indent=2)
        
        # Create annotation instructions
        instructions = f"""
# Ground Truth Annotation Instructions

## Overview
{len(ground_truth_data)} cells need manual annotation across {num_sheets} sheets.

## Files Created
- annotation_sheet_1.png to annotation_sheet_{num_sheets}.png
- annotation_data.json (to be filled with ground truth)

## Instructions
1. Open each annotation sheet image
2. For each cell, determine if it's a 0 or 1 based on visual inspection
3. Fill in the ground truth values in annotation_data.json
4. Look for patterns like:
   - Dark regions = 0
   - Light regions = 1
   - Consider the background pattern structure

## Quality Guidelines
- Be consistent with threshold decisions
- When in doubt, mark as 'uncertain'
- Consider the overall pattern context
- Take breaks to maintain accuracy

## Format for annotation_data.json
For each cell, update "ground_truth_bit" with:
- "0" for dark/zero regions
- "1" for light/one regions  
- "uncertain" for ambiguous cases

## Next Steps
After annotation, run validation to measure extraction accuracy.
"""
        
        with open("documentation/ground_truth/ANNOTATION_INSTRUCTIONS.md", "w") as f:
            f.write(instructions)
        
        print(f"Created {num_sheets} annotation sheets in documentation/ground_truth/")
        print("Please manually annotate the cells and update annotation_data.json")
    
    def generate_research_documentation(self):
        """Generate comprehensive research documentation."""
        
        os.makedirs("documentation/research", exist_ok=True)
        
        # Generate methodology report
        methodology_report = self.create_methodology_report()
        with open("documentation/research/METHODOLOGY_REPORT.md", "w") as f:
            f.write(methodology_report)
        
        # Save complete research log
        with open("documentation/research/research_log.json", "w") as f:
            json.dump(self.research_log, f, indent=2)
        
        # Generate current status report
        status_report = self.create_status_report()
        with open("documentation/research/STATUS_REPORT.md", "w") as f:
            f.write(status_report)
        
        self.log_research("Research documentation generated", {
            'methodology_report': 'documentation/research/METHODOLOGY_REPORT.md',
            'research_log': 'documentation/research/research_log.json',
            'status_report': 'documentation/research/STATUS_REPORT.md'
        })
        
        print("Research documentation generated in documentation/research/")
    
    def create_methodology_report(self):
        """Create comprehensive methodology report."""
        
        report = []
        report.append("# Systematic Binary Extraction Methodology")
        report.append("## Research Documentation for Developers")
        report.append("")
        report.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("**Author**: Claude Code Assistant")
        report.append("**Project**: Satoshi Poster Binary Extraction")
        report.append("")
        
        # Problem statement
        report.append("## Problem Statement")
        report.append("")
        report.append("Previous extraction attempts achieved only ~30-40% accuracy due to:")
        report.append("1. Incorrect grid parameter assumptions")
        report.append("2. Poor threshold selection")
        report.append("3. Lack of systematic validation")
        report.append("4. No ground truth dataset")
        report.append("")
        
        # Methodology
        report.append("## Systematic Methodology")
        report.append("")
        report.append("### Phase 1: Grid Calibration")
        report.append("- **Method**: Exhaustive parameter sweep")
        report.append("- **Parameters**: row_pitch (28-34), col_pitch (22-27), row0 (0-5), col0 (3-7)")
        report.append("- **Evaluation**: Multi-metric alignment scoring")
        report.append("- **Metrics**: Variance, edge density, gradient magnitude")
        report.append("")
        
        report.append("### Phase 2: Ground Truth Creation")
        report.append("- **Method**: Manual annotation of diverse sample")
        report.append("- **Sample Size**: 50 cells across different regions")
        report.append("- **Regions**: Top sparse, middle dense, bottom mixed, random")
        report.append("- **Interface**: Visual annotation sheets")
        report.append("")
        
        report.append("### Phase 3: Validation Framework")
        report.append("- **Method**: Compare extraction vs ground truth")
        report.append("- **Metrics**: Accuracy, precision, recall, F1-score")
        report.append("- **Thresholds**: Region-specific optimization")
        report.append("")
        
        # Grid calibration results
        if self.grid_params['confidence'] > 0:
            report.append("## Grid Calibration Results")
            report.append("")
            report.append(f"**Optimal Parameters**:")
            report.append(f"- Row pitch: {self.grid_params['row_pitch']}")
            report.append(f"- Column pitch: {self.grid_params['col_pitch']}")
            report.append(f"- Row origin: {self.grid_params['row0']}")
            report.append(f"- Column origin: {self.grid_params['col0']}")
            report.append(f"- Confidence score: {self.grid_params['confidence']:.3f}")
            report.append("")
        
        # For other developers
        report.append("## For Other Developers")
        report.append("")
        report.append("### Key Files")
        report.append("- `systematic_extraction_research.py` - Main research code")
        report.append("- `documentation/calibration/` - Grid calibration results")
        report.append("- `documentation/ground_truth/` - Manual annotation data")
        report.append("- `documentation/research/` - Research logs and reports")
        report.append("")
        
        report.append("### Replication Instructions")
        report.append("1. Run grid calibration: `extractor.manual_grid_calibration()`")
        report.append("2. Create ground truth: `extractor.create_ground_truth_dataset()`")
        report.append("3. Manually annotate cells using annotation sheets")
        report.append("4. Validate extraction accuracy")
        report.append("5. Optimize thresholds based on results")
        report.append("")
        
        report.append("### Next Steps")
        report.append("1. Complete manual annotation of ground truth dataset")
        report.append("2. Implement region-specific threshold optimization")
        report.append("3. Create consensus extraction using multiple methods")
        report.append("4. Validate final extraction accuracy > 90%")
        report.append("5. Extract complete binary matrix")
        report.append("")
        
        return "\\n".join(report)
    
    def create_status_report(self):
        """Create current status report."""
        
        report = []
        report.append("# Current Research Status")
        report.append("")
        report.append(f"**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Completed tasks
        report.append("## [COMPLETED] Completed Tasks")
        report.append("")
        if self.grid_params['confidence'] > 0:
            report.append(f"- [x] Grid calibration (confidence: {self.grid_params['confidence']:.3f})")
        else:
            report.append("- [ ] Grid calibration")
        report.append("- [x] Research framework setup")
        report.append("- [x] Ground truth annotation interface")
        report.append("- [x] Methodology documentation")
        report.append("")
        
        # Current tasks
        report.append("## [CURRENT] Current Tasks")
        report.append("")
        report.append("- [ ] Manual annotation of ground truth dataset")
        report.append("- [ ] Validation framework implementation")
        report.append("- [ ] Threshold optimization")
        report.append("")
        
        # Next tasks
        report.append("## [NEXT] Next Tasks")
        report.append("")
        report.append("- [ ] Region-specific extraction")
        report.append("- [ ] Consensus method implementation")
        report.append("- [ ] Final accuracy validation")
        report.append("- [ ] Complete binary matrix extraction")
        report.append("")
        
        # Research log summary
        report.append("## [SUMMARY] Research Log Summary")
        report.append("")
        report.append(f"Total research actions: {len(self.research_log)}")
        
        if self.research_log:
            report.append("Recent actions:")
            for entry in self.research_log[-5:]:
                report.append(f"- {entry['timestamp']}: {entry['action']}")
        
        report.append("")
        
        return "\\n".join(report)
    
    def run_systematic_research(self):
        """Run complete systematic research process."""
        
        print("=== Systematic Binary Extraction Research ===")
        print("Goal: Extract every possible 1 and 0 with rigorous verification")
        print("")
        
        # Phase 1: Grid calibration
        print("Phase 1: Grid Calibration")
        best_params, best_score = self.manual_grid_calibration()
        
        if best_score > 0:
            print(f"[OK] Grid calibration complete - Score: {best_score:.3f}")
        else:
            print("[ERROR] Grid calibration failed")
            return
        
        # Phase 2: Ground truth creation
        print("\\nPhase 2: Ground Truth Creation")
        ground_truth_data = self.create_ground_truth_dataset()
        print(f"[OK] Ground truth interface created - {len(ground_truth_data)} samples")
        
        # Phase 3: Documentation
        print("\\nPhase 3: Documentation")
        self.generate_research_documentation()
        print("[OK] Research documentation generated")
        
        # Summary
        print("\\n=== Research Phase Complete ===")
        print("Next steps:")
        print("1. Complete manual annotation using annotation sheets")
        print("2. Run validation to measure accuracy")
        print("3. Optimize thresholds based on results")
        print("4. Extract final binary matrix")
        print("")
        print("Files created:")
        print("- documentation/calibration/ - Grid calibration results")
        print("- documentation/ground_truth/ - Manual annotation interface")
        print("- documentation/research/ - Research logs and methodology")

def main():
    """Main research function."""
    
    image_path = "satoshi (1).png"
    
    if not os.path.exists(image_path):
        print("Error: satoshi (1).png not found")
        return
    
    # Initialize systematic extractor
    extractor = SystematicExtractor(image_path)
    
    # Run systematic research
    extractor.run_systematic_research()
    
    print("\\n[COMPLETE] Systematic research complete!")
    print("All methodology and results documented for other developers.")
    print("Check documentation/ directory for complete research records.")

if __name__ == "__main__":
    main()