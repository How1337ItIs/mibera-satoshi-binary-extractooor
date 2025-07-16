#!/usr/bin/env python3
"""
Honest validation tools for binary extraction accuracy assessment.

Created by Claude Code
Date: July 16, 2025
Purpose: Provide honest accuracy validation instead of misleading pipeline confidence
"""
import csv
import cv2
import numpy as np
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any

def validate_extraction_accuracy(
    image_path: str,
    extracted_csv: str,
    sample_size: int = 50,
    output_dir: str = "validation_output"
) -> Dict[str, Any]:
    """
    Validate extraction accuracy by manually checking a sample of cells.
    
    Args:
        image_path: Path to original poster image
        extracted_csv: Path to extracted results CSV
        sample_size: Number of cells to validate
        output_dir: Directory to save validation images
        
    Returns:
        Validation results dictionary
    """
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load original image
    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"Could not load image: {image_path}"}
    
    # Load extracted results
    extracted = {}
    with open(extracted_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            r, c, bit = int(row['row']), int(row['col']), row['digit']
            extracted[(r, c)] = bit
    
    if not extracted:
        return {"error": "No extracted data found"}
    
    # Sample cells for validation
    all_cells = list(extracted.keys())
    sample_cells = random.sample(all_cells, min(sample_size, len(all_cells)))
    
    # Grid parameters (should match extraction)
    row_pitch, col_pitch = 31, 25
    row0, col0 = 1, 5
    
    validation_results = []
    matches = 0
    mismatches = 0
    ambiguous = 0
    
    print(f"Validating {len(sample_cells)} sample cells...")
    print("=" * 60)
    
    for i, (r, c) in enumerate(sample_cells):
        # Calculate actual pixel position
        y = row0 + r * row_pitch
        x = col0 + c * col_pitch
        
        # Extract cell region
        cell_size = 10  # ±5 pixels around center
        cell_region = img[max(0, y-5):min(img.shape[0], y+6), 
                         max(0, x-5):min(img.shape[1], x+6)]
        
        # Get extracted bit
        extracted_bit = extracted.get((r, c), 'missing')
        
        # Save cell image for visual inspection
        cell_filename = f"{output_dir}/cell_{r:02d}_{c:02d}_{extracted_bit}.png"
        if cell_region.size > 0:
            # Enlarge for easier inspection
            enlarged = cv2.resize(cell_region, (50, 50), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(cell_filename, enlarged)
        
        # Manual classification using same pipeline logic
        manual_bit = manual_classify_cell(cell_region, extracted_bit)
        
        # Compare results
        if manual_bit == extracted_bit:
            result = "MATCH"
            matches += 1
        elif manual_bit == "ambiguous":
            result = "AMBIGUOUS"
            ambiguous += 1
        else:
            result = "MISMATCH"
            mismatches += 1
        
        validation_results.append({
            "row": r, "col": c, 
            "extracted": extracted_bit,
            "manual": manual_bit,
            "result": result,
            "image": cell_filename
        })
        
        print(f"Cell ({r:2d},{c:2d}): {extracted_bit} -> {manual_bit} [{result}]")
    
    # Calculate accuracy statistics
    total_validated = len(sample_cells)
    match_rate = matches / total_validated * 100
    ambiguous_rate = ambiguous / total_validated * 100
    mismatch_rate = mismatches / total_validated * 100
    
    print("=" * 60)
    print(f"VALIDATION RESULTS (n={total_validated}):")
    print(f"  Matches:    {matches:3d} ({match_rate:5.1f}%)")
    print(f"  Ambiguous:  {ambiguous:3d} ({ambiguous_rate:5.1f}%)")
    print(f"  Mismatches: {mismatches:3d} ({mismatch_rate:5.1f}%)")
    print(f"")
    print(f"ESTIMATED TRUE ACCURACY: {match_rate:.1f}% ± {(match_rate * 0.2):.1f}%")
    
    # Generate warnings
    warnings = []
    if match_rate < 70:
        warnings.append("LOW ACCURACY: True accuracy appears to be below 70%")
    if ambiguous_rate > 30:
        warnings.append("HIGH AMBIGUITY: Many cells are difficult to classify")
    if mismatch_rate > 20:
        warnings.append("HIGH ERROR RATE: Many clear classification errors")
    
    return {
        "total_validated": total_validated,
        "matches": matches,
        "ambiguous": ambiguous,
        "mismatches": mismatches,
        "match_rate": match_rate,
        "ambiguous_rate": ambiguous_rate,
        "mismatch_rate": mismatch_rate,
        "estimated_accuracy": match_rate,
        "warnings": warnings,
        "validation_details": validation_results
    }

def manual_classify_cell(cell_region: np.ndarray, extracted_bit: str) -> str:
    """
    Manually classify a cell using visual inspection logic.
    
    Args:
        cell_region: Cell image region
        extracted_bit: What the pipeline extracted
        
    Returns:
        Manual classification: '0', '1', or 'ambiguous'
    """
    if cell_region.size == 0:
        return "ambiguous"
    
    # Convert to HSV and analyze saturation channel
    hsv = cv2.cvtColor(cell_region, cv2.COLOR_BGR2HSV)
    sat_channel = hsv[:, :, 1]
    
    # Background subtraction (same as pipeline)
    blurred = cv2.GaussianBlur(sat_channel, (0, 0), 25)
    if blurred.size > 0:
        subtracted = cv2.subtract(sat_channel, blurred)
        
        # Threshold (same as pipeline)
        _, binary = cv2.threshold(subtracted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Calculate white percentage
        white_pixels = np.sum(binary == 255)
        total_pixels = binary.size
        white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0
        
        # Conservative manual classification
        if white_ratio > 0.9:  # Very clear 1
            return '1'
        elif white_ratio < 0.1:  # Very clear 0
            return '0'
        else:
            return 'ambiguous'  # Uncertain
    
    return "ambiguous"

def create_validation_report(validation_results: Dict[str, Any], output_file: str = "validation_report.md"):
    """Create a markdown report of validation results."""
    
    with open(output_file, 'w') as f:
        f.write("# Binary Extraction Validation Report\n\n")
        f.write(f"**Date:** {np.datetime64('now')}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Total cells validated:** {validation_results['total_validated']}\n")
        f.write(f"- **Matches:** {validation_results['matches']} ({validation_results['match_rate']:.1f}%)\n")
        f.write(f"- **Ambiguous:** {validation_results['ambiguous']} ({validation_results['ambiguous_rate']:.1f}%)\n")
        f.write(f"- **Mismatches:** {validation_results['mismatches']} ({validation_results['mismatch_rate']:.1f}%)\n")
        f.write(f"- **Estimated True Accuracy:** {validation_results['estimated_accuracy']:.1f}%\n\n")
        
        if validation_results['warnings']:
            f.write("## Warnings\n\n")
            for warning in validation_results['warnings']:
                f.write(f"- ⚠️ {warning}\n")
            f.write("\n")
        
        f.write("## Detailed Results\n\n")
        f.write("| Row | Col | Extracted | Manual | Result | Image |\n")
        f.write("|-----|-----|-----------|--------|--------|-------|\n")
        
        for detail in validation_results['validation_details']:
            f.write(f"| {detail['row']} | {detail['col']} | {detail['extracted']} | {detail['manual']} | {detail['result']} | {detail['image']} |\n")
        
        f.write("\n## Recommendations\n\n")
        if validation_results['match_rate'] < 70:
            f.write("- **Critical:** Extraction accuracy is too low for reliable analysis\n")
            f.write("- **Action Required:** Improve grid alignment and threshold parameters\n")
        elif validation_results['match_rate'] < 85:
            f.write("- **Warning:** Extraction accuracy needs improvement\n")
            f.write("- **Suggested:** Fine-tune parameters for better results\n")
        else:
            f.write("- **Good:** Extraction accuracy is acceptable for analysis\n")
            f.write("- **Optional:** Minor parameter tuning could improve results\n")
    
    print(f"Validation report saved to: {output_file}")