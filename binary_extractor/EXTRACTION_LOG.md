# Satoshi Poster Bit Extraction: Project Log

## **Intention**
Extract every visible ‘0’ and ‘1’ bit from the Satoshi poster image, logging their grid position, and flagging anything obscured or blank. Output must be a CSV with all detected bits and their positions, plus a separate CSV for overlay/unknown cells. The process must be repeatable, test-driven, and highly configurable.

> **Note:** The provided reference CSV is not a gold standard, but a starting point for validation and tuning. The true goal is to extract every visible bit, even if that means finding more than the reference.

---

## **Methods & Pipeline**
- Modular Python pipeline (`pipeline.py`) for image processing, grid detection, cell classification, and CSV export.
- All parameters (color space, blur, threshold, morphology, grid, etc.) exposed in `cfg.yaml`.
- Output files:
  - `recognized_digits.csv` (all detected 0/1 bits, with row/col/digit)
  - `overlay_unknown_cells.csv` (all overlay/blank cells, with row/col)
  - `cells.csv` (all cells, for debugging)
  - Debug images for every processing stage.
- Validation script compares pipeline output to reference CSVs, flags mismatches, and visualizes results.
- **Reference CSV is used for initial validation and tuning, but the pipeline aims to maximize true positive extraction, not just match the reference.**

---

## **Log of Attempts, Breakthroughs, and Setbacks**

### **Initial Implementation**
- Scaffolded repo, implemented pipeline, config, and validation script.
- Set up all output and debug artifacts.

### **Setbacks**
- Initial pipeline output did not match reference: over-detection or under-detection of bits.
- Validation script could not find expected output files due to naming mismatch (`cells.csv` vs. `recognized_digits.csv`).
- Missing config file in expected location prevented pipeline from running.

### **Breakthroughs**
- Updated pipeline to output `recognized_digits.csv` and `overlay_unknown_cells.csv` in the correct format.
- Automated config file management and output directory checks.
- Added explicit logging and error handling at every step.
- Validation script now runs end-to-end, loading and comparing both reference and pipeline outputs.

### **Current Validation Status**
- Pipeline produces all required files.
- Validation script loads and compares reference and pipeline output.
- Distribution and comparison of 0s and 1s is available.
- **Extraction is not yet perfect:**
  - The number of detected bits may be higher than the reference. This is expected and desirable if the extra bits are real, visible digits.
  - Validation script will flag mismatches, missing, or extra bits, but the ultimate goal is to maximize true positive extraction, not just match the reference.

---

## **Areas for Further Research & Tuning**
- **Parameter Tuning:**
  - Thresholding, blur, morphology, grid alignment, and overlay detection all impact accuracy.
  - Systematic grid search or manual tuning may be needed to maximize match rate.
- **Cell Classification:**
  - Improve heuristics or add template matching/deep learning for ambiguous cells.
- **Grid Detection:**
  - Refine auto-detection and alignment to match the poster’s true grid.
- **Overlay/Unknown Handling:**
  - Better distinguish between true overlays and faint digits.
- **Validation Automation:**
  - Automate iterative tuning and validation to approach 100% extraction.
- **Manual/Visual Review:**
  - Use debug artifacts and overlays to confirm that extra bits are real, not false positives.
  - Update the reference with newly discovered bits for future validation if needed.

---

## **Next Steps**
- Review validation output for mismatches or missing bits.
- Tune parameters and iterate until every single 0 and 1 is extracted and matches the visible bits in the poster, not just the reference.
- Document optimal configuration and results.

---

## **Latest Pipeline Pass Results**
- **Date:** [AUTOMATED ENTRY]
- **Pipeline output:**
  - 0s: 1391
  - 1s: 858
  - Total detected: 2249
- **Reference:**
  - 0s: 551
  - 1s: 687
  - Total: 1238
- **Digit Recognition Comparison:**
  - Matches: 221 (17.9% of reference)
  - Mismatches: 265 (21.4% of reference)
  - Extra in pipeline: 1766
  - Missing in pipeline: 752 (60.7% of reference)
  - F1 Score: 0.1266
- **Overlay Detection Comparison:**
  - Total in reference: 923
  - Total in pipeline: 348
  - Matches: 1226 (132.8% of reference)
  - Extra in pipeline: 0
  - Missing in pipeline: 0
  - F1 Score: 1.9292
- **Status:**
  - Over-detection of both 0s and 1s compared to reference.
  - Overlay detection is high, but pipeline overlay count is lower than reference.
  - Next step: Tune grid and threshold parameters to improve match rate and reduce mismatches/missing bits.

---

## **Methods Log: Parameter Changes and Rationale**

### Pass [AUTOMATED ENTRY]
- **Parameters:**
  - `bit_hi`: 0.65
  - `bit_lo`: 0.3
  - `blur_sigma`: 15
  - `morph_k`: 3
  - `morph_iterations`: 2
  - `threshold.method`: otsu
  - `use_color_space`: HSV_S
  - `row0`: 50
  - `col0`: 20
  - `row_pitch`: null (auto)
  - `col_pitch`: null (auto)
- **Rationale:**
  - Lowered `bit_hi` and `blur_sigma` to try to detect more faint 1s and reduce background noise.
  - Kept grid detection in auto mode to allow for best-fit grid.
- **Outcome:**
  - No significant change in output distribution or match rate.
  - Overlay comparison bug fixed in validation script.
  - Next: Consider tuning grid origin, pitch, or switching to adaptive thresholding for next pass.

--- 