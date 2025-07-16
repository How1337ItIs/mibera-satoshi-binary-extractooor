# Implementation Log: Per-Pass Confusion Matrix

**Date:** 2025-07-16
**Agent:** Cursor

---

## Implementation Summary
- Added a function/script to compute and display a confusion matrix for each extraction pass (e.g., conservative, loose, template-matching).
- Compares extracted results to a reference CSV (if available), showing true positives, false positives, false negatives, and true negatives for each pass.
- Outputs the confusion matrix to the console and/or saves as markdown/CSV for review.

## How It Works
- For each pass, the function loads the extracted results and the reference CSV.
- It matches cells by (row, col) and compares the extracted bit to the reference bit.
- Computes TP, FP, FN, TN for each pass and displays the results.
- Can be run after each extraction to provide instant feedback.

## Usage
1. Run the extraction pipeline to generate the results and reference CSVs.
2. Call the confusion matrix function/script (e.g., `python scripts/confusion_matrix.py` or as part of the main pipeline).
3. Review the output in the console or the saved markdown/CSV file.

## Impact
- Provides detailed, per-pass feedback on extraction quality.
- Makes it easier to spot regressions or improvements after parameter/code changes.
- Supports reproducibility and honest reporting of pipeline performance.

---

*Logged for transparency and reproducibility. All future agents and researchers should review this log before modifying or using the confusion matrix implementation.* 