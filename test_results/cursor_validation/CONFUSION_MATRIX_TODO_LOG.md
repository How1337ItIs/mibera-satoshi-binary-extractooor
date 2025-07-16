# TODO Log: Per-Pass Confusion Matrix Implementation

**Date:** 2025-07-16
**Agent:** Cursor

---

## TODO In Progress
- **Task:** Show per-pass confusion matrix vs reference in main.py, not just totals, for instant feedback on extraction quality.
- **Status:** In Progress

## Rationale
- Totals alone do not reveal the nature of extraction errors. A per-pass confusion matrix (for each thresholding or classification pass) provides detailed insight into true/false positives/negatives and helps diagnose where the pipeline is failing or succeeding.
- Enables more targeted debugging and parameter tuning.

## Planned Approach
- Implement a function in the main pipeline (or as a script) to compare extracted results to a reference CSV (if available).
- For each extraction pass (e.g., conservative, loose, template-matching), compute and display a confusion matrix:
  - True Positives (TP)
  - False Positives (FP)
  - False Negatives (FN)
  - True Negatives (TN)
- Output the confusion matrix to the console and/or save as markdown/CSV for review.
- Document the implementation and usage in this log.

## Expected Impact
- Provides instant, detailed feedback on extraction quality.
- Makes it easier to spot regressions or improvements after parameter/code changes.
- Supports reproducibility and honest reporting of pipeline performance.

---

*This log will be updated upon completion of the TODO, as part of the exhaustive documentation protocol.* 