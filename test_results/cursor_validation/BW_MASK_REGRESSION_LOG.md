# bw_mask.png Regression Test Log

**Date:** 2025-07-16
**Agent:** Cursor
**File:** test_bw_mask_regression.py

---

## Context & Motivation

Silent regressions in image processing (e.g., thresholding, morphology) can be hard to detect and may break extraction quality without obvious errors. Hashing the output binary mask (`bw_mask.png`) and comparing it to a reference hash provides a simple, robust regression test to catch any unexpected changes.

## Implementation
- Script: `test_bw_mask_regression.py`
- Computes SHA-256 hash of `test_results/bw_mask.png`.
- Compares to reference hash in `test_results/bw_mask_reference_hash.txt`.
- If the reference hash is missing, prints the current hash and instructions to set it.
- If the hash differs, prints a failure message and both hashes.

## Usage
1. Run the extraction pipeline to generate `bw_mask.png`.
2. To set the current mask as the reference, run:
   ```bash
   python test_bw_mask_regression.py
   # Follow the instructions to set the reference hash:
   echo <hash> > test_results/bw_mask_reference_hash.txt
   ```
3. For future runs, simply run:
   ```bash
   python test_bw_mask_regression.py
   ```
   - [PASS] if the mask matches the reference
   - [FAIL] if the mask has changed (potential regression)

## Rationale
- **Early Detection:** Catches silent changes in image processing that may affect extraction.
- **Reproducibility:** Ensures that pipeline changes do not unintentionally alter core outputs.
- **Simplicity:** Easy to run in CI or as a pre-merge check.

## Next Steps
- Integrate this test into the standard validation workflow.
- Update the reference hash only after intentional, validated improvements.

---

*Logged for transparency and reproducibility. All future agents and researchers should review this log before modifying or updating the regression test or reference hash.* 