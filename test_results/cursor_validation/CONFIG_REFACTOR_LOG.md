# Config-Driven Refactor Log

**Date:** 2025-07-16
**Agent:** Cursor
**Files:** binary_extractor/extractor/classify.py, binary_extractor/extractor/pipeline.py

---

## Context & Motivation

Hardcoded values ("magic numbers") in the pipeline and classification code made tuning and reproducibility difficult. Migrating these to config-driven parameters in `cfg.yaml` ensures all critical settings are visible, documented, and easily adjustable for future agents and researchers.

## Parameters Migrated to Config

- `cell_window` (row_half, col_half): Cell extraction window size
- `dual_pass_thresholds` (bit_lo_1, bit_hi_1, bit_lo_2, bit_hi_2): Dual-pass cell classification thresholds
- `adaptive_block_size`: Block size for adaptive thresholding
- `morph_open_iterations`: Number of open morphology iterations

## Rationale
- **Transparency:** All critical parameters are now visible and documented in one place.
- **Reproducibility:** Future runs can be exactly replicated or tuned by adjusting config, not code.
- **Tuning:** Makes it easy to experiment with different values and document their impact.
- **Collaboration:** Lowers the barrier for new agents or human researchers to understand and improve the pipeline.

## Implementation
- Refactored `classify.py` to use `cell_window` and `dual_pass_thresholds` from `cfg.yaml`.
- Refactored `pipeline.py` to use `adaptive_block_size` and `morph_open_iterations` from `cfg.yaml`.
- All changes are fully documented and reproducible.

## Next Steps
- Continue to log every change, rationale, and result as per the exhaustive documentation protocol.
- Proceed to the next prioritized TODO (e.g., regression test or confusion matrix implementation).

---

*Logged for transparency and reproducibility. All future agents and researchers should review this log before modifying config-driven parameters or related code.* 