# Claude Code Changes Summary

**Created by:** Claude Code  
**Date:** July 16, 2025  
**Purpose:** Track all changes made to fix misleading accuracy reporting

---

## Changes Made

### 1. Fixed Pipeline Accuracy Reporting
**File:** `binary_extractor/extractor/pipeline.py`
- **Function:** `analyze_results()`
- **Changes:** 
  - Replaced misleading "legible_digits" and "success rate" with honest confidence metrics
  - Added "confident_cells", "ambiguous_cells", "confidence_percentage"
  - Added accuracy warnings for high ambiguity, overlay interference, low confidence
  - Added CRITICAL_WARNING about pipeline confidence vs true accuracy

### 2. Improved Cell Classification Logic
**File:** `binary_extractor/extractor/classify.py`
- **Function:** `classify_single_cell()`
- **Changes:**
  - Added confidence gap requirement (default 0.1)
  - More conservative thresholds: must exceed bit_hi + gap for '1', below bit_lo - gap for '0'
  - Prevents false classifications by requiring clear separation

### 3. Updated Configuration
**File:** `binary_extractor/extractor/cfg.yaml`
- **Changes:**
  - Increased bit_hi from 0.7 to 0.8 (more conservative)
  - Decreased bit_lo from 0.3 to 0.2 (more conservative)
  - Added confidence_threshold: 0.1 parameter

### 4. Created Validation Tools
**File:** `binary_extractor/extractor/validation.py` (NEW)
- **Purpose:** Honest accuracy validation through manual cell inspection
- **Features:**
  - Sample-based validation with visual cell inspection
  - Manual classification using same pipeline logic
  - Generates validation reports with match/mismatch/ambiguous rates
  - Saves individual cell images for visual verification

### 5. Updated Extraction Script
**File:** `binary_extractor/scripts/extract.py`
- **Changes:**
  - Updated to use new analysis format
  - Displays accuracy warnings and critical warnings
  - Removed misleading "Legible Digits" metric

### 6. Updated Agent Coordination
**File:** `AGENTS.MD`
- **Added:** Mandatory attribution requirements
- **Rule:** All code and documents must include "Created by [Agent Name]" headers
- **Purpose:** Track who made what changes for coordination

---

## Results

### Before Changes:
- **Reported:** "95.6% extraction success rate"
- **Reality:** Misleading - counted any classification as success

### After Changes:
- **Reported:** "95.3% confidence, 90% ambiguous cells"
- **Validated:** 10% true accuracy on manual inspection
- **Reality:** Honest assessment of actual extraction quality

---

## Attribution Rule Added

**MANDATORY RULE for all agents:**
- **All code files**: Must include header comment with "Created by [Agent Name]" or "Modified by [Agent Name]"
- **All documents**: Must include "Created by [Agent Name]" in header or frontmatter
- **All functions/methods**: Must include comment indicating author if modified
- **Purpose**: Track who made what changes for coordination and accountability

**This document follows the rule** - see header attribution to Claude Code.

---

## Key Insights

1. **Original pipeline was overconfident** - counted any classification as success
2. **Most cells are ambiguous** - fall in uncertain threshold range
3. **True accuracy is ~10%** - only clearly extractable cells
4. **Validation is essential** - pipeline confidence ≠ true accuracy
5. **Conservative thresholds needed** - prevent false classifications

---

## Recommendations

1. **Use validation tools** before trusting extraction results
2. **Focus on high-confidence regions** first
3. **Implement region-based parameter tuning**
4. **Manual parameter adjustment** for different poster areas
5. **Accept lower extraction rates** for higher accuracy

---

## Files Modified

- `binary_extractor/extractor/pipeline.py` - Honest accuracy reporting
- `binary_extractor/extractor/classify.py` - Conservative classification
- `binary_extractor/extractor/cfg.yaml` - Updated thresholds
- `binary_extractor/extractor/validation.py` - NEW validation tools
- `binary_extractor/extractor/__init__.py` - Updated package imports
- `binary_extractor/scripts/extract.py` - Updated reporting
- `AGENTS.MD` - Attribution requirements
- `REGION_BASED_ACCURACY_PROCESS.md` - Process documentation

All changes aimed at providing honest, validated accuracy assessment instead of misleading confidence metrics.