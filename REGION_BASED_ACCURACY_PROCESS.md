# Region-Based Accuracy Process for Satoshi Poster Binary Extraction

**Created by:** Claude Code  
**Document Created:** July 16, 2025  
**Status:** Active Implementation Strategy  
**Priority:** Critical - Foundation for Accurate Extraction

---

## Executive Summary

This document outlines a pragmatic, accuracy-first approach to extract every binary digit from the Satoshi poster background. Rather than pursuing perfect automation, we prioritize manual parameter tuning and region-specific optimization to achieve 100% accuracy across all extractable bits.

**Core Principle:** Accuracy over automation. Better to have 100% correct extraction of 2,000 bits than 60% accuracy on 2,500 bits.

---

## Process Overview

### Phase 1: High-Contrast Region Mastery (Week 1)
**Goal:** Achieve 100% accuracy on the clearest 25% of the poster

1. **Identify Clear Regions**
   - Visual inspection of poster to find high-contrast areas
   - Typically: top-left corner, center sections with minimal overlay
   - Mark coordinates of 3-4 test regions (50-100 cells each)

2. **Perfect Grid Alignment**
   - Extract test regions with current parameters
   - Manually verify grid overlay alignment
   - Adjust `row_pitch`, `col_pitch`, `row0`, `col0` iteratively
   - Target: Grid lines hit digit centers consistently

3. **Threshold Optimization**
   - Test color channels: `HSV_S`, `Lab_b`, `RGB_G`
   - Adjust `bit_hi`, `bit_lo` for clean 0/1 separation
   - Test threshold methods: `otsu`, `adaptive`, `sauvola`
   - Validate: Manual count of 0s/1s vs extracted results

4. **Quality Gate**
   - Extract 100 random cells from optimized regions
   - Manual verification must show >95% accuracy
   - If failed, return to parameter tuning

### Phase 2: Problem Region Analysis (Week 2)
**Goal:** Characterize and solve specific extraction challenges

1. **Problem Categorization**
   - **Blurry regions**: Increase blur_sigma, try different channels
   - **High overlay areas**: Aggressive overlay masking
   - **Faint digits**: Template matching fallback
   - **Inconsistent grid**: Region-specific parameters

2. **Region-Specific Parameter Sets**
   ```yaml
   regions:
     clear_areas:
       use_color_space: HSV_S
       bit_hi: 0.7
       bit_lo: 0.3
     blurry_areas:
       use_color_space: Lab_b
       blur_sigma: 30
       bit_hi: 0.6
       bit_lo: 0.4
     overlay_heavy:
       overlay.saturation_threshold: 30
       overlay.value_threshold: 190
       template_match: true
   ```

3. **Multi-Pass Extraction Strategy**
   - **Pass 1**: Conservative thresholds (high confidence only)
   - **Pass 2**: Moderate thresholds for remaining cells
   - **Pass 3**: Aggressive extraction with template matching
   - **Pass 4**: Manual review of remaining ambiguous cells

### Phase 3: Complete Extraction (Week 3)
**Goal:** Extract every possible bit with validated accuracy

1. **Systematic Coverage**
   - Apply optimized parameters to full poster regions
   - Track extraction statistics per region
   - Flag cells with confidence <0.8 for manual review

2. **Quality Assurance**
   - Random sampling: 200 cells across all regions
   - Manual verification target: >90% accuracy
   - Cross-reference adjacent cells for consistency

3. **Final Validation**
   - Generate complete bit matrix
   - Spot-check against original poster
   - Document any unextractable regions with rationale

---

## Implementation Guidelines

### Visual Validation Process
1. **Grid Overlay Check**
   - Generate `grid_overlay.png` after each parameter change
   - Overlay should center on digit positions
   - Check alignment at corners and edges

2. **Cell Extraction Review**
   - Export 5x5 test regions as individual cell images
   - Manually count 0s and 1s in each cell
   - Compare to algorithm results

3. **Parameter Adjustment Criteria**
   - If accuracy <90%: Adjust thresholds first
   - If grid misaligned: Adjust pitch/origin parameters
   - If high ambiguity: Enable template matching

### Success Metrics
- **Regional Accuracy**: >95% on high-contrast areas
- **Overall Accuracy**: >90% on full poster
- **Coverage**: Extract >95% of visible bits
- **Confidence**: >80% of cells classified with high confidence

### Tools and Scripts
- **Primary**: `binary_extractor/scripts/extract.py`
- **Debug**: Check `grid_overlay.png`, `bw_mask.png` outputs
- **Config**: All parameters in `binary_extractor/extractor/cfg.yaml`
- **Validation**: Manual cell counting and comparison

---

## Risk Management

### Common Issues and Solutions
1. **Grid Drift**: Different regions need different origins
   - Solution: Region-specific parameter sets
2. **Color Variation**: Single channel fails in some areas
   - Solution: Multi-channel approach per region
3. **Overlay Interference**: Aggressive masking hides real bits
   - Solution: Conservative masking + template matching
4. **Threshold Sensitivity**: Small changes break extraction
   - Solution: Multi-pass with graduated thresholds

### Quality Control Checkpoints
- After each parameter change: Test on 20 known cells
- After region completion: Random sample 100 cells
- Before final extraction: Full poster spot-check

---

## Expected Timeline

**Week 1:** Master high-contrast regions (25% of poster)  
**Week 2:** Solve problem regions with targeted approaches  
**Week 3:** Complete extraction with quality assurance  

**Total Time Investment:** 3 weeks of focused, methodical work  
**Expected Outcome:** Complete, validated bit matrix with >90% accuracy

---

## Notes for Implementation

### Priority Order
1. Grid alignment (most critical)
2. Color channel selection
3. Threshold optimization
4. Overlay handling
5. Template matching (fallback only)

### Avoid Over-Engineering
- No complex ML models until basics work
- No advanced image processing until thresholds optimized
- No automation until manual process validated

### Documentation Requirements
- Log all parameter changes with rationale
- Document accuracy results for each region
- Track time spent on each approach
- Record failure modes and solutions

**Remember:** The goal is not to build the perfect system, but to extract every possible bit with maximum accuracy. Sometimes the best solution is careful manual tuning rather than automated optimization.