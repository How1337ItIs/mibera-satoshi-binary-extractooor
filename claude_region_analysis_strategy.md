# Region-Based Extraction Strategy Analysis
**Agent:** Claude Code Agent  
**Date:** July 16, 2025  
**Status:** Strategic Planning Phase  

## Executive Summary

Based on the current 30-40% extraction accuracy and the region-based accuracy process, this document outlines a systematic approach to achieve >90% accuracy through targeted region optimization rather than universal parameter tuning.

## Problem Decomposition

### 1. Grid Alignment Issues
**Root Cause:** Single grid origin (`row0: 40, col0: 10`) doesn't account for poster distortion
**Evidence:** 
- Grid overlay misalignment in different poster regions
- Inconsistent cell extraction across poster areas
- Visual inspection shows digit centers not aligned with grid

**Solution Strategy:**
- Region-specific grid origins
- Visual calibration per poster quadrant
- Adaptive pitch calculation based on local patterns

### 2. Threshold Sensitivity
**Root Cause:** Single threshold pair (`bit_hi: 0.70, bit_lo: 0.35`) fails across varying poster conditions
**Evidence:**
- High-contrast regions: thresholds too conservative
- Low-contrast regions: thresholds too aggressive  
- Overlay-heavy areas: thresholds confused by interference

**Solution Strategy:**
- Multi-pass extraction with graduated thresholds
- Region-specific color channel selection
- Confidence-based threshold adjustment

### 3. Overlay Interference
**Root Cause:** Aggressive overlay masking removes legitimate binary digits
**Evidence:**
- Known binary patterns partially masked
- Inconsistent overlay detection across regions
- Template matching fallback not properly integrated

**Solution Strategy:**
- Conservative overlay masking with template validation
- Region-specific overlay parameters
- Multi-channel overlay detection

## Region Classification Framework

### High-Contrast Regions (Target: 100% accuracy)
**Characteristics:**
- Clear digit boundaries
- Minimal overlay interference
- Consistent lighting
- Strong contrast between 0s and 1s

**Optimal Parameters:**
```yaml
use_color_space: HSV_S
bit_hi: 0.75
bit_lo: 0.30
blur_sigma: 10
overlay:
  saturation_threshold: 20
  value_threshold: 200
```

### Medium-Contrast Regions (Target: 95% accuracy)
**Characteristics:**
- Moderate digit clarity
- Some overlay interference
- Variable lighting conditions
- Mixed contrast levels

**Optimal Parameters:**
```yaml
use_color_space: Lab_b
bit_hi: 0.65
bit_lo: 0.40
blur_sigma: 15
overlay:
  saturation_threshold: 30
  value_threshold: 180
template_match: true
```

### Low-Contrast Regions (Target: 85% accuracy)
**Characteristics:**
- Faint digit boundaries
- Heavy overlay interference
- Poor lighting conditions
- Ambiguous contrast

**Optimal Parameters:**
```yaml
use_color_space: RGB_G
bit_hi: 0.60
bit_lo: 0.45
blur_sigma: 20
overlay:
  saturation_threshold: 40
  value_threshold: 160
template_match: true
tm_thresh: 0.40
```

## Implementation Strategy

### Phase 1: Region Identification (Week 1)
1. **Visual Poster Analysis**
   - Divide poster into 3x3 grid (9 regions)
   - Classify each region by contrast level
   - Identify 2-3 high-contrast test regions (50-100 cells each)

2. **Baseline Accuracy Measurement**
   - Extract test regions with current parameters
   - Manual verification of 100 random cells
   - Document accuracy per region type

3. **Parameter Optimization**
   - Iterate on high-contrast regions first
   - Achieve >95% accuracy before moving to medium-contrast
   - Document successful parameter sets

### Phase 2: Systematic Expansion (Week 2)
1. **Region-Specific Implementation**
   - Apply optimized parameters to similar regions
   - Cross-validate between regions of same type
   - Refine parameters based on regional variations

2. **Multi-Pass Extraction**
   - Pass 1: High-confidence extraction (>0.8 confidence)
   - Pass 2: Moderate confidence (0.6-0.8 confidence)
   - Pass 3: Template matching for remaining cells
   - Pass 4: Manual review of ambiguous cells

### Phase 3: Quality Assurance (Week 3)
1. **Comprehensive Validation**
   - Random sampling: 500 cells across all regions
   - Manual verification target: >90% overall accuracy
   - Region-specific accuracy tracking

2. **Final Optimization**
   - Address remaining problem areas
   - Document extraction limitations
   - Prepare for cryptographic analysis

## Success Metrics

### Accuracy Targets
- **High-contrast regions:** >95% accuracy
- **Medium-contrast regions:** >90% accuracy  
- **Low-contrast regions:** >80% accuracy
- **Overall poster:** >90% accuracy

### Coverage Targets
- **Extractable bits:** >95% of visible binary digits
- **High-confidence cells:** >80% of total extraction
- **Manual review needed:** <5% of total cells

### Validation Requirements
- **Random sampling:** 200+ cells per region type
- **Cross-validation:** Adjacent cell consistency checks
- **Spot verification:** Against original poster

## Risk Mitigation

### Technical Risks
1. **Parameter Overfitting**
   - Solution: Cross-validate between similar regions
   - Monitor: Accuracy consistency across regions

2. **Grid Drift**
   - Solution: Region-specific grid origins
   - Monitor: Visual alignment verification

3. **Template Matching Failure**
   - Solution: Conservative confidence thresholds
   - Monitor: Manual review of low-confidence cells

### Process Risks
1. **Manual Verification Bottleneck**
   - Solution: Systematic sampling strategy
   - Monitor: Time spent on validation vs. extraction

2. **Inconsistent Classification**
   - Solution: Clear region classification criteria
   - Monitor: Inter-rater reliability for region types

## Next Steps

### Immediate Actions (This Week)
1. **Visual Region Analysis**
   - Create poster region classification map
   - Identify high-contrast test regions
   - Document current extraction accuracy baseline

2. **Parameter Optimization Framework**
   - Implement region-specific parameter sets
   - Create systematic testing protocol
   - Develop accuracy measurement tools

3. **Coordination with Cursor Agent**
   - Provide visual validation criteria
   - Define grid alignment verification process
   - Establish parameter adjustment workflow

### Medium-term Goals (Next 2 Weeks)
1. **Complete Region Optimization**
   - Achieve target accuracy for all region types
   - Implement multi-pass extraction strategy
   - Establish quality assurance protocols

2. **Validation Framework**
   - Systematic accuracy measurement
   - Cross-region consistency validation
   - Final quality control procedures

## Conclusion

The region-based approach offers a pragmatic path to >90% extraction accuracy by acknowledging that different poster areas require different optimization strategies. This systematic methodology prioritizes accuracy over automation and provides a clear roadmap for achieving the project's goals.

**Key Success Factors:**
- Methodical region-by-region optimization
- Comprehensive validation at each stage
- Clear coordination between agents
- Honest assessment of limitations and capabilities

This strategy aligns with the project's core principle: "Better to have 100% correct extraction of 2,000 bits than 60% accuracy on 2,500 bits." 