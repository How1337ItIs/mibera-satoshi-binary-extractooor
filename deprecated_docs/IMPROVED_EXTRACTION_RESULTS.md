# Improved Satoshi Poster Binary Extraction Results

## Configuration Improvements Made
1. **Threshold Method**: Changed from Otsu to adaptive with increased C parameter (6)
2. **Bit Classification**: Adjusted thresholds (bit_hi: 0.65, bit_lo: 0.35)
3. **Template Matching**: Enabled with threshold 0.4
4. **Templates**: Extracted 4 templates (2 for '0', 2 for '1')

## Comparison: Original vs Improved Extraction

| Metric | Original | Improved | Change |
|--------|----------|----------|--------|
| **Total Cells** | 2,700 | 2,592 | -108 |
| **Zeros (0)** | 1,790 | 659 | -1,131 |
| **Ones (1)** | 790 | 1,547 | +757 |
| **Blanks** | 61 | 298 | +237 |
| **Overlays** | 59 | 88 | +29 |
| **Legible Digits** | 2,580 | 2,206 | -374 |
| **Success Rate** | 95.6% | 85.1% | -10.5% |

## Analysis of Changes

### Grid Detection
- **Original**: 54×50 grid (2,700 cells)
- **Improved**: 54×48 grid (2,592 cells)
- **Impact**: Adaptive thresholding changed grid detection, resulting in fewer columns

### Binary Distribution
- **Original**: 69.4% zeros, 30.6% ones
- **Improved**: 29.9% zeros, 70.1% ones
- **Impact**: Dramatic shift in bit distribution - suggests different interpretation of the same data

### Data Quality
- **Original**: Higher success rate (95.6%) with more confident classification
- **Improved**: Lower success rate (85.1%) but potentially more accurate template matching

## Key Findings

1. **Grid Sensitivity**: Small parameter changes significantly affect grid detection
2. **Binary Interpretation**: Different thresholds lead to dramatically different bit readings
3. **Template Matching**: Reduces overall success rate but may improve accuracy for ambiguous cells
4. **Data Consistency**: The dramatic shift in 0/1 ratio suggests the original parameters may have been more appropriate

## Recommendations

1. **Validate Original Settings**: The original extraction may have been more accurate
2. **Hybrid Approach**: Use original settings with selective template matching for ambiguous cells
3. **Manual Verification**: Visually compare debug images to validate which approach is more accurate
4. **Ground Truth**: Need manual annotation of sample regions to determine optimal parameters

## Files Generated
- `cells.csv`: 2,592 cells with improved classification
- `recognized_digits.csv`: 2,206 confidently recognized digits
- `overlay_unknown_cells.csv`: 386 problematic cells
- Debug visualizations for comparison analysis