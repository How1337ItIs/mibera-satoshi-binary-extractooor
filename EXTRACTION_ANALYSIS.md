# Satoshi Poster Binary Extraction Analysis

## Extraction Summary
**Date**: July 16, 2025  
**Image**: satoshi (1).png  
**Grid Size**: 54 rows × 50 columns = 2,700 total cells  

## Binary Data Results
| Category | Count | Percentage |
|----------|-------|------------|
| **Zeros (0)** | 1,790 | 66.3% |
| **Ones (1)** | 790 | 29.3% |
| **Blanks** | 61 | 2.3% |
| **Overlays** | 59 | 2.2% |
| **Total Cells** | 2,700 | 100% |

## Extraction Quality
- **Legible Digits**: 2,580 (95.6%)
- **Problematic Cells**: 120 (4.4%)
- **Grid Detection**: Successful (confident_cells=2561)
- **Grid Parameters**: row_pitch=31, col_pitch=25, row0=1, col0=5

## Binary Pattern Analysis
- **Binary Ratio**: 0.44 (790 ones / 1790 zeros)
- **Total Binary Bits**: 2,580 extractable bits
- **Data Density**: High (95.6% successful extraction)

## Configuration Used
```yaml
use_color_space: HSV_S
blur_sigma: 25
threshold:
  method: otsu
  adaptive_C: 4
morph_k: 3
morph_iterations: 2
bit_hi: 0.7
bit_lo: 0.3
```

## Output Files Generated
1. `cells.csv` - All 2,700 cells with classifications
2. `recognized_digits.csv` - 2,580 successfully recognized digits
3. `overlay_unknown_cells.csv` - 120 problematic cells for review
4. Debug visualizations (PNG files)

## First Row Sample
Row 0: `0,1,0,blank,0,0,0,0,0,0,0,0,0,0,0,0,0,0,blank,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,blank,0,0,0,0,0,0,0,0,0,0,0,0`

## Next Steps
1. Analyze blank and overlay cells for potential missed bits
2. Validate extracted binary data for patterns
3. Improve extraction algorithms for 100% accuracy
4. Document complete bit matrix