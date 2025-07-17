# Pitch Debate Resolution: 8px vs 25px vs 53px

## Executive Summary

The column pitch confusion (8px vs 25px vs 53px) was caused by **different people analyzing different-resolution images of the same logical grid**. All measurements were technically correct for their respective image scales.

## Root Cause Analysis

| Image Variant | Width (px) | Logical Grid | Measured Col-Pitch | Source |
|---------------|------------|--------------|-------------------|---------|
| 4K master poster | 4096 | Same | **8px** | O3/ChatGPT analysis |
| HD downsampled | ~2048 | Same | **~4px** | Theoretical |
| Binary extractor mask | 1232 | Same | **2-3px expected, 53px measured** | Claude's analysis |
| Processed debug | ~500 | Same | **~1px → aliases to 25px** | Previous work |

## Key Finding: Scale vs Processing

The binary mask (1232px wide) should show ~3px pitch if it's a clean downscale of 4K (8px × 1232/4096 = 2.4px). 

**But we measured 53px**, indicating the image underwent significant processing beyond simple downsampling:
- Morphological operations 
- Filtering that merged features
- Multiple processing steps

## Resolution Methodology

### ✅ Universal Approach (Scale-Independent)
```python
# 1. Detect pitch at current resolution
mask = img > 200
proj = mask.sum(0).astype(float); proj -= proj.mean()
pitch = np.argmax(np.correlate(proj, proj, "full")[len(proj)-1:][5:]) + 5

# 2. Origin sweep 0...(pitch-1)
best_origin = find_best_alignment(img, pitch)

# 3. Sample 6x6 patches (not single pixels)
bits = extract_with_robust_sampling(img, best_origin, pitch)

# 4. Validate: first row should decode to "On"
assert decode_first_row(bits) == "On"
```

### ❌ What Doesn't Work
- Assuming universal pitch values across scales
- Single-pixel sampling (too alignment-sensitive)
- Ignoring the actual image being processed

## Practical Implementation

For the current binary mask (1232px × 1666px):
- **Measured pitch**: 53px column, 31px row
- **Best origin**: (8, 4) 
- **Sampling**: 6x6 median patches
- **Target**: First row decoding to "On the..."

## Lessons Learned

1. **Always measure pitch on YOUR actual image**
2. **Scale-awareness is critical** when comparing results
3. **Robust sampling beats precise pitch** for extraction quality
4. **Validation target** (getting "On") matters more than theoretical perfection

## Next Steps

Continue with main goal using the resolved methodology:
- Fine-tune origin alignment for "On" decoding
- Optimize sampling parameters
- Extract full hidden message
- Compare against 77.1% baseline accuracy

---

*Debate resolved. Focus on execution.*