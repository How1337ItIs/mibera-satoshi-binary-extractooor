# Critical Analysis Summary: The 8px vs 25px Pitch Debate

## Key Finding: Scale/Source Confusion

The debate between 8px and 25px column pitch was caused by **different people analyzing different images at different scales**.

### What Actually Happened:

1. **O3/ChatGPT**: Likely analyzed a **cropped row at original 4K resolution** where the true pitch is ~8px
2. **Claude's Extractor**: Analyzed a **full downsampled binary mask** (1232px wide) where the apparent pitch is ~53px
3. **Previous Analysis**: Found 25px pitch, probably on a different intermediate scale

### Evidence:

- **Binary mask dimensions**: 1666 x 1232 pixels
- **Measured column spacing**: ~43-53 pixels between digit centers
- **Scale factor**: If original is 4K wide, this is ~30% scale
- **Expected 8px scaled**: 8 * 0.3 = 2.4 pixels (not what we see)

### The Real Issue:

The binary mask appears to be heavily processed, not just downsampled. The digit spacing of ~50px suggests either:
1. Multiple processing steps that changed the effective scale
2. Different source image than o3 was referencing
3. Morphological operations that merged/expanded features

## Practical Resolution:

### ✅ What Works:
1. **Measure pitch on YOUR actual image** - don't assume scales
2. **Use robust 6x6 patch sampling** instead of single pixels  
3. **Origin sweep within detected pitch range** for alignment
4. **Target 'On' in first row** as validation

### ❌ What Doesn't Work:
- Assuming universal pitch values across different image scales
- Single-pixel sampling (too sensitive to alignment)
- Ignoring the actual image you're processing

## Current Status:

- **Grid detected**: 31px row pitch, ~53px column pitch
- **Best origin**: (8, 4) with score 28.0
- **Sample extraction**: Getting bits, but not yet readable text
- **Next step**: Fine-tune origin and sampling to get "On" decoding

## Conclusion:

Both sides were technically correct for their respective images. The lesson is:
**Always measure pitch on the exact image you're processing, not a theoretical reference.**

The methodology (autocorrelation + robust sampling + origin sweep) is sound regardless of scale.