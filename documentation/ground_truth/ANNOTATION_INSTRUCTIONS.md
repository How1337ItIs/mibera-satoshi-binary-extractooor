
# Ground Truth Annotation Instructions

## Overview
40 cells need manual annotation across 2 sheets.

## Files Created
- annotation_sheet_1.png to annotation_sheet_2.png
- annotation_data.json (to be filled with ground truth)

## Instructions
1. Open each annotation sheet image
2. For each cell, determine if it's a 0 or 1 based on visual inspection
3. Fill in the ground truth values in annotation_data.json
4. Look for patterns like:
   - Dark regions = 0
   - Light regions = 1
   - Consider the background pattern structure

## Quality Guidelines
- Be consistent with threshold decisions
- When in doubt, mark as 'uncertain'
- Consider the overall pattern context
- Take breaks to maintain accuracy

## Format for annotation_data.json
For each cell, update "ground_truth_bit" with:
- "0" for dark/zero regions
- "1" for light/one regions  
- "uncertain" for ambiguous cases

## Next Steps
After annotation, run validation to measure extraction accuracy.
