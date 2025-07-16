# Debug Artifacts

**Created by:** Claude Code  
**Date:** July 16, 2025  
**Purpose:** Archive debugging files used to identify and fix grid alignment issues

---

## Contents

### Analysis Files:
- `debug_*.png` - Debug images showing cell extractions and grid analysis
- `test_*.png` - Grid parameter testing images
- `validation_output/` - Manual validation cell images (10% accuracy results)
- `cell_*.png` - Individual cell extraction samples

### Purpose:
These files were created during the debugging process that identified the core problem:
- **Wrong grid parameters** (31×25 vs actual 15×12)
- **Extracting background** instead of digits
- **Grid misalignment** causing poor extraction

### Key Discovery:
Visual inspection of these debug images revealed that the pipeline was extracting spaces between digits rather than the digits themselves.

### Result:
Led to the corrected approach in `corrected_extraction.py` with 54% accuracy vs previous 10%.

---

**These files are kept for reference but are not needed for current extraction work.**