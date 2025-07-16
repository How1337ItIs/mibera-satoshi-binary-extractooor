# DEPRECATED DOCUMENTS

**Created by:** Claude Code  
**Date:** July 16, 2025  
**Status:** DEPRECATED - DO NOT USE

---

## ⚠️ WARNING: THESE DOCUMENTS CONTAIN MISLEADING INFORMATION

**All documents in this directory have been deprecated due to fundamental errors in the extraction approach.**

### Problems with Original Approach:
1. **Wrong grid parameters** - Used 31×25 pitch instead of actual 15×12
2. **Extracting background** instead of digits
3. **Overconfident accuracy claims** - Reported 95.6% when actual was ~10%
4. **Misleading validation** - Internal metrics vs true accuracy

### Documents Moved Here:
- `COMPREHENSIVE_RESEARCH_REPORT.md` - Claims 95.6% success rate (FALSE)
- `COMPLETE_BIT_MATRIX.md` - Claims complete extraction (FALSE)
- `EXTRACTION_ANALYSIS.md` - Based on wrong grid alignment
- `EXTRACTION_CONFIDENCE_ANALYSIS.md` - Overconfident claims
- `IMPROVED_EXTRACTION_RESULTS.md` - Not actually improved

---

## ✅ CURRENT WORKING APPROACH

**See these files for accurate information:**
- `CLAUDE_CODE_CHANGES_SUMMARY.md` - What was fixed and why
- `corrected_extraction.py` - Working extraction method (54% accuracy)
- `fix_grid_alignment.py` - Grid analysis that found the problems
- `HONEST_ASSESSMENT.md` - Reality check on previous claims

### Current Status:
- **Grid parameters fixed**: 15×12 pitch instead of 31×25
- **Actual accuracy**: 54% clear classifications (vs previous 10%)
- **Honest validation**: Manual inspection confirms results
- **Working method**: Blue channel classification with corrected grid

---

## Lessons Learned:
1. **Never trust automated metrics without visual validation**
2. **Grid alignment is fundamental** - wrong grid = wrong everything
3. **Start simple** - basic color classification works when grid is right
4. **Be honest about failures** - deprecate misleading claims immediately

**DO NOT REFERENCE THESE DEPRECATED DOCUMENTS FOR ANALYSIS OR RESEARCH**