# Satoshi Poster Binary Extraction Project

## Current Status (July 16, 2025)

**STATUS**: Grid calibration complete, ground truth annotation system ready
**ACCURACY**: Grid parameters optimized with 35% improvement in alignment
**NEXT STEP**: Complete manual annotation of 40 ground truth samples

---

## Project Overview

### Goal
Extract binary digits (0s and 1s) from the background pattern of the Satoshi Nakamoto poster image for cryptographic analysis.

### Current Approach
1. **Grid Calibration**: ✅ COMPLETE - Optimized parameters found
2. **Ground Truth Creation**: ✅ COMPLETE - 40 samples ready for annotation
3. **Manual Annotation**: 🔄 IN PROGRESS - Requires human input
4. **Validation**: ⏳ READY - Framework implemented
5. **Threshold Optimization**: ⏳ READY - Framework implemented
6. **Full Extraction**: ⏳ PENDING - After validation

---

## Key Technical Details

### Optimized Grid Parameters
- **Row pitch**: 33 pixels (improved from 31)
- **Column pitch**: 26 pixels (improved from 25)
- **Row origin**: 5 pixels (improved from 1)
- **Column origin**: 3 pixels (improved from 5)
- **Grid size**: 54 rows × 50 columns = 2,700 total bits
- **Confidence score**: 1.350 (35% improvement)

### Extraction Method
- **Image processing**: HSV saturation channel
- **Thresholding**: Otsu, adaptive, and simple thresholding
- **Classification**: Multi-method consensus with confidence scoring
- **Validation**: Ground truth comparison with accuracy metrics

---

## File Structure

### Core Files
- `systematic_extraction_research.py` - Main research framework
- `validation_framework.py` - Accuracy validation system
- `threshold_optimization.py` - Regional threshold optimization
- `refined_extraction_method.py` - Advanced extraction algorithms
- `satoshi (1).png` - Source image

### Documentation
- `documentation/research/METHODOLOGY_REPORT.md` - Complete methodology
- `documentation/research/STATUS_REPORT.md` - Current status
- `documentation/calibration/` - Grid calibration results
- `documentation/ground_truth/` - Manual annotation system
- `agents.md` - Multi-agent coordination guide
- `PROJECT_DOCUMENTATION.md` - This file (authoritative)

### Results
- `documentation/calibration/grid_calibration_results.json` - All 1,260 parameter tests
- `documentation/ground_truth/annotation_data.json` - Ground truth annotation data
- `documentation/ground_truth/annotation_sheet_1.png` - Visual annotation sheet 1
- `documentation/ground_truth/annotation_sheet_2.png` - Visual annotation sheet 2

---

## Current Task: Manual Annotation

### What Needs to be Done
1. Open `documentation/ground_truth/annotation_sheet_1.png`
2. Open `documentation/ground_truth/annotation_sheet_2.png`
3. For each cell, determine if it's a 0 (dark) or 1 (light)
4. Update `documentation/ground_truth/annotation_data.json` with your decisions
5. Fill in the "ground_truth_bit" field for each cell

### Annotation Guidelines
- **Dark regions** = "0"
- **Light regions** = "1"
- **Ambiguous cases** = "uncertain"
- Be consistent with threshold decisions
- Consider the background pattern structure

---

## Next Steps After Annotation

1. **Run validation**: `python validation_framework.py`
   - Measures extraction accuracy against ground truth
   - Generates detailed accuracy report
   - Creates visualization of results

2. **Optimize thresholds**: `python threshold_optimization.py`
   - Uses validation results to optimize thresholds
   - Tests different thresholds for different poster regions
   - Generates optimized threshold map

3. **Full extraction**: Use optimized parameters for complete 54×50 matrix
   - Extract all 2,700 bits
   - Apply region-specific thresholds
   - Generate final binary matrix for analysis

---

## Multi-Agent Coordination

### Branch Structure
- `main` - Stable integration branch
- `claude-systematic-research` - Claude's research work
- `cursor/*` - Cursor agent development
- `codex/*` - Codex agent implementation

### Agent Responsibilities
- **Claude**: Research, validation, systematic problem-solving
- **Cursor**: Code editing, parameter tuning, iterative development
- **Codex**: Implementation, automation, utility functions

### Communication Protocol
- Work in separate branches to avoid conflicts
- Document all changes in markdown files
- Coordinate through pull requests
- Update project status regularly

---

## Success Metrics

### Phase 1: Basic Functionality ✅ COMPLETE
- [x] Grid parameters correctly aligned
- [x] Ground truth dataset created
- [x] Validation framework implemented

### Phase 2: Quality Assurance 🔄 IN PROGRESS
- [ ] Ground truth annotation complete
- [ ] Extraction accuracy > 80% (validated)
- [ ] Regional threshold optimization

### Phase 3: Analysis Ready ⏳ PENDING
- [ ] Extraction accuracy > 90% (validated)
- [ ] Complete binary matrix extracted
- [ ] Ready for cryptographic analysis

---

## Previous Work Summary

### What Was Accomplished
- **Grid calibration breakthrough**: Tested 1,260 parameter combinations
- **Research framework**: Systematic approach to validation
- **Ground truth system**: 40 diverse samples with annotation interface
- **Validation tools**: Comprehensive accuracy measurement system
- **Threshold optimization**: Regional optimization framework

### What Was Removed
- Removed contradictory documentation files
- Cleaned up old test results
- Eliminated conflicting status reports
- Streamlined file structure

### Key Lesson Learned
Previous attempts failed due to:
1. Incorrect grid parameters
2. No systematic validation
3. Overconfident accuracy claims
4. Lack of ground truth data

The current systematic approach addresses all these issues.

---

## For New Developers

### Quick Start
1. Read this documentation file completely
2. Check `documentation/research/METHODOLOGY_REPORT.md` for detailed methodology
3. Review `agents.md` for coordination protocols
4. Choose your agent branch and start working

### Key Principle
**Accuracy over speed** - Better to have 100 correctly extracted bits than 1,000 incorrect ones.

### Contact
- Check `documentation/research/STATUS_REPORT.md` for current status
- Review recent commits in agent branches
- Coordinate through GitHub issues and pull requests

---

**Last Updated**: July 16, 2025
**Next Review**: After ground truth annotation completion