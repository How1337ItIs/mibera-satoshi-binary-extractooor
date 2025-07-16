# AI Agent Coordination Guide

## Project: Satoshi Poster Binary Extraction

### 🎯 Current Situation (July 16, 2025)
- **Status**: CRITICAL - Low extraction accuracy (~30-40% estimated)
- **Problem**: Grid alignment issues, poor thresholds, overconfident claims
- **Goal**: Extract binary digits from Satoshi poster background for crypto analysis

---

## 👥 Agent Roles & Responsibilities

### 🎨 Cursor Agent
**Primary Focus**: Visual inspection and manual calibration

**Strengths**:
- Interactive visual debugging
- Real-time code editing
- UI/UX improvements
- Manual parameter tuning

**Recommended Tasks**:
1. **Grid Calibration**: Visually inspect `satoshi (1).png` and manually adjust grid parameters
2. **Threshold Tuning**: Test different threshold values on sample regions
3. **Visual Validation**: Create side-by-side comparisons of extracted vs actual bits
4. **Interactive Tools**: Build visual interfaces for parameter adjustment

**Starting Commands**:
```bash
python startup.py cursor
python visual_validation.py
```

**Key Files to Focus On**:
- `visual_validation.py` - Visual inspection tools
- `binary_extractor/extractor/cfg.yaml` - Configuration parameters
- `test_results/visual_validation/` - Validation outputs

---

### 🧠 Claude Code Agent
**Primary Focus**: Algorithm development and analysis

**Strengths**:
- Complex algorithm design
- Pattern recognition
- Documentation and reporting
- System architecture

**Recommended Tasks**:
1. **Algorithm Refinement**: Improve grid detection and bit classification
2. **Quality Metrics**: Develop proper accuracy measurement systems
3. **Documentation**: Create comprehensive analysis reports
4. **Pattern Analysis**: Analyze extracted patterns for cryptographic significance

**Starting Commands**:
```bash
python startup.py claude
python refined_extraction_method.py
```

**Key Files to Focus On**:
- `refined_extraction_method.py` - Advanced algorithms
- `binary_extractor/extractor/pipeline.py` - Core extraction logic
- `analyze_binary_data.py` - Pattern analysis

---

### ⚡ Codex Agent
**Primary Focus**: Code implementation and optimization

**Strengths**:
- Fast code generation
- Utility functions
- Performance optimization
- Data processing pipelines

**Recommended Tasks**:
1. **Grid Detection**: Implement robust grid finding algorithms
2. **Preprocessing**: Create image enhancement and filtering functions
3. **Validation Pipeline**: Build automated accuracy testing systems
4. **Performance**: Optimize extraction speed and memory usage

**Starting Commands**:
```bash
python startup.py codex
python -c "from startup import codex_quickstart; codex_quickstart()"
```

**Key Functions to Implement**:
```python
def interactive_grid_calibration():
    """Interactive grid parameter adjustment"""
    
def validate_extraction_accuracy():
    """Systematic accuracy validation"""
    
def optimize_thresholds():
    """Automatic threshold optimization"""
    
def create_ground_truth_dataset():
    """Manual annotation for validation"""
```

---

## 🔄 Coordination Protocol

### 1. Communication
- **Status Updates**: Update `project_status.json` after major changes
- **Documentation**: Use markdown files for sharing findings
- **Code Comments**: Clear documentation of algorithm changes

### 2. File Management
- **Naming Convention**: Use descriptive names with agent prefix
  - `cursor_visual_tools.py`
  - `claude_analysis_report.md`
  - `codex_grid_detection.py`
- **Output Structure**: 
  ```
  test_results/
  ├── cursor_validation/
  ├── claude_analysis/
  └── codex_optimization/
  ```

### 3. Version Control
- **Branching**: Each agent works on separate features
- **Commits**: Include agent name and brief description
- **Merging**: Coordinate through main integration

---

## 🎯 Priority Task Matrix

### 🔥 Critical (Do First)
| Task | Best Agent | Estimated Time | Dependencies |
|------|------------|---------------|--------------|
| Grid parameter calibration | Cursor | 2-3 hours | Visual inspection |
| Accuracy validation system | Codex | 1-2 hours | Grid parameters |
| Threshold optimization | Claude | 2-3 hours | Validation system |

### ⚠️ High Priority
| Task | Best Agent | Estimated Time | Dependencies |
|------|------------|---------------|--------------|
| Visual inspection tools | Cursor | 1-2 hours | None |
| Quality metrics | Claude | 1-2 hours | None |
| Performance optimization | Codex | 2-3 hours | Working extraction |

### 📋 Medium Priority
| Task | Best Agent | Estimated Time | Dependencies |
|------|------------|---------------|--------------|
| Pattern analysis | Claude | 3-4 hours | Accurate extraction |
| Documentation | Claude | 2-3 hours | Completed extraction |
| Advanced algorithms | Codex | 3-4 hours | Basic extraction working |

---

## 📊 Success Metrics

### Phase 1: Basic Functionality
- [ ] Grid parameters correctly aligned (visual inspection)
- [ ] Extraction accuracy > 80% (validated)
- [ ] Proper validation pipeline working

### Phase 2: Quality Assurance
- [ ] Extraction accuracy > 90% (validated)
- [ ] Systematic quality metrics
- [ ] Comprehensive documentation

### Phase 3: Analysis Ready
- [ ] Extraction accuracy > 95% (validated)
- [ ] Complete binary matrix extracted
- [ ] Ready for cryptographic analysis

---

## 🚨 Current Critical Issues

### 1. Grid Alignment Problem
- **Issue**: Grid parameters don't match actual poster pattern
- **Impact**: Fundamental extraction failure
- **Solution**: Manual visual calibration
- **Owner**: Cursor Agent

### 2. Threshold Selection
- **Issue**: Single threshold doesn't work across poster regions
- **Impact**: Poor bit classification accuracy
- **Solution**: Adaptive/regional thresholds
- **Owner**: Claude Agent

### 3. Validation Gap
- **Issue**: No systematic accuracy measurement
- **Impact**: Overconfident claims, unknown quality
- **Solution**: Robust validation pipeline
- **Owner**: Codex Agent

---

## 💡 Agent Collaboration Examples

### Example 1: Grid Calibration
1. **Cursor**: Visually identifies correct grid parameters
2. **Codex**: Implements automatic grid detection algorithm
3. **Claude**: Validates and documents the solution

### Example 2: Threshold Optimization
1. **Codex**: Creates systematic threshold testing framework
2. **Claude**: Analyzes results and determines optimal values
3. **Cursor**: Validates visually and fine-tunes parameters

### Example 3: Quality Assurance
1. **Claude**: Designs comprehensive validation approach
2. **Codex**: Implements automated testing pipeline
3. **Cursor**: Performs manual spot-checks and verification

---

## 🔧 Quick Start Commands

```bash
# Project setup
python startup.py setup

# Agent-specific entry points
python startup.py cursor    # Visual tools and calibration
python startup.py claude    # Algorithm development
python startup.py codex     # Code implementation

# Check current status
python startup.py status

# Get help
python startup.py help
```

---

## 📝 Notes for Success

1. **Start Small**: Test on 5x5 grid region first
2. **Validate Early**: Check accuracy before expanding
3. **Document Everything**: Track what works and what doesn't
4. **Coordinate Often**: Share findings regularly
5. **Be Honest**: Acknowledge failures and limitations

Remember: The goal is accuracy, not speed. Better to have 100 correctly extracted bits than 1000 incorrect ones.