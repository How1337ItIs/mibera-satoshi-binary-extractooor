# AI Agent Coordination Guide

## Project: Satoshi Poster Binary Extraction

### 🎯 Current Situation (July 16, 2025)
- **Status**: CRITICAL - Low extraction accuracy (~30-40% estimated)
- **Problem**: Grid alignment issues, poor thresholds, overconfident claims
- **Goal**: Extract binary digits from Satoshi poster background for crypto analysis

---

## 👥 Agent Roles & Responsibilities

### 🖌️ Cursor Agent
**Primary Focus**: Code editing and iterative development

**Strengths**:
- Real-time code editing and debugging
- Interactive parameter adjustment
- Quick iteration cycles
- Code refactoring and optimization

**Recommended Tasks**:
1. **Parameter Tuning**: Interactively adjust grid parameters in code
2. **Code Refinement**: Improve extraction algorithms through rapid iteration
3. **Debugging**: Fix issues found during extraction runs
4. **Code Organization**: Refactor and optimize extraction pipeline
5. **Exhaustive Logging & Documentation**: Log every change, experiment, technique, research need, setback, breakthrough, and rationale in markdown/status files for transparency and reproducibility.

---

### 🧠 Claude Code Agent
**Primary Focus**: Analysis, research, and systematic problem-solving

**Strengths**:
- Complex algorithm design and analysis
- Research methodology and documentation
- Pattern recognition and validation
- System architecture and planning

**Recommended Tasks**:
1. **Research Framework**: Design and implement systematic extraction research
2. **Validation Systems**: Create ground truth datasets and accuracy measurement
3. **Algorithm Analysis**: Analyze extraction methods and identify improvements
4. **Documentation**: Create comprehensive methodology documentation
5. **Exhaustive Logging & Documentation**: Record all research, experiments, findings, open questions, setbacks, breakthroughs, and context in markdown/status files.

---

### ⚡ Codex Agent
**Primary Focus**: Implementation and automation

**Strengths**:
- Fast code generation and implementation
- Utility functions and tools
- Automated testing and validation
- Data processing and analysis

**Recommended Tasks**:
1. **Implementation**: Build specific extraction functions and utilities
2. **Automation**: Create automated testing and validation pipelines
3. **Data Processing**: Handle image processing and bit extraction
4. **Tools**: Build helper functions and analysis scripts
5. **Exhaustive Logging & Documentation**: Log all code changes, automation steps, test results, setbacks, breakthroughs, and context in markdown/status files.

---

## 🔄 Coordination Protocol

### 1. Communication
- **Status Updates**: Update `project_status.json` after major changes
- **Documentation**: Use markdown files for sharing findings
- **Code Comments**: Clear documentation of algorithm changes
- **Exhaustive Logging**: Every agent must log all changes, techniques, research needs, setbacks, breakthroughs, and relevant context in markdown/status files. This is mandatory for both phase 1 (accurate extraction) and phase 2 (data meaning).

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

### 4. Lightweight Conflict Avoidance Practices
- **Announce file/feature work:** Agents should note in `project_status.json` (or a shared log) when they begin work on a file or feature that might overlap with others.
- **Prefer separate modules:** When possible, agents should work on different modules, scripts, or config sections to minimize overlap.
- **Check for recent changes:** Before starting major edits, agents should pull the latest changes from `main` and review recent commits/PRs in other agent branches.
- **Early communication:** If overlap is likely, agents should communicate and coordinate early to avoid unnecessary conflicts.
- **Keep it simple:** These practices are meant to be lightweight and practical, not burdensome—aimed at reducing merge conflicts while maintaining rapid iteration.

---

## 🎯 Priority Task Matrix

### 🔥 Critical (Do First)
| Task | Best Agent | Estimated Time | Dependencies |
|------|------------|---------------|--------------|
| Grid parameter calibration | Cursor | 2-3 hours | Interactive parameter tuning |
| Systematic research framework | Claude | 2-3 hours | Research methodology |
| Automated validation pipeline | Codex | 1-2 hours | Grid parameters |

### ⚠️ High Priority
| Task | Best Agent | Estimated Time | Dependencies |
|------|------------|---------------|--------------|
| Code refactoring and optimization | Cursor | 1-2 hours | Working extraction |
| Ground truth dataset creation | Claude | 2-3 hours | Manual annotation system |
| Batch processing utilities | Codex | 2-3 hours | Core extraction working |

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
- [ ] **All changes, techniques, research needs, setbacks, breakthroughs, and context are exhaustively logged and accessible.**

### Phase 2: Quality Assurance
- [ ] Extraction accuracy > 90% (validated)
- [ ] Systematic quality metrics
- [ ] Comprehensive documentation
- [ ] **Exhaustive, transparent logs and documentation maintained for all research and analysis.**

### Phase 3: Analysis Ready
- [ ] Extraction accuracy > 95% (validated)
- [ ] Complete binary matrix extracted
- [ ] Ready for cryptographic analysis
- [ ] **All context, findings, and rationale are fully documented for future research.**

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
6. **Maintain exhaustive, transparent logs and documentation at every step.**

Remember: The goal is accuracy, not speed. Better to have 100 correctly extracted bits than 1000 incorrect ones.