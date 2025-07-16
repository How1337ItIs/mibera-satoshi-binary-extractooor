# AI Agent Coordination Guide

## Project: Satoshi Poster Binary Extraction

### 🎯 Current Situation (July 16, 2025)
- **Status**: CRITICAL - Low extraction accuracy (~30-40% estimated)
- **Problem**: Grid alignment issues, poor thresholds, overconfident claims
- **Goal**: Extract binary digits from Satoshi poster background for crypto analysis
- **New Strategy**: Region-based accuracy approach (see `REGION_BASED_ACCURACY_PROCESS.md`)

---

## 👥 Agent Roles & Responsibilities

### 🖌️ Cursor Agent
**Primary Focus**: Visual validation and manual parameter tuning

**Native Capabilities**:
- Real-time visual feedback during code editing
- Interactive parameter adjustment with immediate preview
- Quick iteration cycles for visual alignment
- Manual inspection and validation workflows

**Optimized Responsibilities**:
1. **Visual Grid Alignment**: Use visual debugging tools to manually align grid parameters
2. **Region-by-Region Tuning**: Adjust parameters for specific poster regions based on visual inspection
3. **Quality Control**: Manually verify extracted cells against original poster
4. **Parameter Documentation**: Log all visual observations and parameter changes
5. **Immediate Feedback Loop**: Test→View→Adjust→Repeat cycles for rapid optimization

---

### 🧠 Claude Code Agent
**Primary Focus**: Strategic analysis and systematic methodology

**Native Capabilities**:
- Complex problem decomposition and analysis
- Research methodology design and implementation
- Pattern recognition across large datasets
- Strategic planning and process optimization
- Comprehensive documentation and reporting

**Optimized Responsibilities**:
1. **Process Design**: Create systematic approaches for region-based extraction
2. **Accuracy Analysis**: Develop validation frameworks and metrics
3. **Problem Categorization**: Analyze different types of extraction challenges
4. **Strategic Planning**: Design multi-phase approaches for complete extraction
5. **Research Documentation**: Create comprehensive process guides and findings reports

---

### ⚡ Codex Agent
**Primary Focus**: Implementation and automation

**Native Capabilities**:
- Fast code generation and implementation
- Automated testing and validation pipelines
- Batch processing and data handling
- Utility function creation
- System integration and tooling

**Optimized Responsibilities**:
1. **Implementation**: Build region-specific extraction functions and utilities
2. **Automation**: Create automated testing for parameter validation
3. **Batch Processing**: Handle large-scale extraction across poster regions
4. **Tool Creation**: Build helper functions for manual validation workflows
5. **Pipeline Integration**: Automate the region-based extraction process

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

### 3. Version Control & Branching Protocol
- **Per-Agent Branching:** Each agent must work in its own dedicated branch. Example branch names:
  - `cursor/dev`
  - `claude/research`
  - `codex/automation`
- **Branch Workflow:**
  1. Before starting work, check out or create your agent’s branch.
  2. Commit all changes, logs, and documentation to your branch.
  3. When ready to integrate, open a pull request (PR) to `main` (or a shared integration branch).
  4. PRs must include:
     - A summary of changes and rationale
     - Links to relevant logs/markdown files
     - Any context needed for reviewers
  5. After review and testing, merge into `main`.
- **Conflict Resolution:** Coordinate with other agents to resolve merge conflicts and ensure smooth integration.

---

## 📝 Pull Request Template

```
### Summary of Changes
- Briefly describe what was changed and why

### Techniques/Experiments Implemented
- List new methods, algorithms, or experiments

### Research Needs / Open Questions
- Note any unresolved issues or areas for further investigation

### Setbacks / Breakthroughs
- Document any major obstacles or key successes

### Relevant Logs/Documentation
- Link to markdown/status files with detailed context

### Reviewer Notes
- Any special instructions or context for reviewers
```

---

## 🎯 Priority Task Matrix

### 🔥 Critical (Do First)
| Task | Best Agent | Estimated Time | Dependencies |
|------|------------|---------------|--------------|
| Visual grid alignment for high-contrast regions | Cursor | 2-3 hours | Region-based accuracy process |
| Region-based extraction strategy implementation | Claude | 1-2 hours | Process documentation complete |
| Automated region-specific parameter testing | Codex | 1-2 hours | Grid parameters per region |

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