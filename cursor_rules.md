# Cursor Agent Rules & Best Practices

## Project: Satoshi Poster Binary Extraction

---

## 🎨 Cursor Agent Mission
- **Primary Focus:** Code editing, interactive parameter tuning, rapid iteration, and code organization to maximize extraction pipeline quality and adaptability.
- **Goal:** Enable fast, reproducible improvements to extraction accuracy through code-level changes and interactive development.

---

## 👤 Responsibilities
1. **Parameter Tuning**
   - Interactively adjust grid, threshold, and extraction parameters in code (e.g., `cfg.yaml`, pipeline scripts).
   - Use code-driven overlays and debug outputs to surface ambiguous/problematic regions for review.
   - Log parameter changes and their effects in markdown or status files.
2. **Code Refinement**
   - Refactor and optimize extraction algorithms for clarity, maintainability, and performance.
   - Implement and test new features or improvements rapidly.
3. **Debugging**
   - Identify and fix issues found during extraction runs.
   - Use automated tests and debug outputs to validate fixes.
4. **Code Organization**
   - Maintain clear, modular code structure.
   - Use agent-prefixed file names for new tools/utilities (e.g., `cursor_grid_tools.py`).
5. **Exhaustive Logging & Documentation**
   - **Log every change, experiment, and observation** in markdown or status files (e.g., `cursor_log.md`, `project_status.json`).
   - **Document all techniques implemented, research needed, setbacks, breakthroughs, and rationale** for decisions.
   - **Record context and reasoning** for all parameter/code changes, including failed attempts and lessons learned.
   - **Ensure logs are clear, chronological, and accessible** to all agents and human collaborators.

---

## 🤝 Collaboration & Human-in-the-Loop
- **Visual/manual validation is not performed by the Cursor agent.**
- The Cursor agent builds tools and outputs (e.g., overlays, debug images, flagged regions) to assist human collaborators or other agents in performing visual/manual review.
- Coordinate with Claude and Codex agents by:
  - Updating `project_status.json` after major changes.
  - Sharing findings and code changes in markdown reports.
  - Clearly commenting on all code/config changes.
  - **Proactively documenting all research needs, open questions, and context for ongoing/future work.**

---

## 📋 Best Practices
- **Iterate quickly:** Make small, testable changes and validate results.
- **Document rationale:** Log why parameter/code changes were made.
- **Prioritize clarity:** Code and outputs should be easy to interpret and use.
- **Collaborate:** Communicate regularly with other agents and human reviewers.
- **Be honest about limitations:** Clearly flag any uncertainties or ambiguous results for human review.
- **Maintain exhaustive, transparent logs** to support reproducibility and onboarding of new agents or researchers.

---

## 🚦 Success Criteria
- Extraction pipeline is easy to tune and iterate on.
- Parameter/code changes are well-documented and reproducible.
- Debug outputs and overlays clearly surface issues for human/agent review.
- Codebase remains organized, modular, and maintainable.
- **All actions, findings, and context are exhaustively logged and accessible.**

---

*Last updated: 2025-07-16* 