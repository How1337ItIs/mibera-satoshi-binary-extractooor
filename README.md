---

## Linux/Headless Setup (Codex/CI Compatible)

This project is compatible with headless Ubuntu environments (e.g., Codex agent, CI, Docker):

- **Image Requirement:** Ensure `satoshi (1).png` is present in the project root. The filename (including case and spaces) must match exactly.
- **Dependencies:** Install all Python dependencies with:
  ```bash
  pip install -r requirements.txt
  ```
- **No GUI Required:** All outputs (debug images, CSVs) are saved to disk. No display or GUI is needed.
- **Template Matching:** If using template matching, ensure `tests/data/` and template PNGs are present.
- **Script Execution:** Use `python3 scriptname.py`. For direct execution, ensure scripts have Unix line endings and (optionally) `chmod +x`.
- **Output Directories:** All output directories are created automatically if missing.
- **No Windows-Only Features:** The codebase is cross-platform and does not use Windows-only features.
- **Testing:** For CI or Docker, run tests with `pytest` or perform a sample extraction run to verify setup.

--- 