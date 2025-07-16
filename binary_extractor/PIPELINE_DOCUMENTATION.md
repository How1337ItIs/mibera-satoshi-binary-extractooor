# Satoshi Binary Poster Extractor: Pipeline Documentation

## Overview

This pipeline extracts every visible ‘0’ and ‘1’ bit from the Satoshi poster image, even under challenging conditions (grainy background, silver overlays, faint glyphs). The approach is modular, highly configurable, and test-driven, with every step and parameter exposed in `cfg.yaml` for grid-search and reproducibility.

---

## Pipeline Steps

### 1. Color De-mix / Channel Extraction
- **Purpose:** Isolate the channel where cyan digits are most visible.
- **Options:**  
  - `Lab_b`: Lab color space, b channel (blue-yellow)
  - `HSV_S`: HSV color space, saturation channel
  - `RGB_inv`: Inverted green channel (sometimes reveals faint digits)
  - `HSV_S_inv`, `Lab_b_inv`: Inverted versions for faint glyph rescue
- **Config:**  
  `use_color_space: Lab_b | HSV_S | RGB_inv | ...`

### 2. Background Subtraction
- **Purpose:** Remove smooth background variations, enhance digit contrast.
- **Method:**  
  `hi = channel - GaussianBlur(channel, σ = cfg.blur_sigma)`
- **Config:**  
  `blur_sigma: 15` (tunable)

### 3. Thresholding
- **Purpose:** Convert the image to binary (foreground/background).
- **Options:**  
  - `otsu`: Global Otsu threshold
  - `adaptive`: Adaptive Gaussian threshold
  - `sauvola`: Local Sauvola threshold (via scikit-image)
- **Config:**  
  ```yaml
  threshold:
    method: otsu | adaptive | sauvola
    adaptive_C: 4
    sauvola_window_size: 15
    sauvola_k: 0.2
  ```

### 4. Morphology
- **Purpose:** Clean up speckles, connect broken glyphs, thin strokes.
- **Steps:**  
  - Open → Close with rectangular kernel (`morph_k`, `morph_iterations`)
  - Optional: `mahotas.thin` for skeletonization
- **Config:**  
  ```yaml
  morph_k: 3
  morph_iterations: 2
  use_mahotas_thin: false
  ```

### 5. Overlay (Silver Mask) Detection
- **Purpose:** Mask out regions covered by the silver ladder/caduceus overlay.
- **Method:**  
  - Overlay ≈ low saturation & high value in HSV
  - Mask: `(S < 40) & (V > 180)`, then dilate by 2 px
  - Any cell with >20% overlay pixels is labeled `"overlay"`
- **Config:**  
  ```yaml
  overlay:
    saturation_threshold: 40
    value_threshold: 180
    cell_coverage_threshold: 0.2
    dilate_pixels: 2
  ```

### 6. Grid Detection
- **Purpose:** Find the grid of digit cells (row/col pitch, origin).
- **Method:**  
  - Row pitch: argmax of autocorrelation of horizontal projection
  - Col pitch: argmax of autocorrelation of vertical projection
  - User can override with manual values (`row_pitch`, `col_pitch`, `row0`, `col0`)
- **Config:**  
  ```yaml
  row_pitch: null
  col_pitch: null
  row0: 50
  col0: 20
  ```

### 7. Cell Classification
- **Purpose:** Assign each cell as `0`, `1`, `blank`, or `overlay`.
- **Logic:**  
  - If overlay mask > 20%: `"overlay"`
  - Else, if white pixel fraction > `bit_hi`: `"1"`
  - Else, if < `bit_lo`: `"0"`
  - Else: `"blank"`
- **Config:**  
  ```yaml
  bit_hi: 0.7
  bit_lo: 0.3
  ```

### 8. Template Matching Fallback (Optional)
- **Purpose:** Rescue faint digits missed by thresholding.
- **Method:**  
  - If `template_match: true`, use `skimage.feature.match_template` with clear “0” and “1” templates.
  - If match score > `tm_thresh`, overwrite bit classification.
- **Config:**  
  ```yaml
  template_match: false
  tm_thresh: 0.45
  ```

### 9. Output
- **CSV:**  
  - `cells.csv` with columns: `row, col, bit`
- **Debug Artifacts:**  
  - `bw_mask.png`: Binarized glyphs
  - `silver_mask.png`: Overlay mask
  - `grid_overlay.png`: Grid overlay on original
  - `cells_color.png`: Color-coded cell types
  - `cyan_channel.png`, `gaussian_subtracted.png`: Preprocessing steps

---

## Configuration Example (`cfg.yaml`)

```yaml
use_color_space: HSV_S
blur_sigma: 25
threshold:
  method: otsu
  adaptive_C: 4
  sauvola_window_size: 15
  sauvola_k: 0.2
morph_k: 3
morph_iterations: 2
use_mahotas_thin: false
row_pitch: null
col_pitch: null
row0: 50
col0: 20
bit_hi: 0.7
bit_lo: 0.3
overlay:
  saturation_threshold: 40
  value_threshold: 180
  cell_coverage_threshold: 0.2
  dilate_pixels: 2
template_match: false
tm_thresh: 0.45
save_debug: true
debug_artifacts:
  - bw_mask.png
  - silver_mask.png
  - grid_overlay.png
  - cells_color.png
  - cyan_channel.png
  - gaussian_subtracted.png
output:
  csv_encoding: 'utf-8'
```

---

## Debugging & Tuning

- **If all bits are “1” or “0”:**  
  - Try a different color space (`use_color_space`)
  - Increase `blur_sigma`
  - Adjust `bit_hi`/`bit_lo` thresholds
  - Check grid alignment (`row0`, `col0`)
- **If overlays not detected:**  
  - Lower `overlay.saturation_threshold`
  - Increase `overlay.value_threshold`
  - Increase `overlay.dilate_pixels`
- **If faint digits missing:**  
  - Try `threshold.method: sauvola`
  - Enable `template_match: true`

---

## Reference CSVs & Validation

- **recognized_digits.csv**: Best-guess reference for digit positions and values. Use this to:
  - Compare the distribution of 0’s and 1’s (should be roughly even, not all 0’s or all 1’s)
  - Validate that your pipeline is not systematically missing one class
  - Spot-check specific rows/cols for agreement
- **overlay_unknown_cells.csv**: Reference for cells likely covered by overlay. Use this to:
  - Validate overlay detection logic
  - Tune `overlay` parameters for best match

**How to Compare:**
- Use a script or notebook to compare your `cells.csv` output to `recognized_digits.csv` and `overlay_unknown_cells.csv`.
- Focus on:
  - **Distribution:** Are 0’s and 1’s roughly balanced?
  - **Overlay:** Are overlay cells detected in the right places?
  - **False Positives/Negatives:** Are there systematic errors?

---

## Testing & Validation

- **Unit tests** cover:
  - Grid detection
  - Overlay detection
  - Cell classification
  - Template matching fallback
  - Regression guard: hash of `cells.csv` for full poster
- **Test data:**
  - `tests/data/top_row.png`: Should decode to “The Times…”
  - `tests/data/ladder_slice.png`: ≥90% cells = “overlay”
  - `tests/data/faint_digits.png`: More digits with `sauvola` than `otsu`

---

## Philosophy

- **No hard-coded magic numbers:** Everything is in `cfg.yaml`.
- **No decoding/interpretation:** This repo is for bit extraction only.
- **Every run is reproducible:** All parameters and artifacts are saved.
- **Test-driven:** Refactors cannot silently scramble bits.

---

## Next Steps

- Continue tuning parameters for even 0/1 distribution.
- Use the reference CSVs for validation and grid-search.
- Add more test images and assertions as needed.
- Document all changes and results for future contributors. 