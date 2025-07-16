# Satoshi Binary Poster Extractor

A test-driven Python pipeline to extract binary data from the Satoshi poster image.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Extract binary data from the poster
python scripts/extract.py "../satoshi (1).png" output/

# Run tests
pytest tests/
```

## Configuration

All tunable parameters are in `extractor/cfg.yaml`. Modify these to adjust:
- Grid detection sensitivity
- Threshold methods (Otsu vs adaptive)
- Morphology operations
- Bit classification thresholds
- **OCR backend**: `ocr_backend: heuristic` (default), `template`, `easyocr`, etc.
- **Grid detection method**: `grid_detection_method: autocorr` (default), `grid-finder`, `custom`
- **Template matching threshold**: `template_match_threshold` (default: 0.5)

## Template Matching Backend

To use template matching for digit recognition:
1. Set `ocr_backend: template` in `extractor/cfg.yaml`.
2. Place clear digit templates in `tests/data/` as `template_0_0.png`, `template_1_0.png`, etc.
   - Use `python scripts/extract_templates.py` to extract templates from the poster.
3. The pipeline will use these templates to classify ambiguous cells.

## Output

- `cells.csv`: Extracted binary data with row, col, bit values
- `bw_mask.png`: Binary mask for debugging
- Debug visualizations in output directory

## Architecture

- `pipeline.py`: Main orchestration logic
- `grid.py`: Grid detection and pitch calculation
- `classify.py`: Bit classification (0/1/blank/overlay, now supports template matching)
- `ocr_backends.py`: Modular digit recognition backends
- `cfg.yaml`: All configurable parameters

## Extending

- Add new OCR backends (EasyOCR, PaddleOCR, etc.) in `ocr_backends.py` and update `cfg.yaml`.
- Add new grid detection methods and select via `grid_detection_method` in config.
- Add more digit templates to `tests/data/` for improved template matching. 