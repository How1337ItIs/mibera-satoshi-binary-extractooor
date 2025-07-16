import os
from pathlib import Path
import cv2
import numpy as np
import yaml
from binary_extractor.extractor.pipeline import load_config, extract_channel, create_overlay_mask
from binary_extractor.extractor.grid import detect_grid
from binary_extractor.extractor.classify import classify_cell_bits

# --- Config ---
POSTER_IMAGE = 'satoshi_poster.png'  # Use Linux-friendly name
OUT_DIR = Path('training_data')
CROP_SIZE = 32
LABELS = ['0', '1', 'blank']

# --- Ensure output directories ---
for label in LABELS:
    (OUT_DIR / label).mkdir(parents=True, exist_ok=True)

# --- Load config and image ---
cfg = load_config()
img = cv2.imread(POSTER_IMAGE)
if img is None:
    raise FileNotFoundError(f"Could not load image: {POSTER_IMAGE}")

# --- Preprocess image (as in pipeline) ---
channel = extract_channel(img, cfg['use_color_space'])
hi = cv2.subtract(channel, cv2.GaussianBlur(channel, (0, 0), cfg['blur_sigma']))
# Thresholding (Otsu for now)
thresh_val, bw = cv2.threshold(hi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# Morphology
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg['morph_k'], cfg['morph_k']))
bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=cfg['morph_iterations'])
# Overlay mask
overlay_mask = create_overlay_mask(img, cfg)

# --- Grid detection ---
if cfg['row_pitch'] is None or cfg['col_pitch'] is None:
    rows, cols = detect_grid(bw, cfg)
else:
    rows = list(range(cfg['row0'], img.shape[0], cfg['row_pitch']))
    cols = list(range(cfg['col0'], img.shape[1], cfg['col_pitch']))

# --- Classify cells ---
cells = classify_cell_bits(img, bw, overlay_mask, rows, cols, cfg)

# --- Extract and save crops ---
crop_count = {label: 0 for label in LABELS}
for row_idx, col_idx, bit in cells:
    if bit not in LABELS:
        continue  # skip overlays, etc.
    # Compute crop region (centered on grid point)
    r = rows[row_idx]
    c = cols[col_idx]
    r0 = max(0, r - CROP_SIZE // 2)
    r1 = min(img.shape[0], r + CROP_SIZE // 2)
    c0 = max(0, c - CROP_SIZE // 2)
    c1 = min(img.shape[1], c + CROP_SIZE // 2)
    crop = img[r0:r1, c0:c1]
    # Pad if needed
    crop = cv2.copyMakeBorder(
        crop,
        top=max(0, CROP_SIZE//2 - r),
        bottom=max(0, (r + CROP_SIZE//2) - img.shape[0]),
        left=max(0, CROP_SIZE//2 - c),
        right=max(0, (c + CROP_SIZE//2) - img.shape[1]),
        borderType=cv2.BORDER_CONSTANT,
        value=[0,0,0]
    )
    crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
    out_path = OUT_DIR / bit / f"cell_{row_idx}_{col_idx}.png"
    cv2.imwrite(str(out_path), crop)
    crop_count[bit] += 1

# --- Log dataset stats ---
with open(OUT_DIR / 'extraction_log.md', 'w') as f:
    f.write(f"# CNN Training Data Extraction Log\n\n")
    f.write(f"Poster image: {POSTER_IMAGE}\n")
    f.write(f"Crop size: {CROP_SIZE}x{CROP_SIZE}\n")
    f.write(f"Labels: {LABELS}\n")
    f.write(f"\n## Crop counts\n")
    for label in LABELS:
        f.write(f"- {label}: {crop_count[label]}\n")
    f.write(f"\nExtraction complete.\n")

print("Extraction complete. Crops per label:", crop_count) 