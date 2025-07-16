#!/usr/bin/env python3
"""
Extracts clear 0 and 1 digit templates from the Satoshi poster image for use in template matching.
Saves templates as PNGs in tests/data/.
"""
import cv2
import numpy as np
from pathlib import Path

# Paths
POSTER_PATH = Path("../satoshi (1).png")
OUTPUT_DIR = Path("../tests/data/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Manually identified (row, col, w, h) for clear 0 and 1 digits (update as needed)
# These coordinates are approximate and may need adjustment for your image
TEMPLATES = [
    {"label": "0", "coords": (180, 120, 18, 28)},  # (x, y, w, h)
    {"label": "0", "coords": (320, 220, 18, 28)},
    {"label": "1", "coords": (250, 150, 12, 28)},
    {"label": "1", "coords": (400, 300, 12, 28)},
]

def main():
    img = cv2.imread(str(POSTER_PATH))
    if img is None:
        print(f"Error: Could not load image {POSTER_PATH}")
        return
    
    for i, tpl in enumerate(TEMPLATES):
        x, y, w, h = tpl["coords"]
        crop = img[y:y+h, x:x+w]
        out_path = OUTPUT_DIR / f"template_{tpl['label']}_{i}.png"
        cv2.imwrite(str(out_path), crop)
        print(f"Saved {tpl['label']} template to {out_path}")

if __name__ == "__main__":
    main() 