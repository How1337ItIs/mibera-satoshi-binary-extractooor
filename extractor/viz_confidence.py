"""
extractor/viz_confidence.py
Colour‑codes bits onto the original poster: green (high conf 1), blue (high conf 0),
red‑orange gradient for low confidence, magenta overlay.
"""

import cv2
import numpy as np

def overlay_confidence(bgr, rows, cols, bits, conf, overlay_mask):
    out = bgr.copy()
    h_pitch = np.diff(rows).mean().astype(int)
    w_pitch = np.diff(cols).mean().astype(int)

    for i, r in enumerate(rows[:-1]):
        for j, c in enumerate(cols[:-1]):
            x0, y0 = c, r
            x1, y1 = c + w_pitch, r + h_pitch
            if overlay_mask[i, j]:
                cv2.rectangle(out, (x0, y0), (x1, y1), (255, 0, 255), 1)
            elif bits[i, j] == 1:
                color = (0, int(255 * conf[i, j]), 0)
                cv2.rectangle(out, (x0, y0), (x1, y1), color, 1)
            elif bits[i, j] == 0:
                color = (int(255 * conf[i, j]), 0, 0)
                cv2.rectangle(out, (x0, y0), (x1, y1), color, 1)
    return out 