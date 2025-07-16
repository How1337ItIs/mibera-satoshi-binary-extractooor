"""
extractor/adaptive_thresh.py
Block‑wise threshold: for each W×H cell block measure local stdev.
If stdev < sigma_thresh → Otsu, else Sauvola.
Returns binary mask as numpy.bool_ array.
"""

import numpy as np
import cv2
from skimage.filters import threshold_sauvola, threshold_otsu

def blockwise_threshold(gray, blk_size=64, sigma_thresh=12):
    h, w = gray.shape
    out  = np.zeros_like(gray, bool)

    for y0 in range(0, h, blk_size):
        for x0 in range(0, w, blk_size):
            y1, x1 = min(y0 + blk_size, h), min(x0 + blk_size, w)
            roi = gray[y0:y1, x0:x1]

            if roi.std() < sigma_thresh:
                t = threshold_otsu(roi)
            else:
                t = threshold_sauvola(roi, window_size=31)

            out[y0:y1, x0:x1] = roi > t
    return out

def adaptive_mask(gauss_subtracted_img, cfg):
    blk   = cfg.get('adap_block', 64)
    sigma = cfg.get('adap_sigma', 12)
    bw    = blockwise_threshold(gauss_subtracted_img, blk, sigma)
    # simple 3×3 close
    bw    = cv2.morphologyEx(bw.astype('uint8')*255,
                             cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)),
                             iterations=1).astype(bool)
    return bw 