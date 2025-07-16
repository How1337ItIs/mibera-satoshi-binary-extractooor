import cv2
import numpy as np
import pathlib

def harvest_templates(bw_mask, rows, cols, out_dir, cfg, bits, conf_matrix):
    """
    Grab the first N high-confidence 0/1 glyphs and save as PNG templates.
    Assumes you’ve already computed `conf_matrix` 0–1 and `bits` 0/1.
    """
    N = cfg.get("template_harvest_n", 3)
    h_pitch = int(np.diff(rows).mean())
    w_pitch = int(np.diff(cols).mean())

    picked = {0: 0, 1: 0}
    for i, r in enumerate(rows[:-1]):
        for j, c in enumerate(cols[:-1]):
            bit = bits[i, j]
            conf = conf_matrix[i, j]
            if bit in (0, 1) and conf > 0.95 and picked[bit] < N:
                roi = bw_mask[r:r+h_pitch, c:c+w_pitch].astype('uint8') * 255
                fname = pathlib.Path(out_dir) / f"template_{bit}_{picked[bit]}.png"
                cv2.imwrite(str(fname), roi)
                picked[bit] += 1
            if all(p == N for p in picked.values()):
                return 