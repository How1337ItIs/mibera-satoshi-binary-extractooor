"""
extractor/entropy_fill.py
Try both bit patterns for every overlay‑masked cell in a row, pick the one
that reduces byte entropy and maximises printable ASCII ratio.
"""

import numpy as np
from scipy.stats import entropy

def printable_ascii_ratio(byte_arr):
    return np.count_nonzero((byte_arr >= 32) & (byte_arr <= 126)) / len(byte_arr)

def brute_row(bits_row, mask_row):
    """bits_row:  np.array of 0/1 for visible bits, placeholder -1 for overlay
       mask_row:  boolean array True where overlay bit is present
    """
    candidates = []

    # generate all fill combos – limit to ≤12 masked bits per row for feasibility
    masked_idx = np.where(mask_row)[0]
    m = len(masked_idx)
    if m > 12:
        return bits_row  # skip brute if too many

    for fill in range(1 << m):
        candidate = bits_row.copy()
        for j, bit_pos in enumerate(masked_idx):
            candidate[bit_pos] = (fill >> j) & 1
        # group into bytes
        byte_arr = np.packbits(candidate.reshape(-1, 8)[:, ::-1], axis=1).flatten()
        score = (printable_ascii_ratio(byte_arr), -entropy(np.bincount(byte_arr, minlength=256)))
        candidates.append((score, candidate))

    # pick best by tuple comparison
    best_bits = max(candidates, key=lambda x: x[0])[1]
    return best_bits 