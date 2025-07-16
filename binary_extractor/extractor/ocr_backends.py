"""
Modular OCR backends for digit recognition in the Satoshi binary extractor.

This module provides a unified interface for digit recognition, allowing easy switching
between heuristics, template matching, and deep OCR engines (EasyOCR, PaddleOCR, etc).

To add a new backend, implement the recognize_digits function and add a config flag in cfg.yaml.
"""
from typing import List, Tuple, Optional
import numpy as np

# --- Heuristic (current pipeline) ---
def recognize_digits_heuristic(cell_img: np.ndarray, cfg: dict) -> Optional[str]:
    """
    Recognize digit using current thresholding heuristics.
    Returns '0', '1', or None if undecided.
    """
    # TODO: Use the same logic as classify_single_cell, but return only '0' or '1' or None
    pass  # To be implemented

# --- Template Matching (skimage) ---
def recognize_digits_template(cell_img: np.ndarray, zero_template: np.ndarray, one_template: np.ndarray, cfg: dict) -> Optional[str]:
    """
    Recognize digit using template matching.
    Returns '0', '1', or None if undecided.
    """
    from skimage.feature import match_template
    import cv2
    # Convert to grayscale for template matching
    cell_gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY) if cell_img.ndim == 3 else cell_img
    zero_gray = cv2.cvtColor(zero_template, cv2.COLOR_BGR2GRAY) if zero_template.ndim == 3 else zero_template
    one_gray = cv2.cvtColor(one_template, cv2.COLOR_BGR2GRAY) if one_template.ndim == 3 else one_template

    # Resize cell to template size if needed
    zero_gray = cv2.resize(zero_gray, (cell_gray.shape[1], cell_gray.shape[0]))
    one_gray = cv2.resize(one_gray, (cell_gray.shape[1], cell_gray.shape[0]))

    # Normalize
    cell_gray = cell_gray.astype(np.float32) / 255.0
    zero_gray = zero_gray.astype(np.float32) / 255.0
    one_gray = one_gray.astype(np.float32) / 255.0

    # Compute match scores
    score_0 = match_template(cell_gray, zero_gray, pad_input=True).max()
    score_1 = match_template(cell_gray, one_gray, pad_input=True).max()

    # Configurable threshold for minimum confidence
    threshold = cfg.get('template_match_threshold', 0.5)
    if max(score_0, score_1) < threshold:
        return None
    return '0' if score_0 > score_1 else '1'

# --- EasyOCR (deep learning) ---
def recognize_digits_easyocr(cell_img: np.ndarray, reader=None) -> Optional[str]:
    """
    Recognize digit using EasyOCR.
    Returns '0', '1', or None if undecided.
    """
    # TODO: Requires pip install easyocr and model download
    pass  # To be implemented

# --- Unified interface ---
def recognize_digit(cell_img: np.ndarray, cfg: dict, templates: dict = None, ocr_reader=None) -> Optional[str]:
    """
    Unified digit recognition interface. Selects backend based on cfg['ocr_backend'].
    """
    backend = cfg.get('ocr_backend', 'heuristic')
    if backend == 'heuristic':
        return recognize_digits_heuristic(cell_img, cfg)
    elif backend == 'template' and templates is not None:
        return recognize_digits_template(cell_img, templates['0'], templates['1'], cfg)
    elif backend == 'easyocr' and ocr_reader is not None:
        return recognize_digits_easyocr(cell_img, ocr_reader)
    else:
        return None

# TODO: Add PaddleOCR, Kraken, Calamari, YOLO/Detectron2 backends as needed. 