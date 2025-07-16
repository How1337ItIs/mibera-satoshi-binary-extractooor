"""
Grid detection utilities for binary extraction.
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
from scipy import signal


def detect_grid(bw_image: np.ndarray, cfg: Dict[str, Any]) -> Tuple[List[int], List[int]]:
    """
    Detect grid structure using auto-correlation on binary mask projections.
    Implements brute-force (row0, col0) grid origin search for best alignment.
    """
    # If manual pitch values are provided, use them
    if cfg.get('row_pitch') is not None and cfg.get('col_pitch') is not None:
        row0 = cfg.get('row0', 0)
        col0 = cfg.get('col0', 0)
        rows = list(range(row0, bw_image.shape[0], cfg['row_pitch']))
        cols = list(range(col0, bw_image.shape[1], cfg['col_pitch']))
        return rows, cols

    # Auto-detect using auto-correlation
    row_pitch = int(detect_pitch_from_projection(bw_image, axis=0))
    col_pitch = int(detect_pitch_from_projection(bw_image, axis=1))

    best_score = -1
    best_rows, best_cols = None, None
    best_row0, best_col0 = 0, 0
    # Use bit_lo/bit_hi from config for confidence
    bit_lo = cfg.get('bit_lo', 0.3)
    bit_hi = cfg.get('bit_hi', 0.7)
    # Brute-force sweep over all possible (row0, col0) offsets within one pitch
    for row0 in range(row_pitch):
        for col0 in range(col_pitch):
            rows = list(range(row0, bw_image.shape[0], row_pitch))
            cols = list(range(col0, bw_image.shape[1], col_pitch))
            # Score this grid by number of confident cells
            score = 0
            for i, r in enumerate(rows):
                for j, c in enumerate(cols):
                    row_start = max(0, r - 5)
                    row_end = min(bw_image.shape[0], r + 5)
                    col_start = max(0, c - 2)
                    col_end = min(bw_image.shape[1], c + 2)
                    cell = bw_image[row_start:row_end, col_start:col_end]
                    if cell.size == 0:
                        continue
                    white_ratio = np.sum(cell == 255) / cell.size
                    if white_ratio > bit_hi or white_ratio < bit_lo:
                        score += 1
            if score > best_score:
                best_score = score
                best_rows, best_cols = rows, cols
                best_row0, best_col0 = row0, col0
    print(f"[GRID] Brute-force grid search: best_row0={best_row0}, best_col0={best_col0}, row_pitch={row_pitch}, col_pitch={col_pitch}, confident_cells={best_score}")
    return best_rows, best_cols


def detect_pitch_from_projection(bw_image: np.ndarray, axis: int) -> float:
    """
    Detect pitch using auto-correlation on image projection.
    
    Args:
        bw_image: Binary image
        axis: 0 for rows (vertical), 1 for columns (horizontal)
    
    Returns:
        Detected pitch value
    """
    # Project image along the specified axis
    if axis == 0:  # Rows
        projection = np.sum(bw_image, axis=1)
    else:  # Columns
        projection = np.sum(bw_image, axis=0)
    
    # Normalize projection
    projection = projection / np.max(projection) if np.max(projection) > 0 else projection
    
    # Compute auto-correlation
    autocorr = signal.correlate(projection, projection, mode='full')
    autocorr = autocorr[len(projection)-1:]  # Take positive lags only
    
    # Find peaks in auto-correlation
    peaks = find_peaks_in_autocorr(autocorr)
    
    if not peaks:
        # Fallback to reasonable default
        return 31.0 if axis == 0 else 12.5
    
    # Return the first significant peak
    return float(peaks[0])


def find_peaks_in_autocorr(autocorr: np.ndarray) -> List[float]:
    """
    Find peaks in auto-correlation.
    
    Args:
        autocorr: Auto-correlation array
    
    Returns:
        List of peak positions
    """
    peaks = []
    
    # Find local maxima with reasonable constraints
    for i in range(10, min(100, len(autocorr))):  # Look for peaks between 10-100 pixels
        if (i > 0 and i < len(autocorr) - 1 and
            autocorr[i] > autocorr[i-1] and 
            autocorr[i] > autocorr[i+1] and
            autocorr[i] > np.mean(autocorr) * 1.2):  # Significant peak
            peaks.append(i)
    
    return peaks


def generate_grid_lines(
    image_size: int, 
    pitch: float, 
    min_grid_size: int
) -> List[int]:
    """
    Generate grid line positions based on detected pitch.
    
    Args:
        image_size: Size of image dimension
        pitch: Detected pitch
        min_grid_size: Minimum number of grid lines to generate
    
    Returns:
        List of grid line positions
    """
    if pitch <= 0:
        return []
    
    # Generate grid lines
    lines = []
    pos = pitch / 2  # Start at half pitch (center of first cell)
    
    while pos < image_size and len(lines) < min_grid_size:
        lines.append(int(pos))
        pos += pitch
    
    return lines


def refine_grid_origin(
    bw_image: np.ndarray, 
    rows: List[int], 
    cols: List[int],
    cfg: Dict[str, Any]
) -> Tuple[List[int], List[int]]:
    """
    Refine grid origin by maximizing correlation with expected pattern.
    
    Args:
        bw_image: Binary image
        rows: Current row positions
        cols: Current column positions
        cfg: Configuration
    
    Returns:
        Refined (rows, cols) positions
    """
    # This is a placeholder for grid origin refinement
    # TODO: Implement auto-correlation based origin refinement
    # For now, return the original grid
    return rows, cols 