"""
Cell classification utilities for binary extraction.
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
from pathlib import Path

# Import OCR backends
from .ocr_backends import recognize_digit

def load_templates(template_dir: Path) -> Dict[str, np.ndarray]:
    """
    Load 0 and 1 templates from the given directory.
    Returns a dict: {'0': template_img, '1': template_img}
    """
    print(f"[DEBUG] Looking for templates in: {template_dir.resolve()}")
    templates = {}
    for label in ['0', '1']:
        # Find all template files for this label
        files = sorted(template_dir.glob(f"template_{label}_*.png"))
        print(f"[DEBUG] Found {len(files)} files for label '{label}': {[str(f) for f in files]}")
        if files:
            # Use the first template for now (could average or use all in future)
            templates[label] = cv2.imread(str(files[0]))
    return templates

def classify_cell_bits(
    original_img: np.ndarray,
    bw_image: np.ndarray,
    overlay_mask: np.ndarray,
    rows: List[int],
    cols: List[int],
    cfg: Dict[str, Any]
) -> List[Tuple[int, int, str]]:
    """
    Classify each cell as 0, 1, blank, or overlay.
    Uses template matching if cfg['template_match'] == True.
    Implements dual-pass thresholding: first pass conservative, second pass looser on blanks.
    """
    cells = []
    ocr_backend = cfg.get('ocr_backend', 'heuristic')
    templates = None
    if cfg.get('template_match', False):
        # Load templates from tests/data/
        template_dir = Path(__file__).parent.parent / 'tests' / 'data'
        templates = load_templates(template_dir)
        if not templates or '0' not in templates or '1' not in templates:
            raise RuntimeError("Template matching selected but templates not found in tests/data/")

    # Get config-driven parameters
    cell_row_half = cfg.get('cell_window', {}).get('row_half', 5)
    cell_col_half = cfg.get('cell_window', {}).get('col_half', 2)
    thresholds = cfg.get('dual_pass_thresholds', {})
    bit_lo_1 = thresholds.get('bit_lo_1', 0.35)
    bit_hi_1 = thresholds.get('bit_hi_1', 0.70)
    bit_lo_2 = thresholds.get('bit_lo_2', 0.25)
    bit_hi_2 = thresholds.get('bit_hi_2', 0.55)

    # Store cell results and intermediate blank mask
    cell_results = {}
    blank_cells = []
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            row_start = max(0, row - cell_row_half)
            row_end = min(bw_image.shape[0], row + cell_row_half)
            col_start = max(0, col - cell_col_half)
            col_end = min(bw_image.shape[1], col + cell_col_half)
            cell_bw = bw_image[row_start:row_end, col_start:col_end]
            cell_overlay = overlay_mask[row_start:row_end, col_start:col_end]
            cell_img = original_img[row_start:row_end, col_start:col_end]
            # Overlay check
            overlay_coverage = np.sum(cell_overlay > 0) / cell_overlay.size if cell_overlay.size > 0 else 0
            if overlay_coverage > cfg['overlay']['cell_coverage_threshold']:
                bit = 'overlay'
            else:
                # Conservative first pass
                white_pixels = np.sum(cell_bw == 255)
                total_pixels = cell_bw.size
                white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0
                if white_ratio > bit_hi_1:
                    bit = '1'
                elif white_ratio < bit_lo_1:
                    bit = '0'
                else:
                    bit = 'blank'
            cell_results[(i, j)] = bit
            if bit == 'blank':
                blank_cells.append((i, j, cell_bw, cell_overlay, cell_img))
            cells.append((i, j, bit))
    # Second pass: looser thresholds only on blanks
    for i, j, cell_bw, cell_overlay, cell_img in blank_cells:
        # Only update if still blank
        idx = i * len(cols) + j
        if cell_results[(i, j)] == 'blank':
            white_pixels = np.sum(cell_bw == 255)
            total_pixels = cell_bw.size
            white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0
            if white_ratio > bit_hi_2:
                cells[idx] = (i, j, '1')
                cell_results[(i, j)] = '1'
            elif white_ratio < bit_lo_2:
                cells[idx] = (i, j, '0')
                cell_results[(i, j)] = '0'
            # else remain blank
    return cells


def classify_single_cell(
    cell_bw: np.ndarray,
    cell_overlay: np.ndarray,
    cfg: Dict[str, Any]
) -> str:
    """
    Classify a single cell based on its content.
    
    Args:
        cell_bw: Binary cell region
        cell_overlay: Overlay mask cell region
        cfg: Configuration dictionary
    
    Returns:
        Classification: '0', '1', 'blank', or 'overlay'
    """
    # Check for overlay first (silver ladder/caduceus)
    overlay_coverage = np.sum(cell_overlay > 0) / cell_overlay.size
    if overlay_coverage > cfg['overlay']['cell_coverage_threshold']:
        return 'overlay'
    
    # Calculate white pixel percentage in binary cell
    white_pixels = np.sum(cell_bw == 255)
    total_pixels = cell_bw.size
    white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0
    
    # Classify based on white pixel ratio
    if white_ratio > cfg['bit_hi']:
        return '1'
    elif white_ratio < cfg['bit_lo']:
        return '0'
    else:
        return 'blank'


def analyze_cell_patterns(
    cells: List[Tuple[int, int, str]],
    rows: int,
    cols: int
) -> Dict[str, Any]:
    """
    Analyze patterns in classified cells.
    
    Args:
        cells: List of (row, col, bit) tuples
        rows: Number of rows
        cols: Number of columns
    
    Returns:
        Analysis dictionary
    """
    # Create 2D grid for analysis
    grid = [['blank' for _ in range(cols)] for _ in range(rows)]
    
    for row, col, bit in cells:
        if 0 <= row < rows and 0 <= col < cols:
            grid[row][col] = bit
    
    # Analyze patterns
    analysis = {
        'total_cells': len(cells),
        'grid_shape': (rows, cols),
        'row_analysis': [],
        'column_analysis': []
    }
    
    # Analyze each row
    for i in range(rows):
        row_bits = [grid[i][j] for j in range(cols)]
        zeros = row_bits.count('0')
        ones = row_bits.count('1')
        blanks = row_bits.count('blank')
        overlays = row_bits.count('overlay')
        
        analysis['row_analysis'].append({
            'row': i,
            'zeros': zeros,
            'ones': ones,
            'blanks': blanks,
            'overlays': overlays,
            'legible': zeros + ones
        })
    
    # Analyze each column
    for j in range(cols):
        col_bits = [grid[i][j] for i in range(rows)]
        zeros = col_bits.count('0')
        ones = col_bits.count('1')
        blanks = col_bits.count('blank')
        overlays = col_bits.count('overlay')
        
        analysis['column_analysis'].append({
            'col': j,
            'zeros': zeros,
            'ones': ones,
            'blanks': blanks,
            'overlays': overlays,
            'legible': zeros + ones
        })
    
    return analysis


def extract_ascii_from_cells(
    cells: List[Tuple[int, int, str]],
    rows: int,
    cols: int
) -> List[str]:
    """
    Extract ASCII text from classified cells (8-bit row-major).
    
    Args:
        cells: List of (row, col, bit) tuples
        rows: Number of rows
        cols: Number of columns
    
    Returns:
        List of ASCII strings (one per row)
    """
    # Create 2D grid
    grid = [['blank' for _ in range(cols)] for _ in range(rows)]
    
    for row, col, bit in cells:
        if 0 <= row < rows and 0 <= col < cols:
            grid[row][col] = bit
    
    ascii_strings = []
    
    for i in range(rows):
        row_bits = []
        for j in range(cols):
            if grid[i][j] in ['0', '1']:
                row_bits.append(grid[i][j])
            elif grid[i][j] == 'overlay':
                # For overlay, we could try both 0 and 1 later
                row_bits.append('?')
            else:
                # Stop at first blank (end of meaningful data)
                break
        
        if len(row_bits) >= 8:
            # Convert 8-bit chunks to ASCII
            ascii_chars = []
            for k in range(0, len(row_bits), 8):
                if k + 8 <= len(row_bits):
                    byte_str = ''.join(row_bits[k:k+8])
                    if '?' not in byte_str:  # Only convert if no overlays
                        try:
                            ascii_val = int(byte_str, 2)
                            if 32 <= ascii_val <= 126:  # Printable ASCII
                                ascii_chars.append(chr(ascii_val))
                        except ValueError:
                            pass
            
            if ascii_chars:
                ascii_strings.append(''.join(ascii_chars))
    
    return ascii_strings 