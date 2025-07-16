"""
Main pipeline orchestrator for binary extraction from Satoshi poster.

Low-level image alchemy pipeline with comprehensive configurable options:
- Color space de-mixing (Lab_b, HSV_S, RGB_inv, etc.)
- Background subtraction with Gaussian blur
- Multiple thresholding methods (Otsu, adaptive, Sauvola)
- Morphological cleaning with optional skeletonization
- Overlay detection and masking
- Grid detection with auto-correlation
- Template matching fallback for faint digits
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
import csv
import yaml
from skimage import filters
import mahotas

from .grid import detect_grid
from .classify import classify_cell_bits


def load_config(config_path: Path = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "cfg.yaml"
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def extract_channel(img: np.ndarray, color_space: str) -> np.ndarray:
    """
    Extract the specified channel from the image based on color space.
    
    Args:
        img: Input BGR image
        color_space: Channel specification (Lab_b, HSV_S, RGB_R, etc.)
    
    Returns:
        Single-channel image
    """
    # RGB channels
    if color_space == 'RGB_R':
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return rgb[:, :, 0]  # Red channel
    elif color_space == 'RGB_G':
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return rgb[:, :, 1]  # Green channel
    elif color_space == 'RGB_B':
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return rgb[:, :, 2]  # Blue channel
    
    # HSV channels
    elif color_space == 'HSV_H':
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return hsv[:, :, 0]  # Hue channel
    elif color_space == 'HSV_S':
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return hsv[:, :, 1]  # Saturation channel
    elif color_space == 'HSV_V':
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return hsv[:, :, 2]  # Value channel
    
    # LAB channels
    elif color_space == 'LAB_L':
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        return lab[:, :, 0]  # L channel (lightness)
    elif color_space == 'LAB_A':
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        return lab[:, :, 1]  # A channel (green-red)
    elif color_space == 'LAB_B':
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        return lab[:, :, 2]  # B channel (blue-yellow)
    
    # YUV channels
    elif color_space == 'YUV_Y':
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        return yuv[:, :, 0]  # Y channel (luminance)
    elif color_space == 'YUV_U':
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        return yuv[:, :, 1]  # U channel
    elif color_space == 'YUV_V':
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        return yuv[:, :, 2]  # V channel
    
    # HLS channels
    elif color_space == 'HLS_H':
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        return hls[:, :, 0]  # Hue channel
    elif color_space == 'HLS_L':
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        return hls[:, :, 1]  # Lightness channel
    elif color_space == 'HLS_S':
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        return hls[:, :, 2]  # Saturation channel
    
    # Legacy support
    elif color_space == 'Lab_b':
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        return lab[:, :, 2]  # b channel (blue-yellow)
    elif color_space == 'RGB_inv':
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return 255 - rgb[:, :, 1]  # inverted green channel
    elif color_space == 'HSV_S_inv':
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return 255 - hsv[:, :, 1]  # inverted saturation
    elif color_space == 'Lab_b_inv':
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        return 255 - lab[:, :, 2]  # inverted b channel
    else:
        raise ValueError(f"Unknown color space: {color_space}")


def apply_threshold(channel: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """
    Apply thresholding based on configuration.
    
    Args:
        channel: Single-channel image
        cfg: Configuration dictionary
    
    Returns:
        Binary image (0/255)
    """
    method = cfg['threshold']['method']
    adaptive_block_size = cfg.get('adaptive_block_size', 35)  # Configurable block size
    if method == 'otsu':
        _, bw = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'adaptive':
        bw = cv2.adaptiveThreshold(
            channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, adaptive_block_size, cfg['threshold']['adaptive_C'])
    elif method == 'sauvola':
        bw = filters.threshold_sauvola(
            channel, window_size=cfg['threshold']['sauvola_window_size'],
            k=cfg['threshold']['sauvola_k'])
        bw = (channel > bw).astype(np.uint8) * 255
    else:
        raise ValueError(f"Unknown threshold method: {method}")
    
    return bw


def create_overlay_mask(img: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """
    Create mask for silver overlay (ladder/caduceus).
    
    Args:
        img: Original BGR image
        cfg: Configuration dictionary
    
    Returns:
        Binary mask where True = overlay
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    
    # Silver overlay: low saturation, high value
    overlay_mask = (saturation < cfg['overlay']['saturation_threshold']) & \
                   (value > cfg['overlay']['value_threshold'])
    
    # Dilate to prevent cyan leakage through edges
    if cfg['overlay']['dilate_pixels'] > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                         (cfg['overlay']['dilate_pixels'], cfg['overlay']['dilate_pixels']))
        overlay_mask = cv2.dilate(overlay_mask.astype(np.uint8), kernel, iterations=1)
    
    return overlay_mask.astype(np.uint8) * 255


def save_debug_artifacts(img: np.ndarray, bw: np.ndarray, overlay_mask: np.ndarray, 
                        channel: np.ndarray, hi: np.ndarray, out_dir: Path, cfg: Dict[str, Any]):
    """Save debug artifacts for analysis."""
    if not cfg.get('save_debug', True):
        return
    
    # Save binary mask
    cv2.imwrite(str(out_dir / 'bw_mask.png'), bw)
    
    # Save overlay mask
    cv2.imwrite(str(out_dir / 'silver_mask.png'), overlay_mask)
    
    # Save channel and background-subtracted
    cv2.imwrite(str(out_dir / 'cyan_channel.png'), channel)
    cv2.imwrite(str(out_dir / 'gaussian_subtracted.png'), hi)


def save_csv(cells: List[Tuple[int, int, str]], output_path: Path, encoding: str = 'utf-8'):
    """Save extracted cell data to CSV file."""
    with open(output_path, 'w', newline='', encoding=encoding) as f:
        writer = csv.writer(f)
        writer.writerow(['row', 'col', 'bit'])
        for row, col, bit in cells:
            writer.writerow([row, col, bit])


def run(image_path: Path, out_dir: Path, cfg: dict = None):
    """
    Main pipeline execution with low-level image alchemy.
    """
    print(f"[PIPELINE] Starting run() with image_path={image_path}, out_dir={out_dir}")
    try:
        if cfg is None:
            cfg = load_config()
        print(f"[PIPELINE] Loaded config: {cfg}")
        # Ensure output directory exists
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[PIPELINE] Output directory ensured: {out_dir}")
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        print(f"[PIPELINE] Image loaded: shape={img.shape}")
        # Step 1: Color de-mix
        channel = extract_channel(img, cfg['use_color_space'])
        print(f"[PIPELINE] Channel extracted: {cfg['use_color_space']}")
        # Step 2: Background subtraction
        hi = cv2.subtract(channel, cv2.GaussianBlur(channel, (0, 0), cfg['blur_sigma']))
        print(f"[PIPELINE] Background subtraction done")
        # Step 3: Thresholding
        bw = apply_threshold(hi, cfg)
        print(f"[PIPELINE] Thresholding done")
        # Step 4: Morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg['morph_k'], cfg['morph_k']))
        morph_open_iterations = cfg.get('morph_open_iterations', 1)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=morph_open_iterations)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=cfg['morph_iterations'])
        print(f"[PIPELINE] Morphology done")
        # Optional skeletonization
        if cfg.get('use_mahotas_thin', False):
            bw = mahotas.thin(bw)
            print(f"[PIPELINE] Skeletonization done")
        # Step 5: Overlay mask
        overlay_mask = create_overlay_mask(img, cfg)
        print(f"[PIPELINE] Overlay mask created")
        # Save debug artifacts
        save_debug_artifacts(img, bw, overlay_mask, channel, hi, out_dir, cfg)
        print(f"[PIPELINE] Debug artifacts saved")
        # Step 6: Grid detection
        if cfg['row_pitch'] is None or cfg['col_pitch'] is None:
            rows, cols = detect_grid(bw, cfg)
            print(f"[PIPELINE] Grid detected (auto): rows={len(rows)}, cols={len(cols)}")
        else:
            rows = list(range(cfg['row0'], img.shape[0], cfg['row_pitch']))
            cols = list(range(cfg['col0'], img.shape[1], cfg['col_pitch']))
            print(f"[PIPELINE] Grid detected (manual): rows={len(rows)}, cols={len(cols)}")
        # Step 7: Cell classification
        cells = classify_cell_bits(img, bw, overlay_mask, rows, cols, cfg)
        print(f"[PIPELINE] Cell classification done: {len(cells)} cells")
        # Step 8: Template matching fallback (if enabled)
        if cfg.get('template_match', False):
            cells = apply_template_matching_fallback(img, cells, rows, cols, cfg)
            print(f"[PIPELINE] Template matching fallback applied")
        # Save results
        csv_path = out_dir / 'cells.csv'
        save_csv(cells, csv_path, cfg['output']['csv_encoding'])
        print(f"[PIPELINE] cells.csv saved: {csv_path}")
        # Save recognized_digits.csv (only 0/1 bits)
        recognized_digits_path = out_dir / 'recognized_digits.csv'
        with open(recognized_digits_path, 'w', newline='', encoding=cfg['output']['csv_encoding']) as f:
            writer = csv.writer(f)
            writer.writerow(['row', 'col', 'digit'])
            for row, col, bit in cells:
                if bit in ('0', '1'):
                    writer.writerow([row, col, bit])
        print(f"[PIPELINE] recognized_digits.csv saved: {recognized_digits_path}")
        # Save overlay_unknown_cells.csv (only overlay/blank)
        overlay_unknown_path = out_dir / 'overlay_unknown_cells.csv'
        with open(overlay_unknown_path, 'w', newline='', encoding=cfg['output']['csv_encoding']) as f:
            writer = csv.writer(f)
            writer.writerow(['row', 'col'])
            for row, col, bit in cells:
                if bit in ('overlay', 'blank'):
                    writer.writerow([row, col])
        print(f"[PIPELINE] overlay_unknown_cells.csv saved: {overlay_unknown_path}")
        # Create debug visualizations
        create_debug_visualizations(img, cells, rows, cols, out_dir, cfg)
        print(f"[PIPELINE] Debug visualizations created")
        print(f"[PIPELINE] Finished run() successfully")
        return cells
    except Exception as e:
        print(f"[PIPELINE] ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


def apply_template_matching_fallback(img: np.ndarray, cells: List[Tuple[int, int, str]], 
                                   rows: List[int], cols: List[int], cfg: Dict[str, Any]) -> List[Tuple[int, int, str]]:
    """Apply template matching fallback for uncertain cells."""
    # TODO: Implement template matching fallback
    # This would extract templates from high-confidence cells and apply skimage.feature.match_template
    return cells


def create_debug_visualizations(img: np.ndarray, cells: List[Tuple[int, int, str]], 
                              rows: List[int], cols: List[int], out_dir: Path, cfg: Dict[str, Any]):
    """Create debug visualizations."""
    if not cfg.get('save_debug', True):
        return
    
    # Grid overlay
    overlay = img.copy()
    for row in rows[:20]:  # Show first 20 rows
        cv2.line(overlay, (0, row), (img.shape[1], row), (0, 255, 0), 1)
    for col in cols[:20]:  # Show first 20 columns
        cv2.line(overlay, (col, 0), (col, img.shape[0]), (0, 255, 0), 1)
    cv2.imwrite(str(out_dir / 'grid_overlay.png'), overlay)
    
    # Color-coded cells
    cells_img = np.zeros_like(img)
    for row, col, bit in cells:
        if 0 <= row < len(rows) and 0 <= col < len(cols):
            y, x = rows[row], cols[col]
            if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                if bit == '0':
                    cells_img[y, x] = [255, 0, 0]  # Blue
                elif bit == '1':
                    cells_img[y, x] = [0, 255, 0]  # Green
                elif bit == 'overlay':
                    cells_img[y, x] = [255, 0, 255]  # Magenta
                # blank = black
    cv2.imwrite(str(out_dir / 'cells_color.png'), cells_img)


def analyze_results(cells: List[Tuple[int, int, str]]) -> Dict[str, Any]:
    """Analyze extraction results and return statistics."""
    if not cells:
        return {"error": "No cells extracted"}
    
    bits = [bit for _, _, bit in cells]
    zeros = bits.count('0')
    ones = bits.count('1')
    blanks = bits.count('blank')
    overlays = bits.count('overlay')
    
    return {
        "total_cells": len(cells),
        "zeros": zeros,
        "ones": ones,
        "blanks": blanks,
        "overlays": overlays,
        "legible_digits": zeros + ones,
        "overlay_percentage": overlays / len(cells) * 100 if cells else 0
    } 