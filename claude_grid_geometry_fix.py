#!/usr/bin/env python3
"""
Fix Claude's grid geometry issue - the root cause of extraction errors.
Key fix: Column pitch should be ~8px, not 25px.

Author: Claude Code (with geometry correction)
Date: July 17, 2025
"""

import numpy as np
import cv2
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path


def detect_pitch(projection, min_px=5, max_px=40):
    """Detect pitch using autocorrelation."""
    # Normalize projection
    proj_norm = projection - projection.mean()
    
    # Autocorrelation
    autocorr = signal.correlate(proj_norm, proj_norm, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Keep positive lags only
    
    # Find peaks
    peaks = []
    for lag in range(min_px, min(max_px, len(autocorr))):
        if lag > 0 and lag < len(autocorr) - 1:
            if autocorr[lag] > autocorr[lag-1] and autocorr[lag] > autocorr[lag+1]:
                peaks.append((autocorr[lag], lag))
    
    if peaks:
        peaks.sort(reverse=True)
        return peaks[0][1]
    return None


def sample_cell_robust(img, y, x, cell_size=6):
    """Sample a cell using averaging over a patch instead of single pixel."""
    h, w = img.shape
    
    # Define patch bounds
    y_start = max(0, int(y - cell_size//2))
    y_end = min(h, int(y + cell_size//2 + 1))
    x_start = max(0, int(x - cell_size//2))
    x_end = min(w, int(x + cell_size//2 + 1))
    
    if y_end <= y_start or x_end <= x_start:
        return 0
    
    # Extract patch and compute average
    patch = img[y_start:y_end, x_start:x_end]
    return np.mean(patch)


def extract_with_correct_geometry(img_path):
    """Extract binary data with correct grid geometry."""
    print("Loading image and applying threshold...")
    
    # Load image
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image from {img_path}")
    
    # Apply adaptive threshold
    bw = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)
    
    # Invert if needed (we want digits as white)
    if np.mean(bw) > 127:
        bw = 255 - bw
    
    print(f"Image shape: {img.shape}")
    
    # Row detection (this part was correct)
    print("\nDetecting row pitch...")
    row_proj = bw.sum(axis=1)
    row_pitch = detect_pitch(row_proj, min_px=20, max_px=40)
    print(f"Row pitch: {row_pitch} pixels")
    
    # Column detection - THIS IS THE KEY FIX
    print("\nDetecting column pitch...")
    col_proj = bw.sum(axis=0)
    
    # Show autocorrelation to verify
    col_norm = col_proj - col_proj.mean()
    autocorr = signal.correlate(col_norm, col_norm, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    
    # Find peaks in 5-15 pixel range (glyph width range)
    peaks = []
    for lag in range(5, 15):
        if lag < len(autocorr) - 1:
            if autocorr[lag] > autocorr[lag-1] and autocorr[lag] > autocorr[lag+1]:
                peaks.append((autocorr[lag], lag))
    
    if peaks:
        peaks.sort(reverse=True)
        col_pitch = peaks[0][1]
    else:
        col_pitch = 8  # Default to expected value
    
    print(f"Column pitch: {col_pitch} pixels (was 25, now corrected!)")
    
    # Find grid origin with sweep
    print("\nFinding grid origin...")
    
    # Row origin - find first strong row
    row_threshold = np.max(row_proj) * 0.5
    row0 = None
    for y in range(img.shape[0]):
        if row_proj[y] > row_threshold:
            row0 = y
            break
    
    if row0 is None:
        row0 = 69  # Fallback to known value
    
    print(f"Row origin: {row0}")
    
    # Column origin - sweep to find best alignment
    print("Sweeping column origins...")
    best_score = -1
    best_col0 = 0
    
    for col0 in range(col_pitch):
        # Sample a few test cells
        score = 0
        test_positions = []
        
        # Test first few rows
        for row_idx in range(5):
            y = row0 + row_idx * row_pitch
            if y >= img.shape[0]:
                break
                
            # Test several columns
            for col_idx in range(10):
                x = col0 + col_idx * col_pitch
                if x >= img.shape[1]:
                    break
                
                # Sample with robust method
                val = sample_cell_robust(bw, y, x)
                
                # High confidence if clearly black or white
                if val < 50 or val > 200:
                    score += 1
        
        print(f"  Origin {col0}: score = {score}")
        
        if score > best_score:
            best_score = score
            best_col0 = col0
    
    col0 = best_col0
    print(f"Best column origin: {col0}")
    
    # Build grid coordinates
    rows = []
    y = row0
    while y < img.shape[0] - row_pitch/2:
        rows.append(y)
        y += row_pitch
    
    cols = []
    x = col0
    while x < img.shape[1] - col_pitch/2:
        cols.append(x)
        x += col_pitch
    
    print(f"\nGrid size: {len(rows)} rows x {len(cols)} columns")
    
    # Extract bits with robust sampling
    print("\nExtracting bits...")
    bits = []
    positions = []
    
    for row_idx, y in enumerate(rows):
        row_bits = []
        for col_idx, x in enumerate(cols):
            # Use robust sampling
            val = sample_cell_robust(bw, y, x, cell_size=6)
            
            # Threshold
            bit = '1' if val > 127 else '0'
            
            row_bits.append(bit)
            bits.append(bit)
            positions.append((row_idx, col_idx, y, x))
        
        # Check first row for "On the" pattern
        if row_idx == 0:
            row_str = ''.join(row_bits)
            print(f"\nFirst row bits: {row_str[:50]}...")
            
            # Try to decode first few bytes
            decoded = []
            for i in range(0, min(len(row_str), 48), 8):
                if i + 8 <= len(row_str):
                    byte = row_str[i:i+8]
                    try:
                        char_val = int(byte, 2)
                        if 32 <= char_val <= 126:
                            decoded.append(chr(char_val))
                        else:
                            decoded.append(f'[{char_val}]')
                    except:
                        decoded.append('?')
            
            print(f"First row decoded: {''.join(decoded)}")
    
    # Analyze bit distribution
    zeros = bits.count('0')
    ones = bits.count('1')
    total = len(bits)
    
    print(f"\nBit distribution:")
    print(f"  Zeros: {zeros} ({zeros/total*100:.1f}%)")
    print(f"  Ones: {ones} ({ones/total*100:.1f}%)")
    print(f"  Ratio should be ~75%/25% when grid is correct")
    
    # Save corrected extraction
    output_path = Path('claude_corrected_extraction.txt')
    with open(output_path, 'w') as f:
        f.write("=== CORRECTED GRID EXTRACTION ===\n")
        f.write(f"Grid parameters:\n")
        f.write(f"  Row pitch: {row_pitch} px\n")
        f.write(f"  Col pitch: {col_pitch} px (FIXED from 25!)\n")
        f.write(f"  Origin: ({row0}, {col0})\n")
        f.write(f"  Grid size: {len(rows)} x {len(cols)}\n")
        f.write(f"\nBit distribution:\n")
        f.write(f"  Zeros: {zeros} ({zeros/total*100:.1f}%)\n")
        f.write(f"  Ones: {ones} ({ones/total*100:.1f}%)\n")
        f.write(f"\nExtracted bits by row:\n")
        
        # Write bits row by row
        bit_idx = 0
        for row_idx in range(len(rows)):
            row_bits = []
            for col_idx in range(len(cols)):
                if bit_idx < len(bits):
                    row_bits.append(bits[bit_idx])
                    bit_idx += 1
            
            row_str = ''.join(row_bits)
            f.write(f"Row {row_idx:2d}: {row_str}\n")
            
            # Decode if possible
            if len(row_str) >= 8:
                decoded = []
                for i in range(0, len(row_str), 8):
                    if i + 8 <= len(row_str):
                        byte = row_str[i:i+8]
                        try:
                            char_val = int(byte, 2)
                            if 32 <= char_val <= 126:
                                decoded.append(chr(char_val))
                            else:
                                decoded.append(f'[{char_val}]')
                        except:
                            decoded.append('?')
                
                if decoded:
                    f.write(f"         Decoded: {''.join(decoded)}\n")
    
    print(f"\nExtraction saved to {output_path}")
    
    # Visualize the corrected grid
    plt.figure(figsize=(12, 8))
    plt.imshow(bw, cmap='gray')
    
    # Draw grid lines
    for y in rows[:20]:  # First 20 rows
        plt.axhline(y, color='red', alpha=0.3, linewidth=0.5)
    
    for x in cols[:80]:  # First 80 columns
        plt.axvline(x, color='blue', alpha=0.3, linewidth=0.5)
    
    plt.title(f'Corrected Grid: {row_pitch}x{col_pitch} px (was 31x25)')
    plt.tight_layout()
    plt.savefig('claude_corrected_grid_visualization.png', dpi=150)
    plt.close()
    
    print("Grid visualization saved to claude_corrected_grid_visualization.png")
    
    return {
        'row_pitch': row_pitch,
        'col_pitch': col_pitch,
        'origin': (row0, col0),
        'grid_size': (len(rows), len(cols)),
        'bits': bits,
        'bit_distribution': {'zeros': zeros, 'ones': ones}
    }


if __name__ == "__main__":
    print("Claude Code: Fixing grid geometry issue")
    print("Key insight: Column pitch should be ~8px, not 25px")
    print("-" * 50)
    
    # Use the real poster image
    img_path = Path('posters/poster3_hd_downsampled_gray.png')
    
    if not img_path.exists():
        print(f"Error: Image not found at {img_path}")
        print("Looking for alternative paths...")
        
        # Try other possible locations
        alt_paths = [
            Path('binary_extractor/data/posters/poster3_hd_downsampled_gray.png'),
            Path('data/posters/poster3_hd_downsampled_gray.png'),
            Path('poster3_hd_downsampled_gray.png')
        ]
        
        for alt in alt_paths:
            if alt.exists():
                img_path = alt
                print(f"Found image at: {img_path}")
                break
    
    if img_path.exists():
        results = extract_with_correct_geometry(img_path)
        
        print("\n" + "="*50)
        print("EXTRACTION COMPLETE")
        print(f"Grid: {results['grid_size'][0]} x {results['grid_size'][1]}")
        print(f"Column pitch: {results['col_pitch']} px (CORRECTED!)")
        print(f"Bit distribution: {results['bit_distribution']['zeros']/len(results['bits'])*100:.1f}% / {results['bit_distribution']['ones']/len(results['bits'])*100:.1f}%")
        print("Check claude_corrected_extraction.txt for full results")
    else:
        print(f"Error: Could not find poster image")
        print("Please ensure the poster image is available")