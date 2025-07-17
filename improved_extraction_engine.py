#!/usr/bin/env python3
"""
Improved extraction engine based on critical analysis findings.
Key insights:
- Row pitch: 31px (confirmed)
- Column pitch: ~18px (not 8px or 25px)
- Use robust sampling (6x6 patches)
- Origin sweep for best alignment

Author: Claude Code
Date: July 17, 2025
"""

import numpy as np
import cv2
from pathlib import Path
import json
from scipy import signal
from scipy.ndimage import median_filter


class ImprovedExtractor:
    def __init__(self, img_path):
        self.img_path = img_path
        self.img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if self.img is None:
            raise ValueError(f"Could not load image from {img_path}")
        
        # Apply threshold
        self.bw = cv2.adaptiveThreshold(
            self.img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Invert if needed
        if np.mean(self.bw) > 127:
            self.bw = 255 - self.bw
        
        print(f"Loaded image: {self.img.shape}")
        
        # Detection results
        self.row_pitch = None
        self.col_pitch = None
        self.row0 = None
        self.col0 = None
        self.grid = None
    
    def detect_row_pitch(self):
        """Detect row pitch using autocorrelation."""
        row_proj = self.bw.sum(axis=1)
        
        # Normalize
        row_norm = row_proj - row_proj.mean()
        
        # Autocorrelation
        autocorr = signal.correlate(row_norm, row_norm, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find peak in expected range
        best_lag = None
        best_val = -np.inf
        
        for lag in range(28, 35):  # Expected around 31
            if lag < len(autocorr):
                if autocorr[lag] > best_val:
                    best_val = autocorr[lag]
                    best_lag = lag
        
        self.row_pitch = best_lag or 31
        print(f"Detected row pitch: {self.row_pitch}px")
        
        return self.row_pitch
    
    def detect_column_structure(self):
        """Detect column structure - handle complex spacing patterns."""
        col_proj = self.bw.sum(axis=0)
        
        # Find peaks (digit columns)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(col_proj, height=np.max(col_proj)*0.3, distance=5)
        
        if len(peaks) > 1:
            # Analyze spacing patterns
            spacings = np.diff(peaks)
            
            # Look for common spacings
            unique_spacings, counts = np.unique(spacings, return_counts=True)
            
            # Filter out very large gaps
            reasonable_spacings = [(s, c) for s, c in zip(unique_spacings, counts) 
                                 if 10 <= s <= 30]
            
            if reasonable_spacings:
                # Use weighted average based on frequency
                total_count = sum(c for _, c in reasonable_spacings)
                weighted_sum = sum(s * c for s, c in reasonable_spacings)
                self.col_pitch = int(round(weighted_sum / total_count))
            else:
                self.col_pitch = 18  # Fallback to empirical value
        else:
            self.col_pitch = 18
        
        print(f"Detected column pitch: {self.col_pitch}px")
        
        return self.col_pitch
    
    def find_grid_origin(self):
        """Find optimal grid origin through systematic search."""
        # Row origin - find first strong row
        row_proj = self.bw.sum(axis=1)
        row_threshold = np.max(row_proj) * 0.4
        
        for y in range(50, 100):  # Search in expected range
            if row_proj[y] > row_threshold:
                # Check if this aligns with a row pattern
                test_rows = []
                test_y = y
                while test_y < min(y + 5 * self.row_pitch, self.img.shape[0]):
                    if row_proj[test_y] > row_threshold * 0.8:
                        test_rows.append(test_y)
                    test_y += self.row_pitch
                
                if len(test_rows) >= 3:
                    self.row0 = y
                    break
        
        if self.row0 is None:
            self.row0 = 69  # Fallback
        
        print(f"Row origin: {self.row0}")
        
        # Column origin - sweep to find best alignment
        best_score = -1
        best_col0 = 0
        
        # Test a finer range around expected values
        for col0 in range(30, 45):
            score = self._score_column_alignment(col0)
            
            if score > best_score:
                best_score = score
                best_col0 = col0
        
        self.col0 = best_col0
        print(f"Column origin: {self.col0} (score: {best_score:.2f})")
        
        return self.row0, self.col0
    
    def _score_column_alignment(self, col0):
        """Score a column alignment by checking digit presence."""
        score = 0
        
        # Sample first few rows and columns
        for row_idx in range(5):
            y = self.row0 + row_idx * self.row_pitch
            if y >= self.img.shape[0] - 3:
                break
            
            for col_idx in range(20):
                x = col0 + col_idx * self.col_pitch
                if x >= self.img.shape[1] - 3:
                    break
                
                # Check 5x5 region
                region = self.bw[y-2:y+3, x-2:x+3]
                if region.size > 0:
                    # High score for clear digit presence
                    mean_val = np.mean(region)
                    if mean_val > 200:  # Strong white (digit)
                        score += 2
                    elif mean_val < 50:  # Strong black (background)
                        score += 1
                    # Penalty for ambiguous regions
                    else:
                        score -= 0.5
        
        return score
    
    def extract_robust(self, y, x, patch_size=6):
        """Extract bit value using robust patch sampling."""
        # Define patch bounds
        half = patch_size // 2
        y_start = max(0, int(y - half))
        y_end = min(self.bw.shape[0], int(y + half))
        x_start = max(0, int(x - half))
        x_end = min(self.bw.shape[1], int(x + half))
        
        if y_end <= y_start or x_end <= x_start:
            return '0'
        
        # Extract patch
        patch = self.bw[y_start:y_end, x_start:x_end]
        
        # Use median for robustness
        median_val = np.median(patch)
        
        return '1' if median_val > 127 else '0'
    
    def extract_grid(self):
        """Extract the full grid of bits."""
        if self.row_pitch is None:
            self.detect_row_pitch()
        if self.col_pitch is None:
            self.detect_column_structure()
        if self.row0 is None or self.col0 is None:
            self.find_grid_origin()
        
        # Build grid
        rows = []
        cols = []
        
        y = self.row0
        while y < self.img.shape[0] - self.row_pitch/2:
            rows.append(y)
            y += self.row_pitch
        
        x = self.col0
        while x < self.img.shape[1] - self.col_pitch/2:
            cols.append(x)
            x += self.col_pitch
        
        print(f"\nExtracting {len(rows)} x {len(cols)} grid...")
        
        # Extract bits
        grid = []
        all_bits = []
        
        for row_idx, y in enumerate(rows):
            row_bits = []
            for col_idx, x in enumerate(cols):
                bit = self.extract_robust(y, x)
                row_bits.append(bit)
                all_bits.append(bit)
            
            grid.append(row_bits)
            
            # Decode first row to check
            if row_idx == 0:
                row_str = ''.join(row_bits)
                decoded = self._decode_bits(row_str[:48])
                print(f"First row: {row_str[:50]}...")
                print(f"Decoded: {decoded}")
        
        self.grid = grid
        
        # Statistics
        zeros = all_bits.count('0')
        ones = all_bits.count('1')
        print(f"\nExtraction statistics:")
        print(f"  Total bits: {len(all_bits)}")
        print(f"  Zeros: {zeros} ({zeros/len(all_bits)*100:.1f}%)")
        print(f"  Ones: {ones} ({ones/len(all_bits)*100:.1f}%)")
        
        return grid
    
    def _decode_bits(self, bits):
        """Decode a bit string to ASCII."""
        decoded = []
        for i in range(0, len(bits), 8):
            if i + 8 <= len(bits):
                byte = bits[i:i+8]
                try:
                    val = int(byte, 2)
                    if 32 <= val <= 126:
                        decoded.append(chr(val))
                    else:
                        decoded.append(f'[{val}]')
                except:
                    decoded.append('?')
        return ''.join(decoded)
    
    def search_for_text(self, target="On the "):
        """Search for target text in the extracted grid."""
        if self.grid is None:
            self.extract_grid()
        
        target_bits = ''.join(format(ord(c), '08b') for c in target)
        best_matches = []
        
        for row_idx, row in enumerate(self.grid):
            row_str = ''.join(row)
            
            # Try different starting positions
            for start in range(min(len(row_str) - len(target_bits) + 1, 20)):
                test_bits = row_str[start:start + len(target_bits)]
                
                if len(test_bits) == len(target_bits):
                    matches = sum(1 for i in range(len(target_bits)) 
                                if test_bits[i] == target_bits[i])
                    score = matches / len(target_bits)
                    
                    if score > 0.5:
                        decoded = self._decode_bits(test_bits)
                        best_matches.append({
                            'row': row_idx,
                            'col': start,
                            'score': score,
                            'decoded': decoded,
                            'bits': test_bits[:24]
                        })
        
        # Sort by score
        best_matches.sort(key=lambda x: x['score'], reverse=True)
        
        if best_matches:
            print(f"\nBest matches for '{target}':")
            for i, match in enumerate(best_matches[:5]):
                print(f"  {i+1}. Row {match['row']}, Col {match['col']}: "
                      f"score={match['score']:.1%}, decoded='{match['decoded']}'")
        
        return best_matches
    
    def save_results(self, output_path="improved_extraction_results.json"):
        """Save extraction results."""
        if self.grid is None:
            self.extract_grid()
        
        results = {
            'grid_params': {
                'row_pitch': self.row_pitch,
                'col_pitch': self.col_pitch,
                'row0': self.row0,
                'col0': self.col0,
                'rows': len(self.grid),
                'cols': len(self.grid[0]) if self.grid else 0
            },
            'bits': [''.join(row) for row in self.grid],
            'decoded_rows': []
        }
        
        # Decode each row
        for row_idx, row in enumerate(self.grid):
            row_str = ''.join(row)
            decoded = self._decode_bits(row_str)
            results['decoded_rows'].append({
                'row': row_idx,
                'bits': row_str,
                'decoded': decoded
            })
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_path}")
        
        return results


def compare_with_baseline():
    """Compare with Claude's 77.1% baseline."""
    print("\n" + "="*50)
    print("COMPARISON WITH 77.1% BASELINE")
    print("="*50)
    
    # Load Claude's reference data if available
    ref_file = Path("CLAUDE_COMPREHENSIVE_BREAKTHROUGH_RESULTS.txt")
    if ref_file.exists():
        print(f"\nFound reference file: {ref_file}")
        with open(ref_file, 'r') as f:
            content = f.read()
            if "77.1%" in content:
                print("Confirmed: Reference shows 77.1% accuracy")
    
    # The key metrics to match:
    # - Grid should be ~50x148 cells
    # - First row should decode to "On the "
    # - Zero/one ratio should be ~75%/25%
    
    print("\nTarget metrics:")
    print("  - Grid size: ~50x148")
    print("  - First row: 'On the '")
    print("  - Bit distribution: ~75% zeros, ~25% ones")


if __name__ == "__main__":
    print("Improved Extraction Engine")
    print("Based on critical analysis of actual poster structure")
    print("="*50)
    
    # Find poster image
    img_paths = [
        Path('posters/poster3_hd_downsampled_gray.png'),
        Path('binary_extractor/data/posters/poster3_hd_downsampled_gray.png'),
        Path('data/posters/poster3_hd_downsampled_gray.png')
    ]
    
    img_path = None
    for path in img_paths:
        if path.exists():
            img_path = path
            break
    
    if img_path:
        print(f"Using image: {img_path}")
        
        # Create extractor
        extractor = ImprovedExtractor(img_path)
        
        # Extract grid
        extractor.extract_grid()
        
        # Search for "On the "
        extractor.search_for_text("On the ")
        
        # Save results
        extractor.save_results()
        
        # Compare with baseline
        compare_with_baseline()
    else:
        print("Error: Could not find poster image")
        print("Looking for poster3_hd_downsampled_gray.png")