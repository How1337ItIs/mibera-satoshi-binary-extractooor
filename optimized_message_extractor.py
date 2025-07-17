#!/usr/bin/env python3
"""
Optimized message extractor using resolved pitch methodology.
Goal: Extract the complete hidden Satoshi message with >77.1% accuracy.
"""

import cv2
import numpy as np
from scipy import signal
import json
from pathlib import Path


class OptimizedMessageExtractor:
    def __init__(self, img_path):
        self.img_path = Path(img_path)
        self.img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if self.img is None:
            raise ValueError(f"Could not load {img_path}")
        
        print(f"Loaded image: {self.img.shape}")
        
        # Detection results
        self.row_pitch = None
        self.col_pitch = None
        self.best_origin = None
        self.grid = None
        
    def detect_pitches(self):
        """Detect row and column pitches using autocorrelation."""
        
        # Use bright pixel threshold for digit detection
        mask = self.img > 200
        
        # Column pitch
        col_proj = mask.sum(0).astype(float)
        col_proj -= col_proj.mean()
        col_corr = np.correlate(col_proj, col_proj, mode='full')[len(col_proj):]
        self.col_pitch = np.argmax(col_corr[5:]) + 5
        
        # Row pitch  
        row_proj = mask.sum(1).astype(float)
        row_proj -= row_proj.mean()
        row_corr = np.correlate(row_proj, row_proj, mode='full')[len(row_proj):]
        self.row_pitch = np.argmax(row_corr[5:]) + 5
        
        print(f"Detected pitches: row={self.row_pitch}px, col={self.col_pitch}px")
        
        return self.row_pitch, self.col_pitch
    
    def fine_tune_origin_for_on(self):
        """Fine-tune origin specifically to decode 'On' in first row."""
        
        if self.row_pitch is None or self.col_pitch is None:
            self.detect_pitches()
        
        target_bits = "0100111101101110"  # "On" in binary
        best_score = 0
        best_origin = None
        best_decoded = ""
        
        print(f"\nFine-tuning origin to decode 'On' (target: {target_bits})")
        
        # Test different sampling parameters too
        patch_sizes = [4, 6, 8]
        thresholds = [100, 127, 150, 180]
        
        for patch_size in patch_sizes:
            for threshold in thresholds:
                for row_offset in range(-5, 6):
                    for col_offset in range(-10, 11):
                        # Test this configuration
                        row0 = max(50, min(100, 70 + row_offset))
                        col0 = max(20, min(80, 40 + col_offset))
                        
                        # Extract first 16 bits
                        bits = self._extract_bits_region(
                            row0, col0, 1, 16, patch_size, threshold
                        )
                        
                        if len(bits) >= 16:
                            test_bits = ''.join(bits[:16])
                            
                            # Score against target
                            matches = sum(1 for i in range(16) 
                                        if test_bits[i] == target_bits[i])
                            score = matches / 16
                            
                            if score > best_score:
                                best_score = score
                                best_origin = (row0, col0, patch_size, threshold)
                                best_decoded = self._decode_bits(test_bits)
                                
                                if score > 0.8:  # Good enough
                                    print(f"  Found good match: {score:.1%} "
                                          f"at ({row0}, {col0}) "
                                          f"patch={patch_size} thresh={threshold}")
                                    print(f"  Bits: {test_bits}")
                                    print(f"  Decoded: '{best_decoded}'")
        
        if best_origin:
            row0, col0, patch_size, threshold = best_origin
            self.best_origin = (row0, col0)
            self.best_patch_size = patch_size
            self.best_threshold = threshold
            
            print(f"\nBest origin: ({row0}, {col0}) "
                  f"patch={patch_size} thresh={threshold}")
            print(f"Score: {best_score:.1%}, Decoded: '{best_decoded}'")
            
            return best_origin
        else:
            print("Could not find good origin for 'On' decoding")
            self.best_origin = (70, 40)
            self.best_patch_size = 6
            self.best_threshold = 127
            return None
    
    def _extract_bits_region(self, row0, col0, num_rows, num_cols, patch_size, threshold):
        """Extract bits from a specific region with given parameters."""
        
        bits = []
        
        for r in range(num_rows):
            y = row0 + r * self.row_pitch
            if y >= self.img.shape[0] - patch_size//2:
                break
                
            for c in range(num_cols):
                x = col0 + c * self.col_pitch
                if x >= self.img.shape[1] - patch_size//2:
                    break
                
                # Sample patch
                half = patch_size // 2
                y_start = max(0, y - half)
                y_end = min(self.img.shape[0], y + half + 1)
                x_start = max(0, x - half)  
                x_end = min(self.img.shape[1], x + half + 1)
                
                patch = self.img[y_start:y_end, x_start:x_end]
                
                if patch.size > 0:
                    val = np.median(patch)
                    bit = '1' if val > threshold else '0'
                    bits.append(bit)
        
        return bits
    
    def extract_full_message(self):
        """Extract the complete hidden message."""
        
        if self.best_origin is None:
            self.fine_tune_origin_for_on()
        
        row0, col0 = self.best_origin
        
        print(f"\nExtracting full message...")
        print(f"Origin: ({row0}, {col0})")
        print(f"Pitches: {self.row_pitch}x{self.col_pitch}")
        print(f"Sampling: {self.best_patch_size}x{self.best_patch_size} patch, threshold={self.best_threshold}")
        
        # Calculate grid size
        max_rows = (self.img.shape[0] - row0) // self.row_pitch
        max_cols = (self.img.shape[1] - col0) // self.col_pitch
        
        print(f"Maximum grid: {max_rows} x {max_cols}")
        
        # Extract full grid
        self.grid = []
        all_bits = []
        
        for r in range(max_rows):
            y = row0 + r * self.row_pitch
            if y >= self.img.shape[0] - self.best_patch_size//2:
                break
            
            row_bits = []
            for c in range(max_cols):
                x = col0 + c * self.col_pitch
                if x >= self.img.shape[1] - self.best_patch_size//2:
                    break
                
                # Sample patch
                half = self.best_patch_size // 2
                y_start = max(0, y - half)
                y_end = min(self.img.shape[0], y + half + 1)
                x_start = max(0, x - half)
                x_end = min(self.img.shape[1], x + half + 1)
                
                patch = self.img[y_start:y_end, x_start:x_end]
                
                if patch.size > 0:
                    val = np.median(patch)
                    bit = '1' if val > self.best_threshold else '0'
                else:
                    bit = '0'
                
                row_bits.append(bit)
                all_bits.append(bit)
            
            self.grid.append(row_bits)
        
        print(f"Extracted {len(self.grid)} rows x {len(self.grid[0]) if self.grid else 0} cols")
        
        # Statistics
        zeros = all_bits.count('0')
        ones = all_bits.count('1')
        print(f"Bit distribution: {zeros} zeros ({zeros/len(all_bits)*100:.1f}%), "
              f"{ones} ones ({ones/len(all_bits)*100:.1f}%)")
        
        # Decode and show first few rows
        print(f"\nFirst few rows:")
        for i, row in enumerate(self.grid[:10]):
            row_str = ''.join(row)
            decoded = self._decode_bits(row_str)
            print(f"Row {i:2d}: {decoded[:50]}")
        
        return self.grid
    
    def _decode_bits(self, bits):
        """Decode bit string to ASCII."""
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
    
    def search_readable_text(self):
        """Search for readable text patterns in the extracted data."""
        
        if self.grid is None:
            self.extract_full_message()
        
        print(f"\nSearching for readable text patterns...")
        
        readable_segments = []
        
        for row_idx, row in enumerate(self.grid):
            row_str = ''.join(row)
            
            # Try different alignments
            for start in range(min(8, len(row_str))):
                for length in [32, 48, 64, 80]:  # Different message lengths
                    if start + length <= len(row_str):
                        segment = row_str[start:start + length]
                        decoded = self._decode_bits(segment)
                        
                        # Count readable characters
                        readable_chars = sum(1 for c in decoded 
                                           if c.isalnum() or c.isspace() or c in '.,!?')
                        
                        if readable_chars >= 3:  # At least 3 readable chars
                            readable_segments.append({
                                'row': row_idx,
                                'start': start,
                                'length': length,
                                'decoded': decoded,
                                'readable_count': readable_chars,
                                'readable_ratio': readable_chars / (length // 8)
                            })
        
        # Sort by readable ratio
        readable_segments.sort(key=lambda x: x['readable_ratio'], reverse=True)
        
        print(f"Found {len(readable_segments)} potentially readable segments:")
        for i, seg in enumerate(readable_segments[:20]):
            print(f"  {i+1:2d}. Row {seg['row']:2d} start {seg['start']:2d}: "
                  f"{seg['readable_ratio']:.1%} readable -> '{seg['decoded'][:40]}'")
        
        return readable_segments
    
    def save_results(self):
        """Save extraction results."""
        
        if self.grid is None:
            return
        
        results = {
            'extraction_method': 'optimized_pitch_aware',
            'parameters': {
                'row_pitch': self.row_pitch,
                'col_pitch': self.col_pitch,
                'origin': self.best_origin,
                'patch_size': self.best_patch_size,
                'threshold': self.best_threshold
            },
            'grid_size': {
                'rows': len(self.grid),
                'cols': len(self.grid[0]) if self.grid else 0
            },
            'extracted_rows': []
        }
        
        # Save each row
        for row_idx, row in enumerate(self.grid):
            row_str = ''.join(row)
            decoded = self._decode_bits(row_str)
            
            results['extracted_rows'].append({
                'row': row_idx,
                'bits': row_str,
                'decoded': decoded
            })
        
        # Save to files
        with open('optimized_extraction_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        with open('optimized_extraction_results.txt', 'w') as f:
            f.write("=== OPTIMIZED EXTRACTION RESULTS ===\n")
            f.write(f"Method: Pitch-aware with 'On' optimization\n")
            f.write(f"Pitches: {self.row_pitch}x{self.col_pitch} pixels\n")
            f.write(f"Origin: {self.best_origin}\n")
            f.write(f"Sampling: {self.best_patch_size}x{self.best_patch_size} patch, threshold={self.best_threshold}\n")
            f.write(f"Grid: {len(self.grid)} x {len(self.grid[0]) if self.grid else 0}\n\n")
            
            for row_idx, row in enumerate(self.grid):
                row_str = ''.join(row)
                decoded = self._decode_bits(row_str)
                f.write(f"Row {row_idx:2d}: {row_str}\n")
                f.write(f"       {decoded}\n\n")
        
        print(f"\nResults saved to optimized_extraction_results.json/.txt")


def main():
    print("Optimized Message Extractor")
    print("Post-pitch-debate resolution - focus on extracting the hidden message")
    print("="*70)
    
    # Use the binary mask
    img_path = 'binary_extractor/output_real_data/bw_mask.png'
    
    try:
        extractor = OptimizedMessageExtractor(img_path)
        
        # Step 1: Fine-tune for "On" decoding
        extractor.fine_tune_origin_for_on()
        
        # Step 2: Extract full message
        extractor.extract_full_message()
        
        # Step 3: Search for readable text
        extractor.search_readable_text()
        
        # Step 4: Save results
        extractor.save_results()
        
        print("\n" + "="*70)
        print("EXTRACTION COMPLETE")
        print("Check optimized_extraction_results.txt for the decoded message")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()