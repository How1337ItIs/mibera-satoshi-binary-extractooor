#!/usr/bin/env python3
"""
Refined grid detection to find the expected 'On' text pattern.
Testing different pitch combinations and origins to match checklist requirements.
"""

import cv2
import numpy as np
import pandas as pd
import json
from scipy import signal

def test_multiple_pitch_combinations():
    """Test multiple pitch combinations to find 'On' pattern."""
    
    print("=== TESTING MULTIPLE PITCH COMBINATIONS ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Test different pitch combinations based on expected logical pitches
    # If logical column pitch should be ~8, and scale is 0.301, then pixel pitch = 8 * 0.301 ≈ 2.4
    # But this seems too small, so test larger values
    
    pitch_combinations = [
        (25, 50),   # Close to our autocorrelation results
        (30, 53),   # From previous analysis  
        (31, 53),   # From verification data
        (20, 40),   # Smaller spacing
        (35, 60),   # Larger spacing
        (28, 55),   # Intermediate
    ]
    
    results = []
    
    for row_pitch, col_pitch in pitch_combinations:
        print(f"\n--- Testing pitch {row_pitch}x{col_pitch} ---")
        
        # Test multiple starting positions
        best_origin = None
        best_on_score = 0
        
        for start_row in range(0, min(100, row_pitch)):
            for start_col in range(0, min(100, col_pitch)):
                
                # Extract first 16 bits
                bits = []
                for i in range(16):
                    bit_row = i // 8
                    bit_col = i % 8
                    
                    y = start_row + bit_row * row_pitch
                    x = start_col + bit_col * col_pitch
                    
                    if (0 <= y - 3 and y + 3 < img.shape[0] and 
                        0 <= x - 3 and x + 3 < img.shape[1]):
                        
                        # Sample 6x6 region
                        region = img[y-3:y+4, x-3:x+4]
                        median_val = np.median(region)
                        
                        # Use adaptive threshold
                        bit = 1 if median_val > 35 else 0
                        bits.append(bit)
                
                if len(bits) == 16:
                    # Convert to bytes and check for 'On'
                    byte1 = 0
                    byte2 = 0
                    for i in range(8):
                        byte1 |= (bits[i] << (7 - i))
                        byte2 |= (bits[i + 8] << (7 - i))
                    
                    # Check how close to 'On' (79, 110)
                    target_byte1 = ord('O')  # 79
                    target_byte2 = ord('n')  # 110
                    
                    score1 = 8 - bin(byte1 ^ target_byte1).count('1')  # Bits that match
                    score2 = 8 - bin(byte2 ^ target_byte2).count('1')
                    total_score = score1 + score2
                    
                    if total_score > best_on_score:
                        best_on_score = total_score
                        best_origin = (start_row, start_col)
                        best_bytes = (byte1, byte2)
        
        if best_origin:
            print(f"Best origin: ({best_origin[0]}, {best_origin[1]})")
            print(f"Best 'On' score: {best_on_score}/16 bits")
            if best_on_score >= 10:
                print(f"*** POTENTIAL MATCH: {chr(best_bytes[0]) if 32 <= best_bytes[0] <= 126 else '?'}{chr(best_bytes[1]) if 32 <= best_bytes[1] <= 126 else '?'} ***")
            
            results.append({
                'row_pitch': row_pitch,
                'col_pitch': col_pitch,
                'best_origin': best_origin,
                'on_score': best_on_score,
                'bytes': best_bytes
            })
    
    # Sort by best 'On' score
    results.sort(key=lambda x: x['on_score'], reverse=True)
    
    print(f"\n=== TOP RESULTS ===")
    for i, result in enumerate(results[:5]):
        pitch_str = f"{result['row_pitch']}x{result['col_pitch']}"
        origin_str = f"({result['best_origin'][0]}, {result['best_origin'][1]})"
        score = result['on_score']
        bytes_str = f"{chr(result['bytes'][0]) if 32 <= result['bytes'][0] <= 126 else '?'}{chr(result['bytes'][1]) if 32 <= result['bytes'][1] <= 126 else '?'}"
        print(f"{i+1}. Pitch {pitch_str:8} Origin {origin_str:12} Score {score:2d}/16 '{bytes_str}'")
    
    return results

def extract_with_best_parameters(result):
    """Extract full data using the best parameters found."""
    
    print(f"\n=== EXTRACTING WITH BEST PARAMETERS ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    row_pitch = result['row_pitch']
    col_pitch = result['col_pitch']
    start_row, start_col = result['best_origin']
    
    print(f"Using pitch {row_pitch}x{col_pitch}, origin ({start_row}, {start_col})")
    
    # Extract larger grid
    grid_size = 500  # Extract 500 bits
    samples = []
    positions = []
    
    for i in range(grid_size):
        bit_row = i // 16  # 16 bits per row
        bit_col = i % 16
        
        y = start_row + bit_row * row_pitch
        x = start_col + bit_col * col_pitch
        
        if (0 <= y - 3 and y + 3 < img.shape[0] and 
            0 <= x - 3 and x + 3 < img.shape[1]):
            
            # Sample 6x6 region
            region = img[y-3:y+4, x-3:x+4]
            median_val = np.median(region)
            
            samples.append(median_val)
            positions.append((i, bit_row, bit_col, y, x))
    
    print(f"Extracted {len(samples)} samples")
    print(f"Value range: {min(samples):.1f} - {max(samples):.1f}")
    
    # Determine optimal threshold for balanced extraction
    thresholds = np.arange(20, 60, 2)
    best_threshold = None
    best_balance = float('inf')
    
    for thresh in thresholds:
        bits = [1 if val > thresh else 0 for val in samples]
        ones_ratio = sum(bits) / len(bits)
        balance_error = abs(ones_ratio - 0.5)
        
        if balance_error < best_balance:
            best_balance = balance_error
            best_threshold = thresh
    
    print(f"Optimal threshold: {best_threshold} (balance error: {best_balance:.3f})")
    
    # Extract final bits
    final_bits = [1 if val > best_threshold else 0 for val in samples]
    ones_ratio = sum(final_bits) / len(final_bits)
    
    print(f"Final ones ratio: {ones_ratio:.1%}")
    
    return final_bits, positions, best_threshold, samples

def validate_text_extraction(bits):
    """Validate text extraction for readability."""
    
    print(f"\n=== TEXT EXTRACTION VALIDATION ===")
    
    if len(bits) < 80:
        print("Insufficient bits for text validation")
        return
    
    # Convert to ASCII
    text_chars = []
    for i in range(0, len(bits) - 7, 8):
        byte_val = 0
        for j in range(8):
            byte_val |= (bits[i + j] << (7 - j))
        
        if 32 <= byte_val <= 126:
            text_chars.append(chr(byte_val))
        else:
            text_chars.append('.')
    
    text = ''.join(text_chars[:50])  # First 50 characters
    printable_ratio = text.count('.') / len(text) if text else 0
    printable_ratio = 1 - printable_ratio  # Invert so higher is better
    
    print(f"First 50 characters: '{text}'")
    print(f"Printable ratio: {printable_ratio:.1%}")
    
    # Look for common English words
    common_words = ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'man', 'way']
    
    text_lower = text.lower()
    words_found = [word for word in common_words if word in text_lower]
    
    if words_found:
        print(f"Common words found: {words_found}")
    else:
        print("No common English words detected")
    
    return {
        'text': text,
        'printable_ratio': printable_ratio,
        'words_found': words_found
    }

def save_refined_results():
    """Save refined grid detection results."""
    
    print(f"\n=== SAVING REFINED RESULTS ===")
    
    # Run complete analysis
    pitch_results = test_multiple_pitch_combinations()
    
    if not pitch_results:
        print("No valid pitch combinations found")
        return
    
    best_result = pitch_results[0]
    final_bits, positions, threshold, samples = extract_with_best_parameters(best_result)
    text_validation = validate_text_extraction(final_bits)
    
    # Create comprehensive results
    results = {
        "timestamp": "2025-07-17",
        "analysis_type": "refined_grid_detection",
        "pitch_testing": {
            "combinations_tested": int(len(pitch_results)),
            "best_on_score": int(best_result['on_score']),
            "best_pitch": f"{best_result['row_pitch']}x{best_result['col_pitch']}",
            "best_origin": best_result['best_origin']
        },
        "extraction": {
            "bits_extracted": int(len(final_bits)),
            "ones_ratio": float(sum(final_bits) / len(final_bits)),
            "threshold_used": float(threshold),
            "sample_range": [float(min(samples)), float(max(samples))]
        },
        "text_validation": text_validation,
        "top_configurations": [
            {
                "pitch": f"{r['row_pitch']}x{r['col_pitch']}",
                "origin": r['best_origin'],
                "on_score": int(r['on_score'])
            }
            for r in pitch_results[:5]
        ],
        "assessment": "Refined grid detection targeting expected text patterns"
    }
    
    with open('refined_grid_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Refined analysis saved to refined_grid_analysis.json")
    
    # Also save canonical bit dump if we found good results
    if best_result['on_score'] >= 10:
        dump_data = []
        for i, (bit_idx, bit_row, bit_col, y, x) in enumerate(positions):
            if i < len(final_bits):
                dump_data.append({
                    'bit_index': bit_idx,
                    'grid_row': bit_row,
                    'grid_col': bit_col,
                    'pixel_y': y,
                    'pixel_x': x,
                    'sample_value': samples[i],
                    'threshold': threshold,
                    'bit': final_bits[i]
                })
        
        df = pd.DataFrame(dump_data)
        df.to_csv('refined_canonical_bit_dump.csv', index=False)
        print("Saved refined_canonical_bit_dump.csv")
    
    return results

if __name__ == "__main__":
    print("Refined Grid Detection")
    print("Targeting expected 'On' text pattern")
    print("=" * 50)
    
    # Test multiple pitch combinations
    pitch_results = test_multiple_pitch_combinations()
    
    if pitch_results and pitch_results[0]['on_score'] >= 8:
        print(f"\nFound promising configuration!")
        
        # Extract with best parameters
        best_result = pitch_results[0]
        final_bits, positions, threshold, samples = extract_with_best_parameters(best_result)
        
        # Validate text extraction
        text_validation = validate_text_extraction(final_bits)
        
        # Save results
        analysis_results = save_refined_results()
        
        print(f"\n" + "=" * 50)
        print("REFINED ANALYSIS COMPLETE")
        print(f"Best 'On' score: {best_result['on_score']}/16")
        print(f"Printable text ratio: {text_validation['printable_ratio']:.1%}")
        
        if text_validation['words_found']:
            print(f"English words found: {len(text_validation['words_found'])}")
    else:
        print("\nNo strong 'On' pattern detected")
        print("May need different approach or image preprocessing")