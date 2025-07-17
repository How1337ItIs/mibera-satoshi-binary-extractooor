#!/usr/bin/env python3
"""
Visual grid alignment tool to bridge human pattern recognition with automated extraction.
Uses visual analysis to find the exact grid that matches manual extraction.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy import ndimage

def create_visual_grid_overlay():
    """Create visual grid overlay to help identify correct alignment."""
    
    print("=== VISUAL GRID ALIGNMENT ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Test multiple grid configurations visually
    grid_configs = [
        (25, 50, 13, 18),   # From "OF" match
        (31, 53, 101, 53),  # From verification
        (30, 52, 100, 50),  # Close variant
        (28, 55, 95, 45),   # Alternative
    ]
    
    for i, (row_pitch, col_pitch, start_row, start_col) in enumerate(grid_configs):
        print(f"\nAnalyzing grid {i+1}: {row_pitch}x{col_pitch} at ({start_row},{start_col})")
        
        # Create grid overlay
        overlay = img.copy()
        
        # Draw grid lines for first few rows/cols to visualize alignment
        for row in range(8):  # 8 character rows
            for col in range(16):  # 16 bits per row
                y = start_row + row * row_pitch
                x = start_col + col * col_pitch
                
                if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                    # Draw small crosshair at each bit position
                    cv2.drawMarker(overlay, (x, y), 255, cv2.MARKER_CROSS, 3, 1)
        
        # Save overlay for visual inspection
        filename = f'grid_overlay_{i+1}_{row_pitch}x{col_pitch}.png'
        cv2.imwrite(filename, overlay)
        print(f"Saved {filename}")
        
        # Extract sample text for this grid
        sample_text = extract_sample_text(img, row_pitch, col_pitch, start_row, start_col)
        print(f"Sample: '{sample_text[:20]}...'")

def extract_sample_text(img, row_pitch, col_pitch, start_row, start_col, threshold=60):
    """Extract sample text using given grid parameters."""
    
    sample_chars = 15  # Extract 15 characters
    text = ""
    
    for char_idx in range(sample_chars):
        byte_val = 0
        
        for bit_idx in range(8):
            # Calculate bit position in grid
            total_bit_idx = char_idx * 8 + bit_idx
            bit_row = total_bit_idx // 8
            bit_col = total_bit_idx % 8
            
            y = start_row + bit_row * row_pitch
            x = start_col + bit_col * col_pitch
            
            if 0 <= y - 3 and y + 3 < img.shape[0] and 0 <= x - 3 and x + 3 < img.shape[1]:
                # Sample 6x6 region
                region = img[y-3:y+4, x-3:x+4]
                median_val = np.median(region)
                bit = 1 if median_val > threshold else 0
                
                byte_val |= (bit << (7 - bit_idx))
        
        if 32 <= byte_val <= 126:
            text += chr(byte_val)
        else:
            text += '?'
    
    return text

def analyze_character_positions():
    """Analyze where characters should be positioned based on manual extraction."""
    
    print(f"\n=== CHARACTER POSITION ANALYSIS ===")
    
    # We know the first characters should spell "On the winter..."
    target_chars = "On the winter solstice December 21 "
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # For each character, find regions that might contain its bit pattern
    print("Analyzing character patterns in image...")
    
    char_patterns = {}
    
    for i, char in enumerate(target_chars[:10]):  # First 10 characters
        ascii_val = ord(char)
        binary = format(ascii_val, '08b')
        
        print(f"Char {i:2d}: '{char}' = {ascii_val:3d} = {binary}")
        
        # Store expected pattern
        char_patterns[i] = {
            'char': char,
            'ascii': ascii_val,
            'binary': binary,
            'bits': [int(b) for b in binary]
        }
    
    return char_patterns

def test_threshold_variations():
    """Test different thresholds to find optimal value."""
    
    print(f"\n=== THRESHOLD VARIATION TESTING ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Use most promising grid from previous analysis
    row_pitch, col_pitch = 31, 53
    start_row, start_col = 101, 53
    
    thresholds = range(40, 90, 5)
    results = []
    
    for threshold in thresholds:
        sample_text = extract_sample_text(img, row_pitch, col_pitch, start_row, start_col, threshold)
        
        # Calculate quality metrics
        printable_ratio = sum(1 for c in sample_text if c != '?') / len(sample_text)
        
        # Check for target patterns
        target_score = 0
        if 'o' in sample_text.lower() or 'n' in sample_text.lower():
            target_score += 1
        if 'the' in sample_text.lower():
            target_score += 2
        if 'winter' in sample_text.lower():
            target_score += 3
        
        results.append({
            'threshold': threshold,
            'text': sample_text,
            'printable_ratio': printable_ratio,
            'target_score': target_score
        })
        
        print(f"Threshold {threshold:2d}: {printable_ratio:.1%} printable, score {target_score}, '{sample_text[:15]}...'")
    
    # Find best threshold
    best_result = max(results, key=lambda x: (x['target_score'], x['printable_ratio']))
    print(f"\nBest threshold: {best_result['threshold']}")
    print(f"Best text: '{best_result['text']}'")
    
    return best_result

def fine_tune_grid_position():
    """Fine-tune grid position around best estimates."""
    
    print(f"\n=== FINE-TUNING GRID POSITION ===")
    
    img = cv2.imread('mibera-satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Base configuration
    base_row_pitch, base_col_pitch = 31, 53
    base_start_row, base_start_col = 101, 53
    threshold = 60  # From threshold testing
    
    best_matches = []
    
    # Fine adjustment range
    for row_adj in range(-3, 4):
        for col_adj in range(-3, 4):
            start_row = base_start_row + row_adj
            start_col = base_start_col + col_adj
            
            sample_text = extract_sample_text(img, base_row_pitch, base_col_pitch, 
                                            start_row, start_col, threshold)
            
            # Score based on expected patterns
            score = 0
            text_lower = sample_text.lower()
            
            # Look for parts of "On the winter"
            if text_lower.startswith('o'):
                score += 3
            if 'n' == text_lower[1:2]:
                score += 2
            if ' ' in text_lower[2:4]:
                score += 2
            if 'th' in text_lower:
                score += 3
            if 'he' in text_lower:
                score += 2
            if 'win' in text_lower:
                score += 5
            if 'ter' in text_lower:
                score += 3
            
            printable_ratio = sum(1 for c in sample_text if c != '?') / len(sample_text)
            total_score = score + printable_ratio * 10
            
            best_matches.append({
                'row_adj': row_adj,
                'col_adj': col_adj,
                'start_pos': (start_row, start_col),
                'text': sample_text,
                'score': total_score,
                'printable_ratio': printable_ratio
            })
    
    # Sort by score
    best_matches.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"Top 5 fine-tuned positions:")
    for i, match in enumerate(best_matches[:5]):
        adj_str = f"({match['row_adj']:+2d},{match['col_adj']:+2d})"
        pos_str = f"{match['start_pos']}"
        score = match['score']
        text = match['text'][:20]
        print(f"{i+1}. {adj_str} -> {pos_str}: score {score:.1f}, '{text}...'")
    
    return best_matches[0] if best_matches else None

def save_visual_alignment_results():
    """Save visual alignment analysis results."""
    
    print(f"\n=== SAVING VISUAL ALIGNMENT RESULTS ===")
    
    # Run all analyses
    char_patterns = analyze_character_positions()
    best_threshold = test_threshold_variations()
    best_position = fine_tune_grid_position()
    
    results = {
        "timestamp": "2025-07-17",
        "analysis_type": "visual_grid_alignment",
        "target_patterns": {
            "expected_text": "On the winter solstice December 21",
            "character_count": len(char_patterns),
            "binary_patterns": {str(i): p['binary'] for i, p in char_patterns.items()}
        },
        "threshold_optimization": {
            "best_threshold": best_threshold['threshold'],
            "best_text": best_threshold['text'],
            "printable_ratio": best_threshold['printable_ratio']
        },
        "position_fine_tuning": {
            "best_adjustment": (best_position['row_adj'], best_position['col_adj']) if best_position else None,
            "best_position": best_position['start_pos'] if best_position else None,
            "best_score": best_position['score'] if best_position else 0,
            "extracted_text": best_position['text'] if best_position else ""
        },
        "final_config": {
            "row_pitch": 31,
            "col_pitch": 53,
            "start_row": best_position['start_pos'][0] if best_position else 101,
            "start_col": best_position['start_pos'][1] if best_position else 53,
            "threshold": best_threshold['threshold']
        },
        "assessment": "Visual grid alignment to bridge manual extraction with automation"
    }
    
    with open('visual_alignment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Visual alignment results saved")
    
    return results

if __name__ == "__main__":
    print("Visual Grid Alignment Tool")
    print("Bridging manual and automated extraction")
    print("=" * 45)
    
    # Create visual overlays
    create_visual_grid_overlay()
    
    # Analyze character positions
    char_patterns = analyze_character_positions()
    
    # Test threshold variations
    best_threshold = test_threshold_variations()
    
    # Fine-tune grid position
    best_position = fine_tune_grid_position()
    
    # Save comprehensive results
    results = save_visual_alignment_results()
    
    print(f"\n" + "=" * 45)
    print("VISUAL ALIGNMENT COMPLETE")
    
    if best_position:
        print(f"Best position: {best_position['start_pos']}")
        print(f"Best score: {best_position['score']:.1f}")
        print(f"Extracted: '{best_position['text']}'")
        
        # Check if we found recognizable patterns
        if best_position['score'] > 15:
            print("\n*** PROMISING ALIGNMENT FOUND ***")
        else:
            print("\nAlignment needs further refinement")
    
    print(f"Grid overlays saved as PNG files for visual inspection")