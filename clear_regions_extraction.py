#!/usr/bin/env python3
"""
Focus on clear regions extraction - automate what was manually verified.
Target: "On the winter solstice December 21" and expand from there.
"""

import cv2
import numpy as np
import json
from scipy import ndimage
import matplotlib.pyplot as plt

def identify_clear_vs_washed_regions():
    """Identify clear regions vs washed out areas in the image."""
    
    print("=== IDENTIFYING CLEAR VS WASHED REGIONS ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Analyze image contrast and brightness to identify clear regions
    print(f"Image dimensions: {img.shape}")
    print(f"Overall brightness: {np.mean(img):.1f}")
    print(f"Overall contrast: {np.std(img):.1f}")
    
    # Create contrast map using local standard deviation
    kernel_size = 15
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    
    # Local mean and variance
    mean_img = cv2.filter2D(img.astype(np.float32), -1, kernel)
    sqr_img = cv2.filter2D((img.astype(np.float32))**2, -1, kernel)
    contrast_map = np.sqrt(sqr_img - mean_img**2)
    
    # Identify regions with good contrast (clear areas)
    contrast_threshold = np.percentile(contrast_map, 75)  # Top 25% contrast
    clear_regions = contrast_map > contrast_threshold
    
    print(f"Contrast threshold: {contrast_threshold:.1f}")
    print(f"Clear region coverage: {np.sum(clear_regions) / clear_regions.size:.1%}")
    
    # Also check brightness - avoid very bright washed out areas
    brightness_threshold = np.percentile(img, 85)  # Avoid top 15% brightest
    not_washed_out = img < brightness_threshold
    
    # Combine criteria
    good_regions = clear_regions & not_washed_out
    
    print(f"Good extraction regions: {np.sum(good_regions) / good_regions.size:.1%}")
    
    return good_regions, contrast_map

def focus_extraction_on_clear_areas():
    """Focus extraction specifically on clear, high-contrast areas."""
    
    print("\n=== FOCUSED EXTRACTION ON CLEAR AREAS ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    good_regions, contrast_map = identify_clear_vs_washed_regions()
    
    # Test multiple grid configurations in clear areas only
    known_text = "On the winter solstice December 21 "
    target_bits = []
    for char in known_text:
        byte_val = ord(char)
        binary = format(byte_val, '08b')
        target_bits.extend([int(b) for b in binary])
    
    print(f"Target text: '{known_text}'")
    print(f"Target bits needed: {len(target_bits)}")
    
    # Test configurations focused on clear regions
    test_configs = [
        # (row_pitch, col_pitch, start_row, start_col, threshold, name)
        (31, 53, 101, 53, 72, "original_verification"),
        (30, 52, 100, 50, 65, "close_variant"),
        (28, 55, 95, 45, 60, "alternative"),
        (25, 50, 90, 40, 55, "smaller_spacing"),
        (32, 54, 105, 55, 70, "larger_spacing"),
    ]
    
    results = []
    
    for row_pitch, col_pitch, start_row, start_col, threshold, name in test_configs:
        print(f"\nTesting {name}: {row_pitch}x{col_pitch} at ({start_row},{start_col}) t={threshold}")
        
        # Extract bits, but only count those in clear regions
        extracted_bits = []
        clear_bit_count = 0
        
        for bit_idx in range(len(target_bits)):
            bit_row = bit_idx // 8
            bit_col = bit_idx % 8
            
            y = start_row + bit_row * row_pitch
            x = start_col + bit_col * col_pitch
            
            # Check if this position is in a clear region
            if (0 <= y < good_regions.shape[0] and 0 <= x < good_regions.shape[1]):
                is_clear = good_regions[y-3:y+4, x-3:x+4].mean() > 0.5  # Most of 7x7 region is clear
                
                if (0 <= y - 3 and y + 3 < img.shape[0] and 
                    0 <= x - 3 and x + 3 < img.shape[1]):
                    
                    region = img[y-3:y+4, x-3:x+4]
                    median_val = np.median(region)
                    bit = 1 if median_val > threshold else 0
                    extracted_bits.append(bit)
                    
                    if is_clear:
                        clear_bit_count += 1
        
        if len(extracted_bits) == len(target_bits):
            # Convert to text
            extracted_text = ""
            for char_idx in range(len(extracted_bits) // 8):
                byte_val = 0
                for bit_idx in range(8):
                    bit_pos = char_idx * 8 + bit_idx
                    byte_val |= (extracted_bits[bit_pos] << (7 - bit_idx))
                
                if 32 <= byte_val <= 126:
                    extracted_text += chr(byte_val)
                else:
                    extracted_text += '?'
            
            # Calculate match score
            char_matches = sum(1 for i in range(min(len(extracted_text), len(known_text)))
                             if extracted_text[i] == known_text[i])
            match_ratio = char_matches / len(known_text)
            clear_ratio = clear_bit_count / len(target_bits)
            
            print(f"  Extracted: '{extracted_text}'")
            print(f"  Match: {char_matches}/{len(known_text)} ({match_ratio:.1%})")
            print(f"  Clear bits: {clear_bit_count}/{len(target_bits)} ({clear_ratio:.1%})")
            
            results.append({
                'config_name': name,
                'config': (row_pitch, col_pitch, start_row, start_col, threshold),
                'extracted_text': extracted_text,
                'match_ratio': match_ratio,
                'clear_ratio': clear_ratio,
                'quality_score': match_ratio * 0.7 + clear_ratio * 0.3
            })
    
    # Sort by quality score (match + clear region coverage)
    results.sort(key=lambda x: x['quality_score'], reverse=True)
    
    print(f"\n=== CLEAR REGION EXTRACTION RESULTS ===")
    for i, result in enumerate(results):
        name = result['config_name']
        match = result['match_ratio']
        clear = result['clear_ratio']
        quality = result['quality_score']
        text = result['extracted_text'][:20] + ('...' if len(result['extracted_text']) > 20 else '')
        print(f"{i+1}. {name}: Q={quality:.2f} (M={match:.1%}, C={clear:.1%}) '{text}'")
    
    return results

def expand_extraction_from_best_clear():
    """Expand extraction using the best clear region configuration."""
    
    print(f"\n=== EXPANDING EXTRACTION FROM BEST CLEAR CONFIG ===")
    
    # Get best configuration from clear region analysis
    results = focus_extraction_on_clear_areas()
    
    if not results or results[0]['quality_score'] < 0.5:
        print("No good clear region configuration found")
        return None
    
    best_config = results[0]
    row_pitch, col_pitch, start_row, start_col, threshold = best_config['config']
    
    print(f"Using best config: {best_config['config_name']}")
    print(f"Parameters: {row_pitch}x{col_pitch} at ({start_row},{start_col}) t={threshold}")
    print(f"Quality score: {best_config['quality_score']:.2f}")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    good_regions, _ = identify_clear_vs_washed_regions()
    
    # Extract longer sequence, prioritizing clear regions
    max_chars = 100
    extracted_bits = []
    bit_quality_scores = []
    
    for char_idx in range(max_chars):
        for bit_idx in range(8):
            total_bit_idx = char_idx * 8 + bit_idx
            bit_row = total_bit_idx // 8
            bit_col = total_bit_idx % 8
            
            y = start_row + bit_row * row_pitch
            x = start_col + bit_col * col_pitch
            
            if (0 <= y - 3 and y + 3 < img.shape[0] and 
                0 <= x - 3 and x + 3 < img.shape[1]):
                
                # Check if this is in a clear region
                is_clear = good_regions[y-3:y+4, x-3:x+4].mean() > 0.3
                region_contrast = np.std(img[y-3:y+4, x-3:x+4])
                
                region = img[y-3:y+4, x-3:x+4]
                median_val = np.median(region)
                bit = 1 if median_val > threshold else 0
                
                extracted_bits.append(bit)
                
                # Quality score based on clarity and contrast
                quality = (is_clear * 0.7) + (min(region_contrast / 20, 1.0) * 0.3)
                bit_quality_scores.append(quality)
            else:
                break
    
    # Convert to text
    complete_text = ""
    char_qualities = []
    
    for char_idx in range(len(extracted_bits) // 8):
        byte_val = 0
        for bit_idx in range(8):
            bit_pos = char_idx * 8 + bit_idx
            if bit_pos < len(extracted_bits):
                byte_val |= (extracted_bits[bit_pos] << (7 - bit_idx))
        
        if 32 <= byte_val <= 126:
            complete_text += chr(byte_val)
        else:
            complete_text += '?'
        
        # Calculate character quality (average of its 8 bits)
        char_start = char_idx * 8
        char_end = min(char_start + 8, len(bit_quality_scores))
        if char_end > char_start:
            char_quality = np.mean(bit_quality_scores[char_start:char_end])
            char_qualities.append(char_quality)
    
    print(f"\nExtracted text ({len(complete_text)} chars):")
    print(f"'{complete_text}'")
    
    # Show quality breakdown
    avg_quality = np.mean(bit_quality_scores) if bit_quality_scores else 0
    high_quality_chars = sum(1 for q in char_qualities if q > 0.6)
    
    print(f"\nQuality analysis:")
    print(f"Average bit quality: {avg_quality:.2f}")
    print(f"High quality characters: {high_quality_chars}/{len(char_qualities)}")
    print(f"Overall printable ratio: {sum(1 for c in complete_text if c != '?') / len(complete_text):.1%}")
    
    # Highlight high-confidence portions
    print(f"\nHigh-confidence text (quality > 0.6):")
    high_conf_text = ""
    for i, (char, quality) in enumerate(zip(complete_text, char_qualities)):
        if quality > 0.6:
            high_conf_text += char
        else:
            high_conf_text += '_'
    print(f"'{high_conf_text}'")
    
    return {
        'complete_text': complete_text,
        'bit_qualities': bit_quality_scores,
        'char_qualities': char_qualities,
        'config': best_config,
        'avg_quality': avg_quality
    }

def save_clear_region_results():
    """Save clear region extraction results."""
    
    print(f"\n=== SAVING CLEAR REGION RESULTS ===")
    
    # Run complete analysis
    good_regions, contrast_map = identify_clear_vs_washed_regions()
    clear_results = focus_extraction_on_clear_areas()
    
    if clear_results:
        expanded_results = expand_extraction_from_best_clear()
    else:
        expanded_results = None
    
    # Compile results
    results = {
        "timestamp": "2025-07-17",
        "analysis_type": "clear_regions_extraction",
        "focus": "Automate extraction of manually verified text in clear areas",
        "target_text": "On the winter solstice December 21",
        "region_analysis": {
            "clear_region_coverage": float(np.sum(good_regions) / good_regions.size),
            "contrast_threshold": float(np.percentile(contrast_map, 75))
        },
        "clear_region_configs": [
            {
                "name": r['config_name'],
                "quality_score": r['quality_score'],
                "match_ratio": r['match_ratio'],
                "clear_ratio": r['clear_ratio'],
                "extracted_text": r['extracted_text']
            }
            for r in clear_results
        ] if clear_results else [],
        "expanded_extraction": {
            "complete_text": expanded_results['complete_text'] if expanded_results else "",
            "avg_quality": expanded_results['avg_quality'] if expanded_results else 0,
            "config_used": expanded_results['config']['config_name'] if expanded_results else ""
        } if expanded_results else {},
        "assessment": "Focus on clear regions first - smart approach to build confidence before tackling washed areas"
    }
    
    with open('clear_regions_extraction_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Clear region results saved to clear_regions_extraction_results.json")
    
    return results

if __name__ == "__main__":
    print("Clear Regions Extraction")
    print("Focus on automating manually verified text from clear areas")
    print("=" * 60)
    
    # Identify clear vs washed regions
    good_regions, contrast_map = identify_clear_vs_washed_regions()
    
    # Focus extraction on clear areas
    clear_results = focus_extraction_on_clear_areas()
    
    if clear_results and clear_results[0]['quality_score'] > 0.3:
        print(f"\nFound promising clear region configuration!")
        
        # Expand extraction from best clear config
        expanded_results = expand_extraction_from_best_clear()
        
        # Save comprehensive results
        save_clear_region_results()
        
        print(f"\n" + "=" * 60)
        print("CLEAR REGION EXTRACTION COMPLETE")
        
        if expanded_results:
            print(f"Best config: {expanded_results['config']['config_name']}")
            print(f"Average quality: {expanded_results['avg_quality']:.2f}")
            print(f"Text preview: '{expanded_results['complete_text'][:50]}...'")
    else:
        print("\nNo strong clear region matches found")
        print("May need to adjust parameters or preprocessing")