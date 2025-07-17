#!/usr/bin/env python3
"""
ML-based template extraction and pattern recognition.
Using the 84.4% breakthrough position for advanced analysis.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import ndimage
from collections import Counter
import json

def extract_digit_templates():
    """Extract templates from visible digit areas in the image."""
    
    print("=== EXTRACTING DIGIT TEMPLATES ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Find areas with visible text/digits (very bright regions)
    bright_threshold = np.percentile(img, 95)
    bright_mask = img > bright_threshold
    
    # Find connected components
    labeled, num_features = ndimage.label(bright_mask)
    
    templates = {}
    patch_size = 8
    
    print(f"Found {num_features} bright regions to analyze")
    
    # Extract patches from bright regions
    patches = []
    positions = []
    
    for label_id in range(1, min(num_features + 1, 100)):  # Top 100 regions
        region_mask = labeled == label_id
        coords = np.where(region_mask)
        
        if len(coords[0]) > 20:  # Minimum size
            # Sample multiple patches from this region
            for i in range(0, len(coords[0]), 10):
                y, x = coords[0][i], coords[1][i]
                
                # Extract patch
                y_start = max(0, y - patch_size//2)
                y_end = min(img.shape[0], y + patch_size//2 + 1)
                x_start = max(0, x - patch_size//2)
                x_end = min(img.shape[1], x + patch_size//2 + 1)
                
                patch = img[y_start:y_end, x_start:x_end]
                
                if patch.shape[0] >= patch_size//2 and patch.shape[1] >= patch_size//2:
                    # Resize to standard size
                    patch_resized = cv2.resize(patch, (patch_size, patch_size))
                    patches.append(patch_resized.flatten())
                    positions.append((y, x))
    
    print(f"Extracted {len(patches)} patches from bright regions")
    
    if len(patches) > 10:
        # Cluster patches to find common patterns
        patches_array = np.array(patches)
        
        # Use K-means to find common digit patterns
        n_clusters = min(10, len(patches) // 5)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(patches_array)
        
        # Store cluster centers as templates
        for i, center in enumerate(kmeans.cluster_centers_):
            template = center.reshape(patch_size, patch_size)
            templates[f'cluster_{i}'] = template
            
            # Show cluster info
            cluster_size = np.sum(cluster_labels == i)
            print(f"Cluster {i}: {cluster_size} patches, avg_intensity={np.mean(center):.1f}")
        
        # Save templates
        np.savez('digit_templates.npz', **templates)
        print(f"Saved {len(templates)} templates to digit_templates.npz")
    
    return templates, patches, positions

def template_match_extraction():
    """Use template matching for bit extraction."""
    
    print(f"\n=== TEMPLATE MATCHING EXTRACTION ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Load templates if available
    try:
        templates_data = np.load('digit_templates.npz')
        templates = {key: templates_data[key] for key in templates_data.files}
        print(f"Loaded {len(templates)} templates")
    except:
        print("No templates found, extracting first...")
        templates, _, _ = extract_digit_templates()
    
    # Use breakthrough position
    row0, col0 = 101, 53
    row_pitch = 31
    col_pitch = 53
    patch_size = 8
    
    print(f"Template matching at breakthrough position ({row0}, {col0})")
    
    # Extract patches at grid positions
    grid_patches = []
    grid_positions = []
    
    for r in range(8):  # First 8 rows
        for c in range(16):  # 2 characters worth
            y = row0 + r * row_pitch
            x = col0 + c * col_pitch
            
            if 0 <= y < img.shape[0] - patch_size and 0 <= x < img.shape[1] - patch_size:
                patch = img[y:y+patch_size, x:x+patch_size]
                grid_patches.append(patch)
                grid_positions.append((r, c, y, x))
    
    print(f"Extracted {len(grid_patches)} grid patches")
    
    # Match each grid patch against templates
    matches = []
    
    for i, patch in enumerate(grid_patches):
        r, c, y, x = grid_positions[i]
        
        best_match = None
        best_score = -1
        
        for template_name, template in templates.items():
            # Normalize both patch and template
            patch_norm = (patch - patch.mean()) / (patch.std() + 1e-8)
            template_norm = (template - template.mean()) / (template.std() + 1e-8)
            
            # Calculate correlation
            correlation = np.corrcoef(patch_norm.flatten(), template_norm.flatten())[0, 1]
            
            if not np.isnan(correlation) and correlation > best_score:
                best_score = correlation
                best_match = template_name
        
        matches.append({
            'position': (r, c),
            'coords': (y, x),
            'template': best_match,
            'score': best_score,
            'patch_mean': np.mean(patch)
        })
    
    # Analyze matches
    print(f"\n=== TEMPLATE MATCH ANALYSIS ===")
    
    # Group by rows
    for r in range(8):
        row_matches = [m for m in matches if m['position'][0] == r]
        row_matches.sort(key=lambda x: x['position'][1])
        
        if row_matches:
            match_str = ' '.join([f"{m['template'][:8]:8s}" for m in row_matches[:8]])
            score_str = ' '.join([f"{m['score']:4.2f}" for m in row_matches[:8]])
            print(f"Row {r}: {match_str}")
            print(f"       {score_str}")
    
    return matches

def statistical_pattern_analysis():
    """Statistical analysis of bit patterns at breakthrough position."""
    
    print(f"\n=== STATISTICAL PATTERN ANALYSIS ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Breakthrough configuration
    row0, col0 = 101, 53
    row_pitch = 31
    col_pitch = 53
    threshold = 72
    patch_size = 5
    
    # Extract larger bit grid
    bit_grid = []
    value_grid = []
    
    for r in range(20):
        bit_row = []
        value_row = []
        
        for c in range(40):
            y = row0 + r * row_pitch
            x = col0 + c * col_pitch
            
            if 0 <= y < img.shape[0] - patch_size and 0 <= x < img.shape[1] - patch_size:
                half = patch_size // 2
                patch = img[max(0, y-half):min(img.shape[0], y+half+1), 
                           max(0, x-half):min(img.shape[1], x+half+1)]
                
                if patch.size > 0:
                    val = np.median(patch)
                    bit = 1 if val > threshold else 0
                    bit_row.append(bit)
                    value_row.append(val)
                else:
                    bit_row.append(0)
                    value_row.append(0)
            else:
                bit_row.append(0)
                value_row.append(0)
        
        bit_grid.append(bit_row)
        value_grid.append(value_row)
    
    bit_array = np.array(bit_grid)
    value_array = np.array(value_grid)
    
    print(f"Extracted {bit_array.shape[0]}x{bit_array.shape[1]} bit grid")
    
    # Statistical analysis
    print(f"\n--- Bit Statistics ---")
    total_bits = bit_array.size
    ones_count = np.sum(bit_array)
    zeros_count = total_bits - ones_count
    
    print(f"Total bits: {total_bits}")
    print(f"Ones: {ones_count} ({ones_count/total_bits:.1%})")
    print(f"Zeros: {zeros_count} ({zeros_count/total_bits:.1%})")
    
    # Row patterns
    print(f"\n--- Row Patterns ---")
    for r in range(min(10, bit_array.shape[0])):
        row_bits = bit_array[r, :16]  # First 16 bits
        row_str = ''.join(map(str, row_bits))
        row_ones = np.sum(row_bits)
        print(f"Row {r:2d}: {row_str} ({row_ones:2d}/16 ones)")
    
    # Column patterns  
    print(f"\n--- Column Patterns ---")
    for c in range(min(16, bit_array.shape[1])):
        col_bits = bit_array[:10, c]  # First 10 rows
        col_str = ''.join(map(str, col_bits))
        col_ones = np.sum(col_bits)
        print(f"Col {c:2d}: {col_str} ({col_ones:2d}/10 ones)")
    
    # 8-bit byte analysis
    print(f"\n--- 8-bit Byte Analysis ---")
    byte_patterns = Counter()
    
    for r in range(bit_array.shape[0]):
        for c in range(0, bit_array.shape[1] - 7, 8):
            byte_bits = bit_array[r, c:c+8]
            byte_str = ''.join(map(str, byte_bits))
            byte_val = int(byte_str, 2)
            byte_patterns[byte_val] += 1
    
    print(f"Most common byte values:")
    for byte_val, count in byte_patterns.most_common(10):
        char = chr(byte_val) if 32 <= byte_val <= 126 else f'[{byte_val}]'
        print(f"  {byte_val:3d} ('{char}'): {count:2d} times")
    
    # Entropy analysis
    if ones_count > 0 and zeros_count > 0:
        p1 = ones_count / total_bits
        p0 = zeros_count / total_bits
        entropy = -p1 * np.log2(p1) - p0 * np.log2(p0)
        print(f"\nBit entropy: {entropy:.3f} (max=1.0)")
    
    # Save analysis
    analysis = {
        'total_bits': int(total_bits),
        'ones_count': int(ones_count),
        'zeros_count': int(zeros_count),
        'ones_percentage': float(ones_count/total_bits),
        'entropy': float(entropy) if ones_count > 0 and zeros_count > 0 else 0,
        'most_common_bytes': [(int(val), int(count)) for val, count in byte_patterns.most_common(20)]
    }
    
    with open('statistical_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\nStatistical analysis saved to statistical_analysis.json")
    
    return bit_array, value_array, analysis

def advanced_message_search():
    """Advanced search for message patterns using different approaches."""
    
    print(f"\n=== ADVANCED MESSAGE SEARCH ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Multiple search strategies
    strategies = [
        ("Breakthrough position", (101, 53), 72),
        ("Alternative threshold", (101, 53), 85),
        ("Shifted position", (103, 55), 72),
        ("Lower threshold", (101, 53), 60),
        ("Different region", (80, 40), 70)
    ]
    
    message_candidates = []
    
    for strategy_name, (row0, col0), threshold in strategies:
        print(f"\n--- {strategy_name} ---")
        
        # Extract message
        message_lines = []
        for r in range(15):
            y = row0 + r * 31  # Use established row pitch
            
            row_bits = []
            for c in range(32):  # 4 characters worth
                x = col0 + c * 53  # Use established col pitch
                
                if 0 <= y < img.shape[0] - 5 and 0 <= x < img.shape[1] - 5:
                    patch = img[max(0, y-2):min(img.shape[0], y+3), 
                               max(0, x-2):min(img.shape[1], x+3)]
                    
                    if patch.size > 0:
                        val = np.median(patch)
                        bit = '1' if val > threshold else '0'
                        row_bits.append(bit)
            
            # Decode row
            if len(row_bits) >= 8:
                decoded_chars = []
                for i in range(0, len(row_bits), 8):
                    if i + 8 <= len(row_bits):
                        byte = ''.join(row_bits[i:i+8])
                        try:
                            val = int(byte, 2)
                            if 32 <= val <= 126:
                                decoded_chars.append(chr(val))
                            elif val == 0:
                                decoded_chars.append(' ')
                            else:
                                decoded_chars.append(f'[{val}]')
                        except:
                            decoded_chars.append('?')
                
                line = ''.join(decoded_chars)
                message_lines.append(line)
                
                if r < 8:  # Show first 8 lines
                    print(f"  Row {r}: {line}")
        
        # Analyze for readability
        readable_lines = 0
        all_text = ' '.join(message_lines).lower()
        
        for line in message_lines:
            if line.strip():
                readable_chars = sum(1 for c in line if c.isalnum() or c.isspace() or c in '.,!?-')
                if len(line) > 0 and readable_chars / len(line) > 0.4:
                    readable_lines += 1
        
        # Search for keywords
        keywords = ['bitcoin', 'satoshi', 'on the', 'at the', 'message', 'genesis', 'block', 'hash']
        found_keywords = [kw for kw in keywords if kw in all_text]
        
        score = readable_lines + len(found_keywords) * 2
        
        message_candidates.append({
            'strategy': strategy_name,
            'position': (row0, col0),
            'threshold': threshold,
            'score': score,
            'readable_lines': readable_lines,
            'keywords': found_keywords,
            'first_line': message_lines[0] if message_lines else ''
        })
        
        print(f"  Readable lines: {readable_lines}")
        if found_keywords:
            print(f"  Keywords found: {found_keywords}")
    
    # Rank strategies
    message_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\n=== STRATEGY RANKING ===")
    for i, candidate in enumerate(message_candidates):
        print(f"{i+1}. {candidate['strategy']:20s} Score: {candidate['score']:2d} "
              f"(readable: {candidate['readable_lines']}, keywords: {len(candidate['keywords'])})")
        if candidate['first_line'].strip():
            print(f"   First line: {candidate['first_line'][:40]}...")
    
    return message_candidates

if __name__ == "__main__":
    print("ML-Based Template Extraction and Advanced Analysis")
    print("Using 84.4% breakthrough position for advanced techniques")
    print("="*70)
    
    # Extract digit templates from bright regions
    templates, patches, positions = extract_digit_templates()
    
    # Template matching extraction
    if templates:
        matches = template_match_extraction()
    
    # Statistical pattern analysis
    bit_grid, value_grid, stats = statistical_pattern_analysis()
    
    # Advanced message search
    candidates = advanced_message_search()
    
    print("\n" + "="*70)
    print("ADVANCED ANALYSIS COMPLETE")
    
    if candidates and candidates[0]['score'] > 0:
        best = candidates[0]
        print(f"Best strategy: {best['strategy']} (score: {best['score']})")
        if best['keywords']:
            print(f"Keywords found: {best['keywords']}")
    
    print("✓ Template extraction complete")
    print("✓ Statistical analysis complete") 
    print("✓ Advanced search strategies tested")
    print("\nCheck generated files for detailed results")