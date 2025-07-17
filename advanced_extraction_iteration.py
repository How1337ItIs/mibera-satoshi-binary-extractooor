#!/usr/bin/env python3
"""
Advanced iteration: Sub-pixel alignment, alternative sources, template matching.
Continue pushing towards the 'On the...' extraction.
"""

import cv2
import numpy as np
from scipy import ndimage, interpolate
import json
from pathlib import Path

class AdvancedExtractor:
    def __init__(self):
        self.results = []
        
    def load_all_available_images(self):
        """Load all available image sources for testing."""
        
        image_sources = {
            'bw_mask': 'binary_extractor/output_real_data/bw_mask.png',
            'gaussian_subtracted': 'binary_extractor/output_real_data/gaussian_subtracted.png',
            'cyan_channel': 'binary_extractor/output_real_data/cyan_channel.png',
            'silver_mask': 'binary_extractor/output_real_data/silver_mask.png'
        }
        
        loaded_images = {}
        
        for name, path in image_sources.items():
            try:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    loaded_images[name] = img
                    print(f"✓ Loaded {name}: {img.shape}, range {img.min()}-{img.max()}")
                else:
                    print(f"✗ Could not load {name}")
            except Exception as e:
                print(f"✗ Error loading {name}: {e}")
        
        return loaded_images
    
    def sub_pixel_extraction(self, img, row0, col0, row_pitch, col_pitch, sub_precision=0.1):
        """Try sub-pixel grid alignment with interpolation."""
        
        print(f"\n=== SUB-PIXEL EXTRACTION ===")
        print(f"Base grid: ({row0}, {col0}), pitch {row_pitch}x{col_pitch}")
        print(f"Sub-pixel precision: {sub_precision}")
        
        target_bits = "0100111101101110"  # "On"
        best_score = 0
        best_result = None
        
        # Test sub-pixel offsets
        sub_range = np.arange(-1.0, 1.0, sub_precision)
        
        for row_offset in sub_range:
            for col_offset in sub_range:
                actual_row0 = row0 + row_offset
                actual_col0 = col0 + col_offset
                
                # Extract first 16 bits with interpolation
                bits = self.extract_with_interpolation(
                    img, actual_row0, actual_col0, row_pitch, col_pitch, 16
                )
                
                if len(bits) == 16:
                    bits_str = ''.join(bits)
                    
                    # Score against "On" target
                    matches = sum(1 for i in range(16) if bits_str[i] == target_bits[i])
                    score = matches / 16
                    
                    if score > best_score:
                        best_score = score
                        
                        try:
                            char1 = chr(int(bits_str[:8], 2))
                            char2 = chr(int(bits_str[8:16], 2))
                            decoded = f"{char1}{char2}"
                        except:
                            decoded = "??"
                        
                        best_result = {
                            'offset': (row_offset, col_offset),
                            'position': (actual_row0, actual_col0),
                            'score': score,
                            'bits': bits_str,
                            'decoded': decoded
                        }
                        
                        if score > 0.8:
                            print(f"  Excellent match: offset ({row_offset:.1f}, {col_offset:.1f})")
                            print(f"  Score: {score:.1%}, Decoded: '{decoded}'")
        
        if best_result:
            print(f"\nBest sub-pixel result:")
            print(f"  Offset: {best_result['offset']}")
            print(f"  Position: ({best_result['position'][0]:.1f}, {best_result['position'][1]:.1f})")
            print(f"  Score: {best_result['score']:.1%}")
            print(f"  Decoded: '{best_result['decoded']}'")
            
            return best_result
        
        return None
    
    def extract_with_interpolation(self, img, row0, col0, row_pitch, col_pitch, num_bits):
        """Extract bits using bilinear interpolation for sub-pixel positions."""
        
        bits = []
        
        for i in range(num_bits):
            bit_row = i // 8
            bit_col = i % 8
            
            y = row0 + bit_row * row_pitch
            x = col0 + bit_col * col_pitch
            
            if 0 <= y < img.shape[0]-1 and 0 <= x < img.shape[1]-1:
                # Bilinear interpolation
                x_floor = int(x)
                y_floor = int(y)
                x_frac = x - x_floor
                y_frac = y - y_floor
                
                # Get 2x2 neighborhood
                top_left = img[y_floor, x_floor]
                top_right = img[y_floor, x_floor + 1]
                bottom_left = img[y_floor + 1, x_floor]
                bottom_right = img[y_floor + 1, x_floor + 1]
                
                # Bilinear interpolation
                top = top_left * (1 - x_frac) + top_right * x_frac
                bottom = bottom_left * (1 - x_frac) + bottom_right * x_frac
                val = top * (1 - y_frac) + bottom * y_frac
                
                bit = '1' if val > 127 else '0'
                bits.append(bit)
        
        return bits
    
    def template_match_on_pattern(self, img):
        """Use template matching to find 'On' pattern."""
        
        print(f"\n=== TEMPLATE MATCHING FOR 'On' ===")
        
        # Create template for "On" pattern
        on_template = self.create_on_template()
        
        # Multi-scale template matching
        scales = [0.5, 0.75, 1.0, 1.25, 1.5]
        best_matches = []
        
        for scale in scales:
            if scale != 1.0:
                h, w = on_template.shape
                new_h, new_w = int(h * scale), int(w * scale)
                scaled_template = cv2.resize(on_template, (new_w, new_h))
            else:
                scaled_template = on_template
            
            # Template matching
            result = cv2.matchTemplate(img, scaled_template, cv2.TM_CCOEFF_NORMED)
            
            # Find peaks
            threshold = 0.3
            locations = np.where(result >= threshold)
            
            for y, x in zip(locations[0], locations[1]):
                match_value = result[y, x]
                best_matches.append({
                    'position': (y, x),
                    'scale': scale,
                    'confidence': match_value,
                    'template_size': scaled_template.shape
                })
        
        # Sort by confidence
        best_matches.sort(key=lambda x: x['confidence'], reverse=True)
        
        print(f"Found {len(best_matches)} potential matches")
        for i, match in enumerate(best_matches[:10]):
            print(f"  {i+1}. Position {match['position']}, scale {match['scale']:.2f}, "
                  f"confidence {match['confidence']:.3f}")
        
        return best_matches[:5]  # Return top 5
    
    def create_on_template(self):
        """Create a template for 'On' based on expected bit pattern."""
        
        # "On" = 01001111 01101110
        on_bits = "0100111101101110"
        
        # Create 8x2 template (8 bits wide, 2 characters tall)
        template = np.zeros((16, 8), dtype=np.uint8)
        
        for i, bit in enumerate(on_bits):
            row = i // 8
            col = i % 8
            
            if bit == '1':
                template[row*8:(row+1)*8, col] = 255
            else:
                template[row*8:(row+1)*8, col] = 0
        
        # Smooth slightly to match real digit patterns
        template = cv2.GaussianBlur(template, (3, 3), 0.5)
        
        return template
    
    def try_different_bit_orderings(self, img, row0, col0, row_pitch, col_pitch):
        """Try column-major vs row-major bit ordering."""
        
        print(f"\n=== DIFFERENT BIT ORDERINGS ===")
        
        orderings = {
            'row_major': self.extract_row_major,
            'column_major': self.extract_column_major,
            'zigzag': self.extract_zigzag
        }
        
        target_bits = "0100111101101110"
        results = []
        
        for name, extract_func in orderings.items():
            bits = extract_func(img, row0, col0, row_pitch, col_pitch, 16)
            
            if len(bits) >= 16:
                bits_str = ''.join(bits[:16])
                
                matches = sum(1 for i in range(16) if bits_str[i] == target_bits[i])
                score = matches / 16
                
                try:
                    char1 = chr(int(bits_str[:8], 2))
                    char2 = chr(int(bits_str[8:16], 2))
                    decoded = f"{char1}{char2}"
                except:
                    decoded = "??"
                
                results.append({
                    'ordering': name,
                    'score': score,
                    'bits': bits_str,
                    'decoded': decoded
                })
                
                print(f"  {name}: {score:.1%} -> '{decoded}' ({bits_str})")
        
        return sorted(results, key=lambda x: x['score'], reverse=True)
    
    def extract_row_major(self, img, row0, col0, row_pitch, col_pitch, num_bits):
        """Standard row-major extraction."""
        bits = []
        for i in range(num_bits):
            bit_row = i // 8
            bit_col = i % 8
            y = row0 + bit_row * row_pitch
            x = col0 + bit_col * col_pitch
            
            if y < img.shape[0]-2 and x < img.shape[1]-2:
                region = img[y-1:y+2, x-1:x+2]
                val = np.median(region) if region.size > 0 else 0
                bits.append('1' if val > 127 else '0')
        return bits
    
    def extract_column_major(self, img, row0, col0, row_pitch, col_pitch, num_bits):
        """Column-major extraction."""
        bits = []
        for i in range(num_bits):
            bit_col = i // 8
            bit_row = i % 8
            y = row0 + bit_row * row_pitch
            x = col0 + bit_col * col_pitch
            
            if y < img.shape[0]-2 and x < img.shape[1]-2:
                region = img[y-1:y+2, x-1:x+2]
                val = np.median(region) if region.size > 0 else 0
                bits.append('1' if val > 127 else '0')
        return bits
    
    def extract_zigzag(self, img, row0, col0, row_pitch, col_pitch, num_bits):
        """Zigzag extraction pattern."""
        bits = []
        for i in range(num_bits):
            char_idx = i // 8
            bit_idx = i % 8
            
            if char_idx % 2 == 0:  # Even characters: left to right
                bit_col = bit_idx
            else:  # Odd characters: right to left
                bit_col = 7 - bit_idx
            
            y = row0 + char_idx * row_pitch
            x = col0 + bit_col * col_pitch
            
            if y < img.shape[0]-2 and x < img.shape[1]-2:
                region = img[y-1:y+2, x-1:x+2]
                val = np.median(region) if region.size > 0 else 0
                bits.append('1' if val > 127 else '0')
        return bits
    
    def comprehensive_search(self, img_name, img):
        """Comprehensive search on one image source."""
        
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE SEARCH: {img_name}")
        print(f"{'='*60}")
        
        # Get base grid parameters
        row_pitch, col_pitch = self.detect_pitches(img)
        
        results = []
        
        # 1. Sub-pixel alignment
        print(f"\n1. Testing sub-pixel alignment...")
        for row0 in range(60, 80, 5):
            for col0 in range(30, 50, 5):
                sub_result = self.sub_pixel_extraction(img, row0, col0, row_pitch, col_pitch, 0.2)
                if sub_result and sub_result['score'] > 0.6:
                    results.append(('sub_pixel', sub_result))
        
        # 2. Template matching
        print(f"\n2. Template matching...")
        template_matches = self.template_match_on_pattern(img)
        for match in template_matches:
            if match['confidence'] > 0.4:
                results.append(('template', match))
        
        # 3. Different orderings
        print(f"\n3. Different bit orderings...")
        for row0 in range(65, 75, 2):
            for col0 in range(35, 45, 2):
                ordering_results = self.try_different_bit_orderings(img, row0, col0, row_pitch, col_pitch)
                for result in ordering_results:
                    if result['score'] > 0.6:
                        results.append(('ordering', result))
        
        return results
    
    def detect_pitches(self, img):
        """Quick pitch detection."""
        mask = img > 150
        
        # Row pitch
        row_proj = mask.sum(1).astype(float)
        row_proj -= row_proj.mean()
        row_corr = np.correlate(row_proj, row_proj, mode='full')[len(row_proj):]
        row_pitch = np.argmax(row_corr[20:40]) + 20
        
        # Col pitch  
        col_proj = mask.sum(0).astype(float)
        col_proj -= col_proj.mean()
        col_corr = np.correlate(col_proj, col_proj, mode='full')[len(col_proj):]
        col_pitch = np.argmax(col_corr[10:60]) + 10
        
        return row_pitch, col_pitch

def main():
    print("Advanced Extraction Iteration")
    print("Sub-pixel alignment + Template matching + Alternative orderings")
    print("="*70)
    
    extractor = AdvancedExtractor()
    
    # Load all available images
    images = extractor.load_all_available_images()
    
    if not images:
        print("No images available for processing!")
        return
    
    all_results = []
    
    # Test each image source
    for img_name, img in images.items():
        results = extractor.comprehensive_search(img_name, img)
        
        for result_type, result_data in results:
            all_results.append({
                'image': img_name,
                'type': result_type,
                'data': result_data
            })
    
    # Find best overall results
    print(f"\n{'='*70}")
    print("BEST RESULTS ACROSS ALL METHODS")
    print(f"{'='*70}")
    
    # Sort by score (different scoring for different types)
    scored_results = []
    for result in all_results:
        if result['type'] == 'sub_pixel':
            score = result['data']['score']
            text = result['data']['decoded']
        elif result['type'] == 'template':
            score = result['data']['confidence']
            text = f"Template match at {result['data']['position']}"
        elif result['type'] == 'ordering':
            score = result['data']['score']
            text = result['data']['decoded']
        else:
            continue
        
        scored_results.append((score, result['image'], result['type'], text, result['data']))
    
    scored_results.sort(reverse=True)
    
    print(f"\nTop 10 results:")
    for i, (score, img_name, method, text, data) in enumerate(scored_results[:10]):
        print(f"  {i+1:2d}. {img_name:15s} {method:12s} {score:.3f} -> '{text}'")
    
    # Save results
    with open('ADVANCED_ITERATION_RESULTS.json', 'w') as f:
        # Convert numpy types for JSON serialization
        json_results = []
        for result in all_results:
            json_result = {
                'image': result['image'],
                'type': result['type'],
                'data': {}
            }
            
            # Convert data to JSON-serializable format
            for key, value in result['data'].items():
                if isinstance(value, np.ndarray):
                    json_result['data'][key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    json_result['data'][key] = float(value)
                else:
                    json_result['data'][key] = value
                    
            json_results.append(json_result)
        
        json.dump(json_results, f, indent=2)
    
    print(f"\nDetailed results saved to ADVANCED_ITERATION_RESULTS.json")
    
    # Check if we found any strong candidates
    best_score = scored_results[0][0] if scored_results else 0
    if best_score > 0.8:
        print(f"\n🎉 STRONG CANDIDATE FOUND!")
        print(f"Best result: {scored_results[0]}")
    elif best_score > 0.6:
        print(f"\n🔍 PROMISING RESULTS - continue refinement")
    else:
        print(f"\n🔄 CONTINUE ITERATION - try more approaches")

if __name__ == "__main__":
    main()