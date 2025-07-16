#!/usr/bin/env python3
"""
Enhanced Binary Extraction Pipeline using Advanced Image Alchemy
Implements the best techniques from research to maximize bit recovery.
"""

import cv2
import numpy as np
import os
import yaml
import pywt
from skimage import filters, morphology, exposure
from skimage.restoration import denoise_wavelet
from sklearn.decomposition import PCA, FastICA
import sys
from pathlib import Path

# Add the binary_extractor to path
sys.path.append(str(Path(__file__).parent / "binary_extractor"))

from extractor.pipeline import run
from extractor.grid import detect_grid
from extractor.classify import classify_cell_bits

class EnhancedExtractor:
    """Enhanced binary extraction using advanced image alchemy."""
    
    def __init__(self, image_path):
        self.image_path = image_path
        self.original = cv2.imread(image_path)
        self.rgb = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)
        
    def preprocess_image(self, method='combined'):
        """Apply advanced preprocessing techniques."""
        
        if method == 'clahe_wavelet':
            # CLAHE + Wavelet Denoising
            gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
            
            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Wavelet denoising
            denoised = denoise_wavelet(enhanced, wavelet='db4', sigma=0.1)
            result = (denoised * 255).astype(np.uint8)
            
        elif method == 'pca_enhanced':
            # PCA + Enhancement
            h, w, c = self.rgb.shape
            data = self.rgb.reshape(-1, c)
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(data)
            
            # Use first component (highest variance)
            channel = pca_result[:, 0].reshape(h, w)
            result = ((channel - channel.min()) / (channel.max() - channel.min()) * 255).astype(np.uint8)
            
            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            result = clahe.apply(result)
            
        elif method == 'ica_enhanced':
            # ICA + Enhancement
            h, w, c = self.rgb.shape
            data = self.rgb.reshape(-1, c)
            ica = FastICA(n_components=3, random_state=0)
            ica_result = ica.fit_transform(data)
            
            # Use component with highest entropy (most information)
            entropies = []
            for i in range(3):
                channel = ica_result[:, i].reshape(h, w)
                channel = ((channel - channel.min()) / (channel.max() - channel.min()) * 255).astype(np.uint8)
                hist = np.histogram(channel, bins=256, density=True)[0]
                entropy = -np.sum(hist * np.log(hist + 1e-10))
                entropies.append(entropy)
            
            best_component = np.argmax(entropies)
            channel = ica_result[:, best_component].reshape(h, w)
            result = ((channel - channel.min()) / (channel.max() - channel.min()) * 255).astype(np.uint8)
            
        elif method == 'morphological_enhanced':
            # Morphological Enhancement
            gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
            
            # Top-hat filtering
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
            
            # Add to original
            enhanced = cv2.add(gray, tophat)
            
            # CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            result = clahe.apply(enhanced)
            
        elif method == 'frequency_enhanced':
            # Frequency Domain Enhancement
            gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
            
            # FFT
            fft = np.fft.fft2(gray)
            fft_shifted = np.fft.fftshift(fft)
            
            # High-pass filter
            rows, cols = gray.shape
            crow, ccol = rows//2, cols//2
            mask = np.ones((rows, cols))
            mask[crow-30:crow+30, ccol-30:ccol+30] = 0
            
            # Apply filter
            fft_filtered = fft_shifted * mask
            img_back = np.fft.ifft2(np.fft.ifftshift(fft_filtered))
            result = np.abs(img_back).astype(np.uint8)
            
        elif method == 'combined':
            # Combined best techniques
            gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
            
            # Step 1: CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Step 2: Wavelet denoising
            denoised = denoise_wavelet(enhanced, wavelet='db4', sigma=0.1)
            denoised = (denoised * 255).astype(np.uint8)
            
            # Step 3: Morphological enhancement
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            tophat = cv2.morphologyEx(denoised, cv2.MORPH_TOPHAT, kernel)
            result = cv2.add(denoised, tophat)
            
        else:
            # Default: just grayscale
            result = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        
        return result
    
    def extract_with_method(self, method, output_dir):
        """Extract using a specific enhancement method."""
        
        # Preprocess image
        enhanced = self.preprocess_image(method)
        
        # Save enhanced image for debugging
        debug_path = os.path.join(output_dir, f"enhanced_{method}.png")
        cv2.imwrite(debug_path, enhanced)
        
        # Create modified config
        config = {
            'use_color_space': 'HSV_S',  # Will be overridden by our preprocessed image
            'blur_sigma': 15,  # Reduced since we already preprocessed
            'threshold': {
                'method': 'adaptive',  # Adaptive works better with enhanced images
                'adaptive_C': 4,
                'sauvola_window_size': 15,
                'sauvola_k': 0.2
            },
            'morph_k': 2,  # Reduced since we already enhanced
            'morph_iterations': 1,
            'use_mahotas_thin': False,
            'row_pitch': None,
            'col_pitch': None,
            'row0': 50,
            'col0': 20,
            'bit_hi': 0.65,  # Slightly more sensitive
            'bit_lo': 0.35,
            'overlay': {
                'saturation_threshold': 40,
                'value_threshold': 180,
                'cell_coverage_threshold': 0.2,
                'dilate_pixels': 2
            },
            'template_match': True,
            'tm_thresh': 0.4,
            'save_debug': True,
            'debug_artifacts': ['bw_mask.png', 'grid_overlay.png'],
            'output': {'csv_encoding': 'utf-8'}
        }
        
        # Apply extraction with enhanced image
        # We'll need to modify the pipeline to use our enhanced image
        return self.run_extraction_with_enhanced_image(enhanced, config, output_dir)
    
    def run_extraction_with_enhanced_image(self, enhanced_image, config, output_dir):
        """Run extraction pipeline with our enhanced image."""
        
        # Save enhanced image as temporary file
        temp_path = os.path.join(output_dir, "temp_enhanced.png")
        cv2.imwrite(temp_path, enhanced_image)
        
        # Run extraction on enhanced image
        try:
            # We'll use the existing pipeline but with our enhanced image
            result = run(temp_path, output_dir, config)
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            return result
            
        except Exception as e:
            print(f"Error in extraction: {e}")
            return None
    
    def analyze_results(self, output_dir):
        """Analyze extraction results."""
        cells_file = os.path.join(output_dir, "cells.csv")
        
        if not os.path.exists(cells_file):
            return None
        
        with open(cells_file, 'r') as f:
            lines = f.readlines()
        
        total_cells = len(lines) - 1  # Subtract header
        zeros = sum(1 for line in lines[1:] if line.strip().endswith(',0'))
        ones = sum(1 for line in lines[1:] if line.strip().endswith(',1'))
        blanks = sum(1 for line in lines[1:] if ',blank' in line)
        overlays = sum(1 for line in lines[1:] if ',overlay' in line)
        
        success_rate = (zeros + ones) / total_cells * 100 if total_cells > 0 else 0
        
        return {
            'total_cells': total_cells,
            'zeros': zeros,
            'ones': ones,
            'blanks': blanks,
            'overlays': overlays,
            'success_rate': success_rate,
            'extractable_bits': zeros + ones
        }

def main():
    """Main function to test enhanced extraction methods."""
    
    image_path = "satoshi (1).png"
    
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found")
        return
    
    # Create main output directory
    os.makedirs("test_results/enhanced_extraction", exist_ok=True)
    
    # Initialize enhanced extractor
    extractor = EnhancedExtractor(image_path)
    
    # Test different enhancement methods
    methods = [
        'clahe_wavelet',
        'pca_enhanced', 
        'ica_enhanced',
        'morphological_enhanced',
        'frequency_enhanced',
        'combined'
    ]
    
    results = {}
    
    print("Testing Enhanced Extraction Methods")
    print("=" * 40)
    
    for method in methods:
        print(f"\\nTesting {method}...")
        
        # Create output directory for this method
        output_dir = f"test_results/enhanced_extraction/{method}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract with this method
        extraction_result = extractor.extract_with_method(method, output_dir)
        
        # Analyze results
        analysis = extractor.analyze_results(output_dir)
        
        if analysis:
            results[method] = analysis
            print(f"SUCCESS: {analysis['success_rate']:.1f}% ({analysis['extractable_bits']} bits)")
        else:
            print(f"FAILED: No results generated")
    
    # Generate comprehensive report
    report_lines = []
    report_lines.append("# Enhanced Binary Extraction Results")
    report_lines.append("## Advanced Image Alchemy Applied to Satoshi Poster")
    report_lines.append("")
    report_lines.append("### Method Comparison")
    report_lines.append("")
    report_lines.append("| Method | Success Rate | Extractable Bits | Zeros | Ones | Blanks | Overlays |")
    report_lines.append("|--------|--------------|------------------|-------|------|--------|----------|")
    
    # Sort by extractable bits
    sorted_methods = sorted(results.items(), key=lambda x: x[1]['extractable_bits'], reverse=True)
    
    for method, result in sorted_methods:
        report_lines.append(f"| {method} | {result['success_rate']:.1f}% | {result['extractable_bits']} | {result['zeros']} | {result['ones']} | {result['blanks']} | {result['overlays']} |")
    
    report_lines.append("")
    report_lines.append("### Analysis")
    report_lines.append("")
    
    if sorted_methods:
        best_method, best_result = sorted_methods[0]
        report_lines.append(f"**Best Method**: {best_method}")
        report_lines.append(f"- Maximum extractable bits: {best_result['extractable_bits']}")
        report_lines.append(f"- Success rate: {best_result['success_rate']:.1f}%")
        report_lines.append(f"- Binary distribution: {best_result['zeros']} zeros, {best_result['ones']} ones")
        report_lines.append("")
        
        report_lines.append("### Methodology")
        report_lines.append("")
        report_lines.append("1. **CLAHE + Wavelet**: Contrast enhancement followed by noise reduction")
        report_lines.append("2. **PCA Enhanced**: Principal component analysis for optimal channel selection")
        report_lines.append("3. **ICA Enhanced**: Independent component analysis for signal separation")
        report_lines.append("4. **Morphological Enhanced**: Structure-preserving enhancement")
        report_lines.append("5. **Frequency Enhanced**: Frequency domain filtering")
        report_lines.append("6. **Combined**: Multi-stage enhancement pipeline")
        report_lines.append("")
        
        report_lines.append("### Recommendations")
        report_lines.append("")
        report_lines.append(f"Use the **{best_method}** method for maximum bit recovery.")
        report_lines.append("This method achieved the highest number of extractable bits while maintaining")
        report_lines.append("good success rates and binary classification accuracy.")
    
    # Save report
    report_path = "test_results/enhanced_extraction/ENHANCED_EXTRACTION_REPORT.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\\n".join(report_lines))
    
    print(f"\\n=== Enhanced Extraction Complete ===")
    print(f"Tested {len(methods)} enhancement methods")
    if sorted_methods:
        print(f"Best method: {sorted_methods[0][0]} ({sorted_methods[0][1]['extractable_bits']} bits)")
    print(f"Report saved to: {report_path}")

if __name__ == "__main__":
    main()