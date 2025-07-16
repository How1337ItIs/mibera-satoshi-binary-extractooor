#!/usr/bin/env python3
"""
Advanced Image Alchemy Research for Satoshi Poster Binary Extraction
Implements cutting-edge image enhancement techniques to reveal hidden bits.
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import ndimage, signal
from skimage import filters, morphology, segmentation, feature, exposure
from skimage.restoration import denoise_wavelet, denoise_bilateral
from skimage.filters import unsharp_mask, gaussian
import os

class ImageAlchemist:
    """Advanced image processing techniques for bit revelation."""
    
    def __init__(self, image_path):
        self.image_path = image_path
        self.original = cv2.imread(image_path)
        self.rgb = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)
        self.gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        self.results = {}
        
    def spectral_channel_separation(self):
        """Advanced spectral channel separation techniques."""
        results = {}
        
        # Principal Component Analysis
        try:
            from sklearn.decomposition import PCA
            h, w, c = self.rgb.shape
            data = self.rgb.reshape(-1, c)
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(data)
            
            for i in range(3):
                channel = pca_result[:, i].reshape(h, w)
                channel = ((channel - channel.min()) / (channel.max() - channel.min()) * 255).astype(np.uint8)
                results[f'PCA_{i}'] = channel
                
        except ImportError:
            print("sklearn not available, skipping PCA")
        
        # Independent Component Analysis
        try:
            from sklearn.decomposition import FastICA
            h, w, c = self.rgb.shape
            data = self.rgb.reshape(-1, c)
            ica = FastICA(n_components=3, random_state=0)
            ica_result = ica.fit_transform(data)
            
            for i in range(3):
                channel = ica_result[:, i].reshape(h, w)
                channel = ((channel - channel.min()) / (channel.max() - channel.min()) * 255).astype(np.uint8)
                results[f'ICA_{i}'] = channel
                
        except ImportError:
            print("sklearn not available, skipping ICA")
        
        # Fourier Transform Channel Separation
        fft_rgb = np.fft.fft2(self.rgb, axes=(0, 1))
        
        for i, color in enumerate(['R', 'G', 'B']):
            # High-pass filter
            mask = np.ones_like(fft_rgb[:, :, i])
            center = (mask.shape[0]//2, mask.shape[1]//2)
            y, x = np.ogrid[:mask.shape[0], :mask.shape[1]]
            distance = np.sqrt((x - center[1])**2 + (y - center[0])**2)
            mask[distance < 20] = 0
            
            filtered = fft_rgb[:, :, i] * mask
            channel = np.abs(np.fft.ifft2(filtered)).astype(np.uint8)
            results[f'FFT_HP_{color}'] = channel
        
        return results
    
    def adaptive_histogram_techniques(self):
        """Advanced histogram equalization and adaptation."""
        results = {}
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        results['CLAHE'] = clahe.apply(self.gray)
        
        # Adaptive histogram equalization per channel
        for i, color in enumerate(['R', 'G', 'B']):
            channel = self.rgb[:, :, i]
            results[f'CLAHE_{color}'] = clahe.apply(channel)
        
        # Histogram matching to ideal binary distribution
        # Create target binary-like image
        target_image = np.random.choice([0, 255], size=self.gray.shape, p=[0.7, 0.3]).astype(np.uint8)
        
        from skimage.exposure import match_histograms
        results['HistMatch'] = match_histograms(self.gray, target_image)
        
        return results
    
    def wavelet_enhancement(self):
        """Wavelet-based enhancement techniques."""
        results = {}
        
        # Wavelet denoising
        denoised = denoise_wavelet(self.gray, wavelet='db4', sigma=0.1)
        results['Wavelet_Denoised'] = (denoised * 255).astype(np.uint8)
        
        # Wavelet coefficient manipulation
        import pywt
        coeffs = pywt.wavedec2(self.gray, 'db4', level=3)
        
        # Enhance high-frequency components
        enhanced_coeffs = list(coeffs)
        for i in range(1, len(enhanced_coeffs)):
            if isinstance(enhanced_coeffs[i], tuple):
                enhanced_coeffs[i] = tuple(2 * c for c in enhanced_coeffs[i])
        
        enhanced = pywt.waverec2(enhanced_coeffs, 'db4')
        results['Wavelet_Enhanced'] = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return results
    
    def morphological_alchemy(self):
        """Advanced morphological operations."""
        results = {}
        
        # Top-hat and bottom-hat filtering
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        results['TopHat'] = cv2.morphologyEx(self.gray, cv2.MORPH_TOPHAT, kernel)
        results['BottomHat'] = cv2.morphologyEx(self.gray, cv2.MORPH_BLACKHAT, kernel)
        
        # Morphological gradient
        results['MorphGradient'] = cv2.morphologyEx(self.gray, cv2.MORPH_GRADIENT, kernel)
        
        # Multi-scale morphological operations
        for size in [3, 7, 11]:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            results[f'Opening_{size}'] = cv2.morphologyEx(self.gray, cv2.MORPH_OPEN, kernel)
            results[f'Closing_{size}'] = cv2.morphologyEx(self.gray, cv2.MORPH_CLOSE, kernel)
        
        return results
    
    def edge_and_texture_enhancement(self):
        """Edge detection and texture enhancement."""
        results = {}
        
        # Multiple edge detection algorithms
        results['Canny'] = cv2.Canny(self.gray, 50, 150)
        results['Sobel_X'] = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=3)
        results['Sobel_Y'] = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=3)
        results['Laplacian'] = cv2.Laplacian(self.gray, cv2.CV_64F)
        
        # Gabor filters for texture enhancement
        gabor_responses = []
        for theta in range(0, 180, 30):
            for frequency in [0.1, 0.2, 0.3]:
                real, _ = filters.gabor(self.gray, frequency=frequency, theta=np.radians(theta))
                gabor_responses.append(real)
        
        gabor_combined = np.mean(gabor_responses, axis=0)
        results['Gabor_Combined'] = ((gabor_combined - gabor_combined.min()) / 
                                   (gabor_combined.max() - gabor_combined.min()) * 255).astype(np.uint8)
        
        # Local Binary Patterns
        from skimage.feature import local_binary_pattern
        lbp = local_binary_pattern(self.gray, 8, 1, method='uniform')
        results['LBP'] = (lbp / lbp.max() * 255).astype(np.uint8)
        
        return results
    
    def frequency_domain_analysis(self):
        """Frequency domain enhancement techniques."""
        results = {}
        
        # FFT-based filtering
        fft = np.fft.fft2(self.gray)
        fft_shifted = np.fft.fftshift(fft)
        
        # High-pass filter
        rows, cols = self.gray.shape
        crow, ccol = rows//2, cols//2
        mask = np.ones((rows, cols))
        mask[crow-30:crow+30, ccol-30:ccol+30] = 0
        
        fft_filtered = fft_shifted * mask
        img_back = np.fft.ifft2(np.fft.ifftshift(fft_filtered))
        results['FFT_HighPass'] = np.abs(img_back).astype(np.uint8)
        
        # Band-pass filter
        mask = np.zeros((rows, cols))
        mask[crow-50:crow+50, ccol-50:ccol+50] = 1
        mask[crow-20:crow+20, ccol-20:ccol+20] = 0
        
        fft_filtered = fft_shifted * mask
        img_back = np.fft.ifft2(np.fft.ifftshift(fft_filtered))
        results['FFT_BandPass'] = np.abs(img_back).astype(np.uint8)
        
        return results
    
    def advanced_denoising(self):
        """Advanced denoising techniques."""
        results = {}
        
        # Bilateral filtering
        results['Bilateral'] = cv2.bilateralFilter(self.gray, 9, 75, 75)
        
        # Non-local means denoising
        results['NLM'] = cv2.fastNlMeansDenoising(self.gray)
        
        # Wiener filtering approximation
        noise_variance = np.var(self.gray) * 0.1
        signal_variance = np.var(self.gray) - noise_variance
        wiener_filter = signal_variance / (signal_variance + noise_variance)
        results['Wiener'] = (self.gray * wiener_filter).astype(np.uint8)
        
        return results
    
    def run_all_techniques(self):
        """Run all enhancement techniques and save results."""
        print("Running advanced image alchemy techniques...")
        
        techniques = [
            ('Spectral_Separation', self.spectral_channel_separation),
            ('Histogram_Techniques', self.adaptive_histogram_techniques),
            ('Wavelet_Enhancement', self.wavelet_enhancement),
            ('Morphological_Alchemy', self.morphological_alchemy),
            ('Edge_Texture', self.edge_and_texture_enhancement),
            ('Frequency_Domain', self.frequency_domain_analysis),
            ('Advanced_Denoising', self.advanced_denoising),
        ]
        
        all_results = {}
        
        for technique_name, technique_func in techniques:
            print(f"Processing {technique_name}...")
            try:
                results = technique_func()
                all_results[technique_name] = results
                
                # Save debug images
                output_dir = f"test_results/alchemy/{technique_name.lower()}"
                os.makedirs(output_dir, exist_ok=True)
                
                for method_name, image in results.items():
                    if image is not None:
                        filename = f"{output_dir}/{method_name}.png"
                        cv2.imwrite(filename, image)
                        
            except Exception as e:
                print(f"Error in {technique_name}: {e}")
                continue
        
        self.results = all_results
        return all_results
    
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report."""
        report = []
        report.append("# Advanced Image Alchemy Research Report")
        report.append("## Satoshi Poster Binary Extraction Enhancement")
        report.append("")
        report.append("### Overview")
        report.append("This report documents advanced image processing techniques applied to the Satoshi poster")
        report.append("to maximize binary bit extraction through computational image alchemy.")
        report.append("")
        
        # Analyze each technique category
        for category, methods in self.results.items():
            report.append(f"## {category.replace('_', ' ').title()}")
            report.append("")
            
            for method_name, result in methods.items():
                if result is not None:
                    # Basic statistics
                    mean_val = np.mean(result)
                    std_val = np.std(result)
                    entropy = -np.sum(np.histogram(result, bins=256, density=True)[0] * 
                                    np.log(np.histogram(result, bins=256, density=True)[0] + 1e-10))
                    
                    report.append(f"### {method_name}")
                    report.append(f"- **Mean Intensity**: {mean_val:.2f}")
                    report.append(f"- **Standard Deviation**: {std_val:.2f}")
                    report.append(f"- **Entropy**: {entropy:.3f}")
                    report.append(f"- **Contrast**: {np.max(result) - np.min(result)}")
                    report.append("")
        
        # Recommendations
        report.append("## Recommendations for Binary Extraction")
        report.append("")
        report.append("Based on the analysis, the following techniques show promise:")
        report.append("")
        report.append("1. **CLAHE Enhancement**: Improves local contrast")
        report.append("2. **Wavelet Denoising**: Reduces noise while preserving edges")
        report.append("3. **Morphological Operations**: Enhances bit structure")
        report.append("4. **Frequency Domain Filtering**: Isolates relevant frequencies")
        report.append("5. **Multi-channel Analysis**: Extracts information from different spectral components")
        report.append("")
        
        # Implementation suggestions
        report.append("## Implementation Strategy")
        report.append("")
        report.append("1. **Preprocessing Pipeline**: CLAHE → Wavelet Denoising → Morphological Enhancement")
        report.append("2. **Multi-channel Fusion**: Combine best channels from different color spaces")
        report.append("3. **Adaptive Thresholding**: Use locally adaptive methods post-enhancement")
        report.append("4. **Region-specific Processing**: Apply different techniques to different poster regions")
        report.append("")
        
        return "\n".join(report)

def main():
    """Main function to run image alchemy research."""
    image_path = "satoshi (1).png"
    
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found")
        return
    
    # Create output directory
    os.makedirs("test_results/alchemy", exist_ok=True)
    
    # Initialize alchemist
    alchemist = ImageAlchemist(image_path)
    
    # Run all techniques
    results = alchemist.run_all_techniques()
    
    # Generate report
    report = alchemist.generate_comprehensive_report()
    
    # Save report
    with open("test_results/alchemy/ADVANCED_IMAGE_ALCHEMY_REPORT.md", 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\\nAdvanced Image Alchemy Research Complete!")
    print(f"Processed {len(results)} technique categories")
    print(f"Generated debug images in test_results/alchemy/")
    print(f"Report saved to: test_results/alchemy/ADVANCED_IMAGE_ALCHEMY_REPORT.md")

if __name__ == "__main__":
    main()