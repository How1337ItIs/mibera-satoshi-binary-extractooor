#!/usr/bin/env python3
"""
Computer vision-based extraction techniques for hidden message detection.
Testing alternative approaches beyond simple thresholding.
"""

import cv2
import numpy as np
import json
from scipy import ndimage, signal
from skimage import filters, feature, morphology
import matplotlib.pyplot as plt

def edge_detection_extraction():
    """Extract patterns using edge detection techniques."""
    
    print("=== EDGE DETECTION EXTRACTION ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Apply various edge detection methods
    edge_methods = {
        'sobel': filters.sobel(img),
        'canny': feature.canny(img, sigma=1.0),
        'prewitt': filters.prewitt(img),
        'roberts': filters.roberts(img),
        'laplacian': ndimage.laplace(img)
    }
    
    results = {}
    
    for method_name, edges in edge_methods.items():
        print(f"\n--- {method_name.upper()} EDGE DETECTION ---")
        
        # Convert to binary if needed
        if edges.dtype != bool:
            edge_binary = edges > np.mean(edges)
        else:
            edge_binary = edges
        
        # Test grid extraction on edges
        row0, col0 = 110, 56
        row_pitch, col_pitch = 30, 52
        
        bits = []
        edge_strengths = []
        
        for i in range(256):  # Extract 256 bits
            bit_row = i // 16
            bit_col = i % 16
            
            y = row0 + bit_row * row_pitch
            x = col0 + bit_col * col_pitch
            
            if 0 <= y < edge_binary.shape[0] and 0 <= x < edge_binary.shape[1]:
                # Sample edge strength in small region
                region = edge_binary[max(0,y-2):y+3, max(0,x-2):x+3]
                edge_strength = np.mean(region)
                
                bit = 1 if edge_strength > 0.5 else 0
                bits.append(bit)
                edge_strengths.append(edge_strength)
        
        if bits:
            ones_ratio = sum(bits) / len(bits)
            
            # Calculate entropy
            if 0 < sum(bits) < len(bits):
                p1 = ones_ratio
                p0 = 1 - p1
                entropy = -p1 * np.log2(p1) - p0 * np.log2(p0)
            else:
                entropy = 0
            
            print(f"Bits extracted: {len(bits)}")
            print(f"Ones ratio: {ones_ratio:.1%}")
            print(f"Entropy: {entropy:.3f}")
            print(f"Mean edge strength: {np.mean(edge_strengths):.3f}")
            
            results[method_name] = {
                'bits': bits,
                'ones_ratio': ones_ratio,
                'entropy': entropy,
                'edge_strengths': edge_strengths
            }
    
    return results

def frequency_domain_analysis():
    """Analyze image in frequency domain for hidden patterns."""
    
    print("\n=== FREQUENCY DOMAIN ANALYSIS ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # FFT analysis
    fft = np.fft.fft2(img)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.log(np.abs(fft_shifted) + 1)
    
    print(f"FFT magnitude range: {magnitude.min():.2f} - {magnitude.max():.2f}")
    
    # Look for periodic patterns in frequency domain
    center_y, center_x = magnitude.shape[0] // 2, magnitude.shape[1] // 2
    
    # Extract radial profile
    y, x = np.ogrid[:magnitude.shape[0], :magnitude.shape[1]]
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Bin by distance from center
    r_int = r.astype(int)
    radial_profile = ndimage.mean(magnitude, labels=r_int, index=np.arange(0, r_int.max() + 1))
    
    print(f"Radial profile computed with {len(radial_profile)} bins")
    
    # Look for peaks in radial profile (indicating periodic structure)
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(radial_profile[10:], height=np.mean(radial_profile[10:]))
    
    print(f"Found {len(peaks)} peaks in radial frequency profile")
    for i, peak in enumerate(peaks[:5]):
        freq = peak + 10  # offset for starting at index 10
        print(f"  Peak {i+1}: frequency bin {freq}, magnitude {radial_profile[freq]:.2f}")
    
    # Test grid extraction based on frequency analysis
    if len(peaks) >= 2:
        # Use strongest peaks to estimate grid
        strongest_peaks = peaks[np.argsort(radial_profile[peaks + 10])[-2:]] + 10
        estimated_pitch = img.shape[0] // (strongest_peaks[0] + 1)
        
        print(f"Estimated pitch from frequency analysis: {estimated_pitch}")
        
        # Test extraction with estimated parameters
        bits = []
        for i in range(100):
            bit_row = i // 10
            bit_col = i % 10
            
            y = 100 + bit_row * estimated_pitch
            x = 50 + bit_col * estimated_pitch
            
            if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                bit = 1 if img[y, x] > np.mean(img) else 0
                bits.append(bit)
        
        if bits:
            ones_ratio = sum(bits) / len(bits)
            print(f"FFT-based extraction: {len(bits)} bits, {ones_ratio:.1%} ones")
    
    return {
        'fft_magnitude': magnitude,
        'radial_profile': radial_profile,
        'peaks': peaks,
        'estimated_pitch': estimated_pitch if 'estimated_pitch' in locals() else None
    }

def multi_level_grayscale_encoding():
    """Test multi-level grayscale encoding instead of binary."""
    
    print("\n=== MULTI-LEVEL GRAYSCALE ENCODING ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Test different quantization levels
    quantization_levels = [3, 4, 8, 16]
    
    results = {}
    
    for levels in quantization_levels:
        print(f"\n--- {levels}-LEVEL QUANTIZATION ---")
        
        # Extract values at grid positions
        row0, col0 = 110, 56
        row_pitch, col_pitch = 30, 52
        
        values = []
        quantized_values = []
        
        for i in range(256):
            bit_row = i // 16
            bit_col = i % 16
            
            y = row0 + bit_row * row_pitch
            x = col0 + bit_col * col_pitch
            
            if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                pixel_val = img[y, x]
                values.append(pixel_val)
                
                # Quantize to specified levels
                quantized = int(pixel_val * levels / 256)
                if quantized >= levels:
                    quantized = levels - 1
                quantized_values.append(quantized)
        
        if values:
            print(f"Extracted {len(values)} values")
            print(f"Original range: {min(values)} - {max(values)}")
            print(f"Quantized range: {min(quantized_values)} - {max(quantized_values)}")
            
            # Analyze distribution
            from collections import Counter
            distribution = Counter(quantized_values)
            
            print(f"Level distribution:")
            for level in range(levels):
                count = distribution.get(level, 0)
                pct = count / len(quantized_values) * 100 if quantized_values else 0
                print(f"  Level {level}: {count:3d} ({pct:4.1f}%)")
            
            # Calculate entropy for multi-level
            total = len(quantized_values)
            entropy = 0
            for count in distribution.values():
                if count > 0:
                    p = count / total
                    entropy -= p * np.log2(p)
            
            print(f"Entropy: {entropy:.3f} bits (max: {np.log2(levels):.3f})")
            
            # Test ASCII interpretation of multi-level data
            if levels <= 64:  # Can fit in 6 bits
                ascii_chars = []
                for val in quantized_values[:50]:  # First 50 values
                    ascii_val = 32 + val * (126 - 32) // (levels - 1)  # Map to printable ASCII
                    ascii_chars.append(chr(ascii_val))
                
                ascii_text = ''.join(ascii_chars)
                print(f"ASCII interpretation: '{ascii_text[:30]}...'")
            
            results[f"{levels}_level"] = {
                'values': values,
                'quantized': quantized_values,
                'distribution': dict(distribution),
                'entropy': entropy
            }
    
    return results

def spatial_correlation_analysis():
    """Analyze spatial correlations in the extracted data."""
    
    print("\n=== SPATIAL CORRELATION ANALYSIS ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Extract grid of values
    row0, col0 = 110, 56
    row_pitch, col_pitch = 30, 52
    
    grid_size = 16  # 16x16 grid
    grid_values = np.zeros((grid_size, grid_size))
    
    for row in range(grid_size):
        for col in range(grid_size):
            y = row0 + row * row_pitch
            x = col0 + col * col_pitch
            
            if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                grid_values[row, col] = img[y, x]
    
    print(f"Extracted {grid_size}x{grid_size} grid")
    print(f"Value range: {grid_values.min():.1f} - {grid_values.max():.1f}")
    
    # Analyze spatial correlations
    correlations = {}
    
    # Horizontal correlation
    h_corr = []
    for row in range(grid_size):
        for col in range(grid_size - 1):
            if grid_values[row, col] != 0 and grid_values[row, col + 1] != 0:
                h_corr.append((grid_values[row, col], grid_values[row, col + 1]))
    
    if h_corr:
        h_corr_coef = np.corrcoef([x[0] for x in h_corr], [x[1] for x in h_corr])[0, 1]
        correlations['horizontal'] = h_corr_coef
        print(f"Horizontal correlation: {h_corr_coef:.3f}")
    
    # Vertical correlation
    v_corr = []
    for row in range(grid_size - 1):
        for col in range(grid_size):
            if grid_values[row, col] != 0 and grid_values[row + 1, col] != 0:
                v_corr.append((grid_values[row, col], grid_values[row + 1, col]))
    
    if v_corr:
        v_corr_coef = np.corrcoef([x[0] for x in v_corr], [x[1] for x in v_corr])[0, 1]
        correlations['vertical'] = v_corr_coef
        print(f"Vertical correlation: {v_corr_coef:.3f}")
    
    # Diagonal correlations
    d1_corr = []  # Top-left to bottom-right
    d2_corr = []  # Top-right to bottom-left
    
    for row in range(grid_size - 1):
        for col in range(grid_size - 1):
            if (grid_values[row, col] != 0 and grid_values[row + 1, col + 1] != 0):
                d1_corr.append((grid_values[row, col], grid_values[row + 1, col + 1]))
            
            if (col > 0 and grid_values[row, col] != 0 and grid_values[row + 1, col - 1] != 0):
                d2_corr.append((grid_values[row, col], grid_values[row + 1, col - 1]))
    
    if d1_corr:
        d1_corr_coef = np.corrcoef([x[0] for x in d1_corr], [x[1] for x in d1_corr])[0, 1]
        correlations['diagonal1'] = d1_corr_coef
        print(f"Diagonal (↘) correlation: {d1_corr_coef:.3f}")
    
    if d2_corr:
        d2_corr_coef = np.corrcoef([x[0] for x in d2_corr], [x[1] for x in d2_corr])[0, 1]
        correlations['diagonal2'] = d2_corr_coef
        print(f"Diagonal (↙) correlation: {d2_corr_coef:.3f}")
    
    # Look for periodic patterns
    print(f"\n--- PERIODIC PATTERN ANALYSIS ---")
    
    # Autocorrelation
    flat_values = grid_values.flatten()
    autocorr = np.correlate(flat_values, flat_values, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    
    # Find peaks in autocorrelation
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(autocorr[1:20], height=np.max(autocorr) * 0.1)
    
    print(f"Autocorrelation peaks at lags: {peaks + 1}")
    
    return {
        'grid_values': grid_values,
        'correlations': correlations,
        'autocorr_peaks': peaks + 1 if len(peaks) > 0 else []
    }

def save_computer_vision_results():
    """Save all computer vision analysis results."""
    
    print("\n=== SAVING COMPUTER VISION RESULTS ===")
    
    # Run all analyses
    edge_results = edge_detection_extraction()
    freq_results = frequency_domain_analysis()
    multilevel_results = multi_level_grayscale_encoding()
    spatial_results = spatial_correlation_analysis()
    
    # Compile results
    cv_analysis = {
        "timestamp": "2025-07-17",
        "analysis_type": "computer_vision_extraction",
        "edge_detection": {
            method: {
                "ones_ratio": result["ones_ratio"],
                "entropy": result["entropy"],
                "mean_edge_strength": np.mean(result["edge_strengths"])
            }
            for method, result in edge_results.items()
        },
        "frequency_analysis": {
            "peaks_found": len(freq_results.get("peaks", [])),
            "estimated_pitch": freq_results.get("estimated_pitch"),
            "radial_profile_max": np.max(freq_results["radial_profile"]) if "radial_profile" in freq_results else None
        },
        "multi_level_encoding": {
            f"{levels}_level": {
                "entropy": result["entropy"],
                "distribution": result["distribution"]
            }
            for levels, result in multilevel_results.items()
        },
        "spatial_correlations": spatial_results["correlations"],
        "autocorrelation_peaks": spatial_results["autocorr_peaks"],
        "assessment": "Computer vision techniques tested for alternative extraction methods"
    }
    
    with open('computer_vision_analysis.json', 'w') as f:
        json.dump(cv_analysis, f, indent=2)
    
    print("Computer vision analysis saved to computer_vision_analysis.json")
    
    return cv_analysis

if __name__ == "__main__":
    print("Computer Vision Extraction Techniques")
    print("Testing alternative approaches beyond binary thresholding")
    print("=" * 60)
    
    # Edge detection extraction
    edge_results = edge_detection_extraction()
    
    # Frequency domain analysis
    freq_results = frequency_domain_analysis()
    
    # Multi-level grayscale encoding
    multilevel_results = multi_level_grayscale_encoding()
    
    # Spatial correlation analysis
    spatial_results = spatial_correlation_analysis()
    
    # Save comprehensive results
    cv_analysis = save_computer_vision_results()
    
    print("\n" + "=" * 60)
    print("COMPUTER VISION ANALYSIS COMPLETE")
    
    # Summary of findings
    print(f"\nKey findings:")
    if edge_results:
        best_edge = max(edge_results.items(), key=lambda x: x[1]['entropy'])
        print(f"Best edge method: {best_edge[0]} (entropy: {best_edge[1]['entropy']:.3f})")
    
    if multilevel_results:
        best_multilevel = max(multilevel_results.items(), key=lambda x: x[1]['entropy'])
        print(f"Best quantization: {best_multilevel[0]} (entropy: {best_multilevel[1]['entropy']:.3f})")
    
    if spatial_results['correlations']:
        max_corr = max(spatial_results['correlations'].values())
        print(f"Max spatial correlation: {max_corr:.3f}")
    
    print("Alternative extraction methods provide new perspectives on hidden data")