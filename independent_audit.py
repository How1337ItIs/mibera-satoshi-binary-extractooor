#!/usr/bin/env python3
"""
Independent audit of extraction claims.
Fresh analysis without bias from previous "breakthrough" results.
"""

import cv2
import numpy as np
from scipy import signal
import json
from collections import Counter
import hashlib

def fresh_autocorrelation_analysis():
    """Fresh autocorrelation analysis without preconceptions."""
    
    print("=== INDEPENDENT AUTOCORRELATION ANALYSIS ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("Could not load image")
        return None
    
    print(f"Image shape: {img.shape}")
    
    # Test multiple thresholds for mask creation
    thresholds = [127, 150, 180, 200]
    
    for thresh in thresholds:
        print(f"\n--- Threshold {thresh} ---")
        
        mask = img > thresh
        
        # Column projection and autocorrelation
        col_proj = mask.sum(0).astype(float)
        col_proj -= col_proj.mean()
        
        # Full autocorrelation
        autocorr = signal.correlate(col_proj, col_proj, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Keep positive lags only
        
        # Find first significant peak after lag 5
        peaks = []
        for lag in range(5, min(60, len(autocorr))):
            if (lag > 0 and lag < len(autocorr) - 1 and
                autocorr[lag] > autocorr[lag-1] and 
                autocorr[lag] > autocorr[lag+1] and
                autocorr[lag] > 0.1 * autocorr[0]):
                peaks.append((lag, autocorr[lag]))
        
        if peaks:
            # Sort by strength
            peaks.sort(key=lambda x: x[1], reverse=True)
            strongest_peak = peaks[0]
            print(f"  Strongest peak: {strongest_peak[0]} px (strength: {strongest_peak[1]:.1f})")
            
            # Show top 3 peaks
            print(f"  Top peaks: {[(p[0], f'{p[1]:.1f}') for p in peaks[:3]]}")
        else:
            print(f"  No significant peaks found")
    
    # Row projection analysis
    print(f"\n--- Row Analysis ---")
    
    for thresh in [127, 200]:
        mask = img > thresh
        row_proj = mask.sum(1).astype(float)
        row_proj -= row_proj.mean()
        
        autocorr = signal.correlate(row_proj, row_proj, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        peaks = []
        for lag in range(5, min(60, len(autocorr))):
            if (lag > 0 and lag < len(autocorr) - 1 and
                autocorr[lag] > autocorr[lag-1] and 
                autocorr[lag] > autocorr[lag+1] and
                autocorr[lag] > 0.1 * autocorr[0]):
                peaks.append((lag, autocorr[lag]))
        
        if peaks:
            peaks.sort(key=lambda x: x[1], reverse=True)
            print(f"  Threshold {thresh} row peak: {peaks[0][0]} px")
    
    return peaks

def independent_bit_extraction():
    """Independent bit extraction with multiple configurations."""
    
    print(f"\n=== INDEPENDENT BIT EXTRACTION ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Test multiple grid configurations
    configs = [
        ("Previous claim", 101, 53, 31, 53, 72),
        ("Alt position", 100, 50, 31, 53, 127),
        ("Different pitch", 101, 53, 25, 50, 127),
        ("Conservative", 60, 30, 32, 54, 150),
    ]
    
    extraction_results = {}
    
    for config_name, row0, col0, row_pitch, col_pitch, threshold in configs:
        print(f"\n--- {config_name} ---")
        print(f"Position: ({row0}, {col0})")
        print(f"Pitch: {row_pitch} x {col_pitch}")
        print(f"Threshold: {threshold}")
        
        # Extract 200 bits for analysis
        bits = []
        values = []
        positions = []
        
        for i in range(200):
            bit_row = i // 8
            bit_col = i % 8
            
            y = row0 + bit_row * row_pitch
            x = col0 + bit_col * col_pitch
            
            if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                val = img[y, x]
                bit = 1 if val > threshold else 0
                
                bits.append(bit)
                values.append(val)
                positions.append((y, x))
        
        if len(bits) >= 100:
            # Basic statistics
            ones_count = sum(bits)
            ones_ratio = ones_count / len(bits)
            
            # Entropy calculation
            if ones_count > 0 and ones_count < len(bits):
                p1 = ones_count / len(bits)
                p0 = 1 - p1
                entropy = -p1 * np.log2(p1) - p0 * np.log2(p0)
            else:
                entropy = 0
            
            # Value statistics
            val_min, val_max = min(values), max(values)
            val_mean = np.mean(values)
            
            print(f"  Bits extracted: {len(bits)}")
            print(f"  Ones ratio: {ones_count}/{len(bits)} ({ones_ratio:.1%})")
            print(f"  Bit entropy: {entropy:.3f}")
            print(f"  Value range: {val_min}-{val_max} (mean: {val_mean:.1f})")
            
            # Check for bias indicators
            if ones_ratio < 0.2 or ones_ratio > 0.8:
                print(f"    *** EXTRACTION BIAS WARNING ***")
            
            if entropy < 0.8:
                print(f"    *** LOW ENTROPY WARNING ***")
            
            # Store results
            extraction_results[config_name] = {
                'bits': bits,
                'ones_ratio': ones_ratio,
                'entropy': entropy,
                'value_stats': (val_min, val_max, val_mean),
                'config': (row0, col0, row_pitch, col_pitch, threshold)
            }
        else:
            print(f"  Insufficient bits extracted: {len(bits)}")
    
    return extraction_results

def verify_pattern_claims():
    """Verify the claimed 100% pattern matching."""
    
    print(f"\n=== PATTERN MATCHING VERIFICATION ===")
    
    extraction_results = independent_bit_extraction()
    
    # Test patterns claimed to have 100% accuracy
    test_patterns = {
        "satoshi_s": "01110011",
        "version_1": "00000001", 
        "genesis_start": "00000000",
        "difficulty": "11111111"
    }
    
    for config_name, result in extraction_results.items():
        print(f"\n--- {config_name} ---")
        
        bits = result['bits']
        bit_string = ''.join(map(str, bits))
        
        for pattern_name, pattern in test_patterns.items():
            # Search for pattern
            best_score = 0
            best_pos = -1
            
            for start in range(len(bit_string) - len(pattern) + 1):
                segment = bit_string[start:start + len(pattern)]
                matches = sum(1 for i in range(len(pattern)) if segment[i] == pattern[i])
                score = matches / len(pattern)
                
                if score > best_score:
                    best_score = score
                    best_pos = start
            
            print(f"  {pattern_name:12s}: {best_score:.1%} (pos {best_pos})")
            
            if best_score >= 1.0:
                print(f"    *** CLAIMED 100% MATCH VERIFIED ***")
            elif best_score >= 0.8:
                print(f"    *** HIGH MATCH ***")
            elif best_score < 0.6:
                print(f"    *** LOW MATCH - CLAIM QUESTIONABLE ***")

def validate_hash_claims():
    """Independent validation of 9 leading zeros hash claim."""
    
    print(f"\n=== HASH CLAIM VALIDATION ===")
    
    extraction_results = independent_bit_extraction()
    
    for config_name, result in extraction_results.items():
        print(f"\n--- {config_name} ---")
        
        bits = result['bits']
        
        # Convert to bytes
        if len(bits) >= 256:  # Need at least 32 bytes
            byte_data = []
            for i in range(0, len(bits) - 7, 8):
                byte_val = 0
                for j in range(8):
                    if i + j < len(bits):
                        byte_val |= (bits[i + j] << (7 - j))
                byte_data.append(byte_val)
            
            print(f"  Converted to {len(byte_data)} bytes")
            
            # Look for hash candidates (32-byte sequences)
            hash_candidates = []
            
            for start in range(min(len(byte_data) - 31, 10)):
                hash_bytes = bytes(byte_data[start:start + 32])
                hex_hash = hash_bytes.hex()
                leading_zeros = len(hex_hash) - len(hex_hash.lstrip('0'))
                
                if leading_zeros >= 6:
                    hash_candidates.append((start, hex_hash, leading_zeros))
            
            print(f"  Hash candidates with 6+ zeros: {len(hash_candidates)}")
            
            for start, hex_hash, zeros in hash_candidates:
                print(f"    Position {start}: {hex_hash[:32]}... ({zeros} zeros)")
                
                if zeros >= 9:
                    print(f"      *** 9+ ZEROS CLAIM VERIFIED ***")
                elif zeros >= 8:
                    print(f"      *** 8+ ZEROS FOUND ***")
                
                # Calculate probability
                prob = 1 / (16 ** zeros)
                print(f"      Probability: 1 in {16**zeros:,} ({prob:.2e})")

def check_statistical_claims():
    """Check statistical claims about entropy and structure."""
    
    print(f"\n=== STATISTICAL CLAIMS VERIFICATION ===")
    
    extraction_results = independent_bit_extraction()
    
    for config_name, result in extraction_results.items():
        print(f"\n--- {config_name} ---")
        
        bits = result['bits']
        
        if len(bits) >= 100:
            # Byte entropy analysis
            byte_data = []
            for i in range(0, len(bits) - 7, 8):
                byte_val = 0
                for j in range(8):
                    if i + j < len(bits):
                        byte_val |= (bits[i + j] << (7 - j))
                byte_data.append(byte_val)
            
            # Calculate byte entropy
            byte_counter = Counter(byte_data)
            byte_entropy = 0
            for count in byte_counter.values():
                prob = count / len(byte_data)
                if prob > 0:
                    byte_entropy -= prob * np.log2(prob)
            
            # Entropy as percentage of maximum
            entropy_pct = byte_entropy / 8.0 * 100
            
            print(f"  Byte entropy: {byte_entropy:.3f} / 8.0 ({entropy_pct:.1f}%)")
            print(f"  Bit entropy: {result['entropy']:.3f}")
            
            # Chi-squared test for randomness
            expected_freq = len(byte_data) / 256
            chi_squared = sum((count - expected_freq)**2 / expected_freq 
                            for count in byte_counter.values())
            
            print(f"  Chi-squared: {chi_squared:.1f}")
            
            # Evaluate claims
            if entropy_pct > 70:
                print(f"    *** HIGH ENTROPY - STRUCTURED DATA POSSIBLE ***")
            elif entropy_pct < 50:
                print(f"    *** LOW ENTROPY - BIASED EXTRACTION LIKELY ***")
            
            if 50 <= entropy_pct <= 65:
                print(f"    *** MATCHES CLAIMED 56.2% RANGE ***")

def save_audit_results():
    """Save independent audit results."""
    
    print(f"\n=== SAVING AUDIT RESULTS ===")
    
    # Run all analyses
    autocorr_results = fresh_autocorrelation_analysis()
    extraction_results = independent_bit_extraction()
    
    audit_summary = {
        "audit_timestamp": "2024-01-17",
        "autocorrelation_analysis": {
            "strongest_peak": autocorr_results[0][0] if autocorr_results else None,
            "peak_strength": autocorr_results[0][1] if autocorr_results else None
        },
        "extraction_configs_tested": len(extraction_results),
        "extraction_results": {
            config: {
                "ones_ratio": result["ones_ratio"],
                "entropy": result["entropy"],
                "bias_warning": result["ones_ratio"] < 0.2 or result["ones_ratio"] > 0.8,
                "low_entropy_warning": result["entropy"] < 0.8
            } for config, result in extraction_results.items()
        },
        "audit_findings": {
            "autocorr_consistent": autocorr_results[0][0] in [31, 53] if autocorr_results else False,
            "extraction_biased": any(r["ones_ratio"] < 0.2 or r["ones_ratio"] > 0.8 
                                   for r in extraction_results.values()),
            "entropy_suspicious": any(r["entropy"] < 0.8 for r in extraction_results.values())
        }
    }
    
    with open('independent_audit_results.json', 'w') as f:
        json.dump(audit_summary, f, indent=2)
    
    print("Independent audit results saved to independent_audit_results.json")
    
    return audit_summary

if __name__ == "__main__":
    print("Independent Audit of Extraction Claims")
    print("Unbiased verification of reported breakthroughs")
    print("="*60)
    
    # Fresh autocorrelation analysis
    autocorr_results = fresh_autocorrelation_analysis()
    
    # Independent bit extraction
    extraction_results = independent_bit_extraction()
    
    # Verify pattern matching claims
    verify_pattern_claims()
    
    # Validate hash claims
    validate_hash_claims()
    
    # Check statistical claims
    check_statistical_claims()
    
    # Save audit results
    audit_summary = save_audit_results()
    
    print("\n" + "="*60)
    print("INDEPENDENT AUDIT COMPLETE")
    
    # Summary assessment
    if audit_summary["audit_findings"]["extraction_biased"]:
        print("⚠️  EXTRACTION BIAS DETECTED")
    
    if audit_summary["audit_findings"]["entropy_suspicious"]:
        print("⚠️  SUSPICIOUS ENTROPY LEVELS")
    
    if not audit_summary["audit_findings"]["autocorr_consistent"]:
        print("⚠️  AUTOCORRELATION INCONSISTENT WITH CLAIMS")
    
    print("\nAudit complete - check results for verification status")