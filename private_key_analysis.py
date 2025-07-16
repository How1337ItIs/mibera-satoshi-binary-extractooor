#!/usr/bin/env python3
"""
Analyze the validity and implications of a private key with mostly F's.

Created by Claude Code - July 16, 2025
Purpose: Investigate if a private key of pattern fffffffef... is realistic or extraction artifact
"""
import hashlib
import secrets

def analyze_private_key_pattern():
    """Analyze the extracted private key pattern for realism"""
    
    # The extracted private key
    extracted_key = "fffffffeffffffffffefffffffffffffffffffffffffffffffffffffffffffff"
    
    print("=== PRIVATE KEY PATTERN ANALYSIS ===")
    print(f"Extracted key: {extracted_key}")
    print(f"Length: {len(extracted_key)} hex characters ({len(extracted_key)*4} bits)")
    
    # Analyze the pattern
    f_count = extracted_key.count('f')
    e_count = extracted_key.count('e')
    other_count = len(extracted_key) - f_count - e_count
    
    print(f"\nPattern analysis:")
    print(f"  'f' characters: {f_count}/{len(extracted_key)} ({f_count/len(extracted_key)*100:.1f}%)")
    print(f"  'e' characters: {e_count}/{len(extracted_key)} ({e_count/len(extracted_key)*100:.1f}%)")
    print(f"  Other characters: {other_count}/{len(extracted_key)} ({other_count/len(extracted_key)*100:.1f}%)")
    
    # Check if this is realistic
    print(f"\n=== REALISM ASSESSMENT ===")
    
    # secp256k1 curve order (maximum valid private key)
    secp256k1_order = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    secp256k1_max_hex = hex(secp256k1_order - 1)[2:].upper()
    
    print(f"secp256k1 max valid key: {secp256k1_max_hex}")
    
    # Convert our key to integer
    extracted_int = int(extracted_key, 16)
    print(f"Extracted key as int: {extracted_int}")
    print(f"secp256k1 order:      {secp256k1_order}")
    
    # Check proximity to maximum
    distance_from_max = secp256k1_order - 1 - extracted_int
    print(f"Distance from maximum: {distance_from_max}")
    print(f"As percentage of key space: {(1 - extracted_int/(secp256k1_order-1))*100:.10f}%")
    
    if extracted_int >= secp256k1_order:
        print("❌ INVALID: Key exceeds secp256k1 order")
        return False
    elif distance_from_max < 1000:
        print("⚠️  SUSPICIOUS: Key extremely close to maximum (within 1000)")
        return False
    elif f_count > 50:  # More than ~80% f's
        print("⚠️  SUSPICIOUS: Unrealistically high number of 'f' characters")
        return False
    else:
        print("✅ VALID: Key within reasonable range")
        return True

def analyze_extraction_bias():
    """Analyze if the high F pattern could be an extraction artifact"""
    
    print(f"\n=== EXTRACTION BIAS ANALYSIS ===")
    
    # Our extraction shows 90% ones in binary
    binary_ones_pct = 90.0
    
    # In hex, 'f' = 1111 in binary (all ones)
    # If 90% of bits are 1, what would we expect in hex?
    
    print(f"Binary extraction: {binary_ones_pct}% ones")
    print(f"Expected hex pattern analysis:")
    
    # For each hex digit (4 bits), probability of being 'f' (all 1s)
    prob_all_ones = (binary_ones_pct/100) ** 4
    print(f"  Probability of hex 'f' (4 ones): {prob_all_ones*100:.1f}%")
    
    # Expected number of 'f's in 64-character hex string
    expected_fs = 64 * prob_all_ones
    print(f"  Expected 'f's in 64-char hex: {expected_fs:.1f}")
    
    # Probability of hex 'e' (1110 in binary - 3 ones, 1 zero)
    prob_e = (binary_ones_pct/100)**3 * (1 - binary_ones_pct/100)
    expected_es = 64 * prob_e
    print(f"  Expected 'e's in 64-char hex: {expected_es:.1f}")
    
    # Compare with actual
    extracted_key = "fffffffeffffffffffefffffffffffffffffffffffffffffffffffffffffffff"
    actual_fs = extracted_key.count('f')
    actual_es = extracted_key.count('e')
    
    print(f"\nActual vs Expected:")
    print(f"  'f's: {actual_fs} actual vs {expected_fs:.1f} expected")
    print(f"  'e's: {actual_es} actual vs {expected_es:.1f} expected")
    
    # Statistical test
    if actual_fs > expected_fs * 1.5:
        print("⚠️  ARTIFACT LIKELY: Too many 'f's for 90% bias")
        return True
    else:
        print("✅ PLAUSIBLE: 'f' count consistent with extraction bias")
        return False

def generate_realistic_private_keys():
    """Generate some realistic private keys for comparison"""
    
    print(f"\n=== REALISTIC PRIVATE KEY EXAMPLES ===")
    
    # Generate 3 random private keys
    for i in range(3):
        # Generate 32 random bytes
        private_key_bytes = secrets.token_bytes(32)
        private_key_hex = private_key_bytes.hex()
        
        # Count f's for comparison
        f_count = private_key_hex.count('f')
        
        print(f"Random key {i+1}: {private_key_hex}")
        print(f"  'f' count: {f_count}/64 ({f_count/64*100:.1f}%)")
    
    # Generate a biased key (high probability of 1s)
    print(f"\nSimulated biased extraction (90% ones):")
    biased_bits = ""
    for _ in range(256):
        if secrets.randbelow(100) < 90:  # 90% chance of 1
            biased_bits += "1"
        else:
            biased_bits += "0"
    
    biased_hex = hex(int(biased_bits, 2))[2:].zfill(64)
    biased_f_count = biased_hex.count('f')
    
    print(f"Biased simulation: {biased_hex}")
    print(f"  'f' count: {biased_f_count}/64 ({biased_f_count/64*100:.1f}%)")
    print(f"  Binary ones: {biased_bits.count('1')}/256 ({biased_bits.count('1')/256*100:.1f}%)")

def assess_likelihood():
    """Assess overall likelihood of the extracted pattern"""
    
    print(f"\n" + "="*60)
    print(f"OVERALL LIKELIHOOD ASSESSMENT")
    print(f"="*60)
    
    extracted_key = "fffffffeffffffffffefffffffffffffffffffffffffffffffffffffffffffff"
    
    print(f"\nEXTRACTED PATTERN: {extracted_key}")
    
    # Multiple factors analysis
    factors = []
    
    # Factor 1: F count
    f_count = extracted_key.count('f')
    if f_count > 50:
        factors.append("❌ Extremely high 'f' count (extraction artifact likely)")
    else:
        factors.append("✅ Reasonable 'f' count")
    
    # Factor 2: Pattern regularity
    if "ffffff" in extracted_key:
        long_f_runs = len([m for m in extracted_key.split('f') if len(m) == 0])
        factors.append(f"⚠️  Contains long runs of 'f's ({long_f_runs} consecutive)")
    
    # Factor 3: Bias consistency
    binary_equivalent = bin(int(extracted_key, 16))[2:].zfill(256)
    ones_pct = binary_equivalent.count('1') / 256 * 100
    factors.append(f"📊 Binary: {ones_pct:.1f}% ones (consistent with 90% extraction bias)")
    
    # Factor 4: Cryptographic validity
    secp256k1_order = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    if int(extracted_key, 16) < secp256k1_order:
        factors.append("✅ Cryptographically valid for Bitcoin")
    else:
        factors.append("❌ Exceeds secp256k1 curve order")
    
    print(f"\nASSESSMENT FACTORS:")
    for factor in factors:
        print(f"  {factor}")
    
    print(f"\nCONCLUSION:")
    if f_count > 50:
        print("🚨 LIKELY EXTRACTION ARTIFACT")
        print("   The extremely high proportion of 'f' characters (all-1s patterns)")
        print("   is consistent with systematic bias in the extraction process")
        print("   rather than a genuine private key.")
        print("\n   RECOMMENDATION: Review extraction methodology for bias sources")
    else:
        print("🤔 PLAUSIBLE BUT REQUIRES VERIFICATION")
        print("   Pattern could be genuine but warrants careful validation")

if __name__ == "__main__":
    print("=== PRIVATE KEY REALISM ANALYSIS ===")
    
    # Run all analyses
    is_valid = analyze_private_key_pattern()
    is_artifact = analyze_extraction_bias()
    generate_realistic_private_keys()
    assess_likelihood()
    
    print(f"\n" + "="*60)
    print(f"FINAL ASSESSMENT")
    print(f"="*60)
    
    if is_artifact:
        print("🚨 HIGH LIKELIHOOD OF EXTRACTION ARTIFACT")
        print("The pattern shows characteristics consistent with")
        print("systematic bias in the binary extraction process.")
    else:
        print("🤔 PATTERN REQUIRES FURTHER INVESTIGATION")
        print("While unusual, the pattern is not definitively")
        print("ruled out as a genuine private key.")