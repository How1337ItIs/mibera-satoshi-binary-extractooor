#!/usr/bin/env python3
"""
Quick decode of manually extracted bit sequences.
"""

def decode_bits():
    """Decode the manually extracted bit sequences."""
    
    print("=== DECODING MANUAL BIT SEQUENCES ===")
    
    # Manual bit sequences from user
    bit_sequences = [
        "01100101 01100011 01100101 01101101 01100010 01100101 01110010 00100000 00110010 00110001 00100000 00110010",
        "0100111 01100110 00100000 01110100 01101000 01100101 0010000 01110111 01101001 01101110 10100 01100101",
        "01110010 0010000 01110011 01101111 01101100 01110011 01110100 01101001 01100011 01100101 00100000 01000100",
        "01100101 01100011 01100101 01101101 01100010 01100101 01110010 00100000 00110010 00110001 00100000 00110010",
        "00110000 00110010 00110010 00100000 01110111 01101000 0101001 01101100 01110011 01110100 00100000 01100100",
        "01100101 01100101 01110000 00100000 01101001 01101110 00100000 01110100 0110100001100101 00100000 01100110"
    ]
    
    decoded_lines = []
    
    for i, bit_line in enumerate(bit_sequences):
        print(f"\nLine {i+1}: {bit_line}")
        
        # Clean and split bits
        bits = bit_line.replace(" ", "")
        
        # Convert to bytes and decode
        text = ""
        for j in range(0, len(bits), 8):
            if j + 8 <= len(bits):
                byte_str = bits[j:j+8]
                if len(byte_str) == 8:
                    try:
                        byte_val = int(byte_str, 2)
                        if 32 <= byte_val <= 126:
                            text += chr(byte_val)
                        else:
                            text += f"[{byte_val}]"
                    except:
                        text += "?"
        
        print(f"Decoded: '{text}'")
        decoded_lines.append(text)
    
    # Combine all lines
    full_message = "\n".join(decoded_lines)
    print(f"\n=== FULL MESSAGE ===")
    print(full_message)
    
    return decoded_lines, full_message

if __name__ == "__main__":
    decode_bits()