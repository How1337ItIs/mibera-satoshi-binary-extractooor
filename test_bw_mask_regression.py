import hashlib
from pathlib import Path
import sys

BW_MASK_PATH = Path('test_results/bw_mask.png')
REFERENCE_HASH_PATH = Path('test_results/bw_mask_reference_hash.txt')


def compute_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    if not BW_MASK_PATH.exists():
        print(f"[ERROR] {BW_MASK_PATH} not found. Run the extraction pipeline first.")
        sys.exit(1)
    current_hash = compute_sha256(BW_MASK_PATH)
    print(f"Current bw_mask.png SHA-256: {current_hash}")
    if not REFERENCE_HASH_PATH.exists():
        print(f"[INFO] Reference hash not found. To set the current hash as reference, run:")
        print(f"echo {current_hash} > {REFERENCE_HASH_PATH}")
        sys.exit(0)
    with open(REFERENCE_HASH_PATH, 'r') as f:
        reference_hash = f.read().strip()
    if current_hash == reference_hash:
        print("[PASS] bw_mask.png matches the reference hash.")
        sys.exit(0)
    else:
        print("[FAIL] bw_mask.png does NOT match the reference hash!")
        print(f"Reference: {reference_hash}")
        print(f"Current:   {current_hash}")
        sys.exit(1)

if __name__ == "__main__":
    main() 