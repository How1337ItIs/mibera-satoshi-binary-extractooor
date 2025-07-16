import itertools
import subprocess
import yaml
import csv
from pathlib import Path
from typing import List, Dict, Any

# Parameter grid
BLUR_SIGMAS = [10, 15, 20, 25]
BIT_HIS = [0.65, 0.70, 0.75]
BIT_LOS = [0.25, 0.30, 0.35]
COLOR_SPACES = ['HSV_S', 'Lab_b', 'RGB_inv']
THRESH_METHODS = ['otsu', 'adaptive', 'sauvola']

CFG_PATH = Path('binary_extractor/extractor/cfg.yaml')
RESULTS_CSV = Path('test_results/gridsearch_results.csv')
REFERENCE_CSV = Path('test_results/reference_digits.csv')  # Optional

# Utility to update config
def update_cfg(cfg: Dict[str, Any], blur_sigma, bit_hi, bit_lo, color_space, thresh_method):
    cfg['blur_sigma'] = blur_sigma
    cfg['bit_hi'] = bit_hi
    cfg['bit_lo'] = bit_lo
    cfg['use_color_space'] = color_space
    cfg['threshold']['method'] = thresh_method
    return cfg

def run_pipeline():
    # Run the main pipeline (assumes main.py or similar entry point)
    result = subprocess.run(['python', 'binary_extractor/main.py', 'extract'], capture_output=True, text=True)
    return result.returncode == 0

def score_results():
    # Dummy scoring: count printable ASCII in recognized_digits.csv
    recognized_path = Path('test_results/recognized_digits.csv')
    if not recognized_path.exists():
        return 0, 0
    with open(recognized_path, 'r') as f:
        reader = csv.DictReader(f)
        digits = [row['digit'] for row in reader if 'digit' in row]
    ascii_count = sum(32 <= ord(d) <= 126 for d in digits if len(d) == 1)
    ascii_ratio = ascii_count / len(digits) if digits else 0
    # F1 scoring if reference available
    if REFERENCE_CSV.exists():
        with open(REFERENCE_CSV, 'r') as f:
            ref_digits = { (int(row['row']), int(row['col'])): row['digit'] for row in csv.DictReader(f) }
        with open(recognized_path, 'r') as f:
            test_digits = { (int(row['row']), int(row['col'])): row['digit'] for row in csv.DictReader(f) }
        tp = sum(1 for k in test_digits if k in ref_digits and test_digits[k] == ref_digits[k])
        fp = sum(1 for k in test_digits if k in ref_digits and test_digits[k] != ref_digits[k])
        fn = sum(1 for k in ref_digits if k not in test_digits)
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    else:
        f1 = 0
    return f1, ascii_ratio

def main():
    # Load base config
    with open(CFG_PATH, 'r') as f:
        base_cfg = yaml.safe_load(f)
    # Prepare results CSV
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['blur_sigma', 'bit_hi', 'bit_lo', 'color_space', 'thresh_method', 'f1', 'ascii_ratio'])
        # Grid search
        for blur_sigma, bit_hi, bit_lo, color_space, thresh_method in itertools.product(
            BLUR_SIGMAS, BIT_HIS, BIT_LOS, COLOR_SPACES, THRESH_METHODS):
            cfg = update_cfg(dict(base_cfg), blur_sigma, bit_hi, bit_lo, color_space, thresh_method)
            # Save config
            with open(CFG_PATH, 'w') as f_cfg:
                yaml.dump(cfg, f_cfg)
            print(f"[GRIDSEARCH] blur_sigma={blur_sigma}, bit_hi={bit_hi}, bit_lo={bit_lo}, color_space={color_space}, thresh_method={thresh_method}")
            # Run pipeline
            success = run_pipeline()
            if not success:
                print("[GRIDSEARCH] Pipeline failed, skipping...")
                continue
            # Score results
            f1, ascii_ratio = score_results()
            writer.writerow([blur_sigma, bit_hi, bit_lo, color_space, thresh_method, f1, ascii_ratio])
            print(f"[GRIDSEARCH] F1={f1:.3f}, ASCII ratio={ascii_ratio:.3f}")

if __name__ == "__main__":
    main() 