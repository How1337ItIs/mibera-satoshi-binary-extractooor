"""
extractor/sr.py
Super‑resolves the cyan channel of an input PNG using Real‑ESRGAN (x4 model).
Falls back gracefully if weights or torch+cuda are unavailable.
"""

from pathlib import Path
import subprocess, tempfile, shutil
import cv2
import numpy as np

def run_sr(input_png: Path, out_png: Path, cfg):
    # --- 1. read image & split cyan channel ----------------------------------
    bgr = cv2.imread(str(input_png))
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    cyan = lab[:, :, 2]                             # 'b' channel

    # --- 2. save tmp file -----------------------------------------------------
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_in  = tmp_dir / "cyan.png"
    cv2.imwrite(str(tmp_in), cyan)

    # --- 3. call Real‑ESRGAN --------------------------------------------------
    cmd = [
        "realesrgan-ncnn-vulkan",                  # install via Github release
        "-i", str(tmp_in),
        "-o", str(tmp_dir / "sr.png"),
        "-n", "realesrgan-x4plus",
        "-s", "4",
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("[SR] Real‑ESRGAN not available – skipping SR step")
        shutil.copy(str(tmp_in), str(out_png))
        shutil.rmtree(tmp_dir)
        return

    # --- 4. merge back into colour image (optional) ---------------------------
    sr = cv2.imread(str(tmp_dir / "sr.png"), cv2.IMREAD_GRAYSCALE)
    # Resize original to match SR shape, then swap channel
    h, w   = sr.shape
    bgr_sr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_CUBIC)
    lab_sr = cv2.cvtColor(bgr_sr, cv2.COLOR_BGR2Lab)
    lab_sr[:, :, 2] = sr
    out = cv2.cvtColor(lab_sr, cv2.COLOR_Lab2BGR)

    cv2.imwrite(str(out_png), out)
    shutil.rmtree(tmp_dir) 