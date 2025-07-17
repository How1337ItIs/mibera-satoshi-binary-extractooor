#!/usr/bin/env python3
"""
Template matching approach to find and extract the binary grid.
Uses computer vision to locate grid patterns and extract bits.
"""

import cv2
import numpy as np
import json
from scipy import signal
from scipy.ndimage import measurements

def find_grid_using_template_matching():
    """Use template matching to find binary grid patterns."""
    
    print("=== TEMPLATE MATCHING GRID DETECTION ===")
    
    img = cv2.imread('mibera_satoshi_poster_highres.png', cv2.IMREAD_GRAYSCALE)
    
    # Create templates for common bit patterns (dark and light squares)
    template_sizes = [5, 6, 7, 8]  # Different possible bit cell sizes
    
    best_matches = []
    
    for size in template_sizes: