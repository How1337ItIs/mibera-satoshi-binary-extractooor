"""
Unit tests for binary extraction pipeline.
"""
import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import shutil

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from extractor import run, analyze_results, load_config
from extractor.grid import detect_grid, detect_pitch_from_projection
from extractor.classify import classify_cell_bits, classify_single_cell


class TestPipeline:
    """Test the main pipeline functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cfg = load_config()
        
        # Create a simple test image with known binary pattern
        self.test_image = self.create_test_image()
        self.test_image_path = self.temp_dir / "test_image.png"
        cv2.imwrite(str(self.test_image_path), self.test_image)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_image(self) -> np.ndarray:
        """Create a test image with known binary pattern."""
        # Create a 200x200 image with cyan background
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[:, :, 0] = 128  # Blue channel
        img[:, :, 1] = 255  # Green channel (cyan)
        img[:, :, 2] = 128  # Red channel
        
        # Add some binary digits (white on cyan)
        # Row 1: "01010101" pattern
        for i in range(8):
            if i % 2 == 0:
                # Add white pixels for "0"
                img[20:30, 20+i*15:30+i*15] = [255, 255, 255]
            else:
                # Add white pixels for "1" (more pixels)
                img[20:30, 20+i*15:30+i*15] = [255, 255, 255]
                img[25:35, 25+i*15:35+i*15] = [255, 255, 255]
        
        # Row 2: "10101010" pattern
        for i in range(8):
            if i % 2 == 1:
                # Add white pixels for "0"
                img[50:60, 20+i*15:30+i*15] = [255, 255, 255]
            else:
                # Add white pixels for "1" (more pixels)
                img[50:60, 20+i*15:30+i*15] = [255, 255, 255]
                img[55:65, 25+i*15:35+i*15] = [255, 255, 255]
        
        return img
    
    def test_load_config(self):
        """Test configuration loading."""
        cfg = load_config()
        assert isinstance(cfg, dict)
        assert 'blur_sigma' in cfg
        # Removed outdated config key asserts
    
    def test_pipeline_basic(self):
        """Test basic pipeline execution."""
        output_dir = self.temp_dir / "output"
        
        cells = run(self.test_image_path, output_dir, self.cfg)
        
        # Basic assertions
        assert isinstance(cells, list)
        assert len(cells) > 0
        
        # Check that output files were created
        assert (output_dir / "cells.csv").exists()
        assert (output_dir / "bw_mask.png").exists()
    
    def test_analyze_results(self):
        """Test results analysis."""
        # Create sample cells data
        cells = [
            (0, 0, '0'), (0, 1, '1'), (0, 2, '0'),
            (1, 0, '1'), (1, 1, '0'), (1, 2, '1'),
            (2, 0, 'blank'), (2, 1, 'overlay'), (2, 2, '0')
        ]
        
        analysis = analyze_results(cells)
        
        assert analysis["total_cells"] == 9
        assert analysis["zeros"] == 4
        assert analysis["ones"] == 3
        assert analysis["blanks"] == 1
        assert analysis["overlays"] == 1
        assert analysis["legible_digits"] == 7
    
    def test_grid_detection(self):
        """Test grid detection functionality."""
        # Create a simple binary image with regular pattern
        bw_image = np.zeros((100, 100), dtype=np.uint8)
        
        # Add horizontal lines every 20 pixels
        for i in range(0, 100, 20):
            bw_image[i:i+2, :] = 255
        
        # Add vertical lines every 15 pixels
        for j in range(0, 100, 15):
            bw_image[:, j:j+2] = 255
        
        rows, cols = detect_grid(bw_image, self.cfg)
        
        # Should detect some grid lines
        assert len(rows) > 0
        assert len(cols) > 0
    
    def test_pitch_detection(self):
        """Test pitch detection from projections."""
        # Create test projection with known periodicity
        projection = np.zeros(100)
        for i in range(0, 100, 15):  # Period of 15
            projection[i:i+3] = 1
        # Patch: use current signature (projection, axis)
        pitch = detect_pitch_from_projection(projection.reshape(-1, 1), 0)
        
        # Should detect pitch close to 15
        assert 13 <= pitch <= 17
    
    def test_cell_classification(self):
        """Test cell classification."""
        # Test cell with mostly white pixels (should be "1")
        cell_bw = np.ones((10, 10), dtype=np.uint8) * 255
        cell_hsv = np.zeros((10, 10, 3), dtype=np.uint8)
        cell_hsv[:, :, 1] = 128  # Medium saturation
        cell_hsv[:, :, 2] = 128  # Medium value
        # Patch: use top-level bit_hi/bit_lo config
        cfg = load_config()
        overlay = np.zeros((10, 10), dtype=np.uint8)
        result = classify_single_cell(cell_bw, overlay, cfg)
        assert result == '1'
        
        # Test cell with mostly black pixels (should be "0")
        cell_bw = np.zeros((10, 10), dtype=np.uint8)
        result = classify_single_cell(cell_bw, overlay, cfg)
        assert result == '0'
        
        # Test overlay cell (simulate overlay mask)
        overlay[:, :] = 255
        result = classify_single_cell(cell_bw, overlay, cfg)
        assert result == 'overlay'

    def test_classify_matrix_shape(self):
        """Test that classify_cell_bits returns correct matrix shapes, dtypes, and value ranges."""
        # Use a simple test image and grid
        img = np.zeros((40, 40, 3), dtype=np.uint8)
        bw = np.zeros((40, 40), dtype=np.uint8)
        overlay = np.zeros((40, 40), dtype=np.uint8)
        rows = [5, 15, 25, 35]
        cols = [5, 15, 25, 35]
        cfg = load_config()
        cells, bits, conf = classify_cell_bits(img, bw, overlay, rows, cols, cfg, return_matrix=True)
        assert bits.shape == conf.shape == (len(rows), len(cols))
        assert bits.dtype == np.int8
        assert conf.dtype in (np.float32, np.float64)
        allowed = {0, 1, -1, -2}
        assert set(np.unique(bits)).issubset(allowed)
        assert conf.max() <= 1 and conf.min() >= 0
    
    def test_minimum_digits_extracted(self):
        """Test that we extract at least N digits from test image."""
        output_dir = self.temp_dir / "output"
        cells = run(self.test_image_path, output_dir, self.cfg)
        
        # Count legible digits
        digits = [bit for _, _, bit in cells if bit in ['0', '1']]
        
        # Should extract at least 50 digits from our test pattern
        assert len(digits) >= 50, f"Expected >=50 digits, got {len(digits)}"
    
    def test_csv_output_format(self):
        """Test CSV output format."""
        output_dir = self.temp_dir / "output"
        cells = run(self.test_image_path, output_dir, self.cfg)
        
        csv_path = output_dir / "cells.csv"
        assert csv_path.exists()
        
        # Check CSV format
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            assert len(lines) > 1  # Header + data
            assert lines[0].strip() == "row,col,bit"
            
            # Check first data line format
            if len(lines) > 1:
                parts = lines[1].strip().split(',')
                assert len(parts) == 3
                assert parts[0].isdigit()  # row
                assert parts[1].isdigit()  # col
                assert parts[2] in ['0', '1', 'blank', 'overlay']  # bit


class TestConfiguration:
    """Test configuration handling."""
    
    def test_config_parameters(self):
        """Test that all required config parameters exist."""
        cfg = load_config()
        required_keys = [
            'blur_sigma', 'bit_hi', 'bit_lo', 'morph_k',
            'output'
        ]
        for key in required_keys:
            assert key in cfg, f"Missing config key: {key}"
    
    def test_classification_thresholds(self):
        """Test classification threshold configuration."""
        cfg = load_config()
        # Patch: use top-level bit_hi/bit_lo config
        assert 0 < cfg['bit_hi'] < 1
        assert 0 < cfg['bit_lo'] < 1


if __name__ == "__main__":
    pytest.main([__file__]) 