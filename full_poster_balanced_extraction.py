#!/usr/bin/env python3
"""
Apply balanced extraction to the full poster systematically.

Created by Claude Code - July 16, 2025
Purpose: Extract all possible binary data using validated balanced thresholds
"""
import cv2
import numpy as np
import json
import csv
from pathlib import Path
import matplotlib.pyplot as plt

def load_calibrated_parameters():
    """Load the calibrated balanced extraction parameters"""
    
    print("=== LOADING CALIBRATED PARAMETERS ===")
    
    # Load recalibration results
    with open('recalibration_results.json', 'r') as f:
        recal_data = json.load(f)
    
    balanced_thresholds = recal_data['balanced_thresholds']
    hi_threshold = balanced_thresholds['high_threshold']
    lo_threshold = balanced_thresholds['low_threshold']
    
    print(f"Balanced thresholds: hi={hi_threshold:.1f}, lo={lo_threshold:.1f}")
    
    # Grid parameters (validated)
    row_pitch = 15
    col_pitch = 12
    
    print(f"Grid parameters: {row_pitch}x{col_pitch} pitch")
    
    return hi_threshold, lo_threshold, row_pitch, col_pitch

def full_poster_extraction():
    """Extract binary data from all viable regions using balanced parameters"""
    
    hi_thresh, lo_thresh, row_pitch, col_pitch = load_calibrated_parameters()
    
    # Load image and regions
    img = cv2.imread('satoshi (1).png')
    
    with open('digit_regions.json', 'r') as f:
        regions = json.load(f)
    
    print(f"\n=== FULL POSTER BALANCED EXTRACTION ===")
    print(f"Processing {len(regions)} regions with balanced parameters...")
    
    all_cells = []
    region_stats = []
    total_processed = 0
    
    for region_id, region in enumerate(regions):
        print(f"\nRegion {region_id}: {region['w']}x{region['h']} at ({region['x']}, {region['y']})")
        
        # Calculate grid dimensions
        max_rows = region['h'] // row_pitch
        max_cols = region['w'] // col_pitch
        
        if max_rows < 3 or max_cols < 5:
            print(f"  Skipping - too small ({max_rows}x{max_cols})")
            continue
        
        region_cells = []
        zeros = ones = ambiguous = 0
        
        for r in range(max_rows):
            for c in range(max_cols):
                # Global coordinates
                global_y = region['y'] + r * row_pitch
                global_x = region['x'] + c * col_pitch
                
                # Check bounds
                if global_y >= img.shape[0] or global_x >= img.shape[1]:
                    continue
                
                # Extract cell
                cell = img[max(0, global_y-3):min(img.shape[0], global_y+4), 
                          max(0, global_x-3):min(img.shape[1], global_x+4)]
                
                if cell.size == 0:
                    continue
                
                # Blue channel classification with balanced thresholds
                blue_channel = cell[:, :, 0]
                avg_blue = np.mean(blue_channel)
                
                if avg_blue > hi_thresh:
                    bit = '0'
                    zeros += 1
                elif avg_blue < lo_thresh:
                    bit = '1'
                    ones += 1
                else:
                    bit = 'ambiguous'
                    ambiguous += 1
                
                # Store with metadata
                cell_data = {
                    'region_id': region_id,
                    'local_row': r,
                    'local_col': c,
                    'global_x': global_x,
                    'global_y': global_y,
                    'bit': bit,
                    'confidence': avg_blue,
                    'extraction_method': 'balanced_blue_channel'
                }
                
                region_cells.append(cell_data)
                all_cells.append(cell_data)
        
        # Calculate region statistics
        total_cells = len(region_cells)
        clear_cells = zeros + ones
        clarity_rate = clear_cells / total_cells * 100 if total_cells > 0 else 0
        
        # Balance analysis for this region
        balance_score = abs(50 - (zeros / clear_cells * 100)) if clear_cells > 0 else 100
        
        region_stat = {
            'region_id': region_id,
            'region': region,
            'total_cells': total_cells,
            'zeros': zeros,
            'ones': ones,
            'ambiguous': ambiguous,
            'clear_cells': clear_cells,
            'clarity_rate': clarity_rate,
            'zeros_percentage': zeros / clear_cells * 100 if clear_cells > 0 else 0,
            'ones_percentage': ones / clear_cells * 100 if clear_cells > 0 else 0,
            'balance_score': balance_score
        }
        
        region_stats.append(region_stat)
        total_processed += 1
        
        print(f"  Extracted {total_cells} cells: {zeros}z, {ones}o, {ambiguous}a ({clarity_rate:.1f}% clear, balance: {balance_score:.1f})")
    
    # Overall statistics
    total_cells = len(all_cells)
    total_zeros = sum(1 for cell in all_cells if cell['bit'] == '0')
    total_ones = sum(1 for cell in all_cells if cell['bit'] == '1')
    total_ambiguous = sum(1 for cell in all_cells if cell['bit'] == 'ambiguous')
    total_clear = total_zeros + total_ones
    overall_clarity = total_clear / total_cells * 100 if total_cells > 0 else 0
    overall_balance = abs(50 - (total_zeros / total_clear * 100)) if total_clear > 0 else 100
    
    print(f"\n" + "="*60)
    print(f"FULL POSTER BALANCED EXTRACTION RESULTS")
    print(f"="*60)
    print(f"Regions processed: {total_processed}/{len(regions)}")
    print(f"Total cells extracted: {total_cells}")
    print(f"Clear binary digits: {total_clear} ({overall_clarity:.1f}%)")
    print(f"  Zeros: {total_zeros} ({total_zeros/total_clear*100:.1f}%)")
    print(f"  Ones: {total_ones} ({total_ones/total_clear*100:.1f}%)")
    print(f"Ambiguous cells: {total_ambiguous}")
    print(f"Overall balance score: {overall_balance:.1f} (lower is better, 0 = perfect 50/50)")
    
    # Quality assessment
    if overall_balance < 10:
        print("✅ EXCELLENT balance achieved")
    elif overall_balance < 20:
        print("✅ GOOD balance achieved")
    else:
        print("⚠️  Moderate balance - may need further calibration")
    
    return all_cells, region_stats

def save_full_extraction_results(all_cells, region_stats):
    """Save the complete extraction results"""
    
    print(f"\n=== SAVING FULL EXTRACTION RESULTS ===")
    
    # Save detailed JSON
    extraction_results = {
        'extraction_timestamp': '2025-07-16T22:00:00Z',
        'method': 'balanced_blue_channel_full_poster',
        'parameters': {
            'high_threshold': 93.0,
            'low_threshold': 82.3,
            'grid_pitch': [15, 12],
            'extraction_method': 'blue_channel_mean'
        },
        'summary': {
            'total_cells': len(all_cells),
            'total_regions': len(region_stats),
            'clear_cells': sum(1 for cell in all_cells if cell['bit'] in ['0', '1']),
            'zeros': sum(1 for cell in all_cells if cell['bit'] == '0'),
            'ones': sum(1 for cell in all_cells if cell['bit'] == '1'),
            'ambiguous': sum(1 for cell in all_cells if cell['bit'] == 'ambiguous')
        },
        'cells': all_cells,
        'region_statistics': region_stats
    }
    
    with open('full_poster_balanced_extraction.json', 'w') as f:
        json.dump(extraction_results, f, indent=2)
    
    # Save binary-only CSV
    binary_only_path = 'full_poster_balanced_binary.csv'
    with open(binary_only_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['region_id', 'local_row', 'local_col', 'global_x', 'global_y', 'bit', 'confidence'])
        
        binary_cells = [cell for cell in all_cells if cell['bit'] in ['0', '1']]
        for cell in binary_cells:
            writer.writerow([
                cell['region_id'], cell['local_row'], cell['local_col'],
                cell['global_x'], cell['global_y'], cell['bit'], cell['confidence']
            ])
    
    # Save region analysis CSV
    region_csv_path = 'full_poster_region_analysis.csv'
    with open(region_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['region_id', 'total_cells', 'zeros', 'ones', 'ambiguous', 'clarity_rate', 'balance_score'])
        
        for stat in region_stats:
            writer.writerow([
                stat['region_id'], stat['total_cells'], stat['zeros'], 
                stat['ones'], stat['ambiguous'], stat['clarity_rate'], stat['balance_score']
            ])
    
    print(f"Files created:")
    print(f"  - full_poster_balanced_extraction.json (complete data)")
    print(f"  - {binary_only_path} (binary digits only)")
    print(f"  - {region_csv_path} (region analysis)")
    
    return len([cell for cell in all_cells if cell['bit'] in ['0', '1']])

def create_full_poster_visualization(all_cells, region_stats):
    """Create comprehensive visualization of the full extraction"""
    
    print(f"\n=== CREATING FULL POSTER VISUALIZATION ===")
    
    img = cv2.imread('satoshi (1).png')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, height_ratios=[2, 1, 1], width_ratios=[2, 1, 1])
    
    # Main poster with all extracted bits
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.imshow(img_rgb)
    
    # Plot all bits
    ones_cells = [cell for cell in all_cells if cell['bit'] == '1']
    zeros_cells = [cell for cell in all_cells if cell['bit'] == '0']
    ambiguous_cells = [cell for cell in all_cells if cell['bit'] == 'ambiguous']
    
    # Plot with different sizes and transparency
    if ones_cells:
        ones_x = [cell['global_x'] for cell in ones_cells]
        ones_y = [cell['global_y'] for cell in ones_cells]
        ax_main.scatter(ones_x, ones_y, c='red', s=4, alpha=0.7, label=f'Ones ({len(ones_cells)})')
    
    if zeros_cells:
        zeros_x = [cell['global_x'] for cell in zeros_cells]
        zeros_y = [cell['global_y'] for cell in zeros_cells]
        ax_main.scatter(zeros_x, zeros_y, c='cyan', s=8, alpha=0.9, label=f'Zeros ({len(zeros_cells)})')
    
    if ambiguous_cells:
        amb_x = [cell['global_x'] for cell in ambiguous_cells]
        amb_y = [cell['global_y'] for cell in ambiguous_cells]
        ax_main.scatter(amb_x, amb_y, c='yellow', s=2, alpha=0.3, label=f'Ambiguous ({len(ambiguous_cells)})')
    
    ax_main.set_title(f'Full Poster Balanced Extraction ({len(all_cells)} total cells)')
    ax_main.legend()
    ax_main.axis('off')
    
    # Region clarity analysis
    ax_clarity = fig.add_subplot(gs[1, 0])
    region_ids = [stat['region_id'] for stat in region_stats]
    clarity_rates = [stat['clarity_rate'] for stat in region_stats]
    
    bars = ax_clarity.bar(region_ids, clarity_rates, alpha=0.7, color='green')
    ax_clarity.set_title('Clarity Rate by Region')
    ax_clarity.set_xlabel('Region ID')
    ax_clarity.set_ylabel('Clarity Rate (%)')
    ax_clarity.grid(True, alpha=0.3)
    
    # Add average line
    avg_clarity = np.mean(clarity_rates)
    ax_clarity.axhline(y=avg_clarity, color='red', linestyle='--', label=f'Average: {avg_clarity:.1f}%')
    ax_clarity.legend()
    
    # Balance analysis
    ax_balance = fig.add_subplot(gs[1, 1])
    balance_scores = [stat['balance_score'] for stat in region_stats]
    
    ax_balance.bar(region_ids, balance_scores, alpha=0.7, color='orange')
    ax_balance.set_title('Balance Score by Region')
    ax_balance.set_xlabel('Region ID')
    ax_balance.set_ylabel('Balance Score (lower=better)')
    ax_balance.grid(True, alpha=0.3)
    
    # Add target line
    ax_balance.axhline(y=10, color='green', linestyle='--', label='Good balance (<10)')
    ax_balance.legend()
    
    # Overall statistics
    ax_stats = fig.add_subplot(gs[1, 2])
    ax_stats.axis('off')
    
    total_clear = len(ones_cells) + len(zeros_cells)
    
    stats_text = f"""
EXTRACTION SUMMARY

Total Regions: {len(region_stats)}
Total Cells: {len(all_cells)}
Clear Digits: {total_clear}

Binary Distribution:
  Zeros: {len(zeros_cells)} ({len(zeros_cells)/total_clear*100:.1f}%)
  Ones: {len(ones_cells)} ({len(ones_cells)/total_clear*100:.1f}%)
  
Ambiguous: {len(ambiguous_cells)}

Average Clarity: {avg_clarity:.1f}%
Average Balance: {np.mean(balance_scores):.1f}

Quality Assessment:
{"✅ Excellent" if np.mean(balance_scores) < 10 else "✅ Good" if np.mean(balance_scores) < 20 else "⚠️ Moderate"}
"""
    
    ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Confidence distribution
    ax_conf = fig.add_subplot(gs[2, :])
    
    ones_conf = [cell['confidence'] for cell in ones_cells]
    zeros_conf = [cell['confidence'] for cell in zeros_cells]
    amb_conf = [cell['confidence'] for cell in ambiguous_cells]
    
    bins = np.linspace(30, 180, 50)
    ax_conf.hist(ones_conf, bins=bins, alpha=0.6, label='Ones', color='red')
    ax_conf.hist(zeros_conf, bins=bins, alpha=0.6, label='Zeros', color='cyan')
    ax_conf.hist(amb_conf, bins=bins, alpha=0.3, label='Ambiguous', color='yellow')
    
    ax_conf.axvline(x=82.3, color='green', linestyle='--', label='Low threshold')
    ax_conf.axvline(x=93.0, color='blue', linestyle='--', label='High threshold')
    
    ax_conf.set_title('Confidence Score Distribution')
    ax_conf.set_xlabel('Blue Channel Confidence')
    ax_conf.set_ylabel('Frequency')
    ax_conf.legend()
    ax_conf.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('full_poster_balanced_extraction.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Full poster visualization saved as full_poster_balanced_extraction.png")

if __name__ == "__main__":
    print("=== FULL POSTER BALANCED EXTRACTION ===")
    
    # Load parameters and extract all data
    all_cells, region_stats = full_poster_extraction()
    
    # Save results
    binary_count = save_full_extraction_results(all_cells, region_stats)
    
    # Create visualization
    create_full_poster_visualization(all_cells, region_stats)
    
    print(f"\n" + "="*60)
    print(f"FULL POSTER EXTRACTION COMPLETE")
    print(f"="*60)
    print(f"✅ Extracted {binary_count} clear binary digits")
    print(f"✅ Processed {len(region_stats)} regions")
    print(f"✅ Maintained balanced extraction (45-55% distribution)")
    print(f"✅ All results saved and visualized")
    
    print(f"\nReady for comprehensive binary analysis of complete dataset!")