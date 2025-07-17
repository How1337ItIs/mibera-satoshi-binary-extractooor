#!/usr/bin/env python3
"""
CURSOR AGENT: Quick Visual Check
================================

Quick visual check of some extracted cells to verify optimization makes sense.

Created by: Cursor Agent
Date: July 16, 2025
"""

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

def main():
    print("="*50)
    print("CURSOR AGENT: QUICK VISUAL CHECK")
    print("="*50)
    
    # Load data
    try:
        optimized_data = pd.read_csv("cursor_optimized_extraction.csv")
        poster_image = Image.open("satoshi (1).png")
        print("✅ Data and image loaded successfully")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
        
    # Show some sample cells
    print(f"\n=== SAMPLE CELLS ANALYSIS ===")
    
    # Get some zeros and ones
    zeros = optimized_data[optimized_data['bit'] == 0].head(5)
    ones = optimized_data[optimized_data['bit'] == 1].head(5)
    
    print(f"Sample zeros (first 5):")
    for i, (_, row) in enumerate(zeros.iterrows()):
        print(f"  {i+1}. Region {int(row['region_id'])}, Pos ({int(row['local_row'])},{int(row['local_col'])}), "
              f"Blue: {row['blue_mean']:.1f}, Threshold: {row['threshold']}")
              
    print(f"\nSample ones (first 5):")
    for i, (_, row) in enumerate(ones.iterrows()):
        print(f"  {i+1}. Region {int(row['region_id'])}, Pos ({int(row['local_row'])},{int(row['local_col'])}), "
              f"Blue: {row['blue_mean']:.1f}, Threshold: {row['threshold']}")
              
    # Create a simple visual grid
    print(f"\n=== CREATING VISUAL GRID ===")
    
    # Create a 5x2 grid showing some zeros and ones
    cell_size = 80
    grid_width = 5 * cell_size
    grid_height = 2 * cell_size
    
    grid_image = Image.new('RGB', (grid_width, grid_height), 'white')
    draw = ImageDraw.Draw(grid_image)
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
        
    # Add zeros (top row)
    for i, (_, row) in enumerate(zeros.iterrows()):
        if i >= 5:
            break
            
        x = i * cell_size
        y = 0
        
        try:
            cell_img = poster_image.crop((
                int(row['global_x']) - 5, int(row['global_y']) - 5,
                int(row['global_x']) + 6, int(row['global_y']) + 6
            ))
            cell_img = cell_img.resize((cell_size-4, cell_size-4))
            grid_image.paste(cell_img, (x+2, y+2))
            
            # Add labels
            draw.text((x+5, y+5), "0", fill='red', font=font)
            draw.text((x+5, y+cell_size-25), f"B:{row['blue_mean']:.0f}", fill='blue', font=font)
            
        except Exception as e:
            draw.rectangle([x, y, x+cell_size, y+cell_size], outline='red')
            draw.text((x+5, y+5), "ERR", fill='red', font=font)
            
    # Add ones (bottom row)
    for i, (_, row) in enumerate(ones.iterrows()):
        if i >= 5:
            break
            
        x = i * cell_size
        y = cell_size
        
        try:
            cell_img = poster_image.crop((
                int(row['global_x']) - 5, int(row['global_y']) - 5,
                int(row['global_x']) + 6, int(row['global_y']) + 6
            ))
            cell_img = cell_img.resize((cell_size-4, cell_size-4))
            grid_image.paste(cell_img, (x+2, y+2))
            
            # Add labels
            draw.text((x+5, y+5), "1", fill='green', font=font)
            draw.text((x+5, y+cell_size-25), f"B:{row['blue_mean']:.0f}", fill='blue', font=font)
            
        except Exception as e:
            draw.rectangle([x, y, x+cell_size, y+cell_size], outline='red')
            draw.text((x+5, y+5), "ERR", fill='red', font=font)
            
    # Add headers
    draw.text((10, 10), "ZEROS (above threshold 80)", fill='red', font=font)
    draw.text((10, cell_size + 10), "ONES (below threshold 80)", fill='green', font=font)
    
    # Save grid
    filename = "cursor_quick_visual_check.png"
    grid_image.save(filename)
    print(f"Visual check saved: {filename}")
    
    # Analysis
    print(f"\n=== VISUAL ANALYSIS ===")
    
    # Check if blue values make sense
    zero_blue_means = zeros['blue_mean']
    one_blue_means = ones['blue_mean']
    
    print(f"Zero blue means: {zero_blue_means.min():.1f} - {zero_blue_means.max():.1f}")
    print(f"One blue means: {one_blue_means.min():.1f} - {one_blue_means.max():.1f}")
    
    threshold = optimized_data['threshold'].iloc[0]
    print(f"Threshold: {threshold}")
    
    # Check if classification makes sense
    zeros_above_threshold = len(zero_blue_means[zero_blue_means >= threshold])
    ones_below_threshold = len(one_blue_means[one_blue_means < threshold])
    
    print(f"Zeros above threshold: {zeros_above_threshold}/5")
    print(f"Ones below threshold: {ones_below_threshold}/5")
    
    if zeros_above_threshold >= 4 and ones_below_threshold >= 4:
        print("✅ VISUAL CHECK PASSES: Classification appears correct")
    elif zeros_above_threshold >= 3 and ones_below_threshold >= 3:
        print("⚠️ MOSTLY CORRECT: Most classifications appear right")
    else:
        print("❌ VISUAL CHECK FAILS: Classification may be wrong")
        
    print(f"\nReview {filename} to visually confirm the classifications make sense.")

if __name__ == "__main__":
    main() 