#!/usr/bin/env python
"""
Satoshi Poster Binary Extractor - Validation and Tuning Script

This script provides tools to validate the binary extraction pipeline and tune its parameters
to achieve optimal results.

Key objectives:
- Compare extracted data against reference data
- Visualize extraction results and debug artifacts
- Analyze distribution of 0s and 1s
- Tune pipeline parameters for optimal extraction
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
import yaml
import argparse
from rich import print as rprint
from rich.console import Console
from rich.table import Table

# Add the project root to the path so we can import our modules
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import our modules
from binary_extractor.extractor import run

# Set up console for rich output
console = Console()

def load_reference_data():
    """Load reference data from CSV files"""
    reference_digits_path = Path(project_root) / "recognized_digits.csv"
    reference_overlay_path = Path(project_root) / "overlay_unknown_cells.csv"
    
    # Check if reference files exist
    if not reference_digits_path.exists():
        console.print(f"[yellow]Warning:[/] Reference digits file not found at {reference_digits_path}")
        reference_digits = None
    else:
        reference_digits = pd.read_csv(reference_digits_path)
        console.print(f"Loaded reference digits: {len(reference_digits)} entries")
        console.print(reference_digits.head())
        
    if not reference_overlay_path.exists():
        console.print(f"[yellow]Warning:[/] Reference overlay file not found at {reference_overlay_path}")
        reference_overlay = None
    else:
        reference_overlay = pd.read_csv(reference_overlay_path)
        console.print(f"Loaded reference overlay data: {len(reference_overlay)} entries")
        console.print(reference_overlay.head())
        
    return reference_digits, reference_overlay

def load_pipeline_output(base_dir=None):
    """Load the most recent pipeline output"""
    if base_dir is None:
        # Check for output directories in order of preference
        possible_dirs = [
            Path(project_root) / "binary_extractor" / "output",
            Path(project_root) / "binary_extractor" / "output2",
            Path(project_root) / "output3",
        ]
        
        # Find the most recent directory with output files
        for dir_path in possible_dirs:
            if dir_path.exists():
                # Look for recognized_digits.csv in this directory
                digits_file = dir_path / "recognized_digits.csv"
                overlay_file = dir_path / "overlay_unknown_cells.csv"
                
                if digits_file.exists() or overlay_file.exists():
                    base_dir = dir_path
                    break
    
    if base_dir is None:
        console.print("[red]No output directory found with pipeline results[/]")
        return None, None, None
    
    # Load digits file if it exists
    digits_file = base_dir / "recognized_digits.csv"
    if digits_file.exists():
        pipeline_digits = pd.read_csv(digits_file)
        console.print(f"Loaded pipeline digits from {digits_file}: {len(pipeline_digits)} entries")
        console.print(pipeline_digits.head())
    else:
        pipeline_digits = None
        console.print(f"[yellow]No recognized digits file found at {digits_file}[/]")
    
    # Load overlay file if it exists
    overlay_file = base_dir / "overlay_unknown_cells.csv"
    if overlay_file.exists():
        pipeline_overlay = pd.read_csv(overlay_file)
        console.print(f"Loaded pipeline overlay data from {overlay_file}: {len(pipeline_overlay)} entries")
        console.print(pipeline_overlay.head())
    else:
        pipeline_overlay = None
        console.print(f"[yellow]No overlay data file found at {overlay_file}[/]")
    
    return base_dir, pipeline_digits, pipeline_overlay

def analyze_digit_distribution(df, title="Digit Distribution", save_path=None):
    """Analyze distribution of 0s and 1s in dataset"""
    if df is None:
        console.print(f"[red]Cannot analyze distribution for {title}: No data available[/]")
        return None
    
    # Count 0s and 1s
    digit_counts = df['digit'].value_counts().sort_index()
    
    # Calculate percentages
    total = len(df)
    percentages = digit_counts / total * 100
    
    # Create a DataFrame for display
    distribution_df = pd.DataFrame({
        'Count': digit_counts,
        'Percentage': percentages
    })
    
    # Display statistics
    console.print(f"\n[bold]{title}:[/]")
    console.print(f"Total digits: {total}")
    
    # Create a table for display
    table = Table(title=f"{title} Statistics")
    table.add_column("Digit")
    table.add_column("Count")
    table.add_column("Percentage")
    
    for digit, row in distribution_df.iterrows():
        table.add_row(
            str(digit),
            str(row['Count']),
            f"{row['Percentage']:.1f}%"
        )
    
    console.print(table)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=digit_counts.index, y=digit_counts.values)
    
    # Add count and percentage labels on bars
    for i, (count, pct) in enumerate(zip(digit_counts, percentages)):
        ax.text(i, count/2, f"{count}\n({pct:.1f}%)", 
                ha='center', va='center', fontweight='bold')
    
    plt.title(title)
    plt.xlabel('Digit')
    plt.ylabel('Count')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        console.print(f"[green]Saved distribution plot to {save_path}[/]")
    else:
        plt.show()
    
    return distribution_df

def compare_datasets(pipeline_df, reference_df, title="Comparison with Reference Data", save_path=None):
    """Compare pipeline output with reference data"""
    if pipeline_df is None or reference_df is None:
        console.print(f"[red]Cannot perform {title}: Missing data[/]")
        return
    
    # Merge datasets on row and col
    merged = pd.merge(
        pipeline_df, 
        reference_df, 
        on=['row', 'col'], 
        how='outer',
        suffixes=('_pipeline', '_reference')
    )
    
    # Fill NaN values for better analysis
    merged = merged.fillna({
        'digit_pipeline': -1,  # -1 indicates missing in pipeline
        'digit_reference': -1  # -1 indicates missing in reference
    })
    
    # Convert digits to integers for comparison
    merged['digit_pipeline'] = merged['digit_pipeline'].astype(int)
    merged['digit_reference'] = merged['digit_reference'].astype(int)
    
    # Calculate match status
    merged['match_status'] = 'Unknown'
    
    # Both have valid digits (0 or 1)
    valid_mask = (merged['digit_pipeline'].isin([0, 1])) & (merged['digit_reference'].isin([0, 1]))
    merged.loc[valid_mask & (merged['digit_pipeline'] == merged['digit_reference']), 'match_status'] = 'Match'
    merged.loc[valid_mask & (merged['digit_pipeline'] != merged['digit_reference']), 'match_status'] = 'Mismatch'
    
    # One has a digit, other doesn't
    merged.loc[(merged['digit_pipeline'].isin([0, 1])) & (merged['digit_reference'] == -1), 'match_status'] = 'Extra in Pipeline'
    merged.loc[(merged['digit_pipeline'] == -1) & (merged['digit_reference'].isin([0, 1])), 'match_status'] = 'Missing in Pipeline'
    
    # Calculate statistics
    total_reference = len(reference_df)
    total_pipeline = len(pipeline_df)
    
    match_counts = merged['match_status'].value_counts()
    
    # Calculate metrics
    matches = match_counts.get('Match', 0)
    mismatches = match_counts.get('Mismatch', 0)
    extras = match_counts.get('Extra in Pipeline', 0)
    missing = match_counts.get('Missing in Pipeline', 0)
    
    accuracy = matches / total_reference if total_reference > 0 else 0
    precision = matches / (matches + mismatches + extras) if (matches + mismatches + extras) > 0 else 0
    recall = matches / (matches + mismatches + missing) if (matches + mismatches + missing) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Display results
    console.print(f"\n[bold]{title}:[/]")
    
    # Create a table for display
    table = Table(title=f"{title} Statistics")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_column("Percentage")
    
    table.add_row("Total in reference", str(total_reference), "100.0%")
    table.add_row("Total in pipeline", str(total_pipeline), f"{total_pipeline/total_reference*100:.1f}%")
    table.add_row("Matches", str(matches), f"{matches/total_reference*100:.1f}%")
    table.add_row("Mismatches", str(mismatches), f"{mismatches/total_reference*100:.1f}%")
    table.add_row("Extra in pipeline", str(extras), "-")
    table.add_row("Missing in pipeline", str(missing), f"{missing/total_reference*100:.1f}%")
    
    console.print(table)
    
    # Create a table for metrics
    metrics_table = Table(title="Performance Metrics")
    metrics_table.add_column("Metric")
    metrics_table.add_column("Value")
    
    metrics_table.add_row("Accuracy", f"{accuracy:.4f}")
    metrics_table.add_row("Precision", f"{precision:.4f}")
    metrics_table.add_row("Recall", f"{recall:.4f}")
    metrics_table.add_row("F1 Score", f"{f1:.4f}")
    
    console.print(metrics_table)
    
    # Visualize match status
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=match_counts.index, y=match_counts.values)
    
    # Add count labels on bars
    for i, count in enumerate(match_counts.values):
        ax.text(i, count/2, str(count), ha='center', va='center', fontweight='bold')
    
    plt.title(f"{title} - Match Status")
    plt.xlabel('Status')
    plt.ylabel('Count')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        console.print(f"[green]Saved comparison plot to {save_path}[/]")
    else:
        plt.show()
    
    # Return the merged dataframe for further analysis
    return merged

def compare_overlay_detection(pipeline_df, reference_df, title="Overlay Detection Comparison", save_path=None):
    """Compare overlay detection between pipeline output and reference data"""
    if pipeline_df is None or reference_df is None:
        console.print(f"[red]Cannot perform {title}: Missing data[/]")
        return
    
    # Merge datasets on row and col
    merged = pd.merge(
        pipeline_df, 
        reference_df, 
        on=['row', 'col'], 
        how='outer',
        suffixes=('_pipeline', '_reference')
    )
    
    # Fill NaN values to indicate missing entries
    merged = merged.fillna({
        'row': -1,
        'col': -1
    })
    
    # Calculate match status
    merged['match_status'] = 'Unknown'
    
    # Both have the cell
    both_have = (merged['row'] != -1) & (merged['col'] != -1)
    merged.loc[both_have, 'match_status'] = 'Match'
    
    # Only in pipeline
    if 'row_reference' in merged.columns:
        only_pipeline = (merged['row'] != -1) & (merged['col'] != -1) & pd.isna(merged['row_reference'])
        merged.loc[only_pipeline, 'match_status'] = 'Extra in Pipeline'
    else:
        # If reference columns missing, treat all as matches or only in pipeline
        pass
    
    # Only in reference
    if 'row_pipeline' in merged.columns:
        only_reference = (merged['row'] != -1) & (merged['col'] != -1) & pd.isna(merged['row_pipeline'])
        merged.loc[only_reference, 'match_status'] = 'Missing in Pipeline'
    else:
        # If pipeline columns missing, treat all as matches or only in reference
        pass
    
    # Calculate statistics
    total_reference = len(reference_df)
    total_pipeline = len(pipeline_df)
    
    match_counts = merged['match_status'].value_counts()
    
    # Calculate metrics
    matches = match_counts.get('Match', 0)
    extras = match_counts.get('Extra in Pipeline', 0)
    missing = match_counts.get('Missing in Pipeline', 0)
    
    precision = matches / total_pipeline if total_pipeline > 0 else 0
    recall = matches / total_reference if total_reference > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Display results
    console.print(f"\n[bold]{title}:[/]")
    
    # Create a table for display
    table = Table(title=f"{title} Statistics")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_column("Percentage")
    
    table.add_row("Total in reference", str(total_reference), "100.0%")
    table.add_row("Total in pipeline", str(total_pipeline), f"{total_pipeline/total_reference*100:.1f}%")
    table.add_row("Matches", str(matches), f"{matches/total_reference*100:.1f}%")
    table.add_row("Extra in pipeline", str(extras), "-")
    table.add_row("Missing in pipeline", str(missing), f"{missing/total_reference*100:.1f}%")
    
    console.print(table)
    
    # Create a table for metrics
    metrics_table = Table(title="Performance Metrics")
    metrics_table.add_column("Metric")
    metrics_table.add_column("Value")
    
    metrics_table.add_row("Precision", f"{precision:.4f}")
    metrics_table.add_row("Recall", f"{recall:.4f}")
    metrics_table.add_row("F1 Score", f"{f1:.4f}")
    
    console.print(metrics_table)
    
    # Visualize match status
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=match_counts.index, y=match_counts.values)
    
    # Add count labels on bars
    for i, count in enumerate(match_counts.values):
        ax.text(i, count/2, str(count), ha='center', va='center', fontweight='bold')
    
    plt.title(f"{title} - Match Status")
    plt.xlabel('Status')
    plt.ylabel('Count')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        console.print(f"[green]Saved overlay comparison plot to {save_path}[/]")
    else:
        plt.show()
    
    # Return the merged dataframe for further analysis
    return merged

def display_debug_artifacts(output_dir, save_dir=None):
    """Display debug artifacts from the pipeline output"""
    if output_dir is None:
        console.print("[red]No output directory provided[/]")
        return
    
    # List of common debug artifacts
    debug_files = [
        ("grid_overlay.png", "Grid Detection Overlay"),
        ("bw_mask.png", "Binary Mask"),
        ("cells_color.png", "Classified Cells"),
        ("cells_with_digits.png", "Cells with Detected Digits"),
        ("overlay_mask.png", "Overlay Detection Mask")
    ]
    
    for filename, title in debug_files:
        file_path = output_dir / filename
        if file_path.exists():
            console.print(f"\n[bold]Found {title}:[/] {file_path}")
            
            # Load the image
            img = cv2.imread(str(file_path))
            if img is None:
                console.print(f"[red]Failed to load image: {file_path}[/]")
                continue
            
            # Convert BGR to RGB for display
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Display or save the image
            plt.figure(figsize=(12, 10))
            plt.imshow(img_rgb)
            plt.title(title)
            plt.axis('off')
            plt.tight_layout()
            
            if save_dir:
                save_path = save_dir / filename
                plt.savefig(save_path)
                console.print(f"[green]Saved {title} to {save_path}[/]")
            else:
                plt.show()
        else:
            console.print(f"\n[yellow]{title} not found at {file_path}[/]")

def load_config_file():
    """Load the current configuration"""
    config_path = Path(project_root) / "binary_extractor" / "cfg.yaml"
    if not config_path.exists():
        console.print(f"[red]Config file not found at {config_path}[/]")
        return None
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def save_config_file(config, suffix=None):
    """Save a configuration with an optional suffix"""
    if config is None:
        console.print("[red]No config to save[/]")
        return None
    
    if suffix:
        config_path = Path(project_root) / "binary_extractor" / f"cfg_{suffix}.yaml"
    else:
        config_path = Path(project_root) / "binary_extractor" / "cfg.yaml"
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    console.print(f"[green]Config saved to {config_path}[/]")
    return config_path

def display_config(config):
    """Display the current configuration"""
    if config is None:
        console.print("[red]No configuration to display[/]")
        return
    
    console.print("\n[bold]Current Configuration:[/]")
    
    for section, params in config.items():
        console.print(f"\n[bold]{section}:[/]")
        
        if isinstance(params, dict):
            table = Table(title=section)
            table.add_column("Parameter")
            table.add_column("Value")
            
            for key, value in params.items():
                table.add_row(str(key), str(value))
            
            console.print(table)
        else:
            console.print(f"  {params}")

def run_pipeline_with_config(config=None, output_dir=None, suffix=None):
    """Run the pipeline with a specific configuration"""
    if config is None:
        console.print("[red]No configuration provided[/]")
        return None
    
    # Save the configuration if a suffix is provided
    if suffix:
        config_path = save_config_file(config, suffix)
    else:
        config_path = Path(project_root) / "binary_extractor" / "cfg.yaml"
        # Save the current config temporarily
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    # Set up the output directory
    if output_dir is None:
        if suffix:
            output_dir = Path(project_root) / "binary_extractor" / f"output_{suffix}"
        else:
            output_dir = Path(project_root) / "binary_extractor" / "output"
    
    # Make sure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up the pipeline
    try:
        image_path = Path(project_root) / "satoshi (1).png"
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        console.print(f"[bold yellow]Calling pipeline run() with image: {image_path}, output: {output_dir}[/]")
        run(image_path, output_dir, config)
        # Check for output files
        digits_file = output_dir / "recognized_digits.csv"
        overlay_file = output_dir / "overlay_unknown_cells.csv"
        if not digits_file.exists() or not overlay_file.exists():
            raise RuntimeError(f"Pipeline did not produce expected output files in {output_dir}")
        console.print(f"[green]Pipeline completed. Results saved to {output_dir}[/]")
        return output_dir
    except Exception as e:
        console.print(f"[red]Error running pipeline: {e}[/]")
        return None

def create_tuned_config(base_config, changes, suffix):
    """Create a tuned configuration by applying changes to a base config"""
    if base_config is None:
        console.print("[red]No base configuration provided[/]")
        return None
    
    # Create a deep copy of the base config
    import copy
    tuned_config = copy.deepcopy(base_config)
    
    # Apply changes
    for section, params in changes.items():
        if section not in tuned_config:
            tuned_config[section] = {}
        
        if isinstance(params, dict):
            for key, value in params.items():
                tuned_config[section][key] = value
        else:
            tuned_config[section] = params
    
    # Save the tuned config
    config_path = save_config_file(tuned_config, suffix)
    
    return tuned_config

def main():
    """Main function to run validation and tuning"""
    parser = argparse.ArgumentParser(description="Validate and tune the binary extraction pipeline")
    parser.add_argument("--output-dir", type=str, help="Directory to save output files")
    parser.add_argument("--run-pipeline", action="store_true", help="Run the pipeline with current config")
    parser.add_argument("--tune", action="store_true", help="Perform parameter tuning")
    parser.add_argument("--save-plots", action="store_true", help="Save plots instead of displaying them")
    args = parser.parse_args()
    
    # Create output directory if specified
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]Output will be saved to {output_dir}[/]")
    
    # Load reference data
    console.print("[bold]Loading reference data...[/]")
    reference_digits, reference_overlay = load_reference_data()
    
    # Load pipeline output
    console.print("\n[bold]Loading pipeline output...[/]")
    pipeline_dir, pipeline_digits, pipeline_overlay = load_pipeline_output()

    # If no pipeline output, run the pipeline automatically
    if pipeline_digits is None or pipeline_overlay is None:
        console.print("[yellow]No pipeline output found. Running pipeline automatically...[/]")
        config = load_config_file()
        if config:
            display_config(config)
            pipeline_dir = run_pipeline_with_config(config)
            if pipeline_dir:
                _, pipeline_digits, pipeline_overlay = load_pipeline_output(pipeline_dir)
    
    # Analyze digit distribution
    console.print("\n[bold]Analyzing digit distribution...[/]")
    if args.save_plots and output_dir:
        ref_dist_path = output_dir / "reference_distribution.png"
        pipeline_dist_path = output_dir / "pipeline_distribution.png"
    else:
        ref_dist_path = None
        pipeline_dist_path = None
    
    ref_distribution = analyze_digit_distribution(reference_digits, "Reference Data Distribution", ref_dist_path)
    pipeline_distribution = analyze_digit_distribution(pipeline_digits, "Pipeline Output Distribution", pipeline_dist_path)
    
    # Compare datasets
    console.print("\n[bold]Comparing pipeline output with reference data...[/]")
    if args.save_plots and output_dir:
        comparison_path = output_dir / "digit_comparison.png"
        overlay_comparison_path = output_dir / "overlay_comparison.png"
    else:
        comparison_path = None
        overlay_comparison_path = None
    
    comparison_df = compare_datasets(pipeline_digits, reference_digits, "Digit Recognition Comparison", comparison_path)
    overlay_comparison_df = compare_overlay_detection(pipeline_overlay, reference_overlay, "Overlay Detection Comparison", overlay_comparison_path)
    
    # Display debug artifacts
    console.print("\n[bold]Displaying debug artifacts...[/]")
    if args.save_plots and output_dir:
        artifacts_dir = output_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
    else:
        artifacts_dir = None
    
    display_debug_artifacts(pipeline_dir, artifacts_dir)
    
    # Perform parameter tuning if requested
    if args.tune:
        console.print("\n[bold]Performing parameter tuning...[/]")
        config = load_config_file()
        if config:
            # Example tuning: adjust threshold value
            tuned_config_1 = create_tuned_config(
                config,
                {
                    'preprocessing': {
                        'threshold_value': 120  # Lower threshold value
                    }
                },
                'lower_threshold'
            )
            
            # Run the pipeline with the tuned configuration
            output_dir_1 = run_pipeline_with_config(tuned_config_1, suffix='lower_threshold')
            
            # Load and analyze the results
            if output_dir_1:
                _, pipeline_digits_1, pipeline_overlay_1 = load_pipeline_output(output_dir_1)
                analyze_digit_distribution(pipeline_digits_1, "Lower Threshold - Digit Distribution")
                compare_datasets(pipeline_digits_1, reference_digits, "Lower Threshold - Comparison")
    
    console.print("\n[bold green]Validation and analysis complete![/]")

if __name__ == "__main__":
    main() 