#!/usr/bin/env python3
"""
CLI script for binary extraction from Satoshi poster.
"""
import argparse
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import track

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from extractor import run, analyze_results, load_config


def main():
    parser = argparse.ArgumentParser(
        description="Extract binary data from Satoshi poster image"
    )
    parser.add_argument(
        "image_path",
        type=Path,
        help="Path to input image file"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for results"
    )
    parser.add_argument(
        "--cfg",
        type=Path,
        help="Path to configuration file (default: extractor/cfg.yaml)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    console = Console()
    
    # Validate inputs
    if not args.image_path.exists():
        console.print(f"[red]Error: Image file not found: {args.image_path}[/red]")
        sys.exit(1)
    
    # Load configuration
    try:
        cfg = load_config(args.cfg)
        if args.verbose:
            console.print(f"[green]Loaded configuration from: {args.cfg or 'default'}[/green]")
    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")
        sys.exit(1)
    
    # Run extraction
    console.print(f"[blue]Starting binary extraction from: {args.image_path}[/blue]")
    
    try:
        with console.status("[bold green]Processing image..."):
            cells = run(args.image_path, args.output_dir, cfg)
        
        # Analyze results
        analysis = analyze_results(cells)
        
        # Display results
        console.print("\n[bold green]Extraction Complete![/bold green]")
        
        # Create results table
        table = Table(title="Extraction Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Cells", str(analysis["total_cells"]))
        table.add_row("Zeros", str(analysis["zeros"]))
        table.add_row("Ones", str(analysis["ones"]))
        table.add_row("Blanks", str(analysis["blanks"]))
        table.add_row("Overlays", str(analysis["overlays"]))
        table.add_row("Confident Cells", str(analysis["confident_cells"]))
        table.add_row("Confidence %", f"{analysis['confidence_percentage']:.1f}%")
        
        # Show accuracy warnings
        if analysis.get("accuracy_warnings"):
            console.print("\n[bold red]ACCURACY WARNINGS:[/bold red]")
            for warning in analysis["accuracy_warnings"]:
                console.print(f"WARNING: {warning}")
        
        # Show critical warning
        if analysis.get("CRITICAL_WARNING"):
            console.print(f"\n[bold red]CRITICAL: {analysis['CRITICAL_WARNING']}[/bold red]")
        
        console.print(table)
        
        # Show output files
        console.print(f"\n[green]Output files saved to: {args.output_dir}[/green]")
        for file_path in args.output_dir.glob("*"):
            if file_path.is_file():
                console.print(f"  • {file_path.name}")
        
        # Show first few ASCII characters if available
        if analysis["legible_digits"] > 0:
            console.print(f"\n[blue]First row ASCII preview:[/blue]")
            from extractor.classify import extract_ascii_from_cells
            
            # Estimate grid size from cell count
            estimated_cols = min(99, analysis["total_cells"] // 10)  # Conservative estimate
            estimated_rows = min(54, analysis["total_cells"] // estimated_cols)
            
            ascii_strings = extract_ascii_from_cells(cells, estimated_rows, estimated_cols)
            if ascii_strings:
                console.print(f"  {ascii_strings[0][:50]}...")
        
    except Exception as e:
        console.print(f"[red]Error during extraction: {e}[/red]")
        if args.verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main() 