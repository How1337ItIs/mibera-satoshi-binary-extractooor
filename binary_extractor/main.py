#!/usr/bin/env python3
"""
Main entry point for the Satoshi binary extractor.
"""
import sys
import argparse
from pathlib import Path
from rich.console import Console

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from extractor import run, analyze_results, load_config


def test():
    """Run the test suite."""
    console = Console()
    console.print("[blue]Running tests...[/blue]")
    
    try:
        import pytest
        # Run tests in the tests directory
        test_dir = Path(__file__).parent / "tests"
        result = pytest.main([str(test_dir), "-v"])
        
        if result == 0:
            console.print("[green]All tests passed![/green]")
        else:
            console.print("[red]Some tests failed.[/red]")
            sys.exit(1)
            
    except ImportError:
        console.print("[red]pytest not installed. Install with: pip install pytest[/red]")
        sys.exit(1)


def start(image_path: str = None, output_dir: str = None):
    """Start the extraction pipeline."""
    console = Console()
    
    # Default paths
    if image_path is None:
        image_path = Path(__file__).parent.parent / "satoshi (1).png"
    else:
        image_path = Path(image_path)
    
    if output_dir is None:
        output_dir = Path(__file__).parent / "output"
    else:
        output_dir = Path(output_dir)
    
    # Validate image path
    if not image_path.exists():
        console.print(f"[red]Error: Image file not found: {image_path}[/red]")
        console.print("Please provide the path to the satoshi.png file.")
        sys.exit(1)
    
    console.print(f"[blue]Starting extraction from: {image_path}[/blue]")
    console.print(f"[blue]Output directory: {output_dir}[/blue]")
    
    try:
        # Load configuration
        cfg = load_config()
        
        # Run extraction
        with console.status("[bold green]Processing image..."):
            cells = run(image_path, output_dir, cfg)
        
        # Analyze results
        analysis = analyze_results(cells)
        
        # Display results
        console.print("\n[bold green]Extraction Complete![/bold green]")
        console.print(f"Total cells: {analysis['total_cells']}")
        console.print(f"Legible digits: {analysis['legible_digits']}")
        console.print(f"Overlay percentage: {analysis['overlay_percentage']:.1f}%")
        
        # Show output files
        console.print(f"\n[green]Output files saved to: {output_dir}[/green]")
        for file_path in output_dir.glob("*"):
            if file_path.is_file():
                console.print(f"  • {file_path.name}")
        
    except Exception as e:
        console.print(f"[red]Error during extraction: {e}[/red]")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Satoshi binary extractor - extract binary data from poster image"
    )
    parser.add_argument(
        "command",
        choices=["test", "start"],
        help="Command to run: 'test' or 'start'"
    )
    parser.add_argument(
        "--image",
        help="Path to input image (for 'start' command)"
    )
    parser.add_argument(
        "--output",
        help="Output directory (for 'start' command)"
    )
    
    args = parser.parse_args()
    
    if args.command == "test":
        test()
    elif args.command == "start":
        start(args.image, args.output)


if __name__ == "__main__":
    main() 