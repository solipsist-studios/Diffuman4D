#!/usr/bin/env python3
"""
Generate a sequence of experiment YAML files for processing temporal frames in batches.

This script creates multiple YAML configuration files based on a template (like goprotest_tiny.yaml),
with each file configured to process a different batch of 10 frames.

Usage:
    python generate_frame_sequences.py --end-frame 100
    python generate_frame_sequences.py --end-frame 200 --batch-size 15 --template goprotest_small.yaml
"""

import argparse
import yaml
from pathlib import Path
from typing import Dict, Any


def load_yaml(file_path: Path) -> tuple[str, Dict[str, Any]]:
    """Load YAML configuration file, preserving header comments."""
    with open(file_path, 'r') as f:
        content = f.read()
        # Extract header comments (lines starting with #)
        lines = content.split('\n')
        header_lines = []
        yaml_start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('#'):
                header_lines.append(line)
                yaml_start = i + 1
            elif line.strip() and not line.strip().startswith('#'):
                break
        header = '\n'.join(header_lines) if header_lines else ''
        
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return header, config


def save_yaml(data: Dict[str, Any], file_path: Path, header: str = '') -> None:
    """Save YAML configuration file with optional header."""
    with open(file_path, 'w') as f:
        if header:
            f.write(header)
            f.write('\n\n')
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def generate_frame_sequence_configs(
    template_path: Path,
    output_dir: Path,
    end_frame: int,
    batch_size: int = 10,
    start_frame: int = 1,
    output_prefix: str = None
) -> None:
    """
    Generate a sequence of YAML configuration files for frame processing.
    
    Args:
        template_path: Path to the template YAML file
        output_dir: Directory to save generated YAML files
        end_frame: Ending frame number to process
        batch_size: Number of frames to process in each batch (default: 10)
        start_frame: Starting frame number (default: 1)
        output_prefix: Prefix for output filenames (default: template name without extension)
    """
    # Load template with header
    header, template_config = load_yaml(template_path)
    
    # Determine output prefix
    if output_prefix is None:
        output_prefix = template_path.stem
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate configs for each batch
    current_frame = start_frame
    batch_number = 1
    generated_files = []
    
    while current_frame <= end_frame:
        # Calculate frame range for this batch
        batch_end_frame = min(current_frame + batch_size - 1, end_frame)
        
        # Create a deep copy of the template config
        config = yaml.safe_load(yaml.dump(template_config))
        
        # Update the temporal label range
        # Format: [start, end, step]
        config['sampler']['tem_label_range'] = [current_frame, batch_end_frame + 1, 1]
        
        # Generate output filename
        output_filename = f"{output_prefix}_frames_{batch_number:04d}.yaml"
        output_path = output_dir / output_filename
        
        # Save the config with header
        save_yaml(config, output_path, header)
        generated_files.append(output_filename)
        
        print(f"Generated: {output_filename} (frames {current_frame}-{batch_end_frame})")
        
        # Move to next batch
        current_frame = batch_end_frame + 1
        batch_number += 1
    
    print(f"\nTotal files generated: {len(generated_files)}")
    print(f"Output directory: {output_dir}")
    
    # Generate a summary file
    summary_path = output_dir / f"{output_prefix}_sequence_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Frame Sequence Configuration Summary\n")
        f.write(f"=====================================\n\n")
        f.write(f"Template: {template_path.name}\n")
        f.write(f"Total frames: {end_frame-start_frame}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Total batches: {len(generated_files)}\n\n")
        f.write(f"Generated files:\n")
        for filename in generated_files:
            f.write(f"  - {filename}\n")
    
    print(f"\nSummary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a sequence of experiment YAML files for temporal frame processing"
    )
    parser.add_argument(
        '--template',
        type=str,
        default='configs/exp/goprotest_tiny.yaml',
        help='Path to template YAML file (default: configs/exp/goprotest_tiny.yaml)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='configs/exp',
        help='Output directory for generated YAML files (default: configs/exp)'
    )
    parser.add_argument(
        '--end-frame',
        type=int,
        required=True,
        help='Ending frame number to process'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Number of frames per batch (default: 10)'
    )
    parser.add_argument(
        '--start-frame',
        type=int,
        default=1,
        help='Starting frame number (default: 1)'
    )
    parser.add_argument(
        '--output-prefix',
        type=str,
        default=None,
        help='Prefix for output filenames (default: template filename without extension)'
    )
    
    args = parser.parse_args()
    
    # Convert paths
    template_path = Path(args.template)
    output_dir = Path(args.output_dir)
    
    # Validate template exists
    if not template_path.exists():
        print(f"Error: Template file not found: {template_path}")
        return 1
    
    # Generate configs
    generate_frame_sequence_configs(
        template_path=template_path,
        output_dir=output_dir,
        end_frame=args.end_frame,
        batch_size=args.batch_size,
        start_frame=args.start_frame,
        output_prefix=args.output_prefix
    )
    
    return 0


if __name__ == '__main__':
    exit(main())
