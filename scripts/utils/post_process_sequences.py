#!/usr/bin/env python3
"""
Post-process batch inference results by organizing frames and converting transforms.

This script runs post-processing tasks on each output folder generated from batch inference:
1. Organizes camera frames using Organize-Camera-Frames.ps1
2. Converts transforms to COLMAP format using ns2colmap.py

Usage:
    python post_process_sequences.py --input-dir output/results --pattern "goprotest_tiny_frames_*"
    python post_process_sequences.py --input-dir output/results --sequence-name goprotest
    python post_process_sequences.py --config-file configs/exp/sequences/goprotest_tiny_frames_0001-0010.yaml
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional
import re
from datetime import datetime


def extract_frame_range_from_config(config_name: str) -> Optional[tuple]:
    """
    Extract start and end frame numbers from config filename.
    
    Expected format: goprotest_tiny_frames_0001.yaml -> (1, 1)
    Or from reading the config file content if needed.
    """
    # Try to extract from filename pattern like goprotest_tiny_frames_0001
    match = re.search(r'frames_(\d+)', config_name)
    if match:
        frame_num = int(match.group(1))
        # Assume each sequence is 10 frames starting from (frame_num - 1) * 10 + 400
        start_frame = (frame_num - 1) * 10 + 400
        end_frame = start_frame + 9
        return (start_frame, end_frame)
    return None


def run_organize_frames(
    input_path: Path,
    output_base_dir: Path,
    organize_script: Path,
    organize_by: str = "4D"
) -> bool:
    """
    Run the PowerShell frame organization script.
    
    Args:
        input_path: Input path containing the results
        output_base_dir: Base output directory for organized frames
        organize_script: Path to Organize-Camera-Frames.ps1
        organize_by: Organization method (default: "4D")
    
    Returns:
        True if successful, False otherwise
    """
    cmd = [
        "powershell.exe",
        "-ExecutionPolicy", "Bypass",
        "-File", str(organize_script),
        "-InputPath", str(input_path),
        "-OutputBaseDir", str(output_base_dir),
        "-OrganizeBy", organize_by
    ]
    
    print(f"\n{'='*80}")
    print(f"Organizing frames...")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_base_dir}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error organizing frames: {e}", file=sys.stderr)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        return False


def run_ns2colmap(
    transforms_json: Path,
    pose_dir: Path,
    input_dir: Path,
    ns2colmap_script: Path,
    frame_format: str = "undistorted_Frame{0:04d}",
    coordinate_system: str = "OpenCV",
    start_frame: int = 400,
    end_frame: int = 409,
    extract_camera_name: bool = True
) -> bool:
    """
    Run the ns2colmap conversion script.
    
    Args:
        transforms_json: Path to transforms.json file
        pose_dir: Path to directory containing pose files
        input_dir: Output directory for COLMAP files
        ns2colmap_script: Path to ns2colmap.py
        frame_format: Format string for frame naming
        coordinate_system: Coordinate system to use (default: "OpenCV")
        start_frame: Starting frame number
        end_frame: Ending frame number
        extract_camera_name: Whether to extract camera names
    
    Returns:
        True if successful, False otherwise
    """
    cmd = [
        sys.executable,
        str(ns2colmap_script),
        "-i", str(transforms_json),
        "-p", str(pose_dir),
        "-f", frame_format,
        "-o", str(input_dir),
        "--coordinate_system", coordinate_system,
        "--start_frame", str(start_frame),
        "--end_frame", str(end_frame)
    ]
    
    if extract_camera_name:
        cmd.append("--extract_camera_name")
    
    print(f"\n{'='*80}")
    print(f"Converting to COLMAP format...")
    print(f"  Transforms: {transforms_json}")
    print(f"  Frames: {start_frame}-{end_frame}")
    print(f"  Output: {input_dir}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting to COLMAP: {e}", file=sys.stderr)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        return False


def find_result_folders(input_dir: Path, pattern: str = "*") -> List[Path]:
    """Find all output folders matching the pattern."""
    folders = sorted([f for f in input_dir.glob(pattern) if f.is_dir()])
    # Filter out common non-result directories
    folders = [f for f in folders if f.name not in ['logs', 'debug']]
    return folders


def post_process_folder(
    folder: Path,
    sequence_name: str,
    base_output_dir: Path,
    data_dir: Path,
    organize_script: Path,
    ns2colmap_script: Path,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None
) -> bool:
    """
    Post-process a single output folder.
    
    Args:
        folder: Output folder to process
        sequence_name: Name of the sequence (e.g., "goprotest")
        base_output_dir: Base directory for final organized output
        data_dir: Data directory containing source data
        organize_script: Path to Organize-Camera-Frames.ps1
        ns2colmap_script: Path to ns2colmap.py
        start_frame: Starting frame number (if None, will try to extract from folder name)
        end_frame: Ending frame number (if None, will try to extract from folder name)
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'#'*80}")
    print(f"# Post-processing: {folder.name}")
    print(f"{'#'*80}")
    
    # Construct paths
    input_path = folder / sequence_name
    transforms_json = input_path / "transforms.json"
    
    # Check if required files exist
    if not input_path.exists():
        print(f"Warning: Input path does not exist: {input_path}")
        return False
    
    if not transforms_json.exists():
        print(f"Warning: transforms.json not found: {transforms_json}")
        return False
    
    # Extract frame range if not provided
    if start_frame is None or end_frame is None:
        frame_range = extract_frame_range_from_config(folder.name)
        if frame_range:
            start_frame, end_frame = frame_range
            print(f"Extracted frame range: {start_frame}-{end_frame}")
        else:
            print(f"Warning: Could not extract frame range from {folder.name}, using defaults")
            start_frame = start_frame or 400
            end_frame = end_frame or 409
    
    # Run organize frames
    success_organize = run_organize_frames(
        input_path=input_path,
        output_base_dir=base_output_dir,
        organize_script=organize_script
    )
    
    if not success_organize:
        print(f"Failed to organize frames for {folder.name}")
        return False
    
    # Run ns2colmap
    pose_dir = data_dir / sequence_name / "poses_pcd"
    
    if not pose_dir.exists():
        print(f"Warning: Pose directory does not exist: {pose_dir}")
        return False
    
    success_colmap = run_ns2colmap(
        transforms_json=transforms_json,
        pose_dir=pose_dir,
        input_dir=colmap_output,
        ns2colmap_script=ns2colmap_script,
        start_frame=start_frame,
        end_frame=end_frame
    )
    
    if not success_colmap:
        print(f"Failed to convert to COLMAP for {folder.name}")
        return False
    
    print(f"\n✓ Successfully post-processed {folder.name}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Post-process batch inference results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing inference output folders"
    )
    input_group.add_argument(
        "--config-file",
        type=Path,
        help="YAML config file to determine output folders to process"
    )
    
    # Filtering
    parser.add_argument(
        "--pattern",
        type=str,
        default="*",
        help="Pattern to match output folder names (default: '*')"
    )
    
    parser.add_argument(
        "--sequence-name",
        type=str,
        default="goprotest",
        help="Name of the sequence subfolder (default: 'goprotest')"
    )
    
    # Paths
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="Base data directory (default: './data')"
    )
    
    parser.add_argument(
        "--base-output-dir",
        type=Path,
        default=Path("./output/results/goprotest_tiny_4d"),
        help="Base output directory for organized results (default: './output/results/goprotest_tiny_4d')"
    )
    
    parser.add_argument(
        "--organize-script",
        type=Path,
        default=Path("F:/Users/JeffSipko/Dev/utils/Organize-Camera-Frames.ps1"),
        help="Path to Organize-Camera-Frames.ps1"
    )
    
    parser.add_argument(
        "--ns2colmap-script",
        type=Path,
        default=Path("F:/Users/JeffSipko/Dev/utils/ns2colmap.py"),
        help="Path to ns2colmap.py"
    )
    
    # Frame range override
    parser.add_argument(
        "--start-frame",
        type=int,
        help="Override starting frame number (otherwise extracted from folder name)"
    )
    
    parser.add_argument(
        "--end-frame",
        type=int,
        help="Override ending frame number (otherwise extracted from folder name)"
    )
    
    # Behavior
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing even if a folder fails"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without executing"
    )
    
    args = parser.parse_args()
    
    # Determine output folders to process
    if args.input_dir:
        if not args.input_dir.exists():
            print(f"Error: Output directory does not exist: {args.input_dir}", file=sys.stderr)
            return 1
        
        result_folders = find_result_folders(args.input_dir, args.pattern)
    else:
        # Extract from config file
        # Assume output is in output/results/<config_stem>/
        config_stem = args.config_file.stem
        input_dir = Path("output/results")
        result_folders = [input_dir / config_stem] if (input_dir / config_stem).exists() else []
    
    if not result_folders:
        print("No result folders found to process", file=sys.stderr)
        return 1
    
    print(f"Found {len(result_folders)} folder(s) to post-process:")
    for folder in result_folders:
        print(f"  - {folder}")
    
    if args.dry_run:
        print("\nDry run mode - no changes will be made")
        return 0
    
    # Check if scripts exist
    if not args.organize_script.exists():
        print(f"Error: Organize script not found: {args.organize_script}", file=sys.stderr)
        return 1
    
    if not args.ns2colmap_script.exists():
        print(f"Error: ns2colmap script not found: {args.ns2colmap_script}", file=sys.stderr)
        return 1
    
    # Process each folder
    stats = {
        'total': len(result_folders),
        'successful': 0,
        'failed': 0,
        'failed_folders': []
    }
    
    start_time = datetime.now()
    
    for i, folder in enumerate(result_folders, 1):
        print(f"\n[{i}/{stats['total']}] Processing: {folder.name}")
        
        success = post_process_folder(
            folder=folder,
            sequence_name=args.sequence_name,
            base_output_dir=args.base_output_dir,
            data_dir=args.data_dir,
            organize_script=args.organize_script,
            ns2colmap_script=args.ns2colmap_script,
            start_frame=args.start_frame,
            end_frame=args.end_frame
        )
        
        if success:
            stats['successful'] += 1
        else:
            stats['failed'] += 1
            stats['failed_folders'].append(folder.name)
            
            if not args.continue_on_error:
                print("\nStopping due to error")
                break
    
    # Print summary
    elapsed = datetime.now() - start_time
    
    print(f"\n{'='*80}")
    print("POST-PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"Total folders: {stats['total']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Elapsed time: {elapsed}")
    
    if stats['failed_folders']:
        print("\nFailed folders:")
        for folder_name in stats['failed_folders']:
            print(f"  - {folder_name}")
    
    print(f"{'='*80}\n")
    
    return 0 if stats['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
