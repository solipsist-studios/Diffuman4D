"""
Convert COLMAP points3d.bin to PLY format.

This script reads a COLMAP points3d.bin file and exports the 3D points
to a PLY (Polygon File Format) file with color information preserved.

Usage:
    python colmap_points3d_to_ply.py <input_points3d.bin> <output.ply>
"""

import struct
import numpy as np
import argparse
from pathlib import Path
from typing import Tuple, List


def read_points3d_bin(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read COLMAP points3d.bin file.
    
    Args:
        path: Path to points3d.bin file
        
    Returns:
        Tuple of (xyz, rgb, errors) where:
        - xyz: (N, 3) array of 3D point coordinates
        - rgb: (N, 3) array of RGB colors (uint8)
        - errors: (N,) array of reprojection errors
    """
    points = {}
    
    with open(path, "rb") as f:
        num_points = struct.unpack("Q", f.read(8))[0]
        print(f"Reading {num_points} points from {path}")
        
        for _ in range(num_points):
            # Point ID (uint64)
            point_id = struct.unpack("Q", f.read(8))[0]
            
            # XYZ (3x float64)
            xyz = struct.unpack("ddd", f.read(24))
            
            # RGB (3x uint8)
            rgb = struct.unpack("BBB", f.read(3))
            
            # Error (float64)
            error = struct.unpack("d", f.read(8))[0]
            
            # Track length (uint64)
            track_length = struct.unpack("Q", f.read(8))[0]
            
            # Track data (2x uint32 per element)
            track = struct.unpack(f"{2 * track_length}I", f.read(8 * track_length))
            
            points[point_id] = {
                "xyz": xyz,
                "rgb": rgb,
                "error": error,
                "track": track
            }
    
    # Convert to numpy arrays
    point_ids = sorted(points.keys())
    xyz = np.array([points[pid]["xyz"] for pid in point_ids], dtype=np.float64)
    rgb = np.array([points[pid]["rgb"] for pid in point_ids], dtype=np.uint8)
    errors = np.array([points[pid]["error"] for pid in point_ids], dtype=np.float64)
    
    return xyz, rgb, errors


def write_ply(path: str, xyz: np.ndarray, rgb: np.ndarray, errors: np.ndarray = None) -> None:
    """
    Write 3D points to PLY format.
    
    Args:
        path: Output path for PLY file
        xyz: (N, 3) array of 3D coordinates
        rgb: (N, 3) array of RGB colors (uint8)
        errors: (N,) array of reprojection errors (optional)
    """
    num_points = len(xyz)
    
    with open(path, "w") as f:
        # Write PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        
        if errors is not None:
            f.write("property float error\n")
        
        f.write("end_header\n")
        
        # Write vertex data
        if errors is not None:
            for i in range(num_points):
                f.write(f"{xyz[i, 0]:.6f} {xyz[i, 1]:.6f} {xyz[i, 2]:.6f} "
                       f"{rgb[i, 0]} {rgb[i, 1]} {rgb[i, 2]} {errors[i]:.6f}\n")
        else:
            for i in range(num_points):
                f.write(f"{xyz[i, 0]:.6f} {xyz[i, 1]:.6f} {xyz[i, 2]:.6f} "
                       f"{rgb[i, 0]} {rgb[i, 1]} {rgb[i, 2]}\n")
    
    print(f"Wrote {num_points} points to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert COLMAP points3d.bin to PLY format"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to input points3d.bin file"
    )
    parser.add_argument(
        "output",
        type=str,
        help="Path to output PLY file"
    )
    parser.add_argument(
        "--include-errors",
        action="store_true",
        help="Include reprojection errors in PLY output"
    )
    
    args = parser.parse_args()
    
    # Read COLMAP points3d.bin
    xyz, rgb, errors = read_points3d_bin(args.input)
    
    # Write to PLY
    write_ply(
        args.output,
        xyz,
        rgb,
        errors=errors if args.include_errors else None
    )


if __name__ == "__main__":
    main()
