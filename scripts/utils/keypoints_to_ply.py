"""
Convert 3D keypoints from JSON format to PLY (point cloud) format for debugging.

Supports COCO Wholebody skeleton format (133 keypoints) with bone connections.
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple


# COCO Wholebody skeleton bone definitions
COCO_WHOLEBODY_SKELETON_BONES = [
    (15, 13),   # 0: left leg
    (13, 11),   # 1: left leg
    (16, 14),   # 2: right leg
    (14, 12),   # 3: right leg
    (11, 12),   # 4: body
    (5, 11),    # 5: body
    (6, 12),    # 6: body
    (5, 6),     # 7: body
    (5, 7),     # 8
    (6, 8),     # 9
    (7, 9),     # 10
    (8, 10),    # 11
    (1, 2),     # 12: left eye to right eye
    (0, 1),     # 13: nose to left eye
    (0, 2),     # 14: nose to right eye
    (1, 3),     # 15: left eye to ear
    (2, 4),     # 16: right eye to ear
    (3, 5),     # 17: left ear to shoulder
    (4, 6),     # 18: right ear to shoulder
    (15, 17),   # 19: left foot
    (15, 18),   # 20: left foot
    (15, 19),   # 21: left foot
    (16, 20),   # 22: right foot
    (16, 21),   # 23: right foot
    (16, 22),   # 24: right foot
    (91, 92),   # 25: left hand
    (92, 93),   # 26
    (93, 94),   # 27
    (94, 95),   # 28
    (91, 96),   # 29
    (96, 97),   # 30
    (97, 98),   # 31
    (98, 99),   # 32
    (91, 100),  # 33
    (100, 101), # 34
    (101, 102), # 35
    (102, 103), # 36
    (91, 104),  # 37
    (104, 105), # 38
    (105, 106), # 39
    (106, 107), # 40
    (91, 108),  # 41
    (108, 109), # 42
    (109, 110), # 43
    (110, 111), # 44
    (112, 113), # 45: right hand
    (113, 114), # 46
    (114, 115), # 47
    (115, 116), # 48
    (112, 117), # 49
    (117, 118), # 50
    (118, 119), # 51
    (119, 120), # 52
    (112, 121), # 53
    (121, 122), # 54
    (122, 123), # 55
    (123, 124), # 56
    (112, 125), # 57
    (125, 126), # 58
    (126, 127), # 59
    (127, 128), # 60
    (112, 129), # 61
    (129, 130), # 62
    (130, 131), # 63
    (131, 132), # 64
]


def read_keypoints_json(json_path: str) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Read 3D keypoints from JSON file.
    
    Returns:
        keypoints: (N, 3) array of 3D coordinates
        scores: (N,) array of confidence scores, or None
        reprojection: (N, 3) array of reprojection errors, or None
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    instance = data['instance_info'][0]
    keypoints = np.array(instance['keypoints'], dtype=np.float32)
    
    scores = None
    if 'keypoint_scores' in instance:
        scores = np.array(instance['keypoint_scores'], dtype=np.float32)
    
    reprojection = None
    if 'keypoint_reproj' in instance:
        reprojection = np.array(instance['keypoint_reproj'], dtype=np.float32)
    
    return keypoints, scores, reprojection


def create_ply_with_colors(
    keypoints: np.ndarray,
    scores: Optional[np.ndarray] = None,
    reprojection: Optional[np.ndarray] = None,
    include_bones: bool = True,
) -> str:
    """Create PLY file content with vertices and optionally bone edges.
    
    Args:
        keypoints: (N, 3) array of 3D keypoint coordinates
        scores: (N,) array of confidence scores (used for coloring)
        reprojection: (N, 3) array of reprojection errors
        include_bones: If True, add bone connections as edges
    
    Returns:
        PLY file content as string
    """
    N = len(keypoints)
    vertices = []
    edges = []
    
    # Create vertices with colors based on scores
    if scores is not None:
        # Normalize scores to 0-255 for RGB
        normalized_scores = np.clip(scores, 0, 1)
        colors = np.zeros((N, 3), dtype=np.uint8)
        # Green to Red gradient: high confidence = green, low confidence = red
        colors[:, 0] = (normalized_scores * 255).astype(np.uint8)  # R
        colors[:, 1] = ((1 - normalized_scores) * 255).astype(np.uint8)  # G
        colors[:, 2] = 0  # B
    else:
        # Default blue if no scores
        colors = np.full((N, 3), [0, 0, 255], dtype=np.uint8)
    
    # Create vertex entries
    for i, (kp, color) in enumerate(zip(keypoints, colors)):
        vertices.append(f"{kp[0]:.6f} {kp[1]:.6f} {kp[2]:.6f} {int(color[0])} {int(color[1])} {int(color[2])}")
    
    # Add bone edges
    if include_bones:
        for bone_idx, (p1_idx, p2_idx) in enumerate(COCO_WHOLEBODY_SKELETON_BONES):
            if p1_idx < N and p2_idx < N:
                edges.append((p1_idx, p2_idx))
    
    # Build PLY header
    num_vertices = N
    num_edges = len(edges)
    
    ply_content = "ply\n"
    ply_content += "format ascii 1.0\n"
    ply_content += "comment Created from 3D keypoint JSON\n"
    
    if scores is not None:
        ply_content += "comment Vertex colors represent confidence scores (green=high, red=low)\n"
    
    ply_content += f"element vertex {num_vertices}\n"
    ply_content += "property float x\n"
    ply_content += "property float y\n"
    ply_content += "property float z\n"
    ply_content += "property uchar red\n"
    ply_content += "property uchar green\n"
    ply_content += "property uchar blue\n"
    
    if include_bones and num_edges > 0:
        ply_content += f"element edge {num_edges}\n"
        ply_content += "property int vertex1\n"
        ply_content += "property int vertex2\n"
    
    ply_content += "end_header\n"
    
    # Add vertex data
    for vertex in vertices:
        ply_content += vertex + "\n"
    
    # Add edge data
    for vertex1, vertex2 in edges:
        ply_content += f"{vertex1} {vertex2}\n"
    
    return ply_content


def keypoints_json_to_ply(
    json_path: str,
    output_ply_path: str,
    include_bones: bool = True,
    verbose: bool = True,
) -> None:
    """Convert 3D keypoints JSON to PLY file.
    
    Args:
        json_path: Path to input JSON file
        output_ply_path: Path to output PLY file
        include_bones: If True, add skeleton bone connections
        verbose: If True, print debug information
    """
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_ply_path) or '.', exist_ok=True)
    
    # Read keypoints
    keypoints, scores, reprojection = read_keypoints_json(json_path)
    
    if verbose:
        print(f"Read {len(keypoints)} keypoints from {json_path}")
        if scores is not None:
            print(f"  Score range: {scores.min():.3f} - {scores.max():.3f}")
        if reprojection is not None:
            print(f"  Reprojection range: {reprojection.min():.6f} - {reprojection.max():.6f}")
        print(f"  Keypoint coordinate ranges:")
        print(f"    X: {keypoints[:, 0].min():.3f} - {keypoints[:, 0].max():.3f}")
        print(f"    Y: {keypoints[:, 1].min():.3f} - {keypoints[:, 1].max():.3f}")
        print(f"    Z: {keypoints[:, 2].min():.3f} - {keypoints[:, 2].max():.3f}")
    
    # Create PLY content
    ply_content = create_ply_with_colors(
        keypoints,
        scores=scores,
        reprojection=reprojection,
        include_bones=include_bones,
    )
    
    # Write PLY file
    with open(output_ply_path, 'w') as f:
        f.write(ply_content)
    
    if verbose:
        num_edges = len(COCO_WHOLEBODY_SKELETON_BONES) if include_bones else 0
        print(f"Wrote PLY with {len(keypoints)} vertices and {num_edges} edges to {output_ply_path}")


def batch_convert(
    input_dir: str,
    output_dir: str,
    pattern: str = "*.json",
    include_bones: bool = True,
    verbose: bool = True,
) -> None:
    """Batch convert all JSON files in a directory to PLY format.
    
    Args:
        input_dir: Directory containing JSON files
        output_dir: Directory to save PLY files
        pattern: File pattern to match (default: "*.json")
        include_bones: If True, add skeleton bone connections
        verbose: If True, print debug information
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all matching JSON files
    json_files = sorted(input_path.glob(pattern))
    
    if not json_files:
        print(f"No JSON files found matching pattern '{pattern}' in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to convert")
    
    for i, json_file in enumerate(json_files):
        try:
            ply_file = output_path / json_file.relative_to(input_path).with_suffix('.ply')
            ply_file.parent.mkdir(parents=True, exist_ok=True)
            
            if verbose:
                print(f"\n[{i+1}/{len(json_files)}] Converting {json_file.name}...")
            
            keypoints_json_to_ply(str(json_file), str(ply_file), include_bones=include_bones, verbose=verbose)
        
        except Exception as e:
            print(f"Error converting {json_file}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert 3D keypoints from JSON to PLY format for debugging skeleton triangulation"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input JSON file or directory containing JSON files"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output PLY file or directory (default: input path with .ply extension)"
    )
    parser.add_argument(
        "-b", "--bones",
        action="store_true",
        default=True,
        help="Include skeleton bone connections in PLY (default: True)"
    )
    parser.add_argument(
        "--no-bones",
        action="store_true",
        help="Don't include skeleton bone connections"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=True,
        help="Print debug information"
    )
    parser.add_argument(
        "-p", "--pattern",
        type=str,
        default="*.json",
        help="File pattern for batch conversion (default: '*.json')"
    )
    
    args = parser.parse_args()
    
    # Determine if input is file or directory
    input_path = Path(args.input)
    is_directory = input_path.is_dir() or (not input_path.exists() and args.input.endswith('/'))
    
    # Determine output path
    if args.output is None:
        if is_directory:
            output_path = input_path.parent / f"{input_path.name}_ply"
        else:
            output_path = input_path.with_suffix('.ply')
    else:
        output_path = Path(args.output)
    
    include_bones = not args.no_bones
    
    if is_directory:
        batch_convert(str(input_path), str(output_path), pattern=args.pattern, include_bones=include_bones, verbose=args.verbose)
    else:
        keypoints_json_to_ply(str(input_path), str(output_path), include_bones=include_bones, verbose=args.verbose)


if __name__ == "__main__":
    main()
