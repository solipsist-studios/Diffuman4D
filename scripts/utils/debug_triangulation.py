"""
Debug triangulation issues by analyzing camera calibration and 2D keypoint consistency.

This script helps identify whether the problem is with:
1. Camera calibration (intrinsics/extrinsics)
2. 2D keypoint detection quality
3. Keypoint correspondence across views
4. Triangulation algorithm
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.data.utils.camera_parser import parse_cameras
from scripts.preprocess.utils.triang_utils import triangulate_points


def load_2d_keypoints(json_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load 2D keypoints and scores from JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    instance = data['instance_info'][0]
    keypoints = np.array(instance['keypoints'], dtype=np.float32)
    scores = np.array(instance.get('keypoint_scores', np.ones(len(keypoints))), dtype=np.float32)
    return keypoints, scores


def analyze_cameras(camera_path: str, spa_labels: List[str]) -> Dict:
    """Analyze camera parameters for validity."""
    cams = parse_cameras(camera_path, coord_system="opencv", normalize_scene=False)
    
    results = {
        'num_cameras': len(spa_labels),
        'cameras': {}
    }
    
    for label in spa_labels:
        cam = cams[label]
        K = cam['K']
        pose = cam['pose']
        
        # Check camera matrix validity
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        
        # Check pose validity
        R = pose[:3, :3]
        t = pose[:3, 3]
        
        # Verify rotation matrix properties
        det_R = np.linalg.det(R)
        R_check = np.allclose(np.dot(R, R.T), np.eye(3), atol=1e-3)
        
        results['cameras'][label] = {
            'fx': float(fx),
            'fy': float(fy),
            'cx': float(cx),
            'cy': float(cy),
            'det_R': float(det_R),
            'R_is_valid': bool(R_check),
            'translation': t.tolist(),
            'camera_position': (-R.T @ t).tolist(),  # World position of camera
        }
    
    return results


def analyze_2d_consistency(
    kp2d_dir: str,
    tem_label: str,
    spa_labels: List[str],
    threshold: float = 50.0
) -> Dict:
    """Analyze consistency of 2D keypoints across views."""
    
    results = {
        'tem_label': tem_label,
        'num_views': len(spa_labels),
        'keypoints': {}
    }
    
    # Load keypoints from all views
    kp2d_all = []
    scores_all = []
    
    for spa_label in spa_labels:
        json_path = os.path.join(kp2d_dir, spa_label, f"{tem_label}.json")
        if os.path.exists(json_path):
            kp2d, scores = load_2d_keypoints(json_path)
            kp2d_all.append(kp2d)
            scores_all.append(scores)
    
    if not kp2d_all:
        return results
    
    kp2d_all = np.array(kp2d_all)  # (num_views, num_keypoints, 2)
    scores_all = np.array(scores_all)  # (num_views, num_keypoints)
    
    num_keypoints = kp2d_all.shape[1]
    
    # Analyze each keypoint
    for kp_idx in range(num_keypoints):
        kp_2d = kp2d_all[:, kp_idx, :]  # (num_views, 2)
        kp_scores = scores_all[:, kp_idx]  # (num_views,)
        
        # Calculate statistics
        mean_pos = kp_2d.mean(axis=0)
        std_pos = kp_2d.std(axis=0)
        max_dist = np.max([np.linalg.norm(kp_2d[i] - mean_pos) for i in range(len(kp_2d))])
        
        # Check consistency
        is_consistent = max_dist < threshold
        
        results['keypoints'][kp_idx] = {
            'mean_position': mean_pos.tolist(),
            'std_position': std_pos.tolist(),
            'max_distance_from_mean': float(max_dist),
            'is_consistent': bool(is_consistent),
            'scores': kp_scores.tolist(),
            'mean_score': float(kp_scores.mean()),
            'min_score': float(kp_scores.min()),
            'all_positions': [kp_2d[i].tolist() for i in range(len(kp_2d))],
        }
    
    # Summary statistics
    inconsistent_kps = [kp_idx for kp_idx, info in results['keypoints'].items() 
                       if not info['is_consistent']]
    low_score_kps = [kp_idx for kp_idx, info in results['keypoints'].items() 
                     if info['mean_score'] < 0.5]
    
    results['summary'] = {
        'num_inconsistent_keypoints': len(inconsistent_kps),
        'num_low_score_keypoints': len(low_score_kps),
        'inconsistent_kps': inconsistent_kps,
        'low_score_kps': low_score_kps,
    }
    
    return results


def analyze_triangulation(
    camera_path: str,
    kp2d_dir: str,
    tem_label: str,
    spa_labels: List[str],
    score_thr: float = 0.6,
) -> Dict:
    """Analyze triangulation results."""
    
    results = {
        'tem_label': tem_label,
        'score_thr': score_thr,
    }
    
    # Load cameras
    cams = parse_cameras(camera_path, coord_system="opencv", normalize_scene=False)
    Ks = np.array([cams[label]["K"] for label in spa_labels], dtype=np.float32)
    Ts = np.array([np.linalg.inv(cams[label]["pose"]) for label in spa_labels], dtype=np.float32)
    
    # Load 2D keypoints
    kp2d_paths = [os.path.join(kp2d_dir, spa_label, f"{tem_label}.json") for spa_label in spa_labels]
    kp2d_list = []
    scores_list = []
    
    for path in kp2d_paths:
        if os.path.exists(path):
            kp2d, scores = load_2d_keypoints(path)
            kp2d_list.append(kp2d)
            scores_list.append(scores)
    
    if not kp2d_list:
        results['error'] = "No 2D keypoint files found"
        return results
    
    kp2d = np.stack(kp2d_list)
    kp2d_score = np.stack(scores_list)
    
    # Apply hand score scaling (from read_kp2d)
    kp2d_score[:, 92:112] *= kp2d_score[:, 91:92] ** 2
    kp2d_score[:, 113:133] *= kp2d_score[:, 112:113] ** 2
    
    # Triangulate
    kp3d, kp3d_reproj, _ = triangulate_points(
        Ks, Ts, kp2d, kp2d_score, 
        score_thr=score_thr, 
        use_cuda=False
    )
    
    # Analyze results
    results['keypoints'] = {}
    
    for kp_idx in range(len(kp3d)):
        kp_3d = kp3d[kp_idx]
        reproj_err = kp3d_reproj[kp_idx]
        
        # Check if invalid (marked with -100000)
        is_invalid = np.allclose(kp_3d, -100000, atol=1)
        
        # Check if collinear (all on a line)
        # Compute std of normalized coordinates
        kp_3d_norm = kp_3d / (np.linalg.norm(kp_3d) + 1e-6)
        
        results['keypoints'][kp_idx] = {
            'position': kp_3d.tolist(),
            'is_invalid': bool(is_invalid),
            'reprojection_error': float(reproj_err),
            'distance_from_origin': float(np.linalg.norm(kp_3d)),
        }
    
    # Check if points are collinear
    valid_kps = np.array([results['keypoints'][i]['position'] 
                          for i in range(len(results['keypoints']))
                          if not results['keypoints'][i]['is_invalid']])
    
    if len(valid_kps) > 2:
        # Compute covariance
        mean_kp = valid_kps.mean(axis=0)
        centered = valid_kps - mean_kp
        cov = np.cov(centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        results['structure_analysis'] = {
            'num_valid_keypoints': int(len(valid_kps)),
            'eigenvalues': eigenvalues.tolist(),
            'condition_number': float(eigenvalues[0] / (eigenvalues[2] + 1e-6)),
            'is_collinear': bool(eigenvalues[1] < 1e-6 and eigenvalues[2] < 1e-6),
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Debug triangulation issues"
    )
    parser.add_argument("camera_path", type=str, help="Path to camera calibration file")
    parser.add_argument("kp2d_dir", type=str, help="Directory containing 2D keypoints")
    parser.add_argument("--tem_label", type=str, default="000000", help="Temporal label to analyze")
    parser.add_argument("--spa_labels", type=str, nargs="+", default=None, 
                       help="Spatial labels (camera IDs). If not specified, will be auto-detected")
    parser.add_argument("-o", "--output", type=str, default=None,
                       help="Output JSON file for debug info (default: print to console)")
    parser.add_argument("--score_thr", type=float, default=0.6, help="Score threshold for triangulation")
    parser.add_argument("--consistency_threshold", type=float, default=50.0,
                       help="Maximum 2D distance (pixels) for keypoint consistency")
    
    args = parser.parse_args()
    
    # Auto-detect spatial labels if not provided
    if args.spa_labels is None:
        spa_labels = sorted(os.listdir(args.kp2d_dir))
    else:
        spa_labels = args.spa_labels
    
    print(f"Analyzing {len(spa_labels)} cameras: {spa_labels}")
    print(f"Temporal label: {args.tem_label}\n")
    
    debug_info = {}
    
    # Analyze cameras
    print("=" * 60)
    print("CAMERA ANALYSIS")
    print("=" * 60)
    cam_analysis = analyze_cameras(args.camera_path, spa_labels)
    debug_info['cameras'] = cam_analysis
    
    for label, cam_info in cam_analysis['cameras'].items():
        print(f"\n{label}:")
        print(f"  Focal length: fx={cam_info['fx']:.1f}, fy={cam_info['fy']:.1f}")
        print(f"  Principal point: cx={cam_info['cx']:.1f}, cy={cam_info['cy']:.1f}")
        print(f"  Rotation determinant: {cam_info['det_R']:.6f} (should be ~1.0)")
        print(f"  Rotation matrix valid: {cam_info['R_is_valid']}")
        print(f"  Camera position: {[f'{x:.2f}' for x in cam_info['camera_position']]}")
    
    # Analyze 2D keypoint consistency
    print("\n" + "=" * 60)
    print("2D KEYPOINT CONSISTENCY ANALYSIS")
    print("=" * 60)
    consistency = analyze_2d_consistency(args.kp2d_dir, args.tem_label, spa_labels, 
                                        threshold=args.consistency_threshold)
    debug_info['consistency'] = consistency
    
    summary = consistency.get('summary', {})
    print(f"\nTotal keypoints: {len(consistency['keypoints'])}")
    print(f"Inconsistent keypoints: {summary.get('num_inconsistent_keypoints', 0)}")
    print(f"Low score keypoints (< 0.5): {summary.get('num_low_score_keypoints', 0)}")
    
    if summary.get('inconsistent_kps'):
        print(f"Inconsistent keypoint indices: {summary['inconsistent_kps'][:10]}...")
    
    # Analyze triangulation
    print("\n" + "=" * 60)
    print("TRIANGULATION ANALYSIS")
    print("=" * 60)
    triang = analyze_triangulation(args.camera_path, args.kp2d_dir, args.tem_label, 
                                  spa_labels, score_thr=args.score_thr)
    debug_info['triangulation'] = triang
    
    if 'structure_analysis' in triang:
        struct = triang['structure_analysis']
        print(f"\nValid keypoints: {struct['num_valid_keypoints']}")
        print(f"Eigenvalues: {[f'{x:.6f}' for x in struct['eigenvalues']]}")
        print(f"Condition number: {struct['condition_number']:.2e}")
        print(f"Points are collinear: {struct['is_collinear']}")
        
        if struct['is_collinear']:
            print("\n⚠️  WARNING: All 3D points are collinear (on a single line)!")
            print("   This suggests a camera calibration or triangulation issue.")
    
    # Save debug info
    if args.output:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(args.output, 'w') as f:
            json.dump(debug_info, f, indent=2, default=float)
        print(f"\nDebug info saved to: {args.output}")
    else:
        print("\n" + "=" * 60)
        print("FULL DEBUG OUTPUT (JSON)")
        print("=" * 60)
        print(json.dumps(debug_info, indent=2, default=float))


if __name__ == "__main__":
    main()
