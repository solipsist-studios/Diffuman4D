"""
Utility script to visualize Sapiens-generated keypoints as overlaid skeletons on images in a grid.
Useful for evaluating the quality of keypoint predictions.

Usage:
    python scripts/preprocess/evaluate_keypoints.py \\
        --images_dir data/ballet/images \\
        --kp2d_dir data/ballet/poses_2d \\
        --output_dir output/keypoint_evaluation \\
        --grid_rows 3 --grid_cols 4 \\
        --num_samples 12
"""

import os
import json
import cv2
import fire
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from glob import glob
from typing import Optional

try:
    from sapiens.lite.demo.classes_and_palettes import (
        COCO_WHOLEBODY_KPTS_COLORS,
        COCO_WHOLEBODY_SKELETON_INFO,
        BLUE,
    )
except ImportError:
    print("Warning: Could not import SAPIENS color/skeleton info. Using defaults.")
    COCO_WHOLEBODY_KPTS_COLORS = None
    COCO_WHOLEBODY_SKELETON_INFO = None


def score_to_color(rgb, score, low=0.5, high=0.9):
    """Convert a score to an RGB color with intensity based on confidence."""
    score = np.clip(score, low, high)
    norm_score = (score - low) / (high - low)
    rgb = np.array(rgb, dtype=np.float32) * norm_score
    rgb = np.round(rgb, decimals=0).astype(np.uint8).tolist()
    return rgb


def draw_skeleton_on_image(
    image: np.ndarray,
    kpts: np.ndarray,
    scores: np.ndarray,
    depths: Optional[np.ndarray] = None,
    colors_info=None,
    skeleton_info=None,
    low_thr=0.5,
    high_thr=0.9,
    radius=3,
    thickness=2,
):
    """
    Draw skeleton keypoints on an image.

    Args:
        image: Input image as numpy array (H, W, 3) in BGR format
        kpts: Keypoint coordinates as (N, 2) array
        scores: Keypoint confidence scores as (N,) array
        depths: Keypoint depths as (N,) array (optional, for sorting)
        colors_info: Color information for keypoints (COCO_WHOLEBODY_KPTS_COLORS)
        skeleton_info: Skeleton link information (COCO_WHOLEBODY_SKELETON_INFO)
        low_thr: Lower confidence threshold for drawing
        high_thr: Upper confidence threshold for color scaling
        radius: Radius of keypoint circles
        thickness: Thickness of skeleton lines

    Returns:
        Annotated image as numpy array
    """
    canvas = image.copy()
    h, w = image.shape[:2]

    if depths is None:
        depths = np.zeros_like(scores)

    # Draw skeleton lines
    if skeleton_info is not None:
        lines = []
        for skid, link_info in skeleton_info.items():
            i1, i2 = link_info["link"]
            
            # Check bounds
            if i1 >= len(kpts) or i2 >= len(kpts):
                continue
                
            p1_score = scores[i1]
            p2_score = scores[i2]
            line_score = np.min((p1_score, p2_score))

            if line_score < low_thr:
                continue

            # Get colors
            if colors_info is not None:
                p1_color = score_to_color(colors_info[i1], p1_score, low=low_thr, high=high_thr)
                p2_color = score_to_color(colors_info[i2], p2_score, low=low_thr, high=high_thr)
                line_color = score_to_color(link_info["color"], line_score, low=low_thr, high=high_thr)
            else:
                # Default colors if SAPIENS not available
                p1_color = (0, int(p1_score * 255), int((1 - p1_score) * 255))
                p2_color = (0, int(p2_score * 255), int((1 - p2_score) * 255))
                line_color = (0, int(line_score * 255), int((1 - line_score) * 255))

            p1, p2 = kpts[i1], kpts[i2]
            x1, y1 = int(round(p1[0])), int(round(p1[1]))
            x2, y2 = int(round(p2[0])), int(round(p2[1]))

            # Check bounds
            if not (0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h):
                continue

            d1, d2 = float(depths[i1]), float(depths[i2])
            d = (d1 + d2) / 2

            lines.append(
                {
                    "p1": (x1, y1),
                    "p2": (x2, y2),
                    "depth": d,
                    "score": line_score,
                    "p1_color": p1_color[::-1],  # Convert BGR to RGB for cv2
                    "p2_color": p2_color[::-1],
                    "line_color": line_color[::-1],
                    "radius": radius,
                    "thickness": thickness,
                }
            )

        # Sort lines by depth if available
        if (depths != 0.0).any():
            lines = sorted(lines, key=lambda x: x["depth"], reverse=True)
        elif (scores != 1.0).any():
            lines = sorted(lines, key=lambda x: x["score"])

        # Draw lines and joints
        for line in lines:
            cv2.line(canvas, line["p1"], line["p2"], line["line_color"], line["thickness"])
            cv2.circle(canvas, line["p1"], line["radius"], line["p1_color"], -1)
            cv2.circle(canvas, line["p2"], line["radius"], line["p2_color"], -1)

    # Draw keypoint circles (for all keypoints)
    for kid, kpt in enumerate(kpts):
        if scores[kid] < low_thr:
            continue
        
        x, y = int(round(kpt[0])), int(round(kpt[1]))
        
        if not (0 <= x < w and 0 <= y < h):
            continue

        if colors_info is not None and kid < len(colors_info):
            color = score_to_color(colors_info[kid], scores[kid], low=low_thr, high=high_thr)
            color = color[::-1]  # BGR
        else:
            color = (0, int(scores[kid] * 255), int((1 - scores[kid]) * 255))

        cv2.circle(canvas, (x, y), radius, color, -1)

    return canvas


def load_keypoints_from_json(kp2d_path: str):
    """Load keypoints from JSON file generated by predict_keypoints.py"""
    with open(kp2d_path, 'r') as f:
        data = json.load(f)
    
    if "instance_info" not in data or len(data["instance_info"]) == 0:
        return None, None, None
    
    instance = data["instance_info"][0]
    kpts = np.array(instance.get("keypoints", []), dtype=np.float32)
    scores = np.array(instance.get("keypoint_scores", np.ones(len(kpts))), dtype=np.float32)
    depths = np.array(instance.get("keypoint_depths", np.zeros(len(kpts))), dtype=np.float32)
    
    return kpts, scores, depths


def evaluate_keypoints_grid(
    images_dir: str,
    kp2d_dir: str,
    output_dir: str = "./output/keypoint_evaluation",
    grid_rows: int = 3,
    grid_cols: int = 4,
    num_samples: Optional[int] = None,
    image_ext: str = ".jpg",
    output_ext: str = ".jpg",
    low_thr: float = 0.5,
    high_thr: float = 0.9,
    radius: int = 3,
    thickness: int = 2,
    grid_spacing: int = 10,
    cell_size: int = 512,
    draw_labels: bool = True,
    skip_missing_kp: bool = True,
):
    """
    Create a grid of images with overlaid skeleton keypoints for evaluation.

    Args:
        images_dir: Directory containing images
        kp2d_dir: Directory containing 2D keypoints (JSON files)
        output_dir: Output directory for grid visualizations
        grid_rows: Number of rows in the output grid
        grid_cols: Number of columns in the output grid
        num_samples: Maximum number of samples to process (None for all)
        image_ext: Image file extension (e.g., '.jpg', '.png')
        output_ext: Output image extension
        low_thr: Confidence threshold for drawing keypoints
        high_thr: Upper threshold for color scaling
        radius: Keypoint circle radius
        thickness: Skeleton line thickness
        grid_spacing: Spacing between grid cells
        cell_size: Size of each grid cell
        draw_labels: Whether to draw image names on grid
        skip_missing_kp: Skip images without keypoint files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find all image files. Assume images are organized by camera subfolders
    # e.g. images_dir/<camera>/*.jpg and kp2d_dir/<camera>/*.json
    image_files = []  # list of tuples (camera_name, image_path)
    if os.path.isdir(images_dir):
        # list first-level camera directories (ignore files directly under images_dir)
        entries = sorted(os.listdir(images_dir))
        cameras = [e for e in entries if os.path.isdir(os.path.join(images_dir, e))]
        if len(cameras) > 0:
            for cam in cameras:
                cam_dir = os.path.join(images_dir, cam)
                imgs = sorted(glob(os.path.join(cam_dir, f"*{image_ext}")))
                for p in imgs:
                    image_files.append((cam, p))
        else:
            # fallback: images directly under images_dir (no camera subfolders)
            imgs = sorted(glob(os.path.join(images_dir, f"**/*{image_ext}"), recursive=True))
            for p in imgs:
                # empty camera name indicates top-level
                image_files.append(("", p))

    if num_samples is not None:
        image_files = image_files[:num_samples]

    print(f"Found {len(image_files)} images in {images_dir} (camera subfolders assumed)")

    # Get keypoint files that match images
    valid_pairs = []
    for cam, img_path in image_files:
        # Construct corresponding keypoint path. Keep camera subfolder in the relative path
        if cam:
            filename = os.path.splitext(os.path.basename(img_path))[0]
            kp_path = os.path.join(kp2d_dir, cam, filename + ".json")
        else:
            # top-level images
            rel_path = os.path.relpath(img_path, images_dir)
            kp_rel_path = os.path.splitext(rel_path)[0] + ".json"
            kp_path = os.path.join(kp2d_dir, kp_rel_path)

        if os.path.exists(kp_path):
            valid_pairs.append((cam, img_path, kp_path))
        elif not skip_missing_kp:
            print(f"Warning: Missing keypoints for {img_path}")
            valid_pairs.append((cam, img_path, None))

    print(f"Found {len(valid_pairs)} valid image-keypoint pairs")

    if len(valid_pairs) == 0:
        print("No valid pairs found!")
        return

    # Process images and create grids
    num_grids = (len(valid_pairs) + grid_rows * grid_cols - 1) // (grid_rows * grid_cols)
    
    for grid_idx in range(num_grids):
        start_idx = grid_idx * grid_rows * grid_cols
        end_idx = min(start_idx + grid_rows * grid_cols, len(valid_pairs))
        batch = valid_pairs[start_idx:end_idx]

        # Create grid
        grid_h = grid_rows * (cell_size + grid_spacing) + grid_spacing
        grid_w = grid_cols * (cell_size + grid_spacing) + grid_spacing
        grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255

        for idx, (cam, img_path, kp_path) in enumerate(batch):
            row = idx // grid_cols
            col = idx % grid_cols
            
            # Load and resize image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error reading {img_path}")
                continue
            
            # Resize to cell size
            h, w = img.shape[:2]
            aspect = w / h
            if aspect > 1:
                new_w = cell_size
                new_h = int(cell_size / aspect)
            else:
                new_h = cell_size
                new_w = int(cell_size * aspect)
            
            img = cv2.resize(img, (new_w, new_h))

            # Draw keypoints if available
            if kp_path is not None:
                kpts, scores, depths = load_keypoints_from_json(kp_path)
                if kpts is not None:
                    img = draw_skeleton_on_image(
                        img, kpts, scores, depths,
                        colors_info=COCO_WHOLEBODY_KPTS_COLORS,
                        skeleton_info=COCO_WHOLEBODY_SKELETON_INFO,
                        low_thr=low_thr,
                        high_thr=high_thr,
                        radius=radius,
                        thickness=thickness,
                    )

            # Place in grid
            y_start = row * (cell_size + grid_spacing) + grid_spacing
            x_start = col * (cell_size + grid_spacing) + grid_spacing
            
            # Pad image to cell size if needed
            padded = np.ones((cell_size, cell_size, 3), dtype=np.uint8) * 240
            y_offset = (cell_size - new_h) // 2
            x_offset = (cell_size - new_w) // 2
            padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img

            grid[y_start:y_start+cell_size, x_start:x_start+cell_size] = padded

            # Draw label if requested
            if draw_labels:
                # label includes camera if available
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                label = f"{cam}/{base_name}" if cam else base_name
                # Convert to PIL for text drawing
                grid_pil = Image.fromarray(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(grid_pil)
                text_x = x_start + 5
                text_y = y_start + cell_size - 20
                draw.text((text_x, text_y), label, fill=(0, 0, 0))
                grid = cv2.cvtColor(np.array(grid_pil), cv2.COLOR_RGB2BGR)

        # Save grid
        grid_output_path = os.path.join(output_dir, f"grid_{grid_idx:04d}{output_ext}")
        os.makedirs(os.path.dirname(grid_output_path), exist_ok=True)
        cv2.imwrite(grid_output_path, grid)
        print(f"Saved grid to {grid_output_path}")


def evaluate_keypoints_single(
    images_dir: str,
    kp2d_dir: str,
    output_dir: str = "./output/keypoint_evaluation",
    image_ext: str = ".jpg",
    output_ext: str = ".jpg",
    low_thr: float = 0.5,
    high_thr: float = 0.9,
    radius: int = 3,
    thickness: int = 2,
    num_samples: Optional[int] = None,
):
    """
    Create individual annotated images (one per image with keypoints overlaid).

    Args:
        images_dir: Directory containing images
        kp2d_dir: Directory containing 2D keypoints
        output_dir: Output directory for annotated images
        image_ext: Image file extension
        output_ext: Output image extension
        low_thr: Confidence threshold
        high_thr: Upper threshold for color scaling
        radius: Keypoint circle radius
        thickness: Skeleton line thickness
        num_samples: Maximum number of samples to process
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build image list assuming camera subfolders
    image_files = []  # list of tuples (camera_name, image_path)
    if os.path.isdir(images_dir):
        entries = sorted(os.listdir(images_dir))
        cameras = [e for e in entries if os.path.isdir(os.path.join(images_dir, e))]
        if len(cameras) > 0:
            for cam in cameras:
                cam_dir = os.path.join(images_dir, cam)
                imgs = sorted(glob(os.path.join(cam_dir, f"*{image_ext}")))
                for p in imgs:
                    image_files.append((cam, p))
        else:
            imgs = sorted(glob(os.path.join(images_dir, f"**/*{image_ext}"), recursive=True))
            for p in imgs:
                image_files.append(("", p))

    if num_samples is not None:
        image_files = image_files[:num_samples]

    print(f"Found {len(image_files)} images (camera subfolders assumed)")

    for idx, (cam, img_path) in enumerate(image_files):
        # Construct keypoint path matching the camera subfolder
        if cam:
            filename = os.path.splitext(os.path.basename(img_path))[0]
            kp_path = os.path.join(kp2d_dir, cam, filename + ".json")
            rel_path = os.path.join(cam, filename + image_ext)
        else:
            rel_path = os.path.relpath(img_path, images_dir)
            kp_rel_path = os.path.splitext(rel_path)[0] + ".json"
            kp_path = os.path.join(kp2d_dir, kp_rel_path)

        if not os.path.exists(kp_path):
            print(f"Skipping {img_path} - no keypoints")
            continue

        # Load image and keypoints
        img = cv2.imread(img_path)
        kpts, scores, depths = load_keypoints_from_json(kp_path)

        if img is None or kpts is None:
            continue

        # Draw keypoints
        annotated = draw_skeleton_on_image(
            img, kpts, scores, depths,
            colors_info=COCO_WHOLEBODY_KPTS_COLORS,
            skeleton_info=COCO_WHOLEBODY_SKELETON_INFO,
            low_thr=low_thr,
            high_thr=high_thr,
            radius=radius,
            thickness=thickness,
        )

        # Save
        output_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, annotated)

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(image_files)}")


if __name__ == "__main__":
    fire.Fire({
        "grid": evaluate_keypoints_grid,
        "single": evaluate_keypoints_single,
    })
