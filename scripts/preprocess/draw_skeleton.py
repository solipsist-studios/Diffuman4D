from __future__ import annotations
import os
import cv2
import fire
import json
import numpy as np
import importlib
from PIL import Image
from easyvolcap.utils.parallel_utils import parallel_execution

BLUE = [51, 153, 255]


def _resolve_keypoints308_metainfo_file() -> str | None:
    try:
        sapiens_mod = importlib.import_module("sapiens")
    except Exception:
        return None

    sapiens_root = os.path.dirname(sapiens_mod.__file__)
    candidates = [
        os.path.join(sapiens_root, "pose", "configs", "_base_", "keypoints308.py"),
        os.path.join(sapiens_root, "configs", "_base_", "keypoints308.py"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _build_default_colors(n_kpts: int) -> list[list[int]]:
    # Deterministic color palette fallback when metainfo is unavailable.
    colors = []
    for i in range(n_kpts):
        hue = int((179.0 * i) / max(1, n_kpts))
        bgr = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
        colors.append([int(bgr[2]), int(bgr[1]), int(bgr[0])])
    return colors


def _get_layout_info(n_kpts: int):
    """Return (colors_info, skeleton_info) for known keypoint layouts."""
    if n_kpts == 133:
        try:
            from sapiens.lite.demo.classes_and_palettes import (
                COCO_WHOLEBODY_KPTS_COLORS,
                COCO_WHOLEBODY_SKELETON_INFO,
            )

            colors = [list(map(int, c)) for c in COCO_WHOLEBODY_KPTS_COLORS]
            skeleton = {int(k): v for k, v in dict(COCO_WHOLEBODY_SKELETON_INFO).items()}
            return colors, skeleton
        except Exception:
            pass

    if n_kpts == 308:
        try:
            from sapiens.pose.datasets import parse_pose_metainfo

            metainfo_file = _resolve_keypoints308_metainfo_file()
            if metainfo_file is not None:
                metainfo = parse_pose_metainfo(dict(from_file=metainfo_file))
                colors = np.asarray(metainfo["keypoint_colors"], dtype=np.int32)
                links = metainfo["skeleton_links"]
                link_colors = np.asarray(metainfo["skeleton_link_colors"], dtype=np.int32)
                skeleton = {
                    int(i): {
                        "link": (int(link[0]), int(link[1])),
                        "id": int(i),
                        "color": [int(v) for v in link_colors[i].tolist()],
                    }
                    for i, link in enumerate(links)
                }
                return colors.tolist(), skeleton
        except Exception:
            pass

    return _build_default_colors(n_kpts), {}


def score_to_color(rgb, score, low=0.5, high=0.9):
    score = np.clip(score, low, high)
    norm_score = (score - low) / (high - low)
    rgb = np.array(rgb, dtype=np.float32) * norm_score
    rgb = np.round(rgb, decimals=0).astype(np.uint8).tolist()
    return rgb


def draw_one_skeleton(
    kp2d_path,
    out_kpmap_path,
    kp2d_score_path=None,
    kp2d_canvas_shape=(1024, 1024),
    out_kpmap_shape=(1024, 1024),
    low_thr=0.5,
    high_thr=0.9,
    colors_info=None,
    skeleton_info=None,
    radius=2,
    thickness=2,
    image_quality=85,
    draw_face_keypoints=False,
    skip_exists=False,
    max_bone_frac: float = 0.35,  # skip links longer than this fraction of canvas diagonal
):
    if skip_exists and os.path.exists(out_kpmap_path):
        try:
            Image.open(out_kpmap_path).verify()
            return
        except Exception as e:
            print(f"Error reading {out_kpmap_path}: {e}")

    # currently, we only support one instance per image
    kpts_dict = json.load(open(kp2d_path))["instance_info"][0]

    kpts = np.array(kpts_dict["keypoints"], dtype=np.float32)
    n_kpts = len(kpts)

    if colors_info is None or skeleton_info is None:
        auto_colors, auto_skeleton = _get_layout_info(n_kpts)
        if colors_info is None:
            colors_info = auto_colors
        if skeleton_info is None:
            skeleton_info = auto_skeleton

    if kp2d_score_path is not None:
        # override the scores from the kp2d_score_path
        kpts_score_dict = json.load(open(kp2d_score_path))["instance_info"][0]
        scores = np.array(kpts_score_dict["keypoint_scores"], dtype=np.float32)
    elif "keypoint_scores" in kpts_dict:
        scores = np.array(kpts_dict["keypoint_scores"], dtype=np.float32)
    else:
        scores = np.ones(kpts.shape[0], dtype=np.float32)

    if "keypoint_depths" in kpts_dict:
        depths = np.array(kpts_dict["keypoint_depths"], dtype=np.float32)
    else:
        depths = np.zeros_like(scores)

    # update scores for invalid keypoints
    scores[kpts.min(axis=1) < 0] = 0.0

    # scale and shift the keypoints to match the output image shape
    # draw skeleton map at 2048p for anti-aliasing
    drawing_scale = 2048 / max(out_kpmap_shape)
    out_kpmap_shape = (np.array(out_kpmap_shape) * drawing_scale).astype(np.int32)
    kp2d_canvas_shape = np.array(kp2d_canvas_shape)
    scale_ratio = out_kpmap_shape.min() / kp2d_canvas_shape.min()
    kpts = kpts * scale_ratio
    kp2d_canvas_shape = kp2d_canvas_shape * scale_ratio
    kp2d_padding = (out_kpmap_shape.min() - kp2d_canvas_shape.min()) / 2
    kpts += kp2d_padding

    # kp2d_canvas_shape represents the shape of the canvas to draw the skeleton,
    # please ensure the canvas shape matches the input image shape of sapiens poses
    canvas = np.zeros(np.concatenate([out_kpmap_shape, [3]]), dtype=np.uint8)

    if colors_info is None:
        colors_info = _build_default_colors(n_kpts)
    if len(colors_info) < n_kpts:
        colors_info = list(colors_info) + _build_default_colors(n_kpts - len(colors_info))
    elif len(colors_info) > n_kpts:
        colors_info = list(colors_info[:n_kpts])

    # add x links for the body
    skeleton_info = dict(skeleton_info or {})
    if n_kpts > 12:
        skeleton_info.update(
            {
                10065: dict(link=(5, 12), id=10065, color=BLUE),  # left shoulder to right hip
                10066: dict(link=(6, 11), id=10066, color=BLUE),  # right shoulder to left hip
            }
        )

    # reweight the radius and thickness of the skeleton
    base_radius = int(round(radius * scale_ratio))
    base_thickness = int(round(thickness * scale_ratio))
    _max_bone_px = max_bone_frac * np.sqrt(out_kpmap_shape[0] ** 2 + out_kpmap_shape[1] ** 2) if max_bone_frac else None

    # draw skeleton
    lines = []
    for lid, (_, link_info) in enumerate(skeleton_info.items()):
        i1, i2 = link_info["link"]
        if i1 >= n_kpts or i2 >= n_kpts:
            continue
        p1_score = scores[i1]
        p2_score = scores[i2]
        line_score = np.min((p1_score, p2_score))

        if line_score < low_thr:
            continue

        # draw skeleton
        p1_color = score_to_color(colors_info[i1], p1_score, low=low_thr, high=high_thr)
        p2_color = score_to_color(colors_info[i2], p2_score, low=low_thr, high=high_thr)
        line_color = score_to_color(link_info["color"], line_score, low=low_thr, high=high_thr)

        p1, p2 = kpts[i1], kpts[i2]
        x1, y1 = int(round(p1[0])), int(round(p1[1]))
        x2, y2 = int(round(p2[0])), int(round(p2[1]))
        if _max_bone_px is not None and np.hypot(x2 - x1, y2 - y1) > _max_bone_px:
            continue
        d1, d2 = float(depths[i1]), float(depths[i2])
        d = (d1 + d2) / 2

        lines.append(
            {
                "type": "line",
                "p1": (x1, y1),
                "p2": (x2, y2),
                "depth": d,
                "score": line_score,
                "p1_color": p1_color[::-1],
                "p2_color": p2_color[::-1],
                "line_color": line_color[::-1],
                "radius": int(base_radius * (2 if lid < 25 else 1)),
                "thickness": int(base_thickness * (2 if lid < 25 else 1)),
            }
        )

    if (depths != 0.0).any():
        # sort lines by depth
        # ideally, we should use z-buffer to pixel-wise sort the lines
        # here we naively sort the lines by average depth of the two endpoints
        lines = sorted(lines, key=lambda x: x["depth"], reverse=True)
    elif (scores != 1.0).any():
        # sort lines by score
        # lines with higher score are more likely to be at the front
        lines = sorted(lines, key=lambda x: x["score"])

    for line in lines:
        cv2.line(canvas, line["p1"], line["p2"], line["line_color"], line["thickness"])
        cv2.circle(canvas, line["p1"], line["radius"], line["p1_color"], -1)
        cv2.circle(canvas, line["p2"], line["radius"], line["p2_color"], -1)

    # draw face keypoints
    if draw_face_keypoints:
        for kid, kpt in enumerate(kpts):
            if not (23 < kid < 91):
                continue
            if scores[kid] < low_thr:
                continue
            color = score_to_color(colors_info[kid], scores[kid], low=low_thr, high=high_thr)
            x, y = int(round(kpt[0])), int(round(kpt[1]))
            cv2.circle(canvas, (x, y), base_radius, color[::-1], -1)

    os.makedirs(os.path.dirname(out_kpmap_path), exist_ok=True)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(canvas)
    w, h = canvas.size
    canvas = canvas.resize((int(w / drawing_scale), int(h / drawing_scale)))
    canvas.save(out_kpmap_path, quality=image_quality)


def draw_skeleton(
    kp2d_dir: str,
    out_kpmap_dir: str,
    kp2d_score_dir: str | None = None,
    kp2d_canvas_shape: tuple[int, int] = (1024, 1024),
    out_kpmap_shape: tuple[int, int] = (1024, 1024),
    spa_labels: list[int] | None = None,
    tem_labels: list[int] | None = None,
    image_ext: str = ".webp",
    image_quality: int = 85,
    num_workers: int = 16,
    skip_exists: bool = False,
):
    if spa_labels is None:
        spa_labels = sorted(os.listdir(kp2d_dir))
    else:
        spa_labels = [f"{spa_label:02d}" for spa_label in spa_labels]
    if tem_labels is None:
        tem_labels = sorted(os.listdir(f"{kp2d_dir}/{spa_labels[0]}"))
        tem_labels = [tem_label.split(".")[0] for tem_label in tem_labels]
    else:
        tem_labels = [f"{tem_label:06d}" for tem_label in tem_labels]

    kp2d_paths = [f"{kp2d_dir}/{spa_label}/{tem_label}.json" for spa_label in spa_labels for tem_label in tem_labels]
    out_kpmap_paths = [p.replace(kp2d_dir, out_kpmap_dir).replace(".json", image_ext) for p in kp2d_paths]

    if kp2d_score_dir is not None:
        kp2d_score_paths = [p.replace(kp2d_dir, kp2d_score_dir) for p in kp2d_paths]
    else:
        kp2d_score_paths = [None] * len(kp2d_paths)

    parallel_execution(
        kp2d_paths,
        out_kpmap_paths,
        kp2d_score_paths,
        kp2d_canvas_shape=kp2d_canvas_shape,
        out_kpmap_shape=out_kpmap_shape,
        image_quality=image_quality,
        skip_exists=skip_exists,
        action=draw_one_skeleton,
        num_workers=num_workers,
        print_progress=True,
        desc="Drawing skeleton map",
        sequential=False,
    )


if __name__ == "__main__":
    # usage:
    # python scripts/preprocess/draw_skeleton.py --kp2d_dir $DATADIR/poses_2d --out_kpmap_dir $DATADIR/skeletons
    fire.Fire(draw_skeleton)
