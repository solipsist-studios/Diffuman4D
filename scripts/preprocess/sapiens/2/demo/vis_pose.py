# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import sapiens
import warnings
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sapiens.pose.datasets import parse_pose_metainfo, UDPHeatmap
from sapiens.pose.evaluators import nms
from sapiens.pose.models import init_model
from tqdm import tqdm
from transformers import DetrForObjectDetection, DetrImageProcessor
from transformers.utils import logging as hf_logging

from pose_render_utils import visualize_keypoints

# DETR — COCO person = label 1.
_detector_cache: dict = {}
_MASK_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _resolve_keypoints308_metainfo_file() -> str:
    """Resolve keypoints308 metainfo file path from installed sapiens package."""
    candidates = [
        os.path.normpath("configs/_base_/keypoints308.py"),
        os.path.join(os.path.dirname(sapiens.__file__), "pose", "configs", "_base_", "keypoints308.py"),
        os.path.join(os.path.dirname(sapiens.__file__), "configs", "_base_", "keypoints308.py"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.normpath(candidate)
    raise FileNotFoundError(
        "Could not find keypoints308 metainfo file. Tried: " + ", ".join(candidates)
    )


def _get_detector(device, ckpt_dir):
    if "model" not in _detector_cache:
        # Suppress known upstream loading warnings from DETR checkpoints/preprocessor configs.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The `max_size` parameter is deprecated.*",
            )
            _detector_cache["proc"] = DetrImageProcessor.from_pretrained(ckpt_dir)

        prev_verbosity = hf_logging.get_verbosity()
        try:
            hf_logging.set_verbosity_error()
            _detector_cache["model"] = (
                DetrForObjectDetection.from_pretrained(ckpt_dir).eval().to(device)
            )
        finally:
            hf_logging.set_verbosity(prev_verbosity)
    return _detector_cache["proc"], _detector_cache["model"]


def _detect_persons(image_bgr: np.ndarray, args) -> np.ndarray:
    proc, model = _get_detector(args.device, args.det_checkpoint)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    inputs = proc(images=pil_img, return_tensors="pt").to(args.device)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image_rgb.shape[:2]], device=args.device)
    results = proc.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=args.bbox_thr
    )[0]
    person_mask = results["labels"] == 1
    boxes = results["boxes"][person_mask].cpu().numpy()
    scores = results["scores"][person_mask].cpu().numpy().reshape(-1, 1)
    bboxes = np.concatenate([boxes, scores], axis=1)
    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]  # B x 4; x1, y1, x2, y2
    return bboxes  # may be empty; callers handle empty via mask fallback or full-image fallback


def _load_mask_for_image(mask_dir: str | None, image_name: str, image_shape: tuple[int, int, int]) -> np.ndarray | None:
    if not mask_dir:
        return None
    stem = Path(image_name).stem
    for ext in _MASK_SUFFIXES:
        candidate = os.path.join(mask_dir, f"{stem}{ext}")
        if os.path.exists(candidate):
            mask = cv2.imread(candidate, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                return None
            h, w = image_shape[:2]
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            return (mask > 0).astype(np.uint8)
    return None


def _mask_bbox(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _score_bbox_vs_mask(bbox: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Return (overlap_ratio, mask_coverage) for a bbox against a binary mask.

    overlap_ratio  = mask_pixels_in_bbox / bbox_area
                     (what fraction of the bbox is foreground)
    mask_coverage  = mask_pixels_in_bbox / total_mask_pixels
                     (what fraction of the foreground the bbox captures)
    """
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = bbox.astype(np.float32)
    x1 = int(np.clip(np.floor(x1), 0, w - 1))
    y1 = int(np.clip(np.floor(y1), 0, h - 1))
    x2 = int(np.clip(np.ceil(x2), 0, w - 1))
    y2 = int(np.clip(np.ceil(y2), 0, h - 1))
    if x2 < x1 or y2 < y1:
        return 0.0, 0.0
    patch = mask[y1 : y2 + 1, x1 : x2 + 1]
    if patch.size == 0:
        return 0.0, 0.0
    intersect = float(patch.sum())
    bbox_area = float(max(1, (x2 - x1 + 1) * (y2 - y1 + 1)))
    mask_total = float(mask.sum())
    overlap_ratio = intersect / bbox_area
    mask_coverage = intersect / max(1.0, mask_total)
    return overlap_ratio, mask_coverage


def _filter_bboxes_by_mask(bboxes: np.ndarray, mask: np.ndarray | None, min_overlap: float) -> np.ndarray:
    """Filter detections by mask.  When a mask is available the mask bbox is
    always injected as a candidate so a missed DETR detection can still be
    recovered.  Scoring uses mask_coverage (fraction of mask captured by the
    box) which is more discriminative than the previous bbox-density metric."""
    if mask is None:
        if len(bboxes) == 0:
            return bboxes
        return bboxes

    # Always add the tight mask bbox as a synthetic candidate.
    mask_box = _mask_bbox(mask)
    if mask_box is not None:
        bboxes = np.concatenate([bboxes, mask_box[None]], axis=0) if len(bboxes) > 0 else mask_box[None]

    if len(bboxes) == 0:
        return bboxes

    kept = []
    for bbox in bboxes:
        _, mask_coverage = _score_bbox_vs_mask(bbox, mask)
        if mask_coverage >= min_overlap:
            kept.append(bbox)

    if kept:
        return np.asarray(kept, dtype=np.float32)

    # Nothing met the threshold; return the mask box so inference still runs.
    if mask_box is not None:
        return mask_box[None]
    return bboxes


def _bbox_area(bbox: np.ndarray) -> float:
    x1, y1, x2, y2 = bbox.astype(np.float32)
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def _limit_subjects(
    bboxes: np.ndarray,
    mask: np.ndarray | None,
    max_subjects: int | None,
) -> np.ndarray:
    if max_subjects is None or max_subjects <= 0 or len(bboxes) <= max_subjects:
        return bboxes

    scored = []
    for i, bbox in enumerate(bboxes):
        if mask is not None:
            _, mask_coverage = _score_bbox_vs_mask(bbox, mask)
        else:
            mask_coverage = 0.0
        area = _bbox_area(bbox)
        scored.append((i, mask_coverage, area))

    # Prefer high mask coverage; break ties with larger boxes.
    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    keep_ids = sorted([idx for idx, _, _ in scored[:max_subjects]])
    return bboxes[keep_ids]


def process_one_image(args, image, model, fg_mask: np.ndarray | None = None):
    bboxes = _detect_persons(image, args)
    bboxes = _filter_bboxes_by_mask(bboxes, fg_mask, args.mask_bbox_overlap_thr)
    bboxes = _limit_subjects(bboxes, fg_mask, args.max_subjects)

    inputs_list = []
    data_samples_list = []
    for bbox in bboxes:
        data_info = dict(img=image)
        data_info["bbox"] = bbox[None]  # shape (1, 4)
        data_info["bbox_score"] = np.ones(1, dtype=np.float32)  # shape (1,)
        data = model.pipeline(data_info)
        data = model.data_preprocessor(data)
        inputs_list.append(data["inputs"])
        data_samples_list.append(data["data_samples"])

    inputs = torch.cat(inputs_list, dim=0)  # B x 3 x H x W
    with torch.no_grad():
        pred = model(inputs)  # B x 3 x H x W
        if model.cfg.val_cfg is not None and model.cfg.val_cfg.get("flip_test", False):
            pred_flipped = model(inputs.flip(-1))  # B x 3 x H x W
            pred_flipped = pred_flipped.flip(-1)  ## B x K x heatmap_H x heatmap_W
            flip_indices = model.pose_metainfo["flip_indices"]
            assert len(flip_indices) == pred_flipped.shape[1]  ## K
            pred_flipped = pred_flipped[:, flip_indices]
            pred = (pred + pred_flipped) / 2.0

    # ------------------------------------------
    pred = pred.cpu().numpy()  ## B x K x heatmap_H x heatmap_W
    keypoints = []
    keypoint_scores = []
    for i, data_samples in enumerate(data_samples_list):
        ## kps in crop image
        ## keypoints_i is 1 x K x 2
        # keypoint_scores_i is 1 x K
        keypoints_i, keypoint_scores_i = model.codec.decode(pred[i])
        input_size = data_samples["meta"]["input_size"]  ## 1 x 2, 768 x 1024
        bbox_center = data_samples["meta"]["bbox_center"]  ## 1 x 2
        bbox_scale = data_samples["meta"]["bbox_scale"]  ## 1 x 2

        keypoints_i = (
            keypoints_i / input_size * bbox_scale + bbox_center - 0.5 * bbox_scale
        )
        keypoints.append(keypoints_i[0])  ## remove fake batch dim
        keypoint_scores.append(keypoint_scores_i[0])  ## remove fake batch dim

    return keypoints, keypoint_scores, bboxes


# -------------------------------------------------------------------------------
def main():
    parser = ArgumentParser()
    parser.add_argument("det_checkpoint", help="Local DETR snapshot directory")
    parser.add_argument("config", help="Config file")
    parser.add_argument("checkpoint", help="Checkpoint file")
    parser.add_argument("--input", help="Input image dir")
    parser.add_argument("--output", default=None, help="Path to output dir")
    parser.add_argument("--fmasks-dir", default=None, help="Foreground masks directory")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--radius", type=int, default=3, help="Keypoint radius for visualization"
    )
    parser.add_argument(
        "--thickness", type=int, default=1, help="Link thickness for visualization"
    )
    parser.add_argument(
        "--kpt-thr", type=float, default=0.3, help="Visualizing keypoint thresholds"
    )
    parser.add_argument(
        "--bbox-thr", type=float, default=0.3, help="Bounding box score threshold"
    )
    parser.add_argument(
        "--nms-thr", type=float, default=0.3, help="IoU threshold for bounding box NMS"
    )
    parser.add_argument(
        "--mask-bbox-overlap-thr",
        type=float,
        default=0.15,
        help="Minimum fraction of the foreground mask that a bbox must cover to be kept",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Maximum number of detected subjects to keep per frame after mask filtering",
    )
    parser.add_argument(
        "--no-save-json",
        action="store_true",
        help="Disable saving per-video predictions JSON (saved by default).",
    )
    parser.add_argument(
        "--predictions-name",
        default=None,
        help="Override predictions JSON filename (used by helper for per-chunk writes).",
    )

    args = parser.parse_args()

    model = init_model(args.config, args.checkpoint, device=args.device)
    os.makedirs(args.output, exist_ok=True)

    ## add pose metainfo to model
    num_keypoints = model.cfg.num_keypoints
    if num_keypoints == 308:
        metainfo_file = _resolve_keypoints308_metainfo_file()
        model.pose_metainfo = parse_pose_metainfo(
            dict(from_file=metainfo_file)
        )

    ## add codec to model
    codec_type = model.cfg.codec.pop("type")
    assert codec_type == "UDPHeatmap", "Only support UDPHeatmap"
    model.codec = UDPHeatmap(**model.cfg.codec)

    # warm up the bbox detector (loads from args.det_checkpoint)
    _get_detector(args.device, args.det_checkpoint)

    # Get image list
    if os.path.isdir(args.input):
        input_dir = args.input
        image_names = [
            name
            for name in sorted(os.listdir(input_dir))
            if name.endswith((".jpg", ".png", ".jpeg"))
        ]
    else:
        with open(args.input, "r") as f:
            image_paths = [line.strip() for line in f if line.strip()]
        image_names = [os.path.basename(path) for path in image_paths]
        input_dir = os.path.dirname(image_paths[0])

    frames_records = []
    image_size = None
    num_keypoints_seen = None

    for image_name in tqdm(image_names, total=len(image_names)):
        image_path = os.path.join(input_dir, image_name)
        image = cv2.imread(image_path)
        fg_mask = _load_mask_for_image(args.fmasks_dir, image_name, image.shape) if image is not None else None

        try:
            keypoints, keypoint_scores, bboxes = process_one_image(
                args, image, model, fg_mask=fg_mask
            )
        except Exception as e:
            print(f"[vis_pose] inference failed on {image_name}: {e}")
            continue

        if image_size is None:
            image_size = [int(image.shape[0]), int(image.shape[1])]
        if num_keypoints_seen is None and len(keypoints) > 0:
            num_keypoints_seen = int(np.asarray(keypoints[0]).shape[0])

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        vis_image_rgb = visualize_keypoints(
            image=image_rgb,
            keypoints=keypoints,
            keypoints_visible=np.ones_like(keypoint_scores) > 0,
            keypoint_scores=keypoint_scores,
            radius=args.radius,
            thickness=args.thickness,
            kpt_thr=args.kpt_thr,
            skeleton=model.pose_metainfo["skeleton_links"],
            kpt_color=model.pose_metainfo["keypoint_colors"],
            link_color=model.pose_metainfo["skeleton_link_colors"],
        )
        vis_image = cv2.cvtColor(vis_image_rgb, cv2.COLOR_RGB2BGR)
        save_path = os.path.join(args.output, image_name)
        cv2.imwrite(save_path, vis_image)

        if not args.no_save_json:
            try:
                instances = []
                for kpts, scores, bbox in zip(keypoints, keypoint_scores, bboxes):
                    instances.append({
                        "bbox": [float(v) for v in np.asarray(bbox).reshape(-1)[:4]],
                        "keypoints": np.asarray(kpts, dtype=float).tolist(),
                        "keypoint_scores": np.asarray(scores, dtype=float).reshape(-1).tolist(),
                    })
                frames_records.append({
                    "image_name": image_name,
                    "instances": instances,
                })
            except Exception as e:
                print(f"[vis_pose] json record failed on {image_name}: {e}")

    if not args.no_save_json:
        nn = os.path.basename(os.path.normpath(args.output))
        parent_nn = os.path.basename(os.path.dirname(os.path.normpath(args.output)))
        video_label = parent_nn if nn.endswith("_output") else nn
        json_filename = args.predictions_name or f"{video_label}_predictions.json"
        json_path = os.path.join(args.output, json_filename)
        payload = {
            "video": video_label,
            "image_size": image_size,
            "num_keypoints": num_keypoints_seen,
            "kpt_thr_used": float(args.kpt_thr),
            "frames": frames_records,
        }
        with open(json_path, "w") as f:
            json.dump(payload, f)
        print(f"[vis_pose] wrote predictions: {json_path} ({len(frames_records)} frames)")


if __name__ == "__main__":
    main()
