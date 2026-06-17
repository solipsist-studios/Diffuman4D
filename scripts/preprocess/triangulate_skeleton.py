from __future__ import annotations
import os, os.path as osp
import sys
import json
import re
import fire
import numpy as np
from easyvolcap.utils.console_utils import tqdm
from easyvolcap.utils.parallel_utils import parallel_execution


sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
from src.data.utils.camera_parser import parse_cameras
from scripts.preprocess.utils.triang_utils import triangulate_points, project_points


def _extract_tem_label_from_frame(frame: dict) -> str | None:
    for key in ("tem_label", "temporal_label", "frame_label", "time_label", "frame_id"):
        if key in frame and frame[key] is not None:
            return str(frame[key])
    file_path = frame.get("file_path")
    if file_path:
        return osp.splitext(osp.basename(str(file_path)))[0]
    return None


def _candidate_camera_labels(label: str) -> list[str]:
    s = str(label)
    candidates = [s]
    if s.isdigit():
        i = int(s)
        candidates.extend([str(i), f"{i:02d}"])
    if len(s) > 1 and s.startswith("0") and s[1:].isdigit():
        candidates.append(str(int(s)))
    # Preserve order while removing duplicates.
    seen = set()
    out = []
    for c in candidates:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _candidate_tem_labels(label: str) -> list[str]:
    s = str(label)
    candidates = [s]
    if s.isdigit():
        i = int(s)
        candidates.extend([str(i), f"{i:06d}"])
    if len(s) > 1 and s.startswith("0") and s[1:].isdigit():
        candidates.append(str(int(s)))
    seen = set()
    out = []
    for c in candidates:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _detect_kp2d_source(kp2d_dir: str) -> tuple[str, str | None]:
    """Detect whether kp2d input is legacy folder layout or combined predictions JSON.

    Returns:
        ("legacy_dir", None) or ("combined_json", <json_path>)
    """
    if osp.isfile(kp2d_dir) and kp2d_dir.endswith(".json"):
        return "combined_json", kp2d_dir

    if not osp.isdir(kp2d_dir):
        raise FileNotFoundError(f"kp2d_dir not found: {kp2d_dir}")

    # Legacy layout has per-camera subdirectories.
    if any(osp.isdir(osp.join(kp2d_dir, name)) for name in os.listdir(kp2d_dir)):
        return "legacy_dir", None

    # New style from vis_pose writes a *_predictions.json in output directory.
    json_files = [
        osp.join(kp2d_dir, name)
        for name in os.listdir(kp2d_dir)
        if name.endswith(".json")
    ]
    pred_candidates = [p for p in json_files if p.endswith("_predictions.json")]
    for candidate in pred_candidates + json_files:
        try:
            with open(candidate, "r") as f:
                payload = json.load(f)
            if isinstance(payload, dict) and isinstance(payload.get("frames", None), list):
                return "combined_json", candidate
        except Exception:
            continue

    return "legacy_dir", None


def _extract_tem_label_from_image_name(image_name: str) -> str | None:
    # Strip directory component (handles both / and \\ separators).
    basename = str(image_name).replace("\\", "/").split("/")[-1]
    # Strip multi-part extensions like ".mp4.thumb.jpg" by taking only the
    # portion before the first dot.  Without this, "0001_0001.mp4.thumb.jpg"
    # would produce stem "0001_0001.mp4.thumb" and findall would return
    # ["0001", "0001", "4"] — picking "4" from "mp4" as the last token.
    primary = osp.splitext(basename)[0].split(".")[0]
    matches = re.findall(r"\d+", primary)
    if not matches:
        return None
    # Prefer the last numeric token in the primary stem as temporal label.
    return matches[-1]


def _infer_tem_label_from_frame(frame: dict, basename: str, camera_label: str | None) -> str | None:
    """Infer temporal label from a frame record with camera-aware fallback.

    If a frame has no explicit temporal field and its basename stem equals the
    camera label (e.g. ``undistorted_0002.webp`` with camera
    ``undistorted_0002``), treat it as a single-frame capture and use
    ``0001`` so all cameras share the same temporal label.
    """
    for tkey in ("tem_label", "temporal_label", "frame_label", "time_label", "frame_id"):
        if frame.get(tkey) is not None:
            return str(frame[tkey])

    stem = osp.splitext(str(basename))[0]
    if camera_label is not None and stem == str(camera_label):
        return "0001"

    return _extract_tem_label_from_image_name(stem)


def _cross_platform_basename(file_path: str) -> str:
    """Return the basename of a path that may use Windows or POSIX separators."""
    return str(file_path).replace("\\", "/").split("/")[-1]


def _build_image_to_cam_tem_map(camera_path: str) -> tuple[
    dict[str, tuple[str, str]],
    dict[str, tuple[str, str]],
]:
    """Build two lookup tables from transforms frames:
    - exact: basename(file_path) -> (camera_label, tem_label)
    - stem:  stem(file_path)     -> (camera_label, tem_label)

    The stem index provides a fallback for image names that carry extra
    prefixes (e.g. ``undistorted_``) or differ only in extension.
    """
    exact: dict[str, tuple[str, str]] = {}
    stem:  dict[str, tuple[str, str]] = {}

    if not camera_path.endswith(".json") or not osp.isfile(camera_path):
        return exact, stem

    with open(camera_path, "r") as f:
        tfs = json.load(f)

    for frame in tfs.get("frames", []):
        cam = frame.get("camera_label")
        file_path = frame.get("file_path")
        if cam is None or not file_path:
            continue
        basename = _cross_platform_basename(str(file_path))
        stem_key = osp.splitext(basename)[0]
        tem = _infer_tem_label_from_frame(frame, basename=basename, camera_label=str(cam))
        if tem is None:
            continue
        exact[basename] = (str(cam), str(tem))
        stem[stem_key] = (str(cam), str(tem))
    return exact, stem


def _load_combined_kp2d(
    combined_json_path: str,
    camera_path: str,
    dtype=np.float64,
) -> tuple[dict[tuple[str, str], tuple[np.ndarray, np.ndarray | None]], list[str], list[str]]:
    """Load combined predictions JSON and index by (camera_label, tem_label)."""
    with open(combined_json_path, "r") as f:
        payload = json.load(f)
    frames = payload.get("frames", [])
    exact_map, stem_map = _build_image_to_cam_tem_map(camera_path)

    def _lookup_cam_tem(image_name: str) -> tuple[str, str] | None:
        """Try progressively looser matches against the transforms-derived maps."""
        name = str(image_name)
        basename = _cross_platform_basename(name)
        stem_key = osp.splitext(basename)[0]

        # 1. Exact basename match.
        if basename in exact_map:
            return exact_map[basename]
        # 2. Exact stem match (extension differs).
        if stem_key in stem_map:
            return stem_map[stem_key]
        # 3. Suffix match: the transforms entry's basename is a suffix of the
        #    prediction image name.  Handles prefixes like ``undistorted_``.
        for key, val in exact_map.items():
            if basename.endswith(key) or basename.endswith(osp.splitext(key)[0]):
                return val
        # 4. Partial stem match: find a transforms stem that appears inside
        #    the prediction stem (e.g. "0001_0001.mp4.thumb" inside
        #    "undistorted_0001_0001.mp4.thumb").
        for key, val in stem_map.items():
            if key and key in stem_key:
                return val
        return None

    by_cam_tem: dict[tuple[str, str], tuple[np.ndarray, np.ndarray | None]] = {}
    cams_seen: set[str] = set()
    tem_seen: set[str] = set()

    for frame in frames:
        image_name = frame.get("image_name")
        instances = frame.get("instances", [])
        if not image_name or not instances:
            continue

        # Use the first instance for compatibility with legacy single-instance files.
        instance = instances[0]
        keypoints = instance.get("keypoints")
        if keypoints is None:
            continue
        kp = np.array(keypoints, dtype=dtype)

        kp_score = None
        if "keypoint_scores" in instance:
            kp_score = np.array(instance["keypoint_scores"], dtype=dtype)

        # Map to camera/temporal labels.
        cam = frame.get("camera_label")
        tem = frame.get("tem_label") or frame.get("temporal_label")
        if cam is None or tem is None:
            mapped = _lookup_cam_tem(str(image_name))
            if mapped is not None:
                cam, tem = mapped

        if cam is None or tem is None:
            continue

        cam = str(cam)
        tem = str(tem)

        by_cam_tem[(cam, tem)] = (kp, kp_score)
        cams_seen.add(cam)
        tem_seen.add(tem)

    cams = sorted(cams_seen)
    tems = sorted(tem_seen, key=lambda x: int(x) if str(x).isdigit() else str(x))
    return by_cam_tem, cams, tems


def read_kp2d(path, dtype=np.float64):
    with open(path, "r") as f:
        pose = json.load(f)
    instance = pose["instance_info"][0]
    kp = np.array(instance["keypoints"], dtype=dtype)
    kp_depth = kp_score = None
    if "keypoint_depths" in instance:
        kp_depth = np.array(instance["keypoint_depths"], dtype=dtype)
    if "keypoint_scores" in instance:
        kp_score = np.array(instance["keypoint_scores"], dtype=dtype)

    # Re-scale finger scores with hand root confidence for legacy 133-keypoint layout.
    # For 308-keypoint layouts, these indices no longer map to the same semantics.
    if kp_score is not None and kp_score.shape[0] == 133:
        kp_score[92:112] *= kp_score[91] ** 2
        kp_score[113:133] *= kp_score[112] ** 2

    return kp, kp_depth, kp_score


def write_kp2d(path, kp, kp_depth=None, kp_score=None):
    instance = {"keypoints": kp.tolist()}
    if kp_depth is not None:
        instance["keypoint_depths"] = kp_depth.tolist()
    if kp_score is not None:
        instance["keypoint_scores"] = kp_score.tolist()
    pose = {"instance_info": [instance]}
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(pose, f, indent=4)


def write_kp3d(path, kp3d, kp3d_reproj):
    instance = {
        "keypoints": kp3d.tolist(),
        "keypoint_reproj": kp3d_reproj.tolist(),
    }
    pose = {"instance_info": [instance]}
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(pose, f, indent=4)


def write_kp3d_pcd(path, kp3d):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(kp3d)
    os.makedirs(osp.dirname(path), exist_ok=True)
    o3d.io.write_point_cloud(path, pcd)


def triangulate_skeleton(
    camera_path: str,
    kp2d_dir: str,
    out_kp3d_dir: str,
    out_pcd_dir: str = None,
    out_kp2d_proj_dir: str = None,
    spa_labels_range: list[int] = None,
    spa_labels_proj_range: list[int] = None,
    tem_labels_range: list[int] = None,
    spa_labels: list[int] = None,
    spa_labels_proj: list[int] = None,
    tem_labels: list[int] = None,
    spa_label_prefix: str = "",
    spa_label_format: str = None,
    kp2d_padding: list[int, int] = None,
    intri_scale: float = None,
    skip_exists: bool = False,
    num_workers: int = 1,
    use_cuda: bool = False,
    score_thr: float = 0.6,
    dtype=np.float64,
):
    kp2d_mode, combined_json_path = _detect_kp2d_source(kp2d_dir)
    combined_kp2d = None
    combined_spa_labels = None
    combined_tem_labels = None
    if kp2d_mode == "combined_json":
        print(f"Detected combined keypoint JSON: {combined_json_path}")
        combined_kp2d, combined_spa_labels, combined_tem_labels = _load_combined_kp2d(
            combined_json_path, camera_path, dtype=dtype
        )

    # parse labels
    # Determine label format function
    if spa_label_format is not None:
        def format_spa_label(i):
            return spa_label_format.format(i)
    else:
        def format_spa_label(i):
            return f"{spa_label_prefix}{int(i):02d}"
    
    if spa_labels is not None:
        if spa_labels_range is not None:
            raise ValueError("spa_labels and spa_labels_range cannot be specified together")
        spa_labels = [format_spa_label(int(i)) for i in spa_labels]
    elif spa_labels_range is not None:
        b, e, s = spa_labels_range
        spa_labels = [format_spa_label(int(i)) for i in range(b, e, s)]
    else:
        if kp2d_mode == "combined_json":
            spa_labels = combined_spa_labels
        else:
            spa_labels = sorted(os.listdir(kp2d_dir))

    if spa_labels_proj is not None:
        if spa_labels_proj_range is not None:
            raise ValueError("spa_labels_proj and spa_labels_proj_range cannot be specified together")
        spa_labels_proj = [format_spa_label(int(i)) for i in spa_labels_proj]
    elif spa_labels_proj_range is not None:
        b, e, s = spa_labels_proj_range
        spa_labels_proj = [format_spa_label(int(i)) for i in range(b, e, s)]
    else:
        if kp2d_mode == "combined_json":
            spa_labels_proj = combined_spa_labels
        else:
            spa_labels_proj = sorted(os.listdir(kp2d_dir))

    print(f"Using spatial labels: {spa_labels}")
    print(f"Using spatial projection labels: {spa_labels_proj}")

    if tem_labels is not None:
        if tem_labels_range is not None:
            raise ValueError("tem_labels and tem_label_range cannot be specified together")
        tem_labels = [f"{int(i):06d}" for i in tem_labels]
    elif tem_labels_range is not None:
        b, e, s = tem_labels_range
        tem_labels = [f"{int(i):06d}" for i in range(b, e, s)]
    else:
        if kp2d_mode == "combined_json":
            tem_labels = combined_tem_labels
        else:
            tem_labels = sorted(os.listdir(osp.join(kp2d_dir, spa_labels[0])))
            tem_labels = [label.split(".")[0] for label in tem_labels]

    # Load static cameras as baseline. For Nerfstudio-style JSON, optionally
    # override with per-frame intrinsics/extrinsics when available.
    cams = parse_cameras(camera_path, coord_system="opencv", normalize_scene=False)

    frame_cam_map: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    if camera_path.endswith(".json"):
        with open(camera_path, "r") as f:
            tfs = json.load(f)
        for frame in tfs.get("frames", []):
            cam_label = frame.get("camera_label")
            _fp = frame.get("file_path", "")
            basename = _cross_platform_basename(str(_fp)) if _fp else ""
            tem_label = _infer_tem_label_from_frame(
                frame,
                basename=basename,
                camera_label=str(cam_label) if cam_label is not None else None,
            )
            if cam_label is None or tem_label is None:
                continue

            fx = frame.get("fl_x", tfs.get("fl_x"))
            fy = frame.get("fl_y", tfs.get("fl_y"))
            cx = frame.get("cx", tfs.get("cx"))
            cy = frame.get("cy", tfs.get("cy"))
            if any(x is None for x in [fx, fy, cx, cy]):
                continue

            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=dtype)
            pose = np.array(frame["transform_matrix"], dtype=dtype)
            pose[:3, 1:3] *= -1  # convert opengl -> opencv
            T = np.linalg.inv(pose)

            frame_cam_map[(str(cam_label), str(tem_label))] = (K, T)

    def _get_cam_mats(camera_label: str, tem_label: str) -> tuple[np.ndarray, np.ndarray]:
        # Try per-frame cameras first.
        for cam_key in _candidate_camera_labels(camera_label):
            for tem_key in _candidate_tem_labels(tem_label):
                key = (cam_key, tem_key)
                if key in frame_cam_map:
                    return frame_cam_map[key]

        # Fall back to static camera entry.
        for cam_key in _candidate_camera_labels(camera_label):
            if cam_key in cams:
                K = np.array(cams[cam_key]["K"], dtype=dtype)
                T = np.array(np.linalg.inv(cams[cam_key]["pose"]), dtype=dtype)
                return K, T

        raise KeyError(f"Camera label not found: {camera_label}")

    def triangulate_one_skeleton(tem_label):
        out_kp3d_path = osp.join(out_kp3d_dir, f"{tem_label}.json")
        out_pcd_path = osp.join(out_pcd_dir, f"{tem_label}.ply")
        out_kp2d_proj_paths = [osp.join(out_kp2d_proj_dir, spa_label, f"{tem_label}.json") for spa_label in spa_labels_proj]

        Ks, Ts = zip(*[_get_cam_mats(spa_label, tem_label) for spa_label in spa_labels])
        Ks_proj, Ts_proj = zip(*[_get_cam_mats(spa_label, tem_label) for spa_label in spa_labels_proj])
        Ks = np.stack(Ks)
        Ts = np.stack(Ts)
        Ks_proj = np.stack(Ks_proj)
        Ts_proj = np.stack(Ts_proj)

        if intri_scale is not None:
            Ks = Ks * intri_scale
            Ks[:, -1, -1] = 1.0
            Ks_proj = Ks_proj * intri_scale
            Ks_proj[:, -1, -1] = 1.0

        if skip_exists and osp.exists(out_kp3d_path):
            try:
                json.load(open(out_kp3d_path, "r"))
                return
            except Exception as e:
                print(f"Error loading {out_kp3d_path}: {e}, skipping...")

        # read 2d keypoints
        if kp2d_mode == "combined_json":
            rows = []
            for spa_label in spa_labels:
                found = None
                for cam_key in _candidate_camera_labels(spa_label):
                    for tem_key in _candidate_tem_labels(tem_label):
                        pair = (cam_key, tem_key)
                        if pair in combined_kp2d:
                            found = combined_kp2d[pair]
                            break
                    if found is not None:
                        break
                if found is None:
                    raise FileNotFoundError(
                        f"Missing keypoints in combined json for camera={spa_label}, tem={tem_label}"
                    )
                kp, kp_score = found
                rows.append((kp, kp_score))

            kp2d = np.stack([r[0] for r in rows])
            if any(r[1] is None for r in rows):
                kp2d_score = np.ones(kp2d.shape[:2], dtype=dtype)
            else:
                kp2d_score = np.stack([r[1] for r in rows])
        else:
            kp2d_paths = [osp.join(kp2d_dir, spa_label, f"{tem_label}.json") for spa_label in spa_labels]
            kp2d, _, kp2d_score = zip(*[read_kp2d(p, dtype=dtype) for p in kp2d_paths])
            kp2d, kp2d_score = np.stack(kp2d), np.stack(kp2d_score)

        if kp2d_padding is not None:
            kp2d += np.array(kp2d_padding, dtype=dtype)[None]

        # triangulate keypoints
        kp3d, kp3d_reproj, _ = triangulate_points(
            Ks, Ts, kp2d, kp2d_score, 
            score_thr=score_thr, 
            use_cuda=use_cuda
        )

        # save 3d keypoints
        write_kp3d(out_kp3d_path, kp3d, kp3d_reproj)
        if out_pcd_dir is not None:
            write_kp3d_pcd(out_pcd_path, kp3d)

        # project 2d keypoints
        if out_kp2d_proj_dir is not None:
            kp2d_proj, kp2d_depth_proj, _ = project_points(kp3d, Ks_proj, Ts_proj, kp3d_score=None)
            for i in range(len(out_kp2d_proj_paths)):
                write_kp2d(
                    out_kp2d_proj_paths[i],
                    kp=kp2d_proj[i],
                    kp_depth=kp2d_depth_proj[i],
                    kp_score=None,
                )

    if num_workers > 1:
        parallel_execution(
            tem_labels,
            action=triangulate_one_skeleton,
            print_progress=True,
            desc=f"Triangulating skeletons",
            num_workers=num_workers,
            sequential=False,
        )
    else:
        for tem_label in tqdm(tem_labels, desc="Triangulating skeletons"):
            triangulate_one_skeleton(tem_label)


if __name__ == "__main__":
    # usage:
    # python scripts/preprocess/triangulate_skeleton.py \
    # --camera_path $DATADIR/transforms.json --kp2d_dir $DATADIR/poses_sapiens \
    # --out_kp3d_dir $DATADIR/poses_3d --out_pcd_dir $DATADIR/poses_pcd --out_kp2d_proj_dir $DATADIR/poses_2d
    fire.Fire(triangulate_skeleton)
