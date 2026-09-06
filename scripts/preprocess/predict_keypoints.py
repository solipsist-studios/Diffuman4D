from __future__ import annotations
import os
import sys
import importlib
import fire
import torch
import subprocess
import shutil
from huggingface_hub import hf_hub_download, snapshot_download

# Prefer SAPIENS_CHECKPOINT_ROOT; fall back to SAPIENS_2_CHECKPOINT_ROOT; fall back to SAPIENS_LITE_CHECKPOINT_ROOT; default to ./sapiens/2
ckpt_root = os.environ.get("SAPIENS_CHECKPOINT_ROOT") or os.environ.get("SAPIENS_2_CHECKPOINT_ROOT") or os.environ.get("SAPIENS_LITE_CHECKPOINT_ROOT") or os.path.join(".", "sapiens", "2")
ckpt_root = os.path.normpath(ckpt_root)


def _default_pose_repo_candidates(filename: str) -> list[str]:
    """Return best-effort repo candidates for Sapiens2 pose checkpoints."""
    candidates: list[str] = []
    env_repo = os.environ.get("SAPIENS_HF_REPO")
    if env_repo:
        candidates.append(env_repo)

    base_name = os.path.basename(filename)
    model_repo_map = {
        "sapiens2_0.4b_pose.safetensors": "facebook/sapiens2-pose-0.4b",
        "sapiens2_0.8b_pose.safetensors": "facebook/sapiens2-pose-0.8b",
        "sapiens2_1b_pose.safetensors": "facebook/sapiens2-pose-1b",
        "sapiens2_5b_pose.safetensors": "facebook/sapiens2-pose-5b",
    }
    mapped_repo = model_repo_map.get(base_name)
    if mapped_repo:
        candidates.append(mapped_repo)

    # Legacy fallback candidate kept for compatibility with old setups.
    candidates.append("noahcao/sapiens-pose-coco")

    # Keep order, drop duplicates.
    unique_candidates = []
    for repo in candidates:
        if repo not in unique_candidates:
            unique_candidates.append(repo)
    return unique_candidates


def _ensure_hf_file(
    target_path: str,
    filename: str,
    repo_candidates: list[str] | None = None,
    filename_variants: list[str] | None = None,
) -> str:
    """
    Ensure `target_path` exists. If missing, try to download `filename` from a list of Hugging Face
    repo ids using `hf_hub_download`. On success, the file will be placed at `target_path`.

    repo_candidates: optional list of repo_id strings to try. If None, we will try an env var
    `SAPIENS_HF_REPO` (if set) followed by a small list of plausible defaults.
    """
    if os.path.exists(target_path):
        return target_path

    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    if repo_candidates is None:
        repo_candidates = _default_pose_repo_candidates(filename)

    last_exc: Exception | None = None
    for repo in repo_candidates:
        try:
            print(f"Attempting to download {filename} from Hugging Face repo '{repo}'...")
            # Try a couple filename variants: direct filename and any provided subdirectory variants
            # (useful since different repos store checkpoints under different subpaths)
            if filename_variants is None:
                filename_variants = [
                    filename,
                    os.path.basename(filename),
                    f"sapiens_lite_host/torchscript/pose/checkpoints/sapiens_2b/{filename}",
                ]
            downloaded = None
            for fname_var in filename_variants:
                try:
                    print(f"  trying path: {fname_var}")
                    downloaded = hf_hub_download(repo_id=repo, filename=fname_var)
                    # If successful, break out
                    filename = fname_var
                    break
                except Exception as inner_e:
                    # try next variant
                    print(f"    not found at {fname_var}: {inner_e}")
                    continue
            if downloaded is None:
                raise RuntimeError(f"File not found in repo {repo} using variants {filename_variants}")
            # hf_hub_download returns the path to the cached file; copy to target_path if needed
            if os.path.abspath(downloaded) != os.path.abspath(target_path):
                shutil.copyfile(downloaded, target_path)
            print(f"Downloaded {filename} to {target_path}")
            return target_path
        except Exception as e:
            print(f"Download from {repo} failed: {e}")
            last_exc = e

    raise FileNotFoundError(
        f"Could not find or download '{filename}'. Tried repos: {repo_candidates}."
    ) from last_exc


def _require_module(module_name: str, install_hint: str) -> None:
    """Raise a clear error early when required runtime modules are missing."""
    try:
        importlib.import_module(module_name)
    except Exception as e:
        raise ModuleNotFoundError(
            f"Missing required Python module '{module_name}'. {install_hint}"
        ) from e


def _resolve_sapiens_config_path(config_path: str, repo_root: str) -> str:
    """Resolve a Sapiens config file path across local repo and installed package locations."""
    # 1) Directly provided path.
    if os.path.isabs(config_path) and os.path.exists(config_path):
        return os.path.normpath(config_path)

    # 2) Relative to current cwd.
    if os.path.exists(config_path):
        return os.path.normpath(config_path)

    # 3) Relative to repository root.
    repo_candidate = os.path.normpath(os.path.join(repo_root, config_path))
    if os.path.exists(repo_candidate):
        return repo_candidate

    # 4) Installed sapiens package location.
    sapiens_mod = importlib.import_module("sapiens")
    sapiens_root = os.path.dirname(sapiens_mod.__file__)

    package_candidates = [
        os.path.join(sapiens_root, config_path),
        os.path.join(sapiens_root, "pose", config_path),
    ]
    if config_path.startswith("configs/"):
        rel = config_path[len("configs/") :]
        package_candidates.append(os.path.join(sapiens_root, "pose", "configs", rel))

    for candidate in package_candidates:
        candidate = os.path.normpath(candidate)
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not resolve sapiens config path. "
        f"Given='{config_path}'. Tried repo root and installed package locations."
    )


def _ensure_detector_snapshot(detector_ckpt_path: str, detector_repo_id: str) -> str:
    """Ensure local DETR checkpoint snapshot exists; download when missing."""
    if os.path.isdir(detector_ckpt_path) and len(os.listdir(detector_ckpt_path)) > 0:
        return detector_ckpt_path

    os.makedirs(detector_ckpt_path, exist_ok=True)
    print(
        f"Detector checkpoint directory not found; downloading '{detector_repo_id}' to {detector_ckpt_path}..."
    )
    try:
        snapshot_download(repo_id=detector_repo_id, local_dir=detector_ckpt_path)
        print(f"Detector snapshot ready: {detector_ckpt_path}")
        return detector_ckpt_path
    except Exception as e:
        # Fallback: pass repo id directly to from_pretrained in vis_pose.py.
        print(
            f"Warning: detector snapshot download failed ({e}). "
            f"Falling back to remote repo id '{detector_repo_id}'."
        )
        return detector_repo_id


def predict_keypoints(
    images_dir: str,
    out_kp2d_dir: str,
    fmasks_dir: str | None = None,
    max_subjects: int | None = None,
    max_bone_sigma: float = 0.0,
    max_bilateral_ratio: float = 3.5,
    sapiens_ckpt_path: str = f"{ckpt_root}/pose/sapiens2_1b_pose.safetensors",
    config_path: str = None,
    detector_ckpt_path: str = f"{ckpt_root}/detector/detr-resnet-101-dc5",
    gpu_ids: tuple[int, ...] | None = None,
    kpt_thr: float = 0.3,
    save_img: bool | None = None,
    image_ext: str = ".jpg",
    input_height: int | None = None,
    input_width: int | None = None
):
    # Normalize all paths to use the OS-appropriate separators
    images_dir = os.path.normpath(images_dir)
    out_kp2d_dir = os.path.normpath(out_kp2d_dir)
    if fmasks_dir is not None:
        fmasks_dir = os.path.normpath(fmasks_dir)
    image_ext = image_ext if image_ext.startswith(".") else f".{image_ext}"
    sapiens_ckpt_path = os.path.normpath(sapiens_ckpt_path)
    detector_ckpt_path = os.path.normpath(detector_ckpt_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(out_kp2d_dir, exist_ok=True)
    print(f"The results will be saved to: {out_kp2d_dir}")
    
    # Ensure the SAPIENS checkpoint exists locally; if not, try to download from Hugging Face
    sapiens_ckpt_filename = os.path.basename(sapiens_ckpt_path)
    if not os.path.exists(sapiens_ckpt_path):
        try:
            _ensure_hf_file(
                sapiens_ckpt_path,
                sapiens_ckpt_filename,
                repo_candidates=_default_pose_repo_candidates(sapiens_ckpt_filename),
            )
        except Exception as e:
            print(
                "Warning: Could not download the SAPIENS checkpoint automatically:",
                e,
            )
            print(
                "Please download the checkpoint manually and place it at:", sapiens_ckpt_path
            )
    if gpu_ids is None:
        gpu_ids = tuple(range(torch.cuda.device_count()))

    print(os.path.dirname(__file__))
    
    # Get absolute paths for relative paths that depend on repo root
    repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "../.."))
    
    # Resolve relative checkpoint paths from repository root.
    if not os.path.isabs(sapiens_ckpt_path):
        sapiens_ckpt_path = os.path.normpath(os.path.join(repo_root, sapiens_ckpt_path))
    if not os.path.isabs(detector_ckpt_path):
        detector_ckpt_path = os.path.normpath(os.path.join(repo_root, detector_ckpt_path))

    # Determine sapiens 2 paths
    vis_pose_script = os.path.normpath(os.path.join(repo_root, "scripts/preprocess/sapiens/2/demo/vis_pose.py"))

    # Preflight dependency checks so failures are explicit before launching subprocess.
    _require_module(
        "sapiens.pose.datasets",
        "Install the Sapiens2 package into the current environment. "
        "Example: pip install git+https://github.com/facebookresearch/sapiens2.git",
    )
    _require_module(
        "transformers",
        "Install transformers in the current environment. Example: pip install transformers",
    )

    detector_repo_id = os.environ.get("SAPIENS_DETECTOR_HF_REPO", "facebook/detr-resnet-101-dc5")
    detector_ckpt_path = _ensure_detector_snapshot(detector_ckpt_path, detector_repo_id)
    
    # For sapiens2, we need a config file. If not provided, we need to find one or use a default.
    # Sapiens2 configs are part of the sapiens package, so we'll reference them by name
    if config_path is None:
        # Try to find a config in common locations or use a relative path that sapiens will resolve
        config_path = "configs/keypoints308/shutterstock_goliath_3po/sapiens2_1b_keypoints308_shutterstock_goliath_3po-1024x768.py"
    config_path = _resolve_sapiens_config_path(config_path, repo_root)
    
    # Build the command for sapiens2 vis_pose.py
    # Arguments: det_checkpoint config checkpoint --input INPUT --output OUTPUT [--options]
    cmd_parts = [
        sys.executable,
        vis_pose_script,
        detector_ckpt_path,
        config_path,
        sapiens_ckpt_path,
        "--input",
        images_dir,
        "--output",
        out_kp2d_dir,
        "--kpt-thr",
        str(kpt_thr),
    ]

    if fmasks_dir is not None:
        cmd_parts.extend(["--fmasks-dir", fmasks_dir])

    if max_subjects is not None:
        cmd_parts.extend(["--max-subjects", str(max_subjects)])

    cmd_parts.extend(["--max-bone-sigma", str(max_bone_sigma)])
    cmd_parts.extend(["--max-bilateral-ratio", str(max_bilateral_ratio)])

    device = f"cuda:{gpu_ids[0]}" if gpu_ids else "cpu"
    cmd_parts.extend(["--device", device])
    
    # Skeleton overlay images are opt-in (vis_pose.py --save-vis); they are a
    # full-resolution encode per image that no downstream stage reads.
    if save_img:
        cmd_parts.append("--save-vis")

    print(f"Running command from {os.path.dirname(__file__)}")
    print(f"Command: {' '.join(map(str, cmd_parts))}")
    
    # Run from scripts/preprocess directory where vis_pose.py can resolve relative paths.
    subprocess.run(cmd_parts, cwd=os.path.dirname(__file__), check=False)


if __name__ == "__main__":
    # usage:
    # python scripts/preprocess/predict_keypoints.py --images_dir $DATADIR/images --fmasks_dir $DATADIR/fmasks --out_kp2d_dir $DATADIR/poses_2d
    # 
    # NOTE: This script now supports images of arbitrary resolution using a bbox-based approach:
    # 1. Images are loaded at original resolution
    # 2. Person bboxes are detected
    # 3. Each bbox is cropped and resized to 1024x1024 (the TorchScript model's native size)
    # 4. Pose estimation runs on the resized crops
    # 5. Keypoints are automatically remapped back to original image coordinates
    # 
    # The input_height/input_width parameters are DEPRECATED for TorchScript models.
    # They will be ignored as the bbox-based approach handles arbitrary resolutions.
    # For non-TorchScript models (.pth), custom dimensions may work but are untested.
    # 
    # Examples (works with any resolution images):
    # python scripts/preprocess/predict_keypoints.py --images_dir $DATADIR/images --out_kp2d_dir $DATADIR/poses_2d
    fire.Fire(predict_keypoints)
