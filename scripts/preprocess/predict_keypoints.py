from __future__ import annotations
import os
import fire
import torch
import subprocess
import shutil
from huggingface_hub import hf_hub_download

# Prefer SAPIENS_CHECKPOINT_ROOT; fall back to SAPIENS_LITE_CHECKPOINT_ROOT; default to .\sapiens\lite
ckpt_root = os.environ.get("SAPIENS_CHECKPOINT_ROOT") or os.environ.get("SAPIENS_LITE_CHECKPOINT_ROOT") or r".\\sapiens\\lite"
ckpt_root = os.path.normpath(ckpt_root)


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
        repo_candidates = []
        env_repo = os.environ.get("SAPIENS_HF_REPO")
        if env_repo:
            repo_candidates.append(env_repo)
        # common/guess candidates - best-effort only
        repo_candidates.extend([
            "noahcao/sapiens-pose-coco",
        ])

    last_exc: Exception | None = None
    for repo in repo_candidates:
        try:
            print(f"Attempting to download {filename} from Huging Face repo '{repo}'...")
            # Try a couple filename variants: direct filename and any provided subdirectory variants
            # (useful since different repos store checkpoints under different subpaths)
            if filename_variants is None:
                filename_variants = [
                    filename,
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


def predict_keypoints(
    images_dir: str,
    out_kp2d_dir: str,
    fmasks_dir: str | None = None,
    sapiens_ckpt_path: str = f"{ckpt_root}/torchscript/pose/checkpoints/sapiens_2b/sapiens_2b_coco_wholebody_best_coco_wholebody_AP_745_torchscript.pt2",
    detector_ckpt_path: str = f"{ckpt_root}/detector/checkpoints/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth",
    gpu_ids: tuple[int, ...] | None = None,
    num_workers: int = 4,
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
            _ensure_hf_file(sapiens_ckpt_path, sapiens_ckpt_filename)
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
    vis_pose_script = os.path.normpath(os.path.join(repo_root, "scripts/preprocess/sapiens/lite/demo/vis_pose.py"))
    det_config_path = os.path.normpath(os.path.join(repo_root, "scripts/preprocess/sapiens/lite/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person_no_nms.py"))

    # Build the command and only include --gpu_ids when we have at least one GPU id.
    cmd_parts = [
        "python",
        vis_pose_script,
        sapiens_ckpt_path,
        "--det-checkpoint",
        detector_ckpt_path,
        "--det-config",
        det_config_path,
        "--images_dir",
        images_dir,
        "--kpt-thr",
        kpt_thr
    ]

    if fmasks_dir is not None:
        cmd_parts.extend(["--fmasks_dir", fmasks_dir])

    cmd_parts.extend(["--image_ext", image_ext])
    
    # Add shape argument if specified
    if input_height is not None and input_width is not None:
        cmd_parts.extend(["--shape", str(input_height), str(input_width)])
    elif input_height is not None:
        cmd_parts.extend(["--shape", str(input_height)])

    cmd_parts.extend(["--output_dir", out_kp2d_dir, "--skip_exists"])

    if gpu_ids and len(gpu_ids) > 0:
        gpu_arg = ",".join(map(str, gpu_ids))
        cmd_parts.extend(["--gpu_ids", gpu_arg])
    else:
        # No GPU devices detected; do not pass the flag so vis_pose.py will use CPU defaults
        print("No CUDA devices detected — running vis_pose.py on CPU (omitting --gpu_ids flag)")

    cmd_parts.extend(["--num_workers", str(num_workers)])

    if save_img is not None:
        cmd_parts.extend(["--save_image"])

    # Join into a single string for shell execution (keeps previous behavior).
    cmd = " ".join(map(str, cmd_parts))
    print(f"Running command from {os.path.dirname(__file__)}")
    print(f"Command: {cmd}")
    # Run from scripts/preprocess directory where vis_pose.py can resolve relative paths
    subprocess.run(cmd, cwd=os.path.dirname(__file__), check=False, shell=True)


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
