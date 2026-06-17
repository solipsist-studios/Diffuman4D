import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert calibration .pkl file(s) into transforms-style JSON. "
            "If --calibration_path is a directory, all .pkl files are combined "
            "into one output JSON."
        )
    )
    parser.add_argument(
        "--calibration_path",
        "--calibration_pkl",
        dest="calibration_path",
        type=Path,
        required=True,
        help=(
            "Path to calibration .pkl, or a directory containing .pkl files with "
            "camera_matrix and distortion_coefficients."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output transforms.json file.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Optional image width override if not present in calibration data.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Optional image height override if not present in calibration data.",
    )
    parser.add_argument(
        "--camera_model",
        type=str,
        default=None,
        choices=["PINHOLE", "OPENCV", "OPENCV_FISHEYE"],
        help=(
            "Optional transforms camera model override. "
            "If omitted, inferred from calibration model/distortion length."
        ),
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level.",
    )
    parser.add_argument(
        "--file_path_dir",
        type=str,
        default="images",
        help=(
            "Directory prefix to use in each frame file_path. "
            'Example: "images" -> images/<camera_label>.jpg'
        ),
    )
    parser.add_argument(
        "--image_ext",
        type=str,
        default=".jpg",
        help=(
            "Image extension for generated frame file_path values. "
            "Accepts values like .jpg, jpg, .png, webp."
        ),
    )
    return parser.parse_args()


def load_calibration(path: Path) -> dict:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Calibration file does not exist: {path}")
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError("Calibration file must contain a dictionary.")
    if "camera_matrix" not in data or "distortion_coefficients" not in data:
        raise ValueError(
            "Calibration file is missing required keys: camera_matrix and/or distortion_coefficients."
        )
    return data


def collect_calibration_files(calibration_path: Path) -> list[Path]:
    if not calibration_path.exists():
        raise FileNotFoundError(f"calibration_path does not exist: {calibration_path}")

    if calibration_path.is_file():
        return [calibration_path]

    if not calibration_path.is_dir():
        raise ValueError(f"calibration_path must be a file or directory: {calibration_path}")

    files = sorted(p for p in calibration_path.iterdir() if p.is_file() and p.suffix.lower() == ".pkl")
    if not files:
        raise FileNotFoundError(f"No .pkl calibration files found in directory: {calibration_path}")
    return files

SUPPORTED_MODELS = {'OPENCV', 'OPENCV_FISHEYE', 'PINHOLE'}

def infer_camera_model(model, dist):
    if model in SUPPORTED_MODELS:
        return model

    # Backward compatibility for older calibration files that do not include model.
    flat = np.array(dist).reshape(-1)
    return 'OPENCV_FISHEYE' if flat.size == 4 else 'OPENCV'


def build_transforms(
    data: dict,
    width: int | None = None,
    height: int | None = None,
    camera_model: str | None = None,
) -> dict:
    mtx = np.asarray(data["camera_matrix"], dtype=np.float64)
    dist = np.asarray(data["distortion_coefficients"], dtype=np.float64).reshape(-1)

    if mtx.shape != (3, 3):
        raise ValueError(f"camera_matrix must have shape (3, 3), got {mtx.shape}")

    fx = float(mtx[0, 0])
    fy = float(mtx[1, 1])
    cx = float(mtx[0, 2])
    cy = float(mtx[1, 2])

    image_size = data.get("image_size")
    if (width is None or height is None) and image_size is not None and len(image_size) == 2:
        if width is None:
            width = int(image_size[0])
        if height is None:
            height = int(image_size[1])

    if width is None or height is None:
        raise ValueError(
            "Could not determine width/height. Provide overrides, "
            "or include image_size=(w, h) in calibration data."
        )

    resolved_model = camera_model or infer_camera_model(data.get("model"), dist)

    transforms = {
        "camera_model": resolved_model,
        "w": int(width),
        "h": int(height),
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "frames": [],
    }

    if resolved_model == "OPENCV_FISHEYE":
        transforms["k1"] = float(dist[0]) if dist.size > 0 else 0.0
        transforms["k2"] = float(dist[1]) if dist.size > 1 else 0.0
        transforms["k3"] = float(dist[2]) if dist.size > 2 else 0.0
        transforms["k4"] = float(dist[3]) if dist.size > 3 else 0.0
    elif resolved_model == "OPENCV":
        transforms["k1"] = float(dist[0]) if dist.size > 0 else 0.0
        transforms["k2"] = float(dist[1]) if dist.size > 1 else 0.0
        transforms["p1"] = float(dist[2]) if dist.size > 2 else 0.0
        transforms["p2"] = float(dist[3]) if dist.size > 3 else 0.0

    return transforms


def build_frame_from_calibration(
    label: str,
    transforms: dict,
    file_path_dir: str,
    image_ext: str,
) -> dict:
    # Per-frame pose is unknown from calibration alone; use identity as placeholder.
    identity = np.eye(4, dtype=np.float64).tolist()
    prefix = str(file_path_dir).replace("\\", "/").strip("/")
    ext = str(image_ext).strip()
    if not ext:
        ext = ".jpg"
    if not ext.startswith("."):
        ext = f".{ext}"

    if prefix:
        file_path = f"{prefix}/{label}{ext}"
    else:
        file_path = f"{label}{ext}"

    frame = {
        "file_path": file_path,
        "camera_label": label,
        "transform_matrix": identity,
    }

    for key, value in transforms.items():
        if key != "frames":
            frame[key] = value

    return frame


def main() -> None:
    args = parse_args()
    calibration_files = collect_calibration_files(args.calibration_path)

    frames: list[dict] = []
    first_transforms: dict | None = None
    for path in calibration_files:
        data = load_calibration(path)
        transforms = build_transforms(
            data=data,
            width=args.width,
            height=args.height,
            camera_model=args.camera_model,
        )
        if first_transforms is None:
            first_transforms = transforms
        frames.append(
            build_frame_from_calibration(
                path.stem,
                transforms,
                file_path_dir=args.file_path_dir,
                image_ext=args.image_ext,
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    payload = {"frames": frames}
    if first_transforms is not None and len(frames) == 1 and args.calibration_path.is_file():
        for key, value in first_transforms.items():
            if key != "frames":
                payload[key] = value

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=args.indent)

    print(f"Wrote transforms JSON to: {args.output}")
    if first_transforms is not None and len(frames) == 1 and args.calibration_path.is_file():
        print(f"camera_model={first_transforms['camera_model']}, w={first_transforms['w']}, h={first_transforms['h']}")
        print(
            f"fl_x={first_transforms['fl_x']:.6f}, fl_y={first_transforms['fl_y']:.6f}, "
            f"cx={first_transforms['cx']:.6f}, cy={first_transforms['cy']:.6f}"
        )
    else:
        print(f"Converted {len(frames)} calibration file(s) from: {args.calibration_path}")


if __name__ == "__main__":
    main()