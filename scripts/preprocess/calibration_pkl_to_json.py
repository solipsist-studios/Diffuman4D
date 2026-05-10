import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an OpenCV calibration .pkl file into a transforms.json "
            "with global intrinsics for pose prediction."
        )
    )
    parser.add_argument(
        "--calibration_pkl",
        type=Path,
        required=True,
        help="Path to calibration .pkl containing camera_matrix and distortion_coefficients.",
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

SUPPORTED_MODELS = {'OPENCV', 'OPENCV_FISHEYE', 'PINHOLE'}

def infer_camera_model(model, dist):
    if model in SUPPORTED_MODELS:
        return model

    # Backward compatibility for older calibration files that do not include model.
    flat = np.array(dist).reshape(-1)
    return 'OPENCV_FISHEYE' if flat.size == 4 else 'OPENCV'


def main() -> None:
    args = parse_args()
    data = load_calibration(args.calibration_pkl)

    mtx = np.asarray(data["camera_matrix"], dtype=np.float64)
    dist = np.asarray(data["distortion_coefficients"], dtype=np.float64).reshape(-1)

    if mtx.shape != (3, 3):
        raise ValueError(f"camera_matrix must have shape (3, 3), got {mtx.shape}")

    fx = float(mtx[0, 0])
    fy = float(mtx[1, 1])
    cx = float(mtx[0, 2])
    cy = float(mtx[1, 2])

    width = args.width
    height = args.height
    image_size = data.get("image_size")
    if (width is None or height is None) and image_size is not None and len(image_size) == 2:
        if width is None:
            width = int(image_size[0])
        if height is None:
            height = int(image_size[1])

    if width is None or height is None:
        raise ValueError(
            "Could not determine width/height. Provide --width and --height, "
            "or include image_size=(w, h) in calibration pkl."
        )

    camera_model = args.camera_model or infer_camera_model(data.get("model"), dist)

    transforms = {
        "camera_model": camera_model,
        "w": int(width),
        "h": int(height),
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy
    }

    # Map distortion coefficients to the keys expected by predict_poses.py.
    if camera_model == "OPENCV_FISHEYE":
        transforms["k1"] = float(dist[0]) if dist.size > 0 else 0.0
        transforms["k2"] = float(dist[1]) if dist.size > 1 else 0.0
        transforms["k3"] = float(dist[2]) if dist.size > 2 else 0.0
        transforms["k4"] = float(dist[3]) if dist.size > 3 else 0.0

    elif camera_model == "OPENCV":
        transforms["k1"] = float(dist[0]) if dist.size > 0 else 0.0
        transforms["k2"] = float(dist[1]) if dist.size > 1 else 0.0
        transforms["p1"] = float(dist[2]) if dist.size > 2 else 0.0
        transforms["p2"] = float(dist[3]) if dist.size > 3 else 0.0

    transforms["frames"] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(transforms, f, indent=args.indent)

    print(f"Wrote transforms JSON to: {args.output}")
    print(f"camera_model={camera_model}, w={width}, h={height}")
    print(f"fl_x={fx:.6f}, fl_y={fy:.6f}, cx={cx:.6f}, cy={cy:.6f}")


if __name__ == "__main__":
    main()