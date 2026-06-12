import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from calibration_pkl_to_json import build_transforms, load_calibration


DEFAULT_CANDIDATES = (
    "calibration_6k.pkl",
    "calibration_data.pkl",
    "fullframe_calibration/calibration_6k.pkl",
    "fullframe_calibration/calibration_data.pkl",
    "widescreen_calibration/calibration_data.pkl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a transforms.json from a directory of per-camera calibration folders. "
            "Each exported frame receives its own camera intrinsics; global intrinsics can "
            "also be written as the first camera or the mean across cameras."
        )
    )
    parser.add_argument(
        "--calibration_root",
        type=Path,
        required=True,
        help="Root directory containing one subdirectory per camera.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the output transforms.json file.",
    )
    parser.add_argument(
        "--index_file",
        type=Path,
        default=None,
        help=(
            "Optional index file that maps camera numbers to calibration subdirectories, "
            "e.g. '1 - 3761'. Defaults to <calibration_root>/index.txt when present."
        ),
    )
    parser.add_argument(
        "--candidate",
        action="append",
        default=None,
        help=(
            "Relative pickle path to try inside each camera directory. "
            "May be passed multiple times. Defaults prefer fullframe_calibration/calibration_6k.pkl."
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Fallback width when a calibration pickle does not contain image_size.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Fallback height when a calibration pickle does not contain image_size.",
    )
    parser.add_argument(
        "--camera_model",
        type=str,
        default=None,
        choices=["PINHOLE", "OPENCV", "OPENCV_FISHEYE"],
        help="Optional camera model override applied to every calibration.",
    )
    parser.add_argument(
        "--global_intrinsics",
        type=str,
        default="mean",
        choices=["mean", "first", "none"],
        help=(
            "How to populate top-level intrinsics for tools that expect a single camera model. "
            "'mean' averages across cameras, 'first' copies the first camera, 'none' omits them."
        ),
    )
    parser.add_argument(
        "--reindex_labels",
        action="store_true",
        help=(
            "Rename cameras using numeric labels. If an index file is available, its mapping "
            "is used; otherwise labels are assigned by sorted directory order."
        ),
    )
    parser.add_argument(
        "--label_format",
        type=str,
        default="{:02d}",
        help="Python format string used with --reindex_labels, e.g. '{:02d}' or 'cam_{:03d}'.",
    )
    parser.add_argument(
        "--file_path_template",
        type=str,
        default="images/{camera_label}/000000.jpg",
        help="Template for each frame's file_path.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if a camera directory does not contain a usable calibration pickle.",
    )
    return parser.parse_args()


def resolve_calibration_path(camera_dir: Path, candidates: tuple[str, ...]) -> Path | None:
    for candidate in candidates:
        path = camera_dir / candidate
        if path.is_file():
            return path
    return None


def resolve_index_file(calibration_root: Path, index_file: Path | None) -> Path | None:
    if index_file is not None:
        return index_file

    default_index = calibration_root / "index.txt"
    if default_index.is_file():
        return default_index
    return None


def load_index_mapping(index_file: Path | None) -> dict[str, int]:
    if index_file is None:
        return {}
    if not index_file.exists() or not index_file.is_file():
        raise FileNotFoundError(f"Index file does not exist: {index_file}")

    mapping: dict[str, int] = {}
    pattern = re.compile(r"^\s*(\d+)\s*-\s*(.+?)\s*$")
    for line_number, raw_line in enumerate(index_file.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        match = pattern.match(line)
        if match is None:
            raise ValueError(
                f"Invalid line {line_number} in index file {index_file}: {raw_line!r}. "
                "Expected lines like '1 - 3761'."
            )
        camera_number = int(match.group(1))
        directory_name = match.group(2).strip()
        if directory_name in mapping:
            raise ValueError(f"Duplicate directory name in index file {index_file}: {directory_name}")
        mapping[directory_name] = camera_number

    if not mapping:
        raise ValueError(f"Index file {index_file} did not contain any camera mappings.")
    return mapping


def collect_camera_calibrations(
    calibration_root: Path,
    candidates: tuple[str, ...],
    width: int | None,
    height: int | None,
    camera_model: str | None,
    strict: bool,
) -> list[dict]:
    if not calibration_root.exists() or not calibration_root.is_dir():
        raise FileNotFoundError(f"Calibration root does not exist or is not a directory: {calibration_root}")

    camera_dirs = sorted(path for path in calibration_root.iterdir() if path.is_dir())
    if not camera_dirs:
        raise ValueError(f"No camera subdirectories found under {calibration_root}")

    camera_entries = []
    skipped = []
    for camera_dir in camera_dirs:
        calibration_path = resolve_calibration_path(camera_dir, candidates)
        if calibration_path is None:
            skipped.append(camera_dir.name)
            if strict:
                raise FileNotFoundError(
                    f"No calibration pickle found for {camera_dir.name}. Tried: {', '.join(candidates)}"
                )
            continue

        calibration_data = load_calibration(calibration_path)
        transforms = build_transforms(
            data=calibration_data,
            width=width,
            height=height,
            camera_model=camera_model,
        )
        camera_entries.append(
            {
                "camera_dir": camera_dir,
                "calibration_path": calibration_path,
                "transforms": transforms,
            }
        )

    if not camera_entries:
        raise ValueError(
            f"No usable calibration pickles found under {calibration_root}. "
            f"Tried: {', '.join(candidates)}"
        )

    if skipped:
        print(f"Skipped {len(skipped)} camera directories without a matching calibration pickle: {', '.join(skipped)}")

    return camera_entries


def choose_label(
    camera_dir_name: str,
    index: int,
    reindex_labels: bool,
    label_format: str,
    index_mapping: dict[str, int],
) -> str:
    if reindex_labels:
        label_number = index_mapping.get(camera_dir_name, index)
        return label_format.format(label_number)
    return camera_dir_name


def populate_global_intrinsics(camera_entries: list[dict], strategy: str) -> dict:
    if strategy == "none":
        return {"frames": []}

    source_transforms = [entry["transforms"] for entry in camera_entries]
    camera_models = {transforms["camera_model"] for transforms in source_transforms}
    if len(camera_models) != 1:
        raise ValueError(
            "Cannot populate top-level intrinsics when calibrations use different camera models. "
            "Pass --camera_model to normalize them or use --global_intrinsics none."
        )

    scalar_keys = ["w", "h", "fl_x", "fl_y", "cx", "cy"]
    model = source_transforms[0]["camera_model"]
    distortion_keys = {
        "OPENCV_FISHEYE": ["k1", "k2", "k3", "k4"],
        "OPENCV": ["k1", "k2", "p1", "p2"],
        "PINHOLE": [],
    }[model]
    all_keys = ["camera_model", *scalar_keys, *distortion_keys]

    if strategy == "first":
        first = source_transforms[0]
        return {key: first[key] for key in all_keys if key in first} | {"frames": []}

    widths = {transforms["w"] for transforms in source_transforms}
    heights = {transforms["h"] for transforms in source_transforms}
    if len(widths) != 1 or len(heights) != 1:
        raise ValueError(
            "Cannot average top-level intrinsics when camera image sizes differ. "
            "Use --global_intrinsics first or none."
        )

    averaged = {
        "camera_model": model,
        "w": source_transforms[0]["w"],
        "h": source_transforms[0]["h"],
        "frames": [],
    }
    for key in ["fl_x", "fl_y", "cx", "cy", *distortion_keys]:
        values = [float(transforms.get(key, 0.0)) for transforms in source_transforms]
        averaged[key] = math.fsum(values) / len(values)
    return averaged


def main() -> None:
    args = parse_args()
    candidates = tuple(args.candidate or DEFAULT_CANDIDATES)
    index_file = resolve_index_file(args.calibration_root, args.index_file)
    index_mapping = load_index_mapping(index_file)
    camera_entries = collect_camera_calibrations(
        calibration_root=args.calibration_root,
        candidates=candidates,
        width=args.width,
        height=args.height,
        camera_model=args.camera_model,
        strict=args.strict,
    )
    if args.reindex_labels and index_mapping:
        missing_index_entries = [
            entry["camera_dir"].name for entry in camera_entries if entry["camera_dir"].name not in index_mapping
        ]
        if missing_index_entries:
            raise ValueError(
                "The following calibration directories were missing from the index mapping: "
                + ", ".join(missing_index_entries)
            )
        camera_entries = sorted(camera_entries, key=lambda entry: index_mapping[entry["camera_dir"].name])

    output = populate_global_intrinsics(camera_entries, args.global_intrinsics)
    output["frames"] = []

    for index, entry in enumerate(camera_entries):
        camera_label = choose_label(
            camera_dir_name=entry["camera_dir"].name,
            index=index,
            reindex_labels=args.reindex_labels,
            label_format=args.label_format,
            index_mapping=index_mapping,
        )
        frame = {
            "camera_label": camera_label,
            "file_path": args.file_path_template.format(camera_label=camera_label),
        }
        for key, value in entry["transforms"].items():
            if key == "frames":
                continue
            frame[key] = value
        output["frames"].append(frame)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=args.indent)

    print(f"Wrote {len(output['frames'])} camera entries to {args.output}")
    print(f"Candidates tried per camera: {', '.join(candidates)}")
    if args.reindex_labels:
        if index_file is not None:
            print(f"Camera labels assigned from index file: {index_file}")
        else:
            print("Camera labels assigned by sorted directory order (no index file found).")
    if args.global_intrinsics == "none":
        print("Top-level intrinsics omitted.")
    else:
        print(f"Top-level intrinsics strategy: {args.global_intrinsics}")


if __name__ == "__main__":
    main()
