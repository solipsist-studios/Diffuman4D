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
            "Build a transforms.json from calibration pickles. "
            "Supports either per-camera subdirectories or a flat collection of .pkl files "
            "directly under --calibration_root."
        )
    )
    parser.add_argument(
        "--calibration_root",
        type=Path,
        required=True,
        help=(
            "Root directory containing either one subdirectory per camera, "
            "or a flat collection of .pkl calibration files."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Optional path to an existing transforms JSON used as a base. "
            "Computed calibration values always override conflicting entries."
        ),
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
        default="input",
        choices=["input", "mean", "first", "none"],
        help=(
            "How to populate top-level intrinsics for tools that expect a single camera model. "
            "'input' keeps existing values from --input when present (fallbacks to 'mean'), "
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
) -> list[tuple[str, dict, bool]]:
    if not calibration_root.exists() or not calibration_root.is_dir():
        raise FileNotFoundError(f"Calibration root does not exist or is not a directory: {calibration_root}")

    camera_dirs = sorted(path for path in calibration_root.iterdir() if path.is_dir())
    flat_pkls = sorted(path for path in calibration_root.iterdir() if path.is_file() and path.suffix.lower() == ".pkl")
    if not camera_dirs and not flat_pkls:
        raise ValueError(
            f"No camera subdirectories or .pkl files found under {calibration_root}"
        )

    calibrations: list[tuple[str, dict, bool]] = []
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
        calibrations.append((camera_dir.name, transforms, True))

    # Also support a flat collection where calibration files live directly
    # in calibration_root (same style as calibration_pkl_to_json.py batch mode).
    for calibration_path in flat_pkls:
        calibration_data = load_calibration(calibration_path)
        transforms = build_transforms(
            data=calibration_data,
            width=width,
            height=height,
            camera_model=camera_model,
        )
        calibrations.append((calibration_path.stem, transforms, False))

    if not calibrations:
        raise ValueError(
            f"No usable calibration pickles found under {calibration_root}. "
            f"Tried: {', '.join(candidates)}"
        )

    if skipped:
        print(f"Skipped {len(skipped)} camera directories without a matching calibration pickle: {', '.join(skipped)}")

    return calibrations


def choose_label(
    label_source: str,
    index: int,
    reindex_labels: bool,
    label_format: str,
    index_mapping: dict[str, int],
) -> str:
    if reindex_labels:
        label_number = index_mapping.get(label_source, index)
        return label_format.format(label_number)
    return label_source


def populate_global_intrinsics(source_transforms: list[dict], strategy: str) -> dict:
    if strategy == "none":
        return {"frames": []}

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


DISTORTION_KEYS_BY_MODEL: dict[str, set[str]] = {
    "OPENCV_FISHEYE": {"k1", "k2", "k3", "k4"},
    "OPENCV": {"k1", "k2", "p1", "p2"},
    "PINHOLE": set(),
}
ALL_DISTORTION_KEYS: set[str] = set().union(*DISTORTION_KEYS_BY_MODEL.values())


def strip_extraneous_distortion(obj: dict) -> dict:
    model = obj.get("camera_model")
    allowed = DISTORTION_KEYS_BY_MODEL.get(model, ALL_DISTORTION_KEYS)
    for key in ALL_DISTORTION_KEYS - allowed:
        obj.pop(key, None)
    return obj


def infer_intrinsic_keys(source_transforms: list[dict]) -> set[str]:
    keys: set[str] = set()
    for transforms in source_transforms:
        if not isinstance(transforms, dict):
            continue
        keys.update(key for key in transforms.keys() if key != "frames")
    return keys


def has_input_intrinsics(payload: dict, intrinsic_keys: set[str]) -> bool:
    return any(key in payload for key in intrinsic_keys)


def load_base_output(input_path: Path | None) -> dict:
    if input_path is None:
        return {}
    if not input_path.exists() or not input_path.is_file():
        raise FileNotFoundError(f"Input JSON does not exist: {input_path}")

    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Input JSON must be an object: {input_path}")
    return payload


def merge_frames(base_frames: list, computed_frames: list[dict]) -> list[dict]:
    if not isinstance(base_frames, list):
        base_frames = []

    base_by_label: dict[str, dict] = {}
    base_by_path: dict[str, dict] = {}
    for frame in base_frames:
        if not isinstance(frame, dict):
            continue
        camera_label = frame.get("camera_label")
        file_path = frame.get("file_path")
        if isinstance(camera_label, str) and camera_label not in base_by_label:
            base_by_label[camera_label] = frame
        if isinstance(file_path, str) and file_path not in base_by_path:
            base_by_path[file_path] = frame

    merged_frames: list[dict] = []
    for index, frame in enumerate(computed_frames):
        base_frame = None
        camera_label = frame.get("camera_label")
        file_path = frame.get("file_path")
        if isinstance(camera_label, str):
            base_frame = base_by_label.get(camera_label)
        if base_frame is None and isinstance(file_path, str):
            base_frame = base_by_path.get(file_path)
        if base_frame is None and index < len(base_frames) and isinstance(base_frames[index], dict):
            # Positional fallback preserves metadata even when labels/paths are regenerated.
            base_frame = base_frames[index]

        merged = dict(base_frame) if isinstance(base_frame, dict) else {}
        merged.update(frame)
        strip_extraneous_distortion(merged)
        merged_frames.append(merged)

    return merged_frames


def main() -> None:
    args = parse_args()
    candidates = tuple(args.candidate or DEFAULT_CANDIDATES)
    base_output = load_base_output(args.input)
    index_file = resolve_index_file(args.calibration_root, args.index_file)
    index_mapping = load_index_mapping(index_file)
    calibrations = collect_camera_calibrations(
        calibration_root=args.calibration_root,
        candidates=candidates,
        width=args.width,
        height=args.height,
        camera_model=args.camera_model,
        strict=args.strict,
    )
    source_transforms = [transforms for _, transforms, _ in calibrations]
    if args.reindex_labels and index_mapping:
        missing_index_entries = [
            label_source
            for label_source, _, from_directory in calibrations
            if from_directory and label_source not in index_mapping
        ]
        if missing_index_entries:
            raise ValueError(
                "The following calibration directories were missing from the index mapping: "
                + ", ".join(missing_index_entries)
            )
        calibrations = sorted(
            calibrations,
            key=lambda item: index_mapping.get(item[0], 10**9)
        )
        source_transforms = [transforms for _, transforms, _ in calibrations]

    intrinsic_keys = infer_intrinsic_keys(source_transforms)
    intrinsics_strategy = args.global_intrinsics
    keep_input_intrinsics = False
    if intrinsics_strategy == "input":
        if has_input_intrinsics(base_output, intrinsic_keys):
            keep_input_intrinsics = True
            computed_output = {"frames": []}
        else:
            intrinsics_strategy = "mean"
            computed_output = populate_global_intrinsics(source_transforms, intrinsics_strategy)
    else:
        computed_output = populate_global_intrinsics(source_transforms, intrinsics_strategy)
    computed_output["frames"] = []

    for index, (label_source, transforms, _) in enumerate(calibrations):
        camera_label = choose_label(
            label_source=label_source,
            index=index,
            reindex_labels=args.reindex_labels,
            label_format=args.label_format,
            index_mapping=index_mapping,
        )
        frame = {
            "camera_label": camera_label,
            "file_path": args.file_path_template.format(camera_label, camera_label=camera_label),
        }
        for key, value in transforms.items():
            if key == "frames":
                continue
            frame[key] = value
        computed_output["frames"].append(frame)

    output = dict(base_output)
    base_frames = output.get("frames", [])
    output["frames"] = merge_frames(base_frames, computed_output["frames"])

    if keep_input_intrinsics:
        strip_extraneous_distortion(output)
    else:
        for key in intrinsic_keys:
            output.pop(key, None)
        for key, value in computed_output.items():
            if key == "frames":
                continue
            output[key] = value
        strip_extraneous_distortion(output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=args.indent)

    print(f"Wrote {len(output['frames'])} camera entries to {args.output}")
    if args.input is not None:
        print(f"Used base JSON from: {args.input}")
    print(f"Candidates tried per camera: {', '.join(candidates)}")
    if args.reindex_labels:
        if index_file is not None:
            print(f"Camera labels assigned from index file: {index_file}")
        else:
            print("Camera labels assigned by sorted directory order (no index file found).")
    if args.global_intrinsics == "none":
        print("Top-level intrinsics omitted.")
    elif args.global_intrinsics == "input" and keep_input_intrinsics:
        print("Top-level intrinsics preserved from input JSON.")
    elif args.global_intrinsics == "input":
        print("Top-level intrinsics strategy: input (fallback to mean because input had no intrinsics).")
    else:
        print(f"Top-level intrinsics strategy: {args.global_intrinsics}")


if __name__ == "__main__":
    main()
