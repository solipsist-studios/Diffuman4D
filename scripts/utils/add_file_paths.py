import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate file_path entries in a transforms JSON using camera numbers "
            "parsed from camera_label."
        )
    )
    parser.add_argument(
        "transforms_path",
        help="Path to input transforms JSON",
    )
    parser.add_argument(
        "path_format",
        help=(
            "Python format string for file_path, filled with camera number. "
            r"Example: \"images\\Camera_{0:04d}.png\""
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "Optional output JSON path. Defaults to input directory with '_paths' "
            "appended to input filename stem."
        ),
    )
    return parser.parse_args()


def default_output_path(transforms_path: Path) -> Path:
    return transforms_path.with_name(f"{transforms_path.stem}_paths{transforms_path.suffix}")


def parse_camera_number(camera_label: str) -> int:
    match = re.search(r"(\d+)$", camera_label)
    if not match:
        raise ValueError(
            f"Could not parse trailing camera number from camera_label '{camera_label}'."
        )
    return int(match.group(1))


def main() -> None:
    args = parse_args()
    transforms_path = Path(args.transforms_path)
    output_path = Path(args.output) if args.output else default_output_path(transforms_path)

    if not transforms_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {transforms_path}")

    with transforms_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "frames" not in data or not isinstance(data["frames"], list):
        raise ValueError("Invalid transforms JSON: missing 'frames' list")

    for frame in data["frames"]:
        camera_label = frame.get("camera_label")
        if not camera_label:
            raise ValueError("Found frame without 'camera_label'")

        camera_number = parse_camera_number(camera_label)
        try:
            frame["file_path"] = args.path_format.format(camera_number)
        except Exception as exc:
            raise ValueError(
                f"Failed to format path with camera number {camera_number} using "
                f"format '{args.path_format}': {exc}"
            ) from exc

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Wrote {len(data['frames'])} frames to: {output_path}")


if __name__ == "__main__":
    main()
