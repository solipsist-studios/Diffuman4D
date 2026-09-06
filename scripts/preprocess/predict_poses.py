import argparse
import json
import os
import pickle
import re
from pathlib import Path

import numpy as np
from hloc import extract_features, match_features, reconstruction, pairs_from_exhaustive
import pycolmap


SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

def load_camera_from_transforms(transforms: dict) -> pycolmap.Camera:
    width = transforms.get('w', transforms.get('width'))
    height = transforms.get('h', transforms.get('height'))
    fl_x = transforms.get('fl_x')
    fl_y = transforms.get('fl_y', fl_x)
    cx = transforms.get('cx')
    cy = transforms.get('cy')

    required = {
        'w/width': width,
        'h/height': height,
        'fl_x': fl_x,
        'fl_y': fl_y,
        'cx': cx,
        'cy': cy,
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise ValueError(
            f"Missing required camera intrinsics: {', '.join(missing)}"
        )

    camera_model = str(transforms.get('camera_model', 'PINHOLE')).upper()

    if camera_model == 'SIMPLE_PINHOLE':
        params = [
            float(fl_x),
            float(cx),
            float(cy),
        ]
    elif camera_model == 'PINHOLE':
        params = [
            float(fl_x),
            float(fl_y),
            float(cx),
            float(cy),
        ]
    elif camera_model == 'SIMPLE_RADIAL':
        params = [
            float(fl_x),
            float(cx),
            float(cy),
            float(transforms.get('k1', 0.0)),
        ]
    elif camera_model == 'RADIAL':
        params = [
            float(fl_x),
            float(cx),
            float(cy),
            float(transforms.get('k1', 0.0)),
            float(transforms.get('k2', 0.0)),
        ]
    elif camera_model == 'OPENCV':
        params = [
            float(fl_x),
            float(fl_y),
            float(cx),
            float(cy),
            float(transforms.get('k1', 0.0)),
            float(transforms.get('k2', 0.0)),
            float(transforms.get('p1', 0.0)),
            float(transforms.get('p2', 0.0)),
        ]
    elif camera_model == 'OPENCV_FISHEYE':
        params = [
            float(fl_x),
            float(fl_y),
            float(cx),
            float(cy),
            float(transforms.get('k1', 0.0)),
            float(transforms.get('k2', 0.0)),
            float(transforms.get('k3', 0.0)),
            float(transforms.get('k4', 0.0)),
        ]
    else:
        raise ValueError(
            f"Unsupported camera_model '{camera_model}'. "
            "Supported models: SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV, OPENCV_FISHEYE"
        )

    return pycolmap.Camera(
        model=camera_model,
        width=int(width),
        height=int(height),
        params=params,
    )


def _can_load_per_frame_intrinsics(transforms: dict) -> bool:
    frames = transforms.get('frames')
    if not isinstance(frames, list) or not frames:
        return False

    try:
        for frame in frames:
            load_camera_from_transforms(frame)
    except Exception:
        return False

    return True


def _intrinsics_from_colmap_camera(camera: pycolmap.Camera) -> dict:
    model_name = camera.model.name
    params = [float(value) for value in camera.params]

    intrinsics = {
        'camera_model': model_name,
        'w': int(camera.width),
        'h': int(camera.height),
    }

    if model_name == 'SIMPLE_PINHOLE':
        fl_x, cx, cy = params
        intrinsics['fl_x'] = fl_x
        intrinsics['fl_y'] = fl_x
        intrinsics['cx'] = cx
        intrinsics['cy'] = cy
    elif model_name == 'PINHOLE':
        fl_x, fl_y, cx, cy = params
        intrinsics['fl_x'] = fl_x
        intrinsics['fl_y'] = fl_y
        intrinsics['cx'] = cx
        intrinsics['cy'] = cy
    elif model_name == 'SIMPLE_RADIAL':
        fl_x, cx, cy, k1 = params
        intrinsics['fl_x'] = fl_x
        intrinsics['fl_y'] = fl_x
        intrinsics['cx'] = cx
        intrinsics['cy'] = cy
        intrinsics['k1'] = k1
    elif model_name == 'RADIAL':
        fl_x, cx, cy, k1, k2 = params
        intrinsics['fl_x'] = fl_x
        intrinsics['fl_y'] = fl_x
        intrinsics['cx'] = cx
        intrinsics['cy'] = cy
        intrinsics['k1'] = k1
        intrinsics['k2'] = k2
    elif model_name == 'OPENCV':
        fl_x, fl_y, cx, cy, k1, k2, p1, p2 = params
        intrinsics['fl_x'] = fl_x
        intrinsics['fl_y'] = fl_y
        intrinsics['cx'] = cx
        intrinsics['cy'] = cy
        intrinsics['k1'] = k1
        intrinsics['k2'] = k2
        intrinsics['p1'] = p1
        intrinsics['p2'] = p2
    elif model_name == 'OPENCV_FISHEYE':
        fl_x, fl_y, cx, cy, k1, k2, k3, k4 = params
        intrinsics['fl_x'] = fl_x
        intrinsics['fl_y'] = fl_y
        intrinsics['cx'] = cx
        intrinsics['cy'] = cy
        intrinsics['k1'] = k1
        intrinsics['k2'] = k2
        intrinsics['k3'] = k3
        intrinsics['k4'] = k4
    else:
        raise ValueError(
            f"Unsupported camera_model '{model_name}'. "
            'Supported models: SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV, OPENCV_FISHEYE'
        )

    return intrinsics


def _infer_camera_model(model: str | None, dist: np.ndarray) -> str:
    supported = {'OPENCV', 'OPENCV_FISHEYE', 'PINHOLE'}
    if model in supported:
        return model
    return 'OPENCV_FISHEYE' if dist.size == 4 else 'OPENCV'


def _infer_image_size(images_dir: Path) -> tuple[int, int]:
    """Return (width, height) of the first readable image found under images_dir."""
    from PIL import Image  # lazy import — only needed when size must be inferred
    for path in sorted(images_dir.rglob('*')):
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            try:
                with Image.open(path) as img:
                    return img.width, img.height
            except Exception:
                continue
    raise FileNotFoundError(
        f'No readable images found under {images_dir} to infer image dimensions.'
    )


def _count_input_images(image_dir: Path) -> int:
    return sum(1 for path in image_dir.rglob('*') if path.is_file())


def _resize_dimensions(width: int, height: int, resize_max: int | None) -> tuple[int, int]:
    if resize_max is None or resize_max <= 0:
        return int(width), int(height)

    max_dim = max(width, height)
    if max_dim <= resize_max:
        return int(width), int(height)

    scale = resize_max / float(max_dim)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return resized_width, resized_height


def _resolution_issue(message: str, strict: bool) -> None:
    if strict:
        raise ValueError(message)
    print(f'WARNING: {message}')


def _collect_expected_image_sizes(
    images_dir: Path,
    resize_max: int | None,
) -> tuple[dict[str, tuple[int, int]], set[tuple[int, int]]]:
    from PIL import Image  # lazy import — only needed for explicit validation

    expected_by_key: dict[str, tuple[int, int]] = {}
    unique_sizes: set[tuple[int, int]] = set()
    ambiguous_keys: set[str] = set()

    for path in sorted(images_dir.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue

        try:
            with Image.open(path) as img:
                expected_size = _resize_dimensions(img.width, img.height, resize_max)
        except Exception:
            continue

        unique_sizes.add(expected_size)

        rel_key = _normalized_rel_to_images(images_dir, path)
        for lookup_key in _path_lookup_keys(rel_key):
            previous = expected_by_key.get(lookup_key)
            if previous is not None and previous != expected_size:
                ambiguous_keys.add(lookup_key)
                continue
            expected_by_key[lookup_key] = expected_size

    for key in ambiguous_keys:
        expected_by_key.pop(key, None)

    if not expected_by_key:
        raise FileNotFoundError(
            f'No readable images found under {images_dir} to validate calibration resolution.'
        )

    return expected_by_key, unique_sizes


def _validate_calibration_resolution(
    transforms: dict,
    images_dir: Path,
    resize_max: int | None,
    strict: bool,
) -> None:
    expected_by_key, unique_sizes = _collect_expected_image_sizes(images_dir, resize_max)
    use_per_frame_intrinsics = _can_load_per_frame_intrinsics(transforms)

    if use_per_frame_intrinsics:
        mismatches: list[tuple[str, tuple[int, int], tuple[int, int]]] = []
        missing_resolution_count = 0
        unresolved_file_path_count = 0

        for frame in transforms.get('frames', []):
            frame_w = frame.get('w', frame.get('width'))
            frame_h = frame.get('h', frame.get('height'))
            if frame_w is None or frame_h is None:
                missing_resolution_count += 1
                continue

            frame_file_path = frame.get('file_path')
            if not isinstance(frame_file_path, str) or not frame_file_path:
                unresolved_file_path_count += 1
                continue

            expected_size = None
            for lookup_key in _path_lookup_keys(frame_file_path):
                expected_size = expected_by_key.get(lookup_key)
                if expected_size is not None:
                    break

            if expected_size is None:
                unresolved_file_path_count += 1
                continue

            actual_size = (int(frame_w), int(frame_h))
            if actual_size != expected_size:
                mismatches.append((frame_file_path, actual_size, expected_size))

        if missing_resolution_count > 0:
            _resolution_issue(
                'Some per-frame calibration entries are missing w/h fields '
                f'({missing_resolution_count} frame(s)); resolution checks were skipped for those entries.',
                strict,
            )

        if unresolved_file_path_count > 0:
            _resolution_issue(
                'Some per-frame calibration entries could not be mapped to images under images_dir '
                f'({unresolved_file_path_count} frame(s)); resolution checks were skipped for those entries.',
                strict,
            )

        if mismatches:
            preview = ', '.join(
                f"{file_path}: calib={actual[0]}x{actual[1]}, expected={expected[0]}x{expected[1]}"
                for file_path, actual, expected in mismatches[:5]
            )
            _resolution_issue(
                'Per-frame calibration resolution does not match image resolution '
                f'after resize_max processing (mismatched frames: {len(mismatches)}). '
                f'Examples: {preview}',
                strict,
            )
        return

    calib_w = transforms.get('w', transforms.get('width'))
    calib_h = transforms.get('h', transforms.get('height'))
    if calib_w is None or calib_h is None:
        _resolution_issue(
            'Global calibration does not include w/h fields; unable to validate resolution match.',
            strict,
        )
        return

    if len(unique_sizes) != 1:
        samples = ', '.join(f'{w}x{h}' for w, h in sorted(unique_sizes)[:5])
        _resolution_issue(
            'Input images do not have a single effective resolution after resize_max processing; '
            f'found {len(unique_sizes)} distinct size(s). Sample sizes: {samples}',
            strict,
        )
        return

    expected_w, expected_h = next(iter(unique_sizes))
    if (int(calib_w), int(calib_h)) != (expected_w, expected_h):
        _resolution_issue(
            'Global calibration resolution does not match effective image resolution after resize_max processing: '
            f'calibration={int(calib_w)}x{int(calib_h)}, expected={expected_w}x{expected_h}.',
            strict,
        )


def _camera_name_score(name: str) -> int:
    normalized = name.strip().lower()
    if not normalized:
        return -1

    score = 0
    if 'camera' in normalized or normalized.startswith('cam'):
        score += 3
    if 'frame' in normalized:
        score -= 3
    if re.search(r'cam(?:era)?[_-]?\d+', normalized):
        score += 2
    if re.search(r'frame[_-]?\d+', normalized):
        score -= 2
    return score


def _resolve_camera_export_name(
    image_name: str,
    camera_name_source: str,
) -> tuple[str, str]:
    image_path = Path(image_name)
    directory_candidate = image_path.parent.name
    filename_candidate = image_path.stem

    if camera_name_source == 'directory':
        camera_label = directory_candidate or filename_candidate
    elif camera_name_source == 'filename':
        camera_label = filename_candidate
    else:
        directory_score = _camera_name_score(directory_candidate)
        filename_score = _camera_name_score(filename_candidate)
        if directory_score > filename_score:
            camera_label = directory_candidate
        else:
            camera_label = filename_candidate

    if not camera_label:
        camera_label = filename_candidate

    output_suffix = image_path.suffix
    output_file_name = f'{camera_label}{output_suffix}' if output_suffix else camera_label
    return camera_label, output_file_name


def _normalized_path(path_like: str) -> str:
    normalized = str(path_like).replace('\\', '/')
    normalized = normalized.lstrip('./')
    return Path(normalized).as_posix()


def _path_lookup_keys(path_like: str) -> list[str]:
    normalized = _normalized_path(path_like)
    keys = [normalized]
    if normalized.startswith('images/'):
        keys.append(normalized[len('images/'):])
    keys.append(Path(normalized).name)

    deduped: list[str] = []
    for key in keys:
        if key and key not in deduped:
            deduped.append(key)
    return deduped


def _normalize_label_token(label: str) -> str:
    token = str(label).strip().lower()
    if token.isdigit():
        return str(int(token))
    return token


def _camera_label_candidates_from_image_name(image_name: str) -> list[str]:
    normalized = _normalized_path(image_name)
    path = Path(normalized)

    candidates: list[str] = []
    if path.parent.as_posix() not in ('', '.'):
        candidates.append(path.parent.name)

    # Handle names like 0001_0001.mp4.thumb.jpg by reducing to 0001_0001.
    stem = path.stem
    stem_primary = stem.split('.')[0]
    if stem_primary:
        candidates.append(stem_primary)

        # Add tokenized candidates in an order that prefers camera-like suffix tokens.
        # Example: 0001_0012 -> prefer 0012 over 0001.
        split_tokens = [token for token in re.split(r'[_\-]', stem_primary) if token]
        for token in reversed(split_tokens):
            candidates.append(token)
        for token in split_tokens:
            candidates.append(token)

    normalized_candidates: list[str] = []
    for candidate in candidates:
        normalized_candidate = _normalize_label_token(candidate)
        if normalized_candidate and normalized_candidate not in normalized_candidates:
            normalized_candidates.append(normalized_candidate)

    return normalized_candidates


def _camera_label_frequency_candidates_from_image_name(image_name: str) -> list[str]:
    normalized = _normalized_path(image_name)
    path = Path(normalized)

    candidates: list[str] = []
    if path.parent.as_posix() not in ('', '.'):
        candidates.append(path.parent.name)

    stem_primary = path.stem.split('.')[0]
    if stem_primary:
        split_tokens = [token for token in re.split(r'[_\-]', stem_primary) if token]
        candidates.extend(split_tokens)

    normalized_candidates: list[str] = []
    for candidate in candidates:
        normalized_candidate = _normalize_label_token(candidate)
        if normalized_candidate:
            normalized_candidates.append(normalized_candidate)
    return normalized_candidates


def _build_label_token_frequency(image_names: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for image_name in image_names:
        for candidate in _camera_label_frequency_candidates_from_image_name(image_name):
            counts[candidate] = counts.get(candidate, 0) + 1
    return counts


def _build_frame_intrinsics_lookup(frames: list[dict]) -> tuple[dict[str, dict], set[str], dict[str, dict], set[str]]:
    lookup: dict[str, dict] = {}
    ambiguous: set[str] = set()
    label_lookup: dict[str, dict] = {}
    ambiguous_labels: set[str] = set()
    for frame in frames:
        file_path = frame.get('file_path')
        if not isinstance(file_path, str) or not file_path:
            continue
        for key in _path_lookup_keys(file_path):
            previous = lookup.get(key)
            if previous is not None and previous is not frame:
                ambiguous.add(key)
                continue
            lookup[key] = frame

        camera_label = frame.get('camera_label')
        if camera_label is not None:
            normalized_label = _normalize_label_token(camera_label)
            if normalized_label:
                previous_label_frame = label_lookup.get(normalized_label)
                if previous_label_frame is not None and previous_label_frame is not frame:
                    ambiguous_labels.add(normalized_label)
                else:
                    label_lookup[normalized_label] = frame

    for key in ambiguous:
        lookup.pop(key, None)
    for label in ambiguous_labels:
        label_lookup.pop(label, None)
    return lookup, ambiguous, label_lookup, ambiguous_labels


def _find_frame_intrinsics_for_image_name(
    image_name: str,
    frame_lookup: dict[str, dict],
    ambiguous_keys: set[str],
    frame_label_lookup: dict[str, dict],
    ambiguous_labels: set[str],
    label_token_frequency: dict[str, int],
) -> tuple[dict, str]:
    for key in _path_lookup_keys(image_name):
        if key in frame_lookup:
            return frame_lookup[key], 'exact'
        if key in ambiguous_keys:
            raise ValueError(
                f'Image {image_name!r} matched an ambiguous frame key {key!r}. '
                'Use unique file_path values in transforms frames.'
            )

    # Fallback for extracted names that may not match frame file_path exactly,
    # e.g. 0001_0001.mp4.thumb.jpg.
    matched_candidates: list[tuple[int, str]] = []
    for label_candidate in _camera_label_candidates_from_image_name(image_name):
        if label_candidate in ambiguous_labels:
            raise ValueError(
                f'Image {image_name!r} matched an ambiguous camera label {label_candidate!r}. '
                'Provide unique camera_label values in transforms frames.'
            )
        if label_candidate in frame_label_lookup:
            frequency = label_token_frequency.get(label_candidate, 10**9)
            matched_candidates.append((frequency, label_candidate))

    if matched_candidates:
        matched_candidates.sort(key=lambda item: (item[0], item[1]))
        best_frequency, best_label = matched_candidates[0]
        equally_good = [label for freq, label in matched_candidates if freq == best_frequency]
        if len(equally_good) > 1:
            raise ValueError(
                f'Image {image_name!r} matched multiple equally likely camera labels {equally_good}. '
                'Please make frame file_path values match image names more directly.'
            )
        return frame_label_lookup[best_label], 'fallback'

    raise KeyError(
        f'No frame intrinsics found for COLMAP image {image_name!r}. '
        'Ensure transforms frame file_path values match image paths under images_dir.'
    )


def _run_reconstruction_with_per_frame_intrinsics(
    sfm_dir: Path,
    image_dir: Path,
    pairs: Path,
    features: Path,
    matches: Path,
    frames: list[dict],
    camera_model_name: str,
    mapper_options: dict,
) -> pycolmap.Reconstruction:
    assert features.exists(), features
    assert pairs.exists(), pairs
    assert matches.exists(), matches

    sfm_dir.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / 'database.db'

    pycolmap.logging.set_log_destination(pycolmap.logging.INFO, sfm_dir / 'colmap.LOG.')

    reconstruction.create_empty_db(database)
    reconstruction.import_images(
        image_dir,
        database,
        pycolmap.CameraMode.PER_IMAGE,
        image_list=None,
        options={'camera_model': camera_model_name},
    )

    frame_lookup, ambiguous_keys, frame_label_lookup, ambiguous_labels = _build_frame_intrinsics_lookup(frames)
    image_ids = reconstruction.get_image_ids(database)
    image_names = list(image_ids.keys())
    label_token_frequency = _build_label_token_frequency(image_names)

    if len(frames) < len(image_names):
        raise ValueError(
            'Per-frame intrinsics were requested, but input transforms has fewer frames '
            f'({len(frames)}) than imported images ({len(image_names)}). '
            'This usually means the wrong transforms file was provided '
            '(for example, a previously reduced output transforms_hloc.json).'
        )

    exact_match_count = 0
    fallback_match_count = 0

    with pycolmap.Database.open(database) as db:
        for db_image in db.read_all_images():
            frame_intrinsics, match_mode = _find_frame_intrinsics_for_image_name(
                db_image.name,
                frame_lookup,
                ambiguous_keys,
                frame_label_lookup,
                ambiguous_labels,
                label_token_frequency,
            )
            if match_mode == 'exact':
                exact_match_count += 1
            else:
                fallback_match_count += 1
            camera = load_camera_from_transforms(frame_intrinsics)
            camera.camera_id = db_image.camera_id
            db.update_camera(camera)

        reconstruction.import_features(image_ids, db, features)
        reconstruction.import_matches(
            image_ids,
            db,
            pairs,
            matches,
            min_match_score=None,
            skip_geometric_verification=False,
        )

    reconstruction.estimation_and_geometric_verification(database, pairs, verbose=False)
    print(
        'Per-frame intrinsics matching summary: '
        f'{exact_match_count} exact, {fallback_match_count} fallback, '
        f'{len(image_names)} total images.'
    )
    return reconstruction.run_reconstruction(
        sfm_dir,
        database,
        image_dir,
        verbose=False,
        options=mapper_options,
    )


def load_transforms_from_pkl(
    pkl_path: Path,
    camera_model: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> dict:
    """Load a calibration .pkl and return a transforms dict compatible with predict_poses.

    w/h are first taken from the supplied arguments if present; otherwise they are loaded from the pkl if present.
    If neither is available, they are inferred from the first image under images_dir.
    """
    if not pkl_path.exists() or not pkl_path.is_file():
        raise FileNotFoundError(f'Calibration file does not exist: {pkl_path}')
    with pkl_path.open('rb') as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError('Calibration file must contain a dictionary.')
    if 'camera_matrix' not in data or 'distortion_coefficients' not in data:
        raise ValueError(
            'Calibration file is missing required keys: camera_matrix and/or distortion_coefficients.'
        )

    mtx = np.asarray(data['camera_matrix'], dtype=np.float64)
    dist = np.asarray(data['distortion_coefficients'], dtype=np.float64).reshape(-1)
    if mtx.shape != (3, 3):
        raise ValueError(f'camera_matrix must have shape (3, 3), got {mtx.shape}')

    resolved_model = camera_model or _infer_camera_model(data.get('model'), dist)

    transforms: dict = {
        'camera_model': resolved_model,
        'fl_x': float(mtx[0, 0]),
        'fl_y': float(mtx[1, 1]),
        'cx': float(mtx[0, 2]),
        'cy': float(mtx[1, 2]),
        'frames': [],
    }

    image_size = data.get('image_size')
    if image_size is not None and len(image_size) == 2:
        transforms['w'] = int(image_size[0])
        transforms['h'] = int(image_size[1])
    else:
        if width is not None:
            transforms['w'] = int(width)
        if height is not None:
            transforms['h'] = int(height)

    if resolved_model == 'OPENCV_FISHEYE':
        transforms['k1'] = float(dist[0]) if dist.size > 0 else 0.0
        transforms['k2'] = float(dist[1]) if dist.size > 1 else 0.0
        transforms['k3'] = float(dist[2]) if dist.size > 2 else 0.0
        transforms['k4'] = float(dist[3]) if dist.size > 3 else 0.0
    elif resolved_model == 'OPENCV':
        transforms['k1'] = float(dist[0]) if dist.size > 0 else 0.0
        transforms['k2'] = float(dist[1]) if dist.size > 1 else 0.0
        transforms['p1'] = float(dist[2]) if dist.size > 2 else 0.0
        transforms['p2'] = float(dist[3]) if dist.size > 3 else 0.0

    return transforms


def _normalized_rel_to_images(images_dir: Path, image_path: Path) -> str:
    rel = image_path.relative_to(images_dir).as_posix()
    return f'images/{rel}'


def _build_image_stem_maps(images_dir: Path) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    by_stem: dict[str, list[Path]] = {}
    by_primary_stem: dict[str, list[Path]] = {}
    for path in sorted(images_dir.rglob('*')):
        if not path.is_file():
            continue
        by_stem.setdefault(path.stem, []).append(path)
        primary = path.stem.split('.')[0]
        by_primary_stem.setdefault(primary, []).append(path)
    return by_stem, by_primary_stem


def _resolve_image_for_calibration_stem(
    stem: str,
    images_dir: Path,
    by_stem: dict[str, list[Path]],
    by_primary_stem: dict[str, list[Path]],
) -> str:
    exact = by_stem.get(stem, [])
    if len(exact) == 1:
        return _normalized_rel_to_images(images_dir, exact[0])
    if len(exact) > 1:
        raise ValueError(
            f"Calibration stem {stem!r} matched multiple images by exact stem: "
            f"{[p.name for p in exact]}."
        )

    primary = stem.split('.')[0]
    fallback = by_primary_stem.get(primary, [])
    if len(fallback) == 1:
        return _normalized_rel_to_images(images_dir, fallback[0])
    if len(fallback) > 1:
        raise ValueError(
            f"Calibration stem {stem!r} matched multiple images by primary stem {primary!r}: "
            f"{[p.name for p in fallback]}."
        )

    return f'images/{stem}.jpg'


def load_transforms_from_calibration_path(
    calibration_path: Path,
    images_dir: Path,
    camera_model: str | None = None,
) -> dict:
    if not calibration_path.exists():
        raise FileNotFoundError(f'Calibration path does not exist: {calibration_path}')

    if calibration_path.is_file():
        return load_transforms_from_pkl(calibration_path, camera_model=camera_model)

    if not calibration_path.is_dir():
        raise ValueError(f'Calibration path must be a file or directory: {calibration_path}')

    pkl_files = sorted(path for path in calibration_path.iterdir() if path.is_file() and path.suffix.lower() == '.pkl')
    if not pkl_files:
        raise FileNotFoundError(f'No calibration .pkl files found in directory: {calibration_path}')

    inferred_width, inferred_height = _infer_image_size(images_dir)
    by_stem, by_primary_stem = _build_image_stem_maps(images_dir)

    frames: list[dict] = []
    for pkl_path in pkl_files:
        intr = load_transforms_from_pkl(
            pkl_path,
            camera_model=camera_model,
            width=inferred_width,
            height=inferred_height,
        )
        file_path = _resolve_image_for_calibration_stem(
            pkl_path.stem,
            images_dir,
            by_stem,
            by_primary_stem,
        )
        frame = {
            'file_path': file_path,
            'camera_label': pkl_path.stem,
        }
        for key, value in intr.items():
            if key == 'frames':
                continue
            frame[key] = value
        frames.append(frame)

    return {'frames': frames}


def compute_spatial_order(transforms: dict, names: list[str], num_locations: int) -> list[int]:
    """Dynamically determine spatial sequence of camera locations using PCA and angular sorting.

    This finds the largest angular gap to handle open arc rig configurations and orders them sequentially.
    """
    if num_locations <= 1:
        return list(range(num_locations))

    # 1. Map each image name to its transform matrix translation vector
    translation_by_key = {}
    for frame in transforms.get('frames', []):
        file_path = frame.get('file_path')
        matrix = frame.get('transform_matrix')
        if not file_path or matrix is None:
            continue
        c2w = np.array(matrix)
        if c2w.shape == (4, 4):
            translation = c2w[:3, 3]
            for key in _path_lookup_keys(file_path):
                translation_by_key[key] = translation

    # 2. Get translation for each location by averaging its camera translations
    loc_centers = []
    for i in range(num_locations):
        idx1 = 2 * i
        idx2 = 2 * i + 1
        t1, t2 = None, None
        if idx1 < len(names):
            for key in _path_lookup_keys(names[idx1]):
                if key in translation_by_key:
                    t1 = translation_by_key[key]
                    break
        if idx2 < len(names):
            for key in _path_lookup_keys(names[idx2]):
                if key in translation_by_key:
                    t2 = translation_by_key[key]
                    break
        
        if t1 is not None and t2 is not None:
            center = (t1 + t2) / 2.0
        elif t1 is not None:
            center = t1
        elif t2 is not None:
            center = t2
        else:
            center = None
        loc_centers.append(center)

    # If we couldn't resolve translations for all locations, return default numeric order
    if any(c is None for c in loc_centers):
        print("WARNING: Could not resolve initial camera translations for all locations. "
              "Falling back to numeric camera ordering.")
        return list(range(num_locations))

    # 3. Project 3D centers to 2D plane using PCA
    pts = np.array(loc_centers)
    centroid = pts.mean(axis=0)
    centered_pts = pts - centroid
    
    # Run SVD
    _, _, Vt = np.linalg.svd(centered_pts)
    pts_2d = centered_pts @ Vt[:2].T

    # 4. Compute angles relative to centroid
    angles = np.arctan2(pts_2d[:, 1], pts_2d[:, 0])
    
    # 5. Sort locations by angle
    sorted_indices = np.argsort(angles)
    sorted_angles = angles[sorted_indices]

    # 6. Find the largest angular gap
    gaps = []
    n = len(sorted_indices)
    for idx in range(n):
        diff = sorted_angles[(idx + 1) % n] - sorted_angles[idx]
        if diff < 0:
            diff += 2 * np.pi
        gaps.append(diff)
        
    largest_gap_idx = np.argmax(gaps)
    start_idx = (largest_gap_idx + 1) % n
    
    spatial_order = []
    for offset in range(n):
        spatial_order.append(int(sorted_indices[(start_idx + offset) % n]))
        
    return spatial_order


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run HLOC feature matching and COLMAP reconstruction using camera intrinsics from a transforms.json or calibration .pkl.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  python scripts/preprocess/predict_poses.py '
            '--images_dir data/goprotest/images '
            '--transforms_json outputs/transforms/goprotest/transforms.json '
            '--output_transforms outputs/transforms/goprotest/transforms_predicted.json\n\n'
            '  python scripts/preprocess/predict_poses.py '
            '--images_dir data/ballet/images '
            '--calibration_path output/calibration_data.pkl '
            '--output_transforms outputs/transforms/ballet/transforms.json '
            '--output_calibration outputs/transforms/ballet/calibration_refined.pkl'
        ),
    )
    parser.add_argument(
        '--images_dir',
        type=Path,
        required=True,
        help='Directory containing input images for HLOC/COLMAP.',
    )
    parser.add_argument(
        '--outputs_dir',
        type=Path,
        default=Path('output/hloc_colmap'),
        help='Directory where pairs, matches, and reconstruction outputs will be written.',
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--transforms_json',
        type=Path,
        help='Path to transforms.json containing camera_model and intrinsics (fl_x/fl_y/cx/cy and distortion params).',
    )
    input_group.add_argument(
        '--calibration_path',
        '--calibration_pkl',
        dest='calibration_path',
        type=Path,
        help=(
            'Path to calibration source: a single .pkl file, or a directory containing '
            'per-image .pkl files (camera_matrix + distortion_coefficients).'
        ),
    )

    parser.add_argument(
        '--camera_model',
        type=str,
        default=None,
        choices=['PINHOLE', 'OPENCV', 'OPENCV_FISHEYE'],
        help='Camera model override when using --calibration_path. Inferred from distortion count if omitted.',
    )

    parser.add_argument(
        '--output_transforms',
        type=Path,
        required=True,
        help='Path to write the output transforms.json with predicted poses and refined intrinsics.',
    )
    parser.add_argument(
        '--output_calibration',
        type=Path,
        default=None,
        help='Optional path to write a calibration .pkl with the refined intrinsics after reconstruction.',
    )
    parser.add_argument(
        '--lock_focus',
        action='store_true',
        help='Disable focal length refinement during bundle adjustment.',
    )
    parser.add_argument(
        '--lock_params',
        action='store_true',
        help='Disable refinement of extra camera parameters during bundle adjustment.',
    )
    parser.add_argument(
        '--refine_principle_point',
        action='store_true',
        help='Enable principal point refinement during bundle adjustment.',
    )
    parser.add_argument(
        '--camera_name_source',
        type=str,
        default='auto',
        choices=['auto', 'directory', 'filename'],
        help='How to derive the exported camera name from nested COLMAP image paths.',
    )
    parser.add_argument(
        '--feature_type',
        type=str,
        default='superpoint',
        choices=['superpoint', 'aliked'],
        help='Feature extraction method: superpoint or aliked.',
    )
    parser.add_argument(
        '--resize_max',
        type=int,
        default=4096,
        help='Maximum dimension to resize images to before feature extraction. Defaults to 4096. Set to 0 or negative to disable resizing.',
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Treat calibration/image resolution mismatches as errors instead of warnings.',
    )
    parser.add_argument(
        '--max_keypoints',
        type=int,
        default=8192,
        help='Maximum number of keypoints to extract per image. Defaults to 8192.',
    )
    parser.add_argument(
        '--adjacent_matching',
        action='store_true',
        help='Only match adjacent cameras in the rig to prevent false matches from repetitive structures.',
    )
    parser.add_argument(
        '--camera_order',
        type=int,
        nargs='+',
        default=None,
        help='Optional manual spatial sequence of camera location indices (e.g. 3 4 5 0 1 2) to override dynamic order.',
    )
    parser.add_argument(
        '--filter_min_tri_angle',
        type=float,
        default=1.5,
        help='Minimum triangulation angle in degrees to filter out low-parallax points. Defaults to 1.5.',
    )
    parser.add_argument(
        '--filter_max_reproj_error',
        type=float,
        default=4.0,
        help='Maximum reprojection error in pixels to filter out noisy matches. Defaults to 4.0.',
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    images_dir = args.images_dir
    outputs_dir = args.outputs_dir

    if not images_dir.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f'images_dir does not exist or is not a directory: {images_dir}')

    outputs_dir.mkdir(parents=True, exist_ok=True)
    num_input_images = _count_input_images(images_dir)

    sfm_pairs = outputs_dir / 'pairs-exhaustive.txt'
    sfm_dir = outputs_dir / 'sfm_reconstruction'

    resize_val = args.resize_max if args.resize_max > 0 else None

    if args.feature_type == 'superpoint':
        feature_conf = {
            'model': {
                'name': 'superpoint',
                'nms_radius': 3,
                'max_keypoints': args.max_keypoints,
            },
            'preprocessing': {
                'grayscale': True,
                'resize_max': resize_val,
            },
            'output': f"feats-superpoint-n{args.max_keypoints}-r{args.resize_max}",
        }
        matcher_conf = match_features.confs['superpoint+lightglue']
    elif args.feature_type == 'aliked':
        feature_conf = {
            'model': {
                'name': 'aliked',
                'model_name': 'aliked-n16',
                'max_num_keypoints': args.max_keypoints,
            },
            'preprocessing': {
                'grayscale': False,
                'resize_max': resize_val,
            },
            'output': f"feats-aliked-n{args.max_keypoints}-r{args.resize_max}",
        }
        matcher_conf = match_features.confs['aliked+lightglue']

    if args.transforms_json is not None:
        if not args.transforms_json.exists() or not args.transforms_json.is_file():
            raise FileNotFoundError(f'transforms_json does not exist: {args.transforms_json}')
        with args.transforms_json.open('r', encoding='utf-8') as handle:
            transforms = json.load(handle)
    else:
        transforms = load_transforms_from_calibration_path(
            args.calibration_path,
            images_dir=images_dir,
            camera_model=args.camera_model,
        )

    print('Extracting features...')
    feature_path = extract_features.main(feature_conf, images_dir, outputs_dir)

    if args.adjacent_matching:
        print('Generating adjacent pairs for the rig to prevent symmetry mismatch...')
        # Get sorted list of images from images_dir (recursive, relative paths)

        image_paths = [

            p for p in sorted(images_dir.rglob('*'))

            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS

        ]
        names = [p.relative_to(images_dir).as_posix() for p in image_paths]



        num_images = len(names)
        if num_images < 2:

            raise ValueError(f'Adjacent matching requires at least 2 images, got {num_images}.')

        if num_images % 2 != 0:

            raise ValueError(

                'Adjacent matching requires an even number of images (top/bottom per location), '

                f'got {num_images}.'

            )

        num_locations = num_images // 2
        pairs: list[tuple[str, str]] = []

        # 1. Vertical pairs (within same location)
        for i in range(num_locations):
            idx1 = 2 * i
            idx2 = 2 * i + 1
            if idx2 < num_images:
                pairs.append((names[idx1], names[idx2]))
                
        # 2. Adjacent locations sequence determined from manual argument or dynamically from input transforms
        if args.camera_order is not None:
            spatial_order = args.camera_order
            print(f"Using manual spatial camera location order: {spatial_order}")
        else:
            spatial_order = compute_spatial_order(transforms, names, num_locations)
            print(f"Dynamically resolved spatial camera location order: {spatial_order}")
        locations_pairs = [(spatial_order[k], spatial_order[k+1]) for k in range(len(spatial_order) - 1)]

        for i, j in locations_pairs:
            idx_i_top = 2 * i
            idx_i_bot = 2 * i + 1
            idx_j_top = 2 * j
            idx_j_bot = 2 * j + 1
            
            cross_pairs = [
                (idx_i_top, idx_j_top),
                (idx_i_top, idx_j_bot),
                (idx_i_bot, idx_j_top),
                (idx_i_bot, idx_j_bot)
            ]
            
            for idx1, idx2 in cross_pairs:
                if idx1 < num_images and idx2 < num_images:
                    pairs.append((names[idx1], names[idx2]))
                    
        # Write to sfm_pairs
        with open(sfm_pairs, 'w') as f:
            for name1, name2 in pairs:
                f.write(f"{name1} {name2}\n")
        print(f"Generated {len(pairs)} adjacent pairs for {num_images} images.")
    else:
        print('Generating exhaustive pairs for the rig...')
        pairs_from_exhaustive.main(sfm_pairs, image_list=None, features=feature_path)

    print('Matching features with LightGlue...')
    match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs_dir)

    _validate_calibration_resolution(
        transforms=transforms,
        images_dir=images_dir,
        resize_max=resize_val,
        strict=args.strict,
    )

    use_per_frame_intrinsics = _can_load_per_frame_intrinsics(transforms)
    per_frame_model = None

    if use_per_frame_intrinsics:
        frame_models = {str(frame.get('camera_model', '')).upper() for frame in transforms['frames']}
        if len(frame_models) != 1:
            raise ValueError(
                f'Per-frame intrinsics require a single camera_model across all frames, got: {sorted(frame_models)}'
            )

        per_frame_model = next(iter(frame_models))
        print(
            'Using per-frame intrinsics from input transforms frames '
            f'(camera_mode={pycolmap.CameraMode.PER_IMAGE.name}, camera_model={per_frame_model}).'
        )

        if transforms.get('w') is None or transforms.get('h') is None:
            first_frame = transforms['frames'][0]
            transforms['w'] = int(first_frame['w'])
            transforms['h'] = int(first_frame['h'])
    else:
        if transforms.get('w') is None or transforms.get('h') is None:
            print('Image dimensions not found in input — inferring from images_dir...')
            w, h = _infer_image_size(images_dir)
            transforms['w'] = w
            transforms['h'] = h
            print(f'Inferred image size: {w}x{h}')

        global_camera = load_camera_from_transforms(transforms)
        global_image_options = {
            'camera_model': global_camera.model.name,
            'camera_params': ','.join(str(float(value)) for value in global_camera.params)
        }
        print(
            'Using global intrinsics from top-level transforms '
            f'(camera_mode={pycolmap.CameraMode.SINGLE.name}, camera_model={global_camera.model.name}).'
        )

    print("Running COLMAP reconstruction...")

    mapper_options_dict = {
        "ba_refine_focal_length": not args.lock_focus,
        "ba_refine_extra_params": not args.lock_params,
        "ba_refine_principal_point": args.refine_principle_point,
        "mapper": {
            "filter_min_tri_angle": args.filter_min_tri_angle,
            "filter_max_reproj_error": args.filter_max_reproj_error,
        },
        "triangulation": {
            "min_angle": args.filter_min_tri_angle,
            "merge_max_reproj_error": args.filter_max_reproj_error,
            "complete_max_reproj_error": args.filter_max_reproj_error,
        }
    }

    if use_per_frame_intrinsics:
        # In PER_IMAGE mode each camera is typically observed by a single image.
        # Refining intrinsics then becomes ill-posed and can fragment the mapping.
        mapper_options_dict.update(
            {
                "ba_refine_focal_length": False,
                "ba_refine_extra_params": False,
                "ba_refine_principal_point": False,
                "multiple_models": False,
                "max_num_models": 1,
            }
        )
        print(
            "Per-frame mode: locking camera intrinsics and restricting COLMAP to a single model "
            "for stable multi-view registration."
        )

    if use_per_frame_intrinsics:
        model = _run_reconstruction_with_per_frame_intrinsics(
            sfm_dir=sfm_dir,
            image_dir=images_dir,
            pairs=sfm_pairs,
            features=feature_path,
            matches=match_path,
            frames=transforms['frames'],
            camera_model_name=per_frame_model,
            mapper_options=mapper_options_dict,
        )
    else:
        model = reconstruction.main(
            sfm_dir,
            images_dir,
            sfm_pairs,
            feature_path,
            match_path,
            camera_mode=pycolmap.CameraMode.SINGLE,
            image_options=global_image_options,
            mapper_options=mapper_options_dict
        )

    if model is None:
        raise RuntimeError('COLMAP reconstruction failed. No valid model was reconstructed.')

    print(f"Reconstruction complete! Reconstructed {model.num_reg_images()} images.")
    print('Reconstruction statistics:')
    print(model.summary())
    print(f'\tnum_input_images = {num_input_images}')

    refined_intrinsics_by_camera_id = {
        camera_id: _intrinsics_from_colmap_camera(camera)
        for camera_id, camera in model.cameras.items()
    }

    if not refined_intrinsics_by_camera_id:
        raise RuntimeError('No reconstructed cameras were found after COLMAP reconstruction.')

    if use_per_frame_intrinsics:
        expected_camera_count = len(transforms.get('frames', []))
        reconstructed_camera_count = len(refined_intrinsics_by_camera_id)
        if reconstructed_camera_count != expected_camera_count:
            print(
                'WARNING: Reconstructed camera model count does not match per-frame input count: '
                f'expected={expected_camera_count}, reconstructed={reconstructed_camera_count}. '
                'This usually indicates partial registration or mismatched intrinsics.'
            )
        print(
            f'Per-frame mode: reconstructed {len(refined_intrinsics_by_camera_id)} camera model(s), '
            'which is expected for multi-camera input.'
        )
        for key in ['camera_model', 'w', 'h', 'fl_x', 'fl_y', 'cx', 'cy', 'k1', 'k2', 'k3', 'k4', 'p1', 'p2']:
            transforms.pop(key, None)
    else:
        if len(refined_intrinsics_by_camera_id) > 1:
            print(
                'WARNING: Single-camera mode expected one camera model, '
                f'but COLMAP reconstructed {len(refined_intrinsics_by_camera_id)} camera models.'
            )
        first_refined_intrinsics = next(iter(refined_intrinsics_by_camera_id.values()))
        for key, value in first_refined_intrinsics.items():
            transforms[key] = value

    # Extract poses from the PyCOLMAP Reconstruction object
    # The 'model' variable is what was returned by reconstruction.main()
    transforms["frames"] = []

    for image_id, image in model.images.items():
        camera_label, output_file_name = _resolve_camera_export_name(
            image.name,
            args.camera_name_source,
        )
        
        # Extract World-to-Camera rotation and translation
        # (Using the modern PyCOLMAP cam_from_world API)
        cam_from_world = image.cam_from_world()
        R_w2c = cam_from_world.rotation.matrix()
        t_w2c = cam_from_world.translation

        # Convert to Camera-to-World
        R_c2w = R_w2c.T
        t_c2w = -np.matmul(R_c2w, t_w2c)

        # Build the 4x4 transformation matrix
        c2w_matrix = np.eye(4)
        c2w_matrix[:3, :3] = R_c2w
        c2w_matrix[:3, 3] = t_c2w

        refined_intrinsics = refined_intrinsics_by_camera_id[image.camera_id]

        # Append to our frames list
        frame_entry = {
            "file_path": os.path.join("images", output_file_name),
            "camera_label": camera_label,
            "transform_matrix": c2w_matrix.tolist()
        }
        for key, value in refined_intrinsics.items():
            frame_entry[key] = value
        transforms["frames"].append(frame_entry)

    # Convert OpenCV -> OpenGL
    translations = []

    for frame in transforms["frames"]:
        # Convert list to numpy array for math
        c2w = np.array(frame["transform_matrix"])
        
        # Flip the Y and Z axes to match PostShot/OpenGL expectations
        c2w[:3, 1] *= -1
        c2w[:3, 2] *= -1
        
        frame["transform_matrix"] = c2w.tolist()
        translations.append(c2w[:3, 3])

    # Fix the Center Origin (Move the rig center to 0,0,0)
    translations = np.array(translations)
    center_of_mass = translations.mean(axis=0)

    for frame in transforms["frames"]:
        c2w = np.array(frame["transform_matrix"])
        # Subtract the center of mass from the translation vector
        c2w[:3, 3] -= center_of_mass
        frame["transform_matrix"] = c2w.tolist()


    # Auto-Align the Up Vector
    # In our new OpenGL C2W matrices, the second column (index 1) is the camera's local Y-axis (Up)
    up_vectors = [np.array(frame["transform_matrix"])[:3, 1] for frame in transforms["frames"]]
    avg_up = np.mean(up_vectors, axis=0)
    avg_up /= np.linalg.norm(avg_up)  # Normalize the vector

    world_up = np.array([0.0, 1.0, 0.0]) # Standard OpenGL World Up

    # Find the rotation axis and angle needed to align the rig to the world
    axis = np.cross(avg_up, world_up)
    axis_norm = np.linalg.norm(axis)

    if axis_norm > 1e-6: # Prevent mathematical errors if already perfectly aligned
        axis /= axis_norm
        angle = np.arccos(np.clip(np.dot(avg_up, world_up), -1.0, 1.0))
        
        # Build the rotation matrix using Rodrigues' formula
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R_align = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        
        # Apply the global rotation to every camera's translation and orientation
        for frame in transforms["frames"]:
            c2w = np.array(frame["transform_matrix"])
            
            # Rotate the translation vector
            c2w[:3, 3] = R_align @ c2w[:3, 3]
            
            # Rotate the orientation matrix
            c2w[:3, :3] = R_align @ c2w[:3, :3]
            
            frame["transform_matrix"] = c2w.tolist()

    # 3. Save to disk
    args.output_transforms.parent.mkdir(parents=True, exist_ok=True)
    with args.output_transforms.open('w', encoding='utf-8') as f:
        json.dump(transforms, f, indent=4)
    print(f"Saved {len(transforms['frames'])} poses to {args.output_transforms}!")

    # 4. Optionally save refined intrinsics back to a calibration .pkl
    if args.output_calibration is not None:
        if use_per_frame_intrinsics:
            raise ValueError(
                'output_calibration is not supported in per-frame mode because a single calibration .pkl '
                'cannot represent multiple per-camera intrinsics. Omit --output_calibration or run with global intrinsics.'
            )

        fl_x = transforms['fl_x']
        fl_y = transforms.get('fl_y', fl_x)
        cx = transforms['cx']
        cy = transforms['cy']
        refined_mtx = np.array([
            [fl_x,  0.0,  cx],
            [ 0.0, fl_y,  cy],
            [ 0.0,  0.0, 1.0],
        ], dtype=np.float64)

        model_name = transforms.get('camera_model', 'PINHOLE')
        if model_name == 'OPENCV_FISHEYE':
            refined_dist = np.array([
                transforms.get('k1', 0.0),
                transforms.get('k2', 0.0),
                transforms.get('k3', 0.0),
                transforms.get('k4', 0.0),
            ], dtype=np.float64)
        elif model_name == 'OPENCV':
            refined_dist = np.array([
                transforms.get('k1', 0.0),
                transforms.get('k2', 0.0),
                transforms.get('p1', 0.0),
                transforms.get('p2', 0.0),
            ], dtype=np.float64)
        else:
            refined_dist = np.zeros(4, dtype=np.float64)

        refined_calib = {
            'camera_matrix': refined_mtx,
            'distortion_coefficients': refined_dist,
            'image_size': (transforms.get('w'), transforms.get('h')),
            'model': model_name,
            'reprojection_error': None,
            'rotation_vectors': None,
            'translation_vectors': None,
        }
        args.output_calibration.parent.mkdir(parents=True, exist_ok=True)
        with args.output_calibration.open('wb') as f:
            pickle.dump(refined_calib, f)
        print(f'Saved refined calibration to {args.output_calibration}')


if __name__ == '__main__':
    main()