import json
import numpy as np
from pathlib import Path
import math
from scipy.spatial.transform import Rotation

def load_transforms(transforms_path):
    """Load existing camera transforms from JSON file.

    This function tolerates JavaScript-style comments (// and /* */) which
    are commonly present in some transforms.json files. It strips those
    comments before calling json.loads.
    """
    import re

    path = Path(transforms_path)
    text = path.read_text()

    # Remove C-style block comments: /* ... */
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    # Remove C++/JS single-line comments: // ...\n
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)

    return json.loads(text)

def extract_camera_positions(transforms):
    """Extract camera positions from transform matrices."""
    positions = []
    for frame in transforms.get('frames', []):
        # Extract translation vector from the camera->world transform matrix
        if 'transform_matrix' not in frame:
            continue
        transform = np.array(frame['transform_matrix'])
        if transform.shape[0] < 3 or transform.shape[1] < 4:
            continue
        position = transform[:3, 3]
        positions.append(position)
        print(f"Extracted camera position: {position}")
    if not positions:
        return np.empty((0, 3))
    return np.array(positions)

def find_intrinsics_for_label(transforms, camera_label):
    """Find intrinsics for a camera label, falling back to file_path folder names."""
    for frame in transforms['frames']:
        if frame.get('camera_label') == camera_label:
            return frame

    for frame in transforms['frames']:
        if 'file_path' in frame:
            folder = Path(frame['file_path']).parent.name
            if folder == camera_label:
                return frame

    return None

def estimate_sphere_parameters(positions):
    """Estimate sphere center and radius from existing camera positions."""
    if positions.size == 0:
        center = np.array([0.0, 0.0, 0.0])
        radius = 1.0
        print("No existing camera positions found; using origin center and default radius=1.0")
        return center, radius

    center = np.mean(positions, axis=0)
    radius = np.mean(np.linalg.norm(positions - center, axis=1))
    if not np.isfinite(radius) or radius <= 0:
        radius = 1.0
        print("Estimated radius was non-finite or <= 0; falling back to radius=1.0")
    print(f"Estimated sphere center: {center}, radius: {radius}")
    return center, radius

def generate_camera_transforms(num_theta, num_phi, radius, center, max_height=None, min_height=None, stagger_rows=True, min_radius=None, max_radius=None, look_at_height=0):
    """Generate camera transform matrices on a sphere/cylinder/cone using quaternion-based rotations.
    
    Args:
        radius: Base sphere radius (used for spherical geometry calculations)
        min_radius: Minimum axial radius (distance from Y-axis) at top (defaults to calculated from sphere)
        max_radius: Maximum axial radius (distance from Y-axis) at equator (defaults to calculated from sphere)
        When min_radius != max_radius, creates conical/cylindrical arrangements
        
    For spherical arrangements:
        - At phi=0 (top), axial_radius=0
        - At phi=pi/2 (equator), axial_radius=radius
        - At phi=pi (bottom), axial_radius=0
        - axial_radius = radius * sin(phi)
    
    Returns:
        List of 4x4 transform matrices (as nested lists for JSON serialization).
    """
    transforms = []

    # Set default height bounds if not provided
    if max_height is None:
        max_height = radius  # Top of sphere
    if min_height is None:
        min_height = 0  # Equator

    # Check if we're in spherical mode or conical/cylindrical mode
    is_spherical_mode = (min_radius is None and max_radius is None)
    
    if is_spherical_mode:
        # --- Spherical Mode: Use phi-based angular distribution ---
        # Clamp heights to sphere bounds
        max_height = min(max_height, radius)
        min_height = max(min_height, -radius)

        # For a sphere centered at the origin: y = radius * cos(phi), where phi=0 is top (y=radius), phi=pi is bottom (y=-radius)
        phi_start = math.acos(max_height / radius)
        phi_end = math.acos(min_height / radius)

        # Determine the step size to fit 'num_phi' cameras inside this band
        total_angle_span = phi_end - phi_start
        phi_step = total_angle_span / (num_phi)
    else:
        # --- Conical/Cylindrical Mode: Use linear height distribution ---
        # Set defaults for min/max radius if not provided
        if min_radius is None:
            min_radius = 0  # Default to point at bottom
        if max_radius is None:
            max_radius = radius  # Default to sphere radius at top
    
    for j in range(num_phi):
        if is_spherical_mode:
            # Start at the top bound, add padding step, then iterate
            phi = phi_start + (j + 1) * phi_step
            
            # Calculate the spherical axial radius at this phi
            # For a sphere: axial_radius = radius * sin(phi)
            current_axial_radius = radius * math.sin(phi)
            
            # Calculate height from phi
            y = radius * math.cos(phi)
        else:
            # Linear interpolation for height and radius in conical/cylindrical mode
            # j=0 (top) uses max_height/max_radius, j=num_phi-1 (bottom) uses min_height/min_radius
            if num_phi > 1:
                t = j / (num_phi - 1)  # 0 to 1 interpolation
            else:
                t = 0.5
            
            y = max_height + t * (min_height - max_height)
            current_axial_radius = max_radius + t * (min_radius - max_radius)
        
        for i in range(num_theta):
            # Calculate azimuthal angle (0 to 2pi), with optional stagger
            if stagger_rows:
                theta = i * (2 * math.pi / num_theta) + (j * math.pi / (num_theta * 2))
            else:
                theta = i * (2 * math.pi / num_theta)

            # Position in cylindrical coordinates: (axial_radius, y, theta)
            x = current_axial_radius * math.cos(theta)
            z = -current_axial_radius * math.sin(theta)

            position = np.array([x, y, z]) # + center
            # Use look_at_height to offset the target position vertically
            target = np.array([0, look_at_height, 0]) # + center
            print(f"Camera pos: {position}, target: {target}")

            # Calculate rotation matrix (looking at target) using quaternion-based rotation
            forward = target - position
            forward = forward / np.linalg.norm(forward)

            # Calculate up vector (approximating the up direction)
            up = np.array([0, 1, 0])
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)

            # Create rotation matrix from basis vectors
            rotation_matrix = np.array([right, up, -forward]).T  # Transpose to get proper rotation matrix

            # Convert to Rotation object and back for consistency
            rotation = Rotation.from_matrix(rotation_matrix)
            rotation_matrix = rotation.as_matrix()

            # Create 4x4 camera->world transform matrix: T = [R | C]
            transform = np.eye(4)
            transform[:3, :3] = rotation_matrix
            transform[:3, 3] = position

            transforms.append(transform.tolist())
    
    return transforms

def find_nearest_generated_indices(existing_positions, generated_positions):
    """Find indices of generated positions nearest to each existing position."""
    nearest_indices = []
    for existing_pos in existing_positions:
        distances = np.linalg.norm(generated_positions - existing_pos, axis=1)
        nearest_indices.append(np.argmin(distances))
    return nearest_indices

def get_camera_label(frame, used_labels, counter=0, camera_format="{:04d}"):
    """Extract camera label from file path or generate a new unique one.

    camera_format: Python format string used with `counter`, e.g. "{:04d}" or "cam_{:03d}".
    """
    if 'file_path' in frame:
        # Extract folder name from the file path
        path = Path(frame['file_path'])
        label = path.parent.name
        # If the folder name is generic like 'images', fall back to formatted counter
        if label == 'images' or label == 'frames':
            return camera_format.format(counter)
        return label
    else:
        # Generate new label that doesn't collide with existing ones using provided format
        while True:
            label = camera_format.format(counter)
            if label not in used_labels:
                return label
            counter += 1

def integrate_cameras(existing_transforms, generated_transforms, nearest_indices, camera_format="{:04d}", intrinsics=None):
    """Create new transforms list, replacing nearest generated cameras with existing ones.
    Preserves complete transform matrices for existing cameras and omits file_path for new ones."""
    new_frames = []
    used_existing = set()
    used_labels = set()
    counter = 1
    
    # First pass: process existing cameras to collect their labels
    for frame in existing_transforms['frames']:
        label = get_camera_label(frame, used_labels, counter, camera_format)
        used_labels.add(label)
        counter += 1
    
    # Second pass: create new frames with labels
    for i in range(len(generated_transforms)):
        if i in nearest_indices:
            # Use existing camera transform exactly as is
            existing_idx = nearest_indices.index(i)
            frame = dict(existing_transforms['frames'][existing_idx])  # Copy to preserve original
            if 'camera_label' not in frame:
                frame['camera_label'] = get_camera_label(frame, used_labels, counter, camera_format)
                used_labels.add(frame['camera_label'])
            new_frames.append(frame)
            used_existing.add(existing_idx)
        else:
            # Create new camera transform without file_path
            label = get_camera_label({}, used_labels, counter, camera_format)
            print(f"New camera label: {label}")
            frame = {
                "transform_matrix": generated_transforms[i],
                "camera_label": label
            }
            if intrinsics:
                frame.update(intrinsics)
            used_labels.add(label)
            new_frames.append(frame)
            counter += 1
    
    # Add any remaining existing cameras that weren't integrated
    for i in range(len(existing_transforms['frames'])):
        if i not in used_existing:
            frame = dict(existing_transforms['frames'][i])  # Copy to preserve original
            if 'camera_label' not in frame:
                frame['camera_label'] = get_camera_label(frame, used_labels, counter, camera_format)
                used_labels.add(frame['camera_label'])
            new_frames.append(frame)
            counter += 1
    
    return new_frames

def main(transforms_path, num_theta=8, num_phi=6, camera_format="{:04d}", min_height=None, max_height=None, stagger_rows=True, min_radius=None, max_radius=None, look_at_height=None, copy_intrinsics=None):
    # Load existing transforms
    existing_transforms = load_transforms(transforms_path)

    intrinsics = None
    if copy_intrinsics:
        source_frame = find_intrinsics_for_label(existing_transforms, copy_intrinsics)
        if source_frame is None:
            raise ValueError(f"No camera found with label '{copy_intrinsics}' to copy intrinsics from.")

        intrinsics_keys = {
            "fl_x", "fl_y", "cx", "cy",
            "k1", "k2", "k3", "k4", "k5", "k6",
            "p1", "p2", "w", "h",
            "camera_angle_x", "camera_angle_y",
            "fx", "fy", "skew"
        }
        intrinsics = {key: source_frame[key] for key in intrinsics_keys if key in source_frame}
        if not intrinsics:
            raise ValueError(f"Camera '{copy_intrinsics}' does not contain intrinsics to copy.")
    
    # Extract camera positions and estimate sphere parameters
    existing_positions = extract_camera_positions(existing_transforms)
    center, radius = estimate_sphere_parameters(existing_positions)
    
    # Generate new camera transform matrices
    generated_transforms = generate_camera_transforms(
        num_theta, num_phi, radius, center, max_height=max_height, min_height=min_height, 
        stagger_rows=stagger_rows, min_radius=min_radius, max_radius=max_radius, look_at_height=look_at_height
    )
    # Wrap generated transforms in a dict with 'frames' key for extract_camera_positions
    generated_transforms_dict = {'frames': [{'transform_matrix': t} for t in generated_transforms]}
    generated_positions = extract_camera_positions(generated_transforms_dict)
    
    print(f"Generated {len(generated_transforms)} camera transforms")

    # Find nearest generated positions to existing cameras
    # First, extract positions from the generated transforms for comparison
    nearest_indices = find_nearest_generated_indices(existing_positions, generated_positions)
    
    # Integrate existing and generated cameras
    new_frames = integrate_cameras(
        existing_transforms, generated_transforms, nearest_indices, camera_format, intrinsics=intrinsics
    )
    
    # Create new transforms dictionary
    new_transforms = existing_transforms.copy()
    new_transforms['frames'] = new_frames
    
    # Save to new file
    output_path = Path(transforms_path)
    output_path = output_path.parent / f"{output_path.stem}_enhanced{output_path.suffix}"
    with open(output_path, 'w') as f:
        json.dump(new_transforms, f, indent=2)
    
    print(f"Enhanced transforms saved to: {output_path}")
    print(f"Total cameras: {len(new_frames)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate spherical camera transforms")
    parser.add_argument("transforms_path", help="Path to existing transforms.json")
    parser.add_argument("--num-theta", type=int, default=8,
                        help="Number of cameras around the axis (default: 8)")
    parser.add_argument("--num-phi", type=int, default=6,
                        help="Number of camera levels above/below equator (default: 6)")
    parser.add_argument("--camera-format", type=str, default="{:04d}",
                        help="Python format string for camera_label, e.g. '{:04d}' or 'cam_{:03d}'")
    parser.add_argument("--min-height", type=float, default=None,
                        help="Minimum height (Z coordinate) for camera placement")
    parser.add_argument("--max-height", type=float, default=None,
                        help="Maximum height (Z coordinate) for camera placement")
    parser.add_argument("--stagger-rows", action="store_true",
                        help="Enable staggering of cameras at each row")
    parser.add_argument("--min-radius", type=float, default=None,
                        help="Minimum radius for camera placement (enables cylindrical/conical arrangements)")
    parser.add_argument("--max-radius", type=float, default=None,
                        help="Maximum radius for camera placement (enables cylindrical/conical arrangements)")
    parser.add_argument("--look-at-height", type=float, default=None,
                        help="Vertical offset for look-at position.")
    parser.add_argument("--copy-intrinsics", type=str, default=None,
                        help="Camera label to copy intrinsics from (applied to generated cameras only).")

    args = parser.parse_args()
    main(args.transforms_path, args.num_theta, args.num_phi, args.camera_format, args.min_height, args.max_height,
        stagger_rows=args.stagger_rows, min_radius=args.min_radius, max_radius=args.max_radius,
        look_at_height=args.look_at_height, copy_intrinsics=args.copy_intrinsics)