"""
Blender script to import cameras from transforms.json (NeRF-format camera data)

Usage:
1. Open Blender
2. Open the Scripting workspace
3. Create a new text file and paste this script
4. Modify the TRANSFORMS_JSON_PATH variable to point to your transforms.json
5. Run the script (Alt+P)

Or run from command line:
    blender --python import_cameras.py --background
"""

import bpy
import json
import numpy as np
from pathlib import Path
from math import radians
from mathutils import Matrix, Euler


TRANSFORMS_JSON_PATH = r"F:\Users\JeffSipko\Dev\GitHub\Diffuman4D\data\ariana_16\transforms_16_paths.json"

# Optional test-time camera jitter.
ADD_JITTER = True
JITTER_MAX_DISPLACEMENT = 0.05  # Scene units.
JITTER_MAX_ROTATION = 5.0  # Degrees.
JITTER_SEED = 0  # Set to None for non-deterministic jitter.

# Camera convention: x'OPENCV', 'OPENGL', or 'NONE'
# OPENCV: X right, Y down, Z forward
# OPENGL: X right, Y up, Z back (NeRF convention)
# NONE: No conversion, use transforms directly (for debugging)
CAMERA_CONVENTION = 'OPENGL'


def clear_existing_cameras():
    """Remove all existing cameras from the scene."""
    for obj in bpy.data.objects:
       if obj.type == 'CAMERA':
           bpy.data.objects.remove(obj, do_unlink=True)


def camera_to_blender(matrix, convention='OPENGL'):
    """Convert 4x4 matrix to Blender format.
    
    Converts from camera convention to Blender convention (X right, Y forward, Z up).
    
    Args:
        matrix: 4x4 transformation matrix
        convention: 'OPENCV' (X right, Y down, Z forward), 
                   'OPENGL' (X right, Y up, Z back),
                   or 'NONE' (no conversion, for debugging)
    """
    
    if convention == 'NONE':
        # No conversion - use matrix as-is
        return matrix
    elif convention == 'OPENCV':
        # OpenCV convention to Blender: X->X, Y_down->-Z, Z_forward->Y
        cv2_to_blender = np.array([
            [1,  0,  0,  0],
            [0,  0,  1,  0],  # forward (Z) -> Blender +Y
            [0, -1,  0,  0],  # down (Y) -> Blender -Z (up is +Z)
            [0,  0,  0,  1]
        ])
        blender_matrix = cv2_to_blender @ matrix
    else:  # OPENGL
        # OpenGL/NeRF convention to Blender: X->X, Y_up->forward, Z_back->up
        opengl_to_blender = np.array([
            [1,  0,  0,  0],
            [0,  0, -1,  0],
            [0,  1,  0,  0],
            [0,  0,  0,  1]
        ])
        blender_matrix = opengl_to_blender @ matrix
    
    return blender_matrix


def set_camera_intrinsics(camera, focal_length_px, image_width, image_height, cx=None, cy=None):
    """Set camera intrinsic parameters using actual sensor size and accurate lens shift."""
    
    # Use your physical GoPro sensor size! No more 36mm hacks.
    sensor_width = 6.74
    sensor_height = 5.05
    
    # Calculate focal length in mm
    focal_length_mm = (sensor_width * focal_length_px) / image_width
    
    camera.lens = focal_length_mm
    camera.sensor_width = sensor_width
    camera.sensor_height = sensor_height
    
    # Handle Principal Point (Lens Shift)
    if cx is None:
        cx = image_width / 2.0
    if cy is None:
        cy = image_height / 2.0

    # Blender shift is normalized by the largest dimension
    max_dim = max(image_width, image_height)
    
    # OpenCV cx is from left. If cx > w/2, optical center is to the right.
    # To move optical center right in the image, we shift the camera box left (negative).
    camera.shift_x = -(cx - (image_width / 2.0)) / max_dim
    
    # OpenCV cy is from top. If cy > h/2, optical center is lower.
    # To move optical center lower in the image, we shift the camera box up (positive).
    camera.shift_y = (cy - (image_height / 2.0)) / max_dim

    # Set render output resolution
    bpy.context.scene.render.resolution_x = image_width
    bpy.context.scene.render.resolution_y = image_height
    bpy.context.scene.render.resolution_percentage = 100

    return sensor_width, sensor_height


def convert_opencv_fisheye_to_blender(k0, k1, k2, k3, focal_length_px, sensor_width, image_width, max_fov_degrees=157):
    """
    Fits OpenCV fisheye coefficients to Blender's Lens Polynomial 
    by mapping sensor radius (r_mm) to incident angles (theta).
    """
    px_to_mm = sensor_width / image_width
    
    # 1. Generate positive theta samples
    max_theta = np.radians(max_fov_degrees / 2.0)
    theta_samples = np.linspace(0, max_theta, 1000) 
    
    # 2. Calculate the corresponding sensor radius (r_mm) using OpenCV's formula
    theta_d = theta_samples + k0*(theta_samples**3) + k1*(theta_samples**5) + k2*(theta_samples**7) + k3*(theta_samples**9)
    r_px = focal_length_px * theta_d
    r_mm_samples = r_px * px_to_mm
    
    # 3. Fit Blender's inverse mapping: r_mm -> NEGATIVE theta
    # Blender expects the angle theta to grow in the negative direction!
    coeffs = np.polyfit(r_mm_samples, -theta_samples, 4)
    
    # Reverse the array to match Blender's [K0, K1, K2, K3, K4] order
    blender_k = coeffs[::-1]
    
    return tuple(blender_k)


def set_camera_fisheye_polynomial(camera, coeffs):
    """Configure camera for Cycles fisheye lens polynomial."""        
    camera.type = 'PANO'
    if hasattr(camera, 'panorama_type'):
        camera.panorama_type = 'FISHEYE_LENS_POLYNOMIAL'
    if hasattr(camera, 'cycles'):
        camera.cycles.panorama_type = 'FISHEYE_LENS_POLYNOMIAL'
        camera.cycles.fisheye_polynomial_k0 = coeffs[0]
        camera.cycles.fisheye_polynomial_k1 = coeffs[1] #* -1 if FISHEYE_FLIP_Y else coeffs[1]
        camera.cycles.fisheye_polynomial_k2 = coeffs[2]
        camera.cycles.fisheye_polynomial_k3 = coeffs[3]
        camera.cycles.fisheye_polynomial_k4 = coeffs[4]
    if hasattr(camera, 'fisheye_polynomial_k0'):
        camera.fisheye_polynomial_k0 = coeffs[0]
        camera.fisheye_polynomial_k1 = coeffs[1] #* -1 if FISHEYE_FLIP_Y else coeffs[1]
        camera.fisheye_polynomial_k2 = coeffs[2]
        camera.fisheye_polynomial_k3 = coeffs[3]
        camera.fisheye_polynomial_k4 = coeffs[4]


def get_opencv_fisheye_coeffs(frame, data):
    """Return OpenCV fisheye coeffs as (k0..k3) even if stored as k1..k4."""
    k0 = frame.get('k0', data.get('k0'))
    k1 = frame.get('k1', data.get('k1'))
    k2 = frame.get('k2', data.get('k2'))
    k3 = frame.get('k3', data.get('k3'))
    if None not in (k0, k1, k2, k3):
        return k0, k1, k2, k3
    k1 = frame.get('k1', data.get('k1'))
    k2 = frame.get('k2', data.get('k2'))
    k3 = frame.get('k3', data.get('k3'))
    k4 = frame.get('k4', data.get('k4'))
    if None not in (k1, k2, k3, k4):
        return k1, k2, k3, k4
    return None


def ensure_cycles_render_engine():
    """Ensure Cycles is enabled so panoramic fisheye settings are respected."""
    if bpy.context.scene.render.engine != 'CYCLES':
        bpy.context.scene.render.engine = 'CYCLES'


def apply_camera_jitter(location, rotation_matrix, rng, max_displacement, max_rotation_degrees):
    """Apply bounded random translation/rotation jitter for testing."""
    jittered_location = np.array(location, dtype=float).copy()
    jittered_rotation = np.array(rotation_matrix, dtype=float).copy()

    if max_displacement > 0.0:
        direction = rng.normal(size=3)
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction /= norm
            displacement = rng.uniform(0.0, max_displacement)
            jittered_location += direction * displacement

    if max_rotation_degrees > 0.0:
        rx = radians(rng.uniform(-max_rotation_degrees, max_rotation_degrees))
        ry = radians(rng.uniform(-max_rotation_degrees, max_rotation_degrees))
        rz = radians(rng.uniform(-max_rotation_degrees, max_rotation_degrees))
        base_rotation = Matrix(jittered_rotation.tolist())
        jitter_rotation = Euler((rx, ry, rz), 'XYZ').to_matrix()
        jittered_rotation = np.array((jitter_rotation @ base_rotation).to_3x3())

    return jittered_location, jittered_rotation


def import_cameras_from_transforms(transforms_path, convention='OPENGL'):
    """Import cameras from transforms.json file.
    
    Args:
        transforms_path: Path to transforms.json
        convention: Camera convention - 'OPENCV' or 'OPENGL'
    """
    
    with open(transforms_path, 'r') as f:
        data = json.load(f)

    # Get global intrinsics if available
    focal_length = data.get('fl_x', 512)
    camera_model = data.get('camera_model', 'OPENCV')

    frames = data.get('frames', [])
    
    if not frames:
        print("No frames found in transforms.json")
        return
    
    # Get image dimensions if available
    image_width = data.get('w', 1024)
    image_height = data.get('h', 1024)
    
    print(f"Importing {len(frames)} cameras from {transforms_path}")
    print(f"Camera convention: {convention}")
    print(f"Image dimensions: {image_width}x{image_height}")

    rng = None
    if ADD_JITTER:
        rng = np.random.default_rng(JITTER_SEED)
        print(
            "Camera jitter enabled: "
            f"max_displacement={JITTER_MAX_DISPLACEMENT}, "
            f"max_rotation={JITTER_MAX_ROTATION} degrees, "
            f"seed={JITTER_SEED}"
        )
    
    # Clear existing cameras
    #clear_existing_cameras()
    
    if camera_model == 'OPENCV_FISHEYE':
        ensure_cycles_render_engine()

    # Create a collection for cameras
    camera_collection = bpy.data.collections.new("Cameras")
    bpy.context.scene.collection.children.link(camera_collection)
    
    for i, frame in enumerate(frames):
        file_path = frame.get('file_path', f'frame_{i:06d}')
        transform_matrix = np.array(frame.get('transform_matrix', np.eye(4)))
        
        # Create camera object
        camera_label = frame.get('camera_label', f"Camera_{i:04d}")
        camera_data = bpy.data.cameras.new(name=camera_label)
        camera_obj = bpy.data.objects.new(camera_label, camera_data)
        
        # Link to collection
        camera_collection.objects.link(camera_obj)
        bpy.context.view_layer.objects.active = camera_obj
        
        # Set camera intrinsics

        # if camera_angle_x is not None:
        #     # Calculate focal length from field of view
        #     fx = image_width / (2 * np.tan(camera_angle_x / 2))
        #     fy = image_height / (2 * np.tan(camera_angle_y / 2 if camera_angle_y else camera_angle_x / 2))
        #     K = np.array([
        #         [fx, 0, image_width / 2],
        #         [0, fy, image_height / 2],
        #         [0, 0, 1]
        #     ])
        # else:
        # Use identity if not provided (assumes 50mm lens)
        #K = np.eye(3)
        
        focal_length = frame.get('fl_x', focal_length)
        frame_camera_model = frame.get('camera_model', camera_model)

        # Principal point.  Default to image center
        cx = frame.get('cx', data.get('cx', image_width / 2.0))
        cy = frame.get('cy', data.get('cy', image_height / 2.0))
        
        sensor_width, _ = set_camera_intrinsics(camera_data, focal_length, image_width, image_height, cx, cy)

        if frame_camera_model == 'OPENCV_FISHEYE':
            opencv_coeffs = get_opencv_fisheye_coeffs(frame, data)
            if opencv_coeffs is None:
                set_camera_fisheye_polynomial(camera_data, (0.0, 0.0, 0.0, 0.0, 0.0))
                print("Warning: Missing fisheye coefficients; using zeroed polynomial")
            else:
                blender_coeffs = convert_opencv_fisheye_to_blender(
                    opencv_coeffs[0], opencv_coeffs[1], opencv_coeffs[2], opencv_coeffs[3],
                    focal_length, sensor_width, image_width
                )
                set_camera_fisheye_polynomial(camera_data, blender_coeffs)
        
        # Convert and set extrinsics (pose matrix)
        blender_matrix = camera_to_blender(transform_matrix, convention=convention)
        
        # Extract location and rotation
        location = blender_matrix[:3, 3]
        rotation_matrix = blender_matrix[:3, :3]

        if ADD_JITTER:
            location, rotation_matrix = apply_camera_jitter(
                location,
                rotation_matrix,
                rng,
                JITTER_MAX_DISPLACEMENT,
                JITTER_MAX_ROTATION,
            )

        # Set location and rotation separately for reliability
        camera_obj.location = location
        
        # Set rotation from rotation matrix
        rot_mat = Matrix(rotation_matrix.tolist())
        camera_obj.rotation_euler = rot_mat.to_euler()
        
        print(f"  [{i+1}/{len(frames)}] {file_path}: position={location.tolist()}")
    
    print(f"Successfully imported {len(frames)} cameras")
    return camera_collection


def import_cameras_from_camera_dict(camera_dict_path):
    """Alternative: Import from a camera dictionary (like transforms.json with camera_dict format)."""
    
    with open(camera_dict_path, 'r') as f:
        data = json.load(f)
    
    cameras = data.get('cameras', {})
    
    if not cameras:
        print("No cameras found in file")
        return
    
    print(f"Importing {len(cameras)} cameras from {camera_dict_path}")
    
    # Clear existing cameras
    #clear_existing_cameras()
    
    # Create a collection for cameras
    camera_collection = bpy.data.collections.new("Cameras")
    bpy.context.scene.collection.children.link(camera_collection)
    
    for cam_name, cam_data in cameras.items():
        # Create camera object
        camera = bpy.data.cameras.new(name=cam_name)
        camera_obj = bpy.data.objects.new(cam_name, camera)
        
        # Link to collection
        camera_collection.objects.link(camera_obj)
        bpy.context.view_layer.objects.active = camera_obj
        
        # Get camera parameters
        K = np.array(cam_data.get('K', np.eye(3)))
        pose = np.array(cam_data.get('pose', np.eye(4)))
        
        # Get image dimensions
        image_width = cam_data.get('w', 1024)
        image_height = cam_data.get('h', 1024)
        
        # Set intrinsics
        set_camera_intrinsics(camera, K, image_width, image_height)
        
        # Set extrinsics
        # Convert from camera-to-world pose to world-to-camera for Blender
        T = np.linalg.inv(pose)
        blender_matrix = opengl_to_blender(T)
        
        location = blender_matrix[:3, 3]
        rotation_matrix = blender_matrix[:3, :3]
        
        # Set camera matrix directly
        # from mathutils import Matrix
        # mat4 = np.eye(4)
        # mat4[:3, :3] = rotation_matrix
        # mat4[:3, 3] = location
        # blender_mat = Matrix(mat4)
        #camera_obj.matrix_world = blender_mat
        camera_obj.location = location
        
        print(f"  {cam_name}: position={location.tolist()}")
    
    print(f"Successfully imported {len(cameras)} cameras")
    return camera_collection


def main():
    """Main import function."""
    transforms_path = Path(TRANSFORMS_JSON_PATH)
    
    if not transforms_path.exists():
        print(f"Error: {transforms_path} not found")
        return
    
    # Try to determine format and import
    with open(transforms_path, 'r') as f:
        data = json.load(f)
    
    if 'frames' in data:
        print("Detected NeRF frames format")
        import_cameras_from_transforms(str(transforms_path), convention=CAMERA_CONVENTION)
    elif 'cameras' in data:
        print("Detected camera dictionary format")
        import_cameras_from_camera_dict(str(transforms_path))
    else:
        print("Unknown format in transforms.json")


if __name__ == "__main__":
    main()
