import json
import numpy as np

transforms_file = "data/ariana_16/transforms_16_hloc.json"

# 1. Load the existing JSON
with open(transforms_file, 'r') as f:
    data = json.load(f)

translations = []

# 2. Fix the "Backwards" Cameras (OpenCV -> OpenGL)
for frame in data["frames"]:
    # Convert list to numpy array for math
    c2w = np.array(frame["transform_matrix"])
    
    # Flip the Y and Z axes to match PostShot/OpenGL expectations
    c2w[:3, 1] *= -1
    c2w[:3, 2] *= -1
    
    frame["transform_matrix"] = c2w.tolist()
    translations.append(c2w[:3, 3])

# 3. Fix the Center Origin (Move the rig center to 0,0,0)
translations = np.array(translations)
center_of_mass = translations.mean(axis=0)

for frame in data["frames"]:
    c2w = np.array(frame["transform_matrix"])
    # Subtract the center of mass from the translation vector
    c2w[:3, 3] -= center_of_mass
    frame["transform_matrix"] = c2w.tolist()

# 4. Save it back
with open(transforms_file, 'w') as f:
    json.dump(data, f, indent=4)

print("Poses flipped to OpenGL and rig centered!")