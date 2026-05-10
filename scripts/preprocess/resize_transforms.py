import argparse
import copy
import json
import os

# --- CONFIGURATION ---
TARGET_WIDTH = 1024
TARGET_HEIGHT = 1024
# ---------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Resize intrinsic parameters in a transforms.json file.")
    parser.add_argument("input_json", nargs="?", default="transforms.json", help="Path to the input transforms JSON file")
    return parser.parse_args()

def resize_intrinsics(data, target_w, target_h):
    # Detect original resolution (assuming it's in the first frame or global)
    if "w" in data and "h" in data:
        orig_w = data["w"]
        orig_h = data["h"]
    else:
        # Fallback: try to find it in the first frame
        orig_w = data["frames"][0]["w"]
        orig_h = data["frames"][0]["h"]
    
    # Calculate scale factors
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h
    
    print(f"Original: {orig_w}x{orig_h} -> Target: {target_w}x{target_h}")
    print(f"Scale X: {scale_x:.4f}, Scale Y: {scale_y:.4f}")

    # Function to update a single set of intrinsics
    def update_params(params):
        if "fl_x" in params: params["fl_x"] *= scale_x
        if "fl_y" in params: params["fl_y"] *= scale_y
        if "cx" in params: params["cx"] *= scale_x
        if "cy" in params: params["cy"] *= scale_y
        params["w"] = target_w
        params["h"] = target_h
        return params

    # 1. Update Global Intrinsics (if they exist at top level)
    data = update_params(data)

    # 2. Update Per-Frame Intrinsics (if they exist inside frames)
    for frame in data["frames"]:
        frame = update_params(frame)
        
    return data

if __name__ == "__main__":
    args = parse_args()

    input_json = args.input_json
    base, ext = os.path.splitext(os.path.basename(input_json))
    output_json = os.path.join(os.path.dirname(input_json), f"{base}_resized{ext}")

    with open(input_json, "r") as f:
        data = json.load(f)

    new_data = resize_intrinsics(copy.deepcopy(data), TARGET_WIDTH, TARGET_HEIGHT)

    with open(output_json, "w") as f:
        json.dump(new_data, f, indent=4)

    print(f"Done! Saved to {output_json}")