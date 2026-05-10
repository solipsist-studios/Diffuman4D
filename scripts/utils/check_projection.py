import os, json, math, argparse
import numpy as np
import cv2

try:
    from plyfile import PlyData
    def load_ply_points(path):
        pd = PlyData.read(path)
        verts = pd['vertex'].data
        pts = np.vstack([verts['x'], verts['y'], verts['z']]).T
        return pts.astype(np.float64)
except Exception:
    import trimesh
    def load_ply_points(path):
        m = trimesh.load(path, force='mesh')
        return np.asarray(m.vertices, dtype=np.float64)

def project_points(pts_world, T_world_to_cam, intr, img_shape):
    # pts_world: (N,3), T: (4,4) maps world->camera
    pts_h = np.concatenate([pts_world, np.ones((len(pts_world),1))], axis=1).T  # 4xN
    cam_h = (T_world_to_cam @ pts_h).T  # N x 4
    cam = cam_h[:, :3]
    zs = cam[:, 2].copy()
    valid = zs > 1e-6
    u = (intr['fx'] * (cam[:,0] / zs) + intr['cx'])
    v = (intr['fy'] * (cam[:,1] / zs) + intr['cy'])
    uv = np.stack([u,v], axis=1)
    return uv, valid

def draw_overlay(img, uv, valid, color=(0,255,0), size=1, skip=4):
    out = img.copy()
    H,W = out.shape[:2]
    for i in range(0, len(uv), skip):
        if not valid[i]: continue
        x,y = int(round(uv[i,0])), int(round(uv[i,1]))
        if 0 <= x < W and 0 <= y < H:
            cv2.circle(out, (x,y), size, color, -1)
    return out

def main(args):
    with open(args.transforms, 'r') as f:
        data = json.load(f)
    pts = load_ply_points(args.ply)
    out_dir = os.path.abspath(args.output)
    os.makedirs(out_dir, exist_ok=True)

    for frame in data['frames']:
        img_path = os.path.join(os.path.dirname(args.transforms), frame['file_path'])
        if not os.path.exists(img_path):
            # try relative to images root
            img_path = os.path.join(args.images_root, os.path.basename(frame['file_path']))
        if not os.path.exists(img_path):
            print("Missing image for", img_path)
            continue
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print("Failed to read", img_path); continue
        H_img, W_img = img.shape[:2]
        w_decl, h_decl = frame.get('w', W_img), frame.get('h', H_img)
        sx = W_img / w_decl
        sy = H_img / h_decl

        intr = {
            'fx': frame['fl_x'] * sx,
            'fy': frame['fl_y'] * sy,
            'cx': frame['cx'] * sx,
            'cy': frame['cy'] * sy,
        }

        T = np.array(frame['transform_matrix'], dtype=np.float64)
        variants = []
        # as given: treat T as world->camera
        variants.append(('as_given_w2c', T.copy()))
        # invert: treat T as camera->world and invert to get world->camera
        try:
            variants.append(('inv_T', np.linalg.inv(T)))
        except Exception:
            pass
        # axis flips on camera frame (after we have world->camera)
        for name, baseT in list(variants):
            # flip X
            flipX = np.diag([-1.0,1.0,1.0,1.0])
            variants.append((f'{name}_flipX', baseT @ flipX))
            # flip Z
            flipZ = np.diag([1.0,1.0,-1.0,1.0])
            variants.append((f'{name}_flipZ', baseT @ flipZ))

        # create images for each variant (avoid duplicates)
        seen = set()
        for name, T_w2c in variants:
            if name in seen: continue
            seen.add(name)
            uv, valid = project_points(pts, T_w2c, intr, (H_img,W_img))
            ov = draw_overlay(img, uv, valid, color=(0,255,0), size=1, skip=max(1, len(uv)//20000))
            outfn = os.path.join(out_dir, f"{os.path.basename(frame['file_path']).replace('/','_')}_{name}.png")
            cv2.imwrite(outfn, ov)
            print("Wrote", outfn)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--transforms', required=True, help='transforms.json path')
    p.add_argument('--ply', required=True, help='sparse_pcd ply file path')
    p.add_argument('--images-root', default='', help='fallback root for image files')
    p.add_argument('--output', default='output/debug', help='output folder')
    args = p.parse_args()
    main(args)