# require miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATADIR=""
IMAGE_EXT=".webp"
CAMERA_FORMAT="{0:04d}"
CAMERA_COUNT=12
ACTIONS=()
ALL_ACTIONS=("remove_background" "carve_vhull" "predict_keypoints" "triangulate_skeleton" "draw_skeleton")

while [[ $# -gt 0 ]]; do
  case $1 in
    --data_dir)
      DATADIR=$2
      shift 2
      ;;
    --actions)
      IFS=',' read -r -a ACTIONS <<< "$2"
      shift 2
      ;;
    --image_ext)
      IMAGE_EXT=$2
      shift 2
      ;;
    --camera_format)
      CAMERA_FORMAT=$2
      shift 2
      ;;
    --camera_count)
      CAMERA_COUNT=$2
      shift 2
      ;;
    *)
      echo ">> Unknown arg: $1"; exit 1;;
  esac
done

# run all actions if not specified
if [ ${#ACTIONS[@]} -eq 0 ]; then
  ACTIONS=("${ALL_ACTIONS[@]}")
fi

# validate required arguments
if [ -z "$DATADIR" ]; then
  echo ">> Error: --data_dir is required" >&2
  exit 1
fi

if [ ! -d "$DATADIR" ]; then
  echo ">> Error: --data_dir does not exist: $DATADIR" >&2
  exit 1
fi
DATADIR="$(cd "$DATADIR" && pwd)"

echo ">> Data directory: $DATADIR"
echo ">> Actions: ${ACTIONS[@]}"

for act in "${ACTIONS[@]}"; do
  case "$act" in
    remove_background)
      python "${SCRIPT_DIR}/remove_background.py" \
        --images_dir "$DATADIR/images" \
        --out_fmasks_dir "$DATADIR/masks" \
        --model_name ZhengPeng7/BiRefNet \
        --image_ext "$IMAGE_EXT" \
        --batch_size 8 # decrease it if OOM
      ;;
    carve_vhull)
      conda activate diffuman4d
      python scripts/preprocess/carve_visual_hull.py \
        --fmasks_dir "$DATADIR/fmasks" \
        --cameras_path "$DATADIR/transforms.json" \
        --out_vhull_dir "$DATADIR/surfs"
      cp "$DATADIR/surfs/000000.ply" "$DATADIR/sparse_pcd.ply"
      ;;
    predict_keypoints)
      # it is recommend to use a seperate conda environment to run sapiens-lite
      # because sapiens-lite requires pytorch<=2.4.1, https://github.com/open-mmlab/mmdetection/issues/12008
      python "${SCRIPT_DIR}/predict_keypoints.py" \
        --images_dir "$DATADIR/images" \
        --fmasks_dir "$DATADIR/masks" \
        --image_ext "$IMAGE_EXT" \
        --out_kp2d_dir "$DATADIR/poses_sapiens"
      ;;
    triangulate_skeleton)
      python "${SCRIPT_DIR}/triangulate_skeleton.py" \
        --camera_path "$DATADIR/transforms.json" \
        --kp2d_dir "$DATADIR/poses_sapiens" \
        --out_kp3d_dir "$DATADIR/poses_3d" \
        --out_pcd_dir "$DATADIR/poses_pcd" \
        --out_kp2d_proj_dir "$DATADIR/poses_2d" \
        --spa_labels_proj_range "1,$((CAMERA_COUNT+1)),1" \
        --spa_label_format "$CAMERA_FORMAT" \
        --score_thr 0.3
      ;;
    draw_skeleton)
      python "${SCRIPT_DIR}/draw_skeleton.py" \
        --kp2d_dir "$DATADIR/poses_2d" \
        --out_kpmap_dir "$DATADIR/skeletons"
      ;;
    *)
      echo "Invalid action: $act" >&2
      exit 1
      ;;
  esac
done
