# Require miniconda3
# Initialize conda
#$condaBase = & conda info --json | ConvertFrom-Json | Select-Object -ExpandProperty conda_prefix
#& "$condaBase\Scripts\conda.ps1" init PowerShell | Out-Null

$DataDir = ""
$Actions = @()
$AllActions = @("remove_background", "predict_keypoints", "triangulate_skeleton", "draw_skeleton")
$CameraFormat = "{0:02d}"
$CameraCount = 48
$ImageExt = ".webp"

# Parse command line arguments
$i = 0
while ($i -lt $args.Count) {
    switch ($args[$i]) {
        "--data_dir" {
            $DataDir = $args[$i + 1]
            $i += 2
        }
        "--actions" {
            $Actions = $args[$i + 1] -split ','
            $i += 2
        }
        "--camera_format" {
            $CameraFormat = $args[$i + 1]
            $i += 2
        }
        "--camera_count" {
            $CameraCount = $args[$i + 1]
            $i += 2
        }
        "--image_ext" {
            $ImageExt = $args[$i + 1]
            $i += 2
        }
        default {
            Write-Host ">> Unknown arg: $($args[$i])"
            exit 1
        }
    }
}

# Run all actions if not specified
if ($Actions.Count -eq 0) {
    $Actions = $AllActions
}

Write-Host ">> Data directory: $DataDir"
Write-Host ">> Actions: $($Actions -join ', ')"

foreach ($act in $Actions) {
    switch ($act) {
        "remove_background" {
            python scripts/preprocess/remove_background.py `
                --images_dir "$DataDir/images" `
                --out_fmasks_dir "$DataDir/fmasks" `
                --model_name ZhengPeng7/BiRefNet `
                --image_ext "${ImageExt}" `
                --batch_size 8 # decrease it if OOM
        }
        "predict_keypoints" {
            # it is recommend to use a seperate conda environment to run sapiens-lite
            # because sapiens-lite requires pytorch<=2.4.1, https://github.com/open-mmlab/mmdetection/issues/12008
            conda run --live-stream -n sapiens python scripts/preprocess/predict_keypoints.py `
                --images_dir "$DataDir/images" `
                --fmasks_dir "$DataDir/fmasks" `
                --out_kp2d_dir "$DataDir/poses_sapiens" `
                --image_ext "${ImageExt}" `
                --save_img
        }
        "triangulate_skeleton" {
            python scripts/preprocess/triangulate_skeleton.py `
                --camera_path "$DataDir/transforms.json" `
                --kp2d_dir "$DataDir/poses_sapiens" `
                --out_kp3d_dir "$DataDir/poses_3d" `
                --out_pcd_dir "$DataDir/poses_pcd" `
                --out_kp2d_proj_dir "$DataDir/poses_2d" `
                --spa_labels_proj_range "1,${CameraCount},1" `
                --spa_label_format "$CameraFormat" `
                --score_thr 0.3
        }
        "draw_skeleton" {
            python scripts/preprocess/draw_skeleton.py `
                --kp2d_dir "$DataDir/poses_2d" `
                --out_kpmap_dir "$DataDir/skeletons"
        }
        default {
            Write-Host "Invalid action: $act" -ForegroundColor Red
            exit 1
        }
    }
}
