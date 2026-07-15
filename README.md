# Soccer Homography Benchmark

Benchmarking 11 pose estimation architectures for soccer field keypoint detection and homography matrix prediction.

**Target publication:** Scopus Q2 (Computer Vision / Sports Analytics)

## Models evaluated (11 total)

| Family | Variants | Framework |
|---|---|---|
| YOLO11-Pose | Small, Medium, XLarge | Ultralytics |
| HRNet | W18, W32, W48 | MMPose |
| ViTPose | Small, Base, Large | MMPose |
| Deformable DETR | R50, R101 | Custom |

## Runtime: Colab atau Kaggle

Semua notebook di `notebooks/` mendukung **auto-detect environment** — bisa jalan
di Google Colab maupun Kaggle tanpa modifikasi. Kalau kena limit GPU Colab,
lihat `KAGGLE_GUIDE.md` untuk panduan lengkap menjalankan seluruh pipeline di Kaggle.

## Quick start (Colab)

```bash
# 1. Setup project structure (jalankan sekali di Colab)
python scripts/setup_colab.py --drive_root "/content/drive/MyDrive/Colab Notebooks"

# 2. Konversi dataset (CPU-only, ~10 menit)
# Buka: notebooks/01_dataset_conversion.ipynb

# 3. Training per keluarga model (butuh GPU)
# notebooks/02a_train_yolo11.ipynb
# notebooks/02b_train_hrnet.ipynb
# notebooks/02c_train_vitpose.ipynb
# notebooks/02d_train_detr.ipynb

# 4. Evaluasi unified
# notebooks/03_evaluation.ipynb
# notebooks/04_homography_pipeline.ipynb
```

## Evaluation metrics

- **Pose accuracy:** mAP@0.5, mAP@0.5:0.95, Mean Pixel Error (MPE ± std)
- **Efficiency:** FPS, Latency (ms ± std), Model size (MB)
- **Downstream:** Homography validity rate, condition number, inlier/outlier ratio
- **Statistical:** Wilcoxon signed-rank test, Kruskal-Wallis, Cohen's d

## Key config

Semua hyperparameter evaluasi ada di `configs/eval_global.yaml`.
Jangan ubah tanpa mendiskusikan implikasi ke seluruh pipeline.

```yaml
conf_threshold: 0.8
ransac_threshold: 5.0
min_keypoints_for_homography: 6
latency_warmup_runs: 5
```

## Dataset

Soccer Field Localization v9 dari Roboflow Universe  
Source: https://universe.roboflow.com/adzuki/soccer-field-localization  
32 keypoints, canonical pitch: 120 × 80 px

## Citation

```bibtex
@article{xxx2025soccer,
  title   = {Benchmarking Pose Estimation Architectures for Soccer Field
             Keypoint Detection and Homography Prediction},
  author  = {xxx, F. and Anas, M.},
  journal = {xxx},
  year    = {2025}
}
```
