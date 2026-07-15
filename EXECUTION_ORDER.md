# Execution Order — Soccer Homography Benchmark

Ikuti urutan ini **tepat**. Jangan skip langkah.

---

## Sebelum mulai (sekali saja)

```bash
# 1. Upload dataset Roboflow ke Google Drive
#    Path: My Drive/Colab Notebooks/Dataset/soccer-field-localization.v9i.yolov8/

# 2. Upload seluruh folder soccer-homography-benchmark/ ke Google Drive
#    Path: My Drive/Colab Notebooks/soccer-homography-benchmark/

# 3. Buat folder hierarchy
python scripts/setup_colab.py

# 4. Kunci test set (WAJIB sebelum training apapun)
python scripts/splits_metadata.py \
  --yolo_dir ".../soccer-field-localization.v9i.yolov8" \
  --output   "data/splits_metadata.json"
```

---

## Phase 1 — Dataset (CPU, ~10 menit)

```
01_dataset_conversion.ipynb
  Input  : data/raw/soccer-field-localization.v9i.yolov8/
  Output : data/converted/annotations/{train,val,test}.json
  Verify : semua cell harus output ✅
```

---

## Phase 2–5 — Training (GPU, hari-hari)

Jalankan satu notebook per session. **Restart runtime antar variant.**

| Notebook | GPU | Total waktu |
|---|---|---|
| `02a_train_yolo11.ipynb` | T4 | ~9 jam |
| `02b_train_hrnet.ipynb`  | V100 | ~40 jam (3 run) |
| `02c_train_vitpose.ipynb`| V100 | ~54 jam (3 run) |
| `02d_train_detr.ipynb`   | V100 | ~25 jam (2 run) |

Setelah setiap run, pastikan `artifacts/weights/{family}/{variant}/train_log.json` tersimpan.

---

## Phase 6 — Evaluation (GPU, ~6 jam)

```
# Verifikasi splits masih konsisten sebelum eval
python scripts/splits_metadata.py --verify \
  --yolo_dir ".../soccer-field-localization.v9i.yolov8" \
  --output   "data/splits_metadata.json"

# Jalankan evaluation pipeline
03_evaluation.ipynb          → artifacts/logs/evaluation/results_master.csv
04_homography_pipeline.ipynb → artifacts/logs/evaluation/homography_results.csv

# Atau otomatis overnight di Kaggle:
bash scripts/run_eval_all.sh
```

---

## Phase 7 — Paper (CPU, ~2 minggu)

```
05_results_visualization.ipynb → artifacts/results/figures/*.pdf
                                → artifacts/results/tables/results_table.tex

# LaTeX paper
paper/main.tex  (target: IEEE Access / EAAI / Applied Sciences)
```

---

## Checklist weights sebelum evaluation

```
artifacts/weights/
  yolo11/small/run/weights/best.pt       ✅
  yolo11/medium/run/weights/best.pt      ✅
  yolo11/xlarge/run/weights/best.pt      ✅
  hrnet/w18/best_coco_AP_epoch_*.pth     ✅
  hrnet/w32/best_coco_AP_epoch_*.pth     ✅
  hrnet/w48/best_coco_AP_epoch_*.pth     ✅
  vitpose/small/best_coco_AP_epoch_*.pth ✅
  vitpose/base/best_coco_AP_epoch_*.pth  ✅
  vitpose/large/best_coco_AP_epoch_*.pth ✅
  detr/r50/checkpoint_best.pth           ✅
  detr/r101/checkpoint_best.pth          ✅
```

---

## Phase 8 — Video Demo (GPU, ~30 menit per video)

```
07_video_demo.ipynb
  Input  : video pertandingan .mp4 + best model dari Phase 6
  Output : artifacts/results/videos/{nama}_demo.mp4
           artifacts/results/videos/{nama}_preview.png
```

Upload video ke: `My Drive/Colab Notebooks/Videos/`
Isi `VIDEO_INPUT` di Cell 4 dengan path video kamu, lalu Run All.

---

## Jalur Alternatif — Kaggle-Only (Tanpa Colab)

Kalau kena limit GPU Colab, seluruh pipeline ini juga bisa dijalankan **100% di Kaggle**,
karena semua notebook sudah mendukung auto-detect environment (Kaggle/Colab/local).

**Baca panduan lengkapnya di:** `KAGGLE_GUIDE.md`

Ringkasan super singkat:
1. Push kode ke GitHub (sekali)
2. Upload dataset mentah sebagai Kaggle Dataset bernama `soccer-field-raw`
3. Jalankan tiap notebook di Kaggle, attach input yang sesuai (lihat tabel dependency di KAGGLE_GUIDE.md)
4. Setiap notebook: **Save Version (Commit)** setelah selesai, supaya hasilnya bisa di-attach ke notebook berikutnya

Kuota gratis Kaggle: ~30 jam GPU/minggu, sesi maksimal 12 jam — reset tiap minggu.
