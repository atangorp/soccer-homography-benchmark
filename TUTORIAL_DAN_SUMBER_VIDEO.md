# Panduan Lengkap: Eksekusi Project + Sumber Video Sepak Bola

Dokumen ini punya dua bagian: (A) tutorial bertahap menjalankan zip project,
dan (B) sumber video/dataset sepak bola yang legal dan layak untuk riset.

---

# BAGIAN A — Tutorial Eksekusi Project

## Tahap 0: Persiapan Akun & Storage

**Yang kamu butuhkan sebelum mulai:**

| Kebutuhan | Free tier | Catatan |
|---|---|---|
| Google Drive | 15 GB | Cukup untuk code + beberapa weights kecil |
| Google Colab | Gratis (T4, limit ~4 jam/sesi) | Untuk training YOLO11 |
| Colab Pro (opsional) | $9.99/bulan | Lebih lama runtime, akses V100/A100 |
| Kaggle | Gratis (P100, 30 jam GPU/minggu) | Alternatif untuk HRNet/ViTPose/DETR |

> 💡 Karena HRNet, ViTPose, dan DETR butuh V100 dengan durasi training panjang,
> sangat disarankan pakai **Kaggle Notebooks** untuk fase training berat
> (kuota 30 jam GPU/minggu gratis, reset tiap Senin) dan **Colab** untuk
> dataset conversion + evaluation yang lebih ringan.

---

## Tahap 1: Extract & Upload ke Google Drive

1. Download `soccer-homography-benchmark-FINAL.zip`
2. Extract di komputer lokal kamu
3. Upload folder hasil extract ke Google Drive, dengan path **persis**:
   ```
   My Drive/Colab Notebooks/soccer-homography-benchmark/
   ```
4. Buat juga folder kosong untuk dataset:
   ```
   My Drive/Colab Notebooks/Dataset/
   ```
   (Dataset akan diisi di Tahap 2)

**Cara verifikasi upload sudah benar:**
Buka Google Drive di browser → masuk ke `Colab Notebooks` → harus terlihat
folder `soccer-homography-benchmark` berisi `notebooks/`, `src/`, `configs/`, dll.

---

## Tahap 2: Download Dataset

1. Buka Roboflow Universe: https://universe.roboflow.com/adzuki/soccer-field-localization
2. Klik **Download Dataset**
3. Pilih format **YOLOv8** (bukan YOLOv5 atau format lain)
4. Pilih opsi **download zip ke komputer** (bukan langsung ke Colab)
5. Extract zip hasil download
6. Upload folder hasil extract ke Google Drive:
   ```
   My Drive/Colab Notebooks/Dataset/soccer-field-localization.v9i.yolov8/
   ```

**Struktur yang harus terlihat setelah upload:**
```
Dataset/soccer-field-localization.v9i.yolov8/
  ├── train/images/   train/labels/
  ├── valid/images/   valid/labels/   ← Roboflow biasa pakai 'valid' bukan 'val'
  ├── test/images/    test/labels/
  └── data.yaml
```

> ⚠️ **Cek nama folder split.** Roboflow kadang export dengan nama `valid/`
> bukan `val/`. Kalau begini, rename manual jadi `val/` di Google Drive,
> karena seluruh notebook project ini mengasumsikan nama `val/`.

---

## Tahap 3: Buka Google Colab & Mount Drive

1. Buka https://colab.research.google.com
2. Klik **File → Open notebook → Google Drive**
3. Navigasi ke `Colab Notebooks/soccer-homography-benchmark/notebooks/`
4. Buka `00_EDA.ipynb`
5. Klik **Runtime → Change runtime type** → pilih **None** (CPU saja, EDA tidak butuh GPU)
6. Jalankan cell pertama — ini akan minta izin akses Google Drive, klik **Allow**

---

## Tahap 4: Jalankan Setup Folder (sekali saja)

Buka notebook baru kosong di Colab, jalankan:

```python
from google.colab import drive
drive.mount('/content/drive')

!python "/content/drive/MyDrive/Colab Notebooks/soccer-homography-benchmark/scripts/setup_colab.py"
```

Ini akan otomatis membuat semua subfolder (`artifacts/weights/...`, `data/converted/...`, dll) yang dibutuhkan notebook-notebook berikutnya.

---

## Tahap 5: Kunci Test Set (WAJIB sebelum training)

Masih di notebook yang sama:

```python
!python "/content/drive/MyDrive/Colab Notebooks/soccer-homography-benchmark/scripts/splits_metadata.py" \
  --yolo_dir "/content/drive/MyDrive/Colab Notebooks/Dataset/soccer-field-localization.v9i.yolov8" \
  --output   "/content/drive/MyDrive/Colab Notebooks/soccer-homography-benchmark/data/splits_metadata.json"
```

Ini mencatat daftar gambar test set sehingga semua 11 model nanti dievaluasi
pada set yang identik — jangan skip langkah ini.

---

## Tahap 6: Jalankan Notebook Berurutan

Buka tiap notebook lewat **File → Open notebook → Google Drive**, lalu **Runtime → Run all**.

| Urutan | Notebook | Runtime type | Estimasi waktu |
|---|---|---|---|
| 1 | `00_EDA.ipynb` | CPU | 10 menit |
| 2 | `01_dataset_conversion.ipynb` | CPU | 10 menit |
| 3 | `02a_train_yolo11.ipynb` | T4 GPU | ~9 jam (3 variant) |
| 4 | `02b_train_hrnet.ipynb` | V100 GPU* | ~40 jam (jalankan 3×, ganti `VARIANT` tiap kali) |
| 5 | `02c_train_vitpose.ipynb` | V100 GPU* | ~54 jam (jalankan 3×) |
| 6 | `02d_train_detr.ipynb` | V100 GPU* | ~25 jam (jalankan 2×) |
| 7 | `03_evaluation.ipynb` | T4 GPU | ~6 jam |
| 8 | `04_homography_pipeline.ipynb` | T4 GPU | ~4 jam |
| 9 | `05_results_visualization.ipynb` | CPU | 30 menit |

*Untuk notebook 02b/02c/02d, pertimbangkan jalankan di **Kaggle** (lihat Tahap 7) karena durasinya melebihi limit sesi gratis Colab.

**Cara ganti runtime GPU di Colab:**
`Runtime → Change runtime type → Hardware accelerator → GPU → pilih T4/A100/dsb`
(tipe GPU yang tersedia gratis tergantung kuota harian Google, biasanya T4)

---

## Tahap 7: (Disarankan) Pindah Training Berat ke Kaggle

Untuk notebook `02b`, `02c`, `02d` yang super panjang:

1. Buka https://www.kaggle.com/code → **New Notebook**
2. **File → Import Notebook** → upload file `.ipynb` dari Google Drive kamu
3. Di sidebar kanan: **Settings → Accelerator → GPU P100**
4. Ganti baris mount Google Drive jadi koneksi via Kaggle Secrets atau upload manual dataset sebagai Kaggle Dataset (lebih stabil dari mount Drive di Kaggle)
5. Kaggle kasih kuota **30 jam GPU/minggu** — cukup untuk 1-2 model besar per minggu

> 💡 Trik: precompute hasil training di Kaggle, lalu download checkpoint `.pth`/`.pt` dan upload manual ke Google Drive project kamu di folder `artifacts/weights/{family}/{variant}/`.

---

## Tahap 8: Cek Progress & Troubleshooting

**Setelah training selesai, verifikasi weights ada:**

```python
import os
W = "/content/drive/MyDrive/Colab Notebooks/soccer-homography-benchmark/artifacts/weights"
for family, variants in [
    ("yolo11", ["small","medium","xlarge"]),
    ("hrnet", ["w18","w32","w48"]),
    ("vitpose", ["small","base","large"]),
    ("detr", ["r50","r101"]),
]:
    for v in variants:
        d = f"{W}/{family}/{v}"
        has_weight = any(f.endswith(('.pt','.pth')) for f in os.listdir(d)) if os.path.exists(d) else False
        print(f"  {'✅' if has_weight else '⏳'} {family}-{v}")
```

**Kalau ada error, paste error message lengkapnya ke chat ini** — sertakan nama notebook dan nomor cell yang error.

---

## Tahap 9: Hasil Akhir

Setelah `05_results_visualization.ipynb` selesai, kamu akan punya:
- `artifacts/results/figures/*.pdf` — siap masuk paper
- `artifacts/results/tables/results_table.tex` — tabel LaTeX siap pakai
- `artifacts/logs/evaluation/results_master.csv` — data mentah semua model

Buka `paper/main.tex`, isi semua `[VALUE]` dengan angka dari CSV, lalu compile.
Sebelum submit, cek `SUBMISSION_CHECKLIST.md`.

---

---

# BAGIAN B — Sumber Video Sepak Bola untuk Riset

Catatan penting di awal: jangan scrape video broadcast resmi (Liga 1, Premier League,
dsb) dari YouTube/streaming tanpa izin — itu melanggar hak siar dan bisa bermasalah
secara hukum maupun etik akademik. Gunakan dataset riset yang memang didedikasikan
untuk computer vision research. Berikut opsi yang aman dan kredibel untuk paper Q2.

## 1. SoccerNet — sumber paling relevan untuk project ini ⭐

SoccerNet adalah dataset riset terbesar untuk soccer video understanding, berisi 550 pertandingan broadcast lengkap dari liga-liga top Eropa (Serie A, La Liga, Premier League, Ligue 1, Bundesliga, Champions League), dengan total hampir 800 jam footage dari musim 2014-2017 dan seterusnya, dilengkapi ratusan ribu anotasi untuk berbagai task computer vision.

**Yang paling relevan untuk project kamu: SoccerNet Camera Calibration task.** Task ini secara spesifik menyediakan gambar dengan anotasi keypoint/garis lapangan untuk task lokalisasi elemen lapangan dan kalibrasi kamera otomatis — persis use case homography kalian.

Cara akses:
```python
pip install SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader as SNdl
soccerNetDownloader = SNdl(LocalDirectory="path/to/SoccerNet")
soccerNetDownloader.downloadDataTask(task="calibration-2023", split=["train","valid","test","challenge"])
```
Sumber: GitHub `SoccerNet/sn-calibration`, situs resmi `soccer-net.org`

> ⚠️ Sebagian data SoccerNet (video full match) butuh password yang didapat dengan mengisi NDA (Non-Disclosure Agreement) riset di situs resmi mereka — ini standar untuk melindungi hak siar broadcaster. Data calibration/keypoint biasanya tidak butuh NDA seketat video penuh. Cek dokumentasi resmi di soccer-net.org untuk detail per-task.

## 2. Dataset turunan SoccerNet yang sudah jadi keypoint (siap pakai)

Ada dataset turunan yang sudah mengekstrak keypoint dari SoccerNet Calibration dengan format yang dekat dengan kebutuhan kalian — kombinasi teknik line intersection dan deteksi area lapangan untuk menghasilkan keypoint siap pakai untuk task deteksi keypoint lapangan sepak bola.

Tersedia di Hugging Face: `Adit-jain/Soccana_Keypoint_detection_v1`
(perlu download SoccerNet Calibration Dataset dulu sebagai basis, lalu repo ini menyediakan label keypoint tambahan)

## 3. DFL — Bundesliga Data Shootout (Kaggle)

Kaggle pernah menjalankan kompetisi resmi bekerja sama dengan Bundesliga (DFL) yang menyediakan footage asli pertandingan Bundesliga untuk riset computer vision, dipakai sebagai basis berbagai tutorial keypoint detection dan homography untuk sepak bola.

Cari di Kaggle: **"DFL - Bundesliga Data Shootout"** — datanya legal untuk riset karena memang dirilis resmi oleh Bundesliga untuk kompetisi.

## 4. SoccerTrack v2

Dataset multi-view full-pitch terbaru (2025) untuk game state reconstruction, menyediakan video dari sudut fisheye dan drone yang dirancang khusus untuk riset tracking dan analisis lapangan sepak bola — cocok kalau kalian ingin menambah variasi sudut kamera scouting di luar broadcast.

## 5. Roboflow Universe (yang sudah kalian pakai)

Selain dataset yang sudah kalian punya, Roboflow Universe punya puluhan dataset soccer field/pitch lainnya yang sudah dianotasi dan siap pakai — cari dengan keyword "soccer field", "football pitch keypoints", atau "soccer homography" di universe.roboflow.com. Banyak yang berasal dari ekstraksi frame video broadcast yang sudah diberi lisensi terbuka oleh uploader-nya.

---

## Rekomendasi konkret untuk paper kalian

Karena dataset utama kalian sudah dari Roboflow (32 keypoints custom), opsi paling realistis untuk **memperkaya generalisasi model** (bukan mengganti dataset utama) adalah:

1. **Tambahkan subset kecil dari SoccerNet Calibration** sebagai *external validation set* — ini akan memperkuat klaim novelty kalian soal "broadcast vs scouting" karena SoccerNet representasi broadcast yang sangat standar dan diakui industri.
2. Sebutkan di paper bagian Limitations bahwa primary dataset adalah Roboflow, dan SoccerNet dipakai sebagai cross-dataset generalization check — reviewer Q2 biasanya sangat suka melihat ini karena menunjukkan robustness di luar training distribution.

Kalau mau, saya bisa bantu buatkan notebook tambahan `06_external_validation_soccernet.ipynb` untuk uji generalisasi model kalian di SoccerNet Calibration — bilang saja kalau mau lanjut ke situ.
