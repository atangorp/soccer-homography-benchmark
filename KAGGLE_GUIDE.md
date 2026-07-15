# Panduan Lengkap: Kaggle-Only (Tanpa Colab)

Dokumen ini menjawab semua pertanyaanmu dan menjadi panduan definitif untuk
menjalankan seluruh project di Kaggle, tanpa Colab sama sekali.

---

## 0. Jawaban Singkat Dulu

| Pertanyaan | Jawaban |
|---|---|
| Aman pakai full Kaggle? | **Ya.** Kuota GPU gratis Kaggle adalah **~30 jam/minggu**, kadang naik ke 40 jam tergantung demand (tidak dijamin). Satu sesi maksimal **12 jam** untuk GPU/CPU. Reset tiap minggu. |
| Apakah notebook berubah? | **Ya**, tapi hanya bagian "environment setup" (2-3 cell paling atas). Semua notebook (00–07) sudah dimodifikasi — auto-detect Kaggle atau Colab, sisanya tidak berubah. |
| Apakah struktur folder berubah? | **Konsepnya sama, lokasi fisiknya beda.** Di Colab semua ada di satu folder Drive. Di Kaggle, kode+output baru ada di `/kaggle/working/`, sedangkan data & hasil dari notebook lain ada di `/kaggle/input/` (read-only). |
| Ada file yang harus berkesinambungan? | **Ya, jelas** — lihat tabel dependency lengkap di Bagian 3. |
| Tiap notebook punya output yang dipakai notebook lain? | **Ya** — setiap notebook training/evaluasi menghasilkan file yang WAJIB "diteruskan" ke notebook berikutnya lewat mekanisme Kaggle bernama **Notebook Output**. |

---

## 1. Konsep Inti: Kenapa Kaggle Berbeda dari Colab

Di **Colab**, kamu mount Google Drive sekali — setelah itu satu folder (`PROJECT_ROOT`) berfungsi untuk baca DAN tulis, dan isinya permanen selamanya.

Di **Kaggle**, tidak ada "Drive" yang di-mount. Sebagai gantinya ada dua folder yang terpisah fungsinya:

```
/kaggle/input/     ← READ-ONLY. Isinya dataset yang kamu "attach" (tempel).
                      Tidak bisa ditulis dari notebook manapun.

/kaggle/working/   ← WRITABLE. Tempat kamu nulis file baru.
                      TAPI: isinya HILANG setiap sesi berakhir,
                      KECUALI kamu klik "Save Version" (commit).
```

Begitu kamu commit/Save Version, isi `/kaggle/working/` di titik itu disimpan sebagai **"Output"** dari notebook tersebut — dan Output ini bisa kamu **attach ke notebook lain** sebagai input baru. Itulah mekanisme relay: Notebook A commit → jadi Dataset Output → Notebook B attach → muncul di `/kaggle/input/nama-notebook-A/`.

**Kesimpulan praktis:** kamu tidak lagi punya "satu folder ajaib" seperti di Drive. Sebagai gantinya, setiap notebook harus secara eksplisit:
1. **Clone kode** dari GitHub (karena `src/`, `configs/` perlu ada di setiap sesi baru)
2. **Attach dataset mentah** (sekali upload, dipakai berkali-kali)
3. **Attach output notebook sebelumnya** (kalau notebook ini butuh hasil training/eval dari notebook lain)

Semua 10 notebook **sudah dimodifikasi** untuk melakukan ini otomatis — tinggal attach dataset yang benar lewat UI, sisanya jalan sendiri.

---

## 2. Yang Berubah di Tiap Notebook (Konkret)

Setiap notebook sekarang punya cell baru di awal yang **auto-detect environment**:

```python
def _detect_env():
    if os.path.exists('/kaggle/working'):
        return 'kaggle'
    if os.path.exists('/content'):
        return 'colab'
    return 'local'
```

Kalau terdeteksi **Kaggle**:
- Clone kode dari GitHub ke `/kaggle/working/soccer-homography-benchmark`
- Symlink dataset mentah dari `/kaggle/input/soccer-field-raw` ke `{PROJECT_ROOT}/data/raw/...`
- Symlink output notebook sebelumnya (kalau notebook ini butuh) ke lokasi yang sama seperti sebelumnya

Kalau terdeteksi **Colab**: perilaku lama (mount Drive) tetap jalan — jadi kalau suatu saat kuota Colab pulih atau kamu mau pakai Colab Pro, notebook yang SAMA masih bisa jalan tanpa edit apapun.

**Variable downstream tidak berubah sama sekali** — `PROJECT_ROOT`, `YOLO_DIR`, `COCO_ANN_DIR`, `TEST_IMAGES`, dst semuanya tetap ada dengan nama yang sama. Jadi cell-cell setelah environment setup (training loop, evaluation loop, dst) **tidak perlu disentuh** — otomatis jalan karena variable-nya sudah benar.

---

## 3. Tabel Dependency Lengkap — File Apa Butuh File Apa

Ini jawaban paling penting untuk pertanyaan soal "file yang harus berkesinambungan":

| Notebook | Butuh attach (input) | Menghasilkan (harus di-commit) | Dipakai oleh |
|---|---|---|---|
| `00_EDA` | Dataset `soccer-field-raw` | Cuma figure, tidak perlu diteruskan | — |
| `01_dataset_conversion` | Dataset `soccer-field-raw` | `data/converted/annotations/*.json` | 02b, 02c, 02d, 03, 04 |
| `02a_train_yolo11` | Dataset `soccer-field-raw` | `artifacts/weights/yolo11/*` | 03, 04, 07 |
| `02b_train_hrnet` | `soccer-field-raw` + Output `01` | `artifacts/weights/hrnet/*` | 03, 04, 07 |
| `02c_train_vitpose` | `soccer-field-raw` + Output `01` | `artifacts/weights/vitpose/*` | 03, 04, 07 |
| `02d_train_detr` | `soccer-field-raw` + Output `01` | `artifacts/weights/detr/*` | 03, 04, 07 |
| `03_evaluation` | `soccer-field-raw` + Output `02a,02b,02c,02d` | `results_master.csv`, `results_per_image.csv` | 05, 07 |
| `04_homography_pipeline` | `soccer-field-raw` + Output `02a,02b,02c,02d` | `homography_results.csv` | 05 |
| `05_results_visualization` | Output `03` + Output `04` | figures `.pdf`, tables `.tex` | (masuk paper) |
| `07_video_demo` | Output `03` + Output `02x` (model terbaik) + video kamu sendiri | video `.mp4` | (qualitative result paper) |

**Cara baca tabel ini:** kolom "Butuh attach" adalah yang harus kamu tempelkan lewat tombol **"+ Add Input"** di sidebar kanan Kaggle sebelum menjalankan notebook itu. Kolom "Menghasilkan" adalah yang harus ada di `/kaggle/working/` saat kamu klik **Save Version** — supaya notebook di kolom "Dipakai oleh" bisa attach hasilnya nanti.

---

## 4. Setup Awal — Langkah Persis dari Nol

### Langkah 1: Push kode ke GitHub (sekali saja)

Kaggle butuh cara mengambil `src/`, `configs/`, `scripts/` di setiap sesi baru. Cara paling rapi adalah GitHub (bukan upload manual berkali-kali).

```bash
cd soccer-homography-benchmark
git init
git add .
git commit -m "Initial commit"
```

Buat repo baru di https://github.com/new (bisa **Private**, tidak masalah — Kaggle notebook tetap bisa clone repo private kalau kamu attach Personal Access Token lewat Kaggle Secrets, tapi untuk simpelnya di awal, buat **Public** dulu supaya `git clone` polos langsung jalan tanpa token).

```bash
git remote add origin https://github.com/USERNAME/soccer-homography-benchmark.git
git branch -M main
git push -u origin main
```

Lalu **edit SEKALI** baris `GITHUB_REPO = '...'` di tiap notebook, ganti `USERNAME` dengan username GitHub kamu. (10 notebook, 10 kali edit satu baris — searah dan cepat.)

### Langkah 2: Upload dataset mentah sebagai Kaggle Dataset (sekali saja)

1. Buka https://www.kaggle.com/datasets → **New Dataset**
2. Upload folder `soccer-field-localization.v9i.yolov8/` (hasil download dari Roboflow)
3. Set nama dataset (slug) persis: **`soccer-field-raw`**
   *(kalau kamu pakai nama lain, ganti semua kemunculan `'soccer-field-raw'` di notebook)*
4. Visibility: **Private** (rekomendasi, karena ini dataset riset kamu dengan Pak Anas)
5. Klik **Create**

### Langkah 3: Jalankan notebook pertama — `00_EDA`

1. Buka https://www.kaggle.com/code → **New Notebook**
2. Klik **File → Import Notebook** → upload `00_EDA.ipynb`
3. **Beri judul notebook: `00-EDA`** (huruf kecil, strip — penting untuk konsistensi slug)
4. Di sidebar kanan, klik **+ Add Input** → cari `soccer-field-raw` → attach
5. Set **Accelerator: None (CPU)** (00_EDA tidak butuh GPU)
6. Edit `GITHUB_REPO` di cell environment setup (isi URL repo kamu)
7. **Run All**
8. Setelah selesai, klik **Save Version** (kanan atas) → pilih **Save & Run All (Commit)**

### Langkah 4: Kunci test split — `splits_metadata.py`

Karena semua model harus dievaluasi di test set yang sama persis, jalankan sekali di sel manapun (bisa ditambahkan sebagai cell baru di `00_EDA`):

```python
!python {PROJECT_ROOT}/scripts/splits_metadata.py \
  --yolo_dir "{PROJECT_ROOT}/data/raw/soccer-field-localization.v9i.yolov8" \
  --output   "{PROJECT_ROOT}/data/splits_metadata.json"
```

Lalu commit ulang (**Save Version**) supaya `splits_metadata.json` ikut tersimpan di Output.

### Langkah 5: Jalankan `01_dataset_conversion`

1. New Notebook → Import `01_dataset_conversion.ipynb`
2. **Beri judul: `01-dataset-conversion`**
3. **+ Add Input** → attach `soccer-field-raw`
4. Accelerator: **None (CPU)**
5. Edit `GITHUB_REPO`
6. **Run All** → **Save Version**

Kaggle otomatis membuat slug output sesuai judul notebook (lowercase, spasi jadi strip). **Beri nama judul notebook persis sesuai kolom pertama tabel Bagian 3** supaya semua path attach di kode sudah cocok otomatis tanpa perlu edit lagi.

### Langkah 6: Training — `02a` sampai `02d`

Untuk tiap notebook training:
1. Import notebook, beri judul sesuai konvensi (`02a-train-yolo11`, `02b-train-hrnet`, dst)
2. **+ Add Input** → attach `soccer-field-raw`
3. Untuk 02b/02c/02d, **+ Add Input** juga → attach Output `01-dataset-conversion`
4. Accelerator: **GPU T4 x2** atau **P100** (pilih di dropdown Settings)
5. Edit `GITHUB_REPO`
6. **Run All** (bisa berjam-jam — Kaggle tetap jalan walau tab browser ditutup, asal jangan close dari halaman "Edit", cukup biarkan running di background)
7. **Save Version** setelah selesai

**Strategi kuota 30 jam/minggu:** karena satu sesi maksimal 12 jam dan kuota mingguan cuma 30 jam, kamu perlu mencicil training di beberapa hari:

```
Senin    : 02a_train_yolo11 (3 variant, ~9 jam) → sisa kuota ~21 jam
Selasa   : 02b_train_hrnet W18 (~8 jam)          → sisa kuota ~13 jam
Rabu     : 02b_train_hrnet W32 (~12 jam)         → kuota abis, tunggu reset
[reset minggu berikutnya]
Senin #2 : 02b_train_hrnet W48 + mulai 02c ...
```

Cek halaman **Settings → GPU** di akun Kaggle untuk melihat sisa kuota dan kapan reset-nya.

### Langkah 7: Evaluasi — `03` dan `04`

1. Import `03_evaluation.ipynb`, beri judul `03-evaluation`
2. **+ Add Input** → attach: `soccer-field-raw`, `02a-train-yolo11`, `02b-train-hrnet`, `02c-train-vitpose`, `02d-train-detr` (5 dataset sekaligus, semua lewat tombol yang sama)
3. Accelerator: GPU
4. **Run All** → **Save Version**
5. Ulangi persis sama untuk `04_homography_pipeline.ipynb` (judul: `04-homography-pipeline`)

### Langkah 8: Visualisasi — `05`

1. Import `05_results_visualization.ipynb`
2. **+ Add Input** → attach Output `03-evaluation` dan `04-homography-pipeline`
3. Accelerator: **None (CPU)** — cukup
4. **Run All** → **Save Version**
5. Download hasil figure/table dari tab **Output** di notebook ini

### Langkah 9 (opsional): Video demo — `07`

1. Upload video pertandinganmu sebagai Kaggle Dataset baru (slug bebas, misal `my-match-video`)
2. Import `07_video_demo.ipynb`
3. **+ Add Input** → attach: `soccer-field-raw`, semua output `02x`, output `03-evaluation`, dan `my-match-video`
4. Edit `VIDEO_INPUT` di Cell 4 mengarah ke `/kaggle/input/my-match-video/nama_file.mp4`
5. Accelerator: GPU
6. **Run All** → download video hasil dari tab Output

---

## 5. Perbedaan Struktur Folder — Sebelum vs Sesudah

```
KONSEP LAMA (Colab)                      KONSEP BARU (Kaggle)
─────────────────────                    ─────────────────────
Satu folder Drive                        Tiga sumber terpisah:
PROJECT_ROOT/
  |-- src/           \                     1. GitHub (kode)
  |-- configs/        |- semua di             -> clone ke /kaggle/working/...
  |-- notebooks/      |  satu tempat,
  |-- data/           |  baca+tulis           2. Kaggle Dataset "soccer-field-raw"
  `-- artifacts/      /  bebas                   -> /kaggle/input/soccer-field-raw/
                                                  (read-only, sekali upload)

                                            3. Output notebook lain (weights, CSV)
                                               -> /kaggle/input/{notebook-slug}/
                                               (read-only, muncul setelah attach)

                                          Semuanya di-symlink otomatis ke
                                          PROJECT_ROOT/data/... dan
                                          PROJECT_ROOT/artifacts/...
                                          oleh cell environment setup --
                                          jadi KODE downstream tetap
                                          membaca dari PROJECT_ROOT seperti biasa.
```

**Jadi jawabannya:** struktur folder LOGIS (nama-nama folder: `data/`, `artifacts/`, `src/`) **tidak berubah**. Yang berubah adalah dari MANA folder-folder itu berasal secara fisik — sebagian dari git clone, sebagian symlink ke dataset yang di-attach. Setelah environment-setup cell selesai jalan, semuanya terasa identik dengan Colab dari sudut pandang kode di bawahnya.

---

## 6. `setup_colab.py` dan `run_eval_all.sh` — Masih Perlu?

**`setup_colab.py`**: Tidak perlu lagi di Kaggle. Fungsinya (membuat folder kosong) sudah otomatis ter-cover oleh `os.makedirs(..., exist_ok=True)` yang ada di tiap notebook, dan struktur folder repo sudah ikut ter-clone dari GitHub.

**`run_eval_all.sh`**: Script ini didesain untuk trigger otomatis via `papermill`, yang mengasumsikan akses shell langsung ke satu filesystem persisten (cocok untuk Colab/server sendiri). Di Kaggle, cara paralelnya adalah pakai **Kaggle API** (`kaggle kernels push`) untuk menjalankan notebook secara terprogram — tapi untuk kebutuhan sekarang (jalan manual satu-satu lewat UI Kaggle), script ini **tidak perlu dipakai**. Cukup jalankan tiap notebook manual sesuai Langkah 7–8 di atas.

---

## 7. Troubleshooting Umum

| Gejala | Penyebab | Solusi |
|---|---|---|
| `FileNotFoundError: /kaggle/input/soccer-field-raw` | Lupa attach dataset | Klik "+ Add Input" di sidebar kanan, cari nama dataset kamu |
| `git clone` gagal / permission denied | Repo masih private tanpa token | Set repo jadi Public, atau tambahkan Personal Access Token via Kaggle Secrets |
| Notebook 03 tidak nemu weights model tertentu | Lupa attach output salah satu notebook 02x, atau notebook 02x itu belum di-commit | Cek tab "Data" di sidebar kanan — pastikan semua 4 output 02a-d ter-attach dan berwarna hijau (artinya berhasil di-load) |
| Kuota GPU habis di tengah training | Wajar — training besar > 12 jam per sesi | Pecah training per variant jadi beberapa sesi terpisah (sudah didesain begini di notebook 02b/02c/02d — tinggal ganti `VARIANT` tiap run) |
| Path attached dataset tidak sesuai dugaan | Kaggle kadang menaruh file di subfolder tambahan sesuai struktur asli saat commit | Cek isi folder lewat `!find /kaggle/input/{slug} -maxdepth 3` untuk lihat struktur persis, sesuaikan path `_link()` di notebook kalau beda |

---

## 8. Ringkasan Perubahan Mental Model

```
Colab : "Satu folder ajaib, semua ada di situ selamanya."

Kaggle: "Kode datang dari GitHub. Data mentah datang dari
         Dataset yang di-attach. Hasil training/eval datang dari
         Output notebook lain yang di-attach. Semuanya disatukan
         lagi jadi satu PROJECT_ROOT oleh cell pertama setiap notebook."
```

Begitu paham mekanisme "attach output sebagai input" ini, semuanya langsung jalan dengan sendirinya — dan kamu bebas dari limit GPU Colab yang selama ini bikin frustrasi.
