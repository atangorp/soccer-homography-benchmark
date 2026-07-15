# Submission Checklist — IEEE Access

Centang semua item ini sebelum upload ke IEEE Author Portal.

---

## A. Kode & Data (sebelum paper dikirim)

- [ ] GitHub repository dibuat (public atau akan di-public saat accepted)
- [ ] README.md repository mencantumkan: cara install, cara run, dataset link
- [ ] Semua 11 model weights tersedia via Google Drive / Hugging Face link
- [ ] `data/splits_metadata.json` di-commit ke repo (memastikan reproducibility)
- [ ] `configs/eval_global.yaml` final (tidak ada lagi inkonsistensi threshold)
- [ ] Semua `[VALUE]` di `paper/main.tex` sudah diisi angka aktual
- [ ] URL GitHub sudah dimasukkan ke paper (Abstract + Conclusion + Acknowledgment)

---

## B. Figures (resolusi 300 DPI, format PDF/EPS)

- [ ] `fig_mpe_comparison.pdf` — dari `05_results_visualization.ipynb`
- [ ] `fig_accuracy_efficiency_tradeoff.pdf`
- [ ] `fig_homography_validity.pdf`
- [ ] `pitch_reference.png` — dari `00_EDA.ipynb`
- [ ] `visibility_distribution.png`
- [ ] `sample_annotations.png`
- [ ] Semua figure punya caption yang self-contained (bisa dipahami tanpa baca paper)
- [ ] Font di figure konsisten dengan font paper (serif, min 9pt)

---

## C. Tables

- [ ] Table 1 (Models): parameter count terisi dari model aktual
- [ ] Table 2 (Results main): semua [VALUE] terisi dari `results_master.csv`
- [ ] Table 3 (Source breakdown): broadcast vs scouting dari `homography_results.csv`
- [ ] Table statistical tests: Wilcoxon p-values dari `statistical_tests.csv`
- [ ] Semua tabel punya caption di atas (IEEE style: caption di atas tabel)

---

## D. Paper content (sebelum submit)

- [ ] **Abstract**: angka spesifik tercantum (bukan hanya "kami menunjukkan bahwa")
- [ ] **Introduction**: 3 kontribusi masih relevan dengan hasil aktual
- [ ] **Related Work**: minimal 15 referensi, tidak ada referensi di bawah 2018 kecuali foundational
- [ ] **Methods**: semua formula dikonfirmasi konsisten dengan implementasi
- [ ] **Dataset**: N_train, N_val, N_test diisi angka aktual
- [ ] **Results**: semua pernyataan kuantitatif ada backing data-nya
- [ ] **Discussion**: menyebutkan limitations (mandatory untuk Q2 review)
- [ ] **Conclusion**: tidak mengulang Abstract word-for-word
- [ ] Tidak ada `\cite{}` yang kosong atau broken
- [ ] Tidak ada referensi di .bib yang tidak di-cite di paper

---

## E. Format IEEE Access

- [ ] Template: `IEEEtran.cls` (bukan template konferensi)
- [ ] Double column layout dikonfirmasi
- [ ] Page limit: IEEE Access tidak ada page limit — tapi usahakan < 14 halaman
- [ ] Section numbering: Roman numerals (I, II, III...)
- [ ] Equation numbering: per-section bukan global (opsional, ikuti template)
- [ ] Acknowledgment section ada (bukan optional, standar IEEE)
- [ ] Keywords: 5–10 keywords, huruf kecil, dipisah koma

---

## F. Author information

- [ ] Nama semua author sudah benar (tidak ada typo)
- [ ] Affiliation sudah benar: "Department of Informatics Engineering, Faculty of Engineering, Universitas Negeri Surabaya (UNESA), Indonesia"
- [ ] Email corresponding author valid
- [ ] ORCID semua author (strongly recommended oleh IEEE Access)
- [ ] Author contribution statement (IEEE Access meminta ini):
  - Contoh: "F.X. conceived the study, designed the experiments, implemented the codebase, and wrote the manuscript. M.A. supervised the research, provided conceptual guidance, and revised the manuscript."

---

## G. Pre-submission self-review

Tanya diri sendiri untuk setiap klaim di paper:

| Klaim | Bukti |
|---|---|
| "Model X mencapai Y% HVR" | Ada di `homography_results.csv` baris yang sesuai |
| "Perbedaan signifikan (p < 0.05)" | Ada di `statistical_tests.csv` |
| "X% lebih cepat dari Y" | Ada di `results_master.csv` kolom fps |
| "Pertama yang membandingkan N arsitektur" | Sudah cek Google Scholar / Semantic Scholar |

---

## H. Submission di IEEE Author Portal

- [ ] Login ke <https://ieee.atypon.com/> (atau portal terbaru IEEE Access)
- [ ] Upload: `main.pdf` (compiled), `main.tex`, semua figure files, `references.bib`
- [ ] Cover letter: upload `cover_letter.pdf`
- [ ] Pilih: Article Type = **Regular Paper**
- [ ] Pilih: Section/Topic Area = **Computing and Processing** atau **Signal Processing and Analysis**
- [ ] Confirm: Open Access (APC required — cek apakah ada fee waiver untuk afiliasi UNESA)
- [ ] Suggested reviewers: isi minimal 3 nama (Cell C di atas)

---

## I. Setelah submit

- [ ] Screenshot confirmation email disimpan
- [ ] Catat manuscript ID untuk tracking
- [ ] Estimasi waktu review IEEE Access: **6–12 minggu**
- [ ] Kalau mendapat "Major Revision": jangan panik — itu normal dan bukan rejection
- [ ] Siapkan response letter template (akan dibuat saat revision datang)

---

**Target submission date:** __________ (isi sendiri)

**IEEE Access APC (2025):** ~$1,850 USD
*(Cek apakah UNESA atau Kemendikbud punya institutional agreement dengan IEEE untuk fee waiver)*
