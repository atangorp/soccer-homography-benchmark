"""
src/evaluation/benchmarks.py
==============================
Unified evaluation runner untuk semua 11 model.
Dipanggil dari notebooks/03_evaluation.ipynb.

Satu fungsi run_benchmark() menghasilkan DataFrame lengkap
yang jadi basis tabel di paper.
"""

from __future__ import annotations
import os
import glob
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
from tqdm import tqdm

from src.models.base_model import BasePoseModel
from src.geometry.metrics import mean_pixel_error, pck, classify_image_source
from src.evaluation.latency import benchmark_model


# ── Core benchmark runner ─────────────────────────────────────────────────────

def run_benchmark(
    model: BasePoseModel,
    test_images_dir: str,
    test_labels_dir: str,
    config: Dict[str, Any],
    n_latency_samples: int = 50,
) -> pd.DataFrame:
    """
    Jalankan evaluasi lengkap untuk satu model pada test set.

    Parameters
    ----------
    model : BasePoseModel
        Model wrapper yang sudah di-load (load_model() sudah dipanggil).
    test_images_dir : str
        Direktori gambar test.
    test_labels_dir : str
        Direktori label YOLO .txt test (untuk ground truth keypoints).
    config : dict
        Isi dari eval_global.yaml:
        conf_threshold, ransac_threshold, latency_warmup_runs, imgsz, dll.
    n_latency_samples : int
        Jumlah gambar untuk latency measurement.

    Returns
    -------
    pd.DataFrame
        Satu baris per gambar. Kolom:
        image_name, image_source, model_family, model_variant,
        mpe, n_valid_kpts, pck_10, instance_detected,
        [latency dan model_size di baris terakhir sebagai metadata]
    """
    info = model.get_model_info()
    img_paths = sorted(
        glob.glob(os.path.join(test_images_dir, "*.jpg")) +
        glob.glob(os.path.join(test_images_dir, "*.png"))
    )

    if not img_paths:
        raise FileNotFoundError(f"Tidak ada gambar di: {test_images_dir}")

    print(f"\n{'='*55}")
    print(f"  Evaluating: {info.full_name}")
    print(f"  Images    : {len(img_paths)}")
    print(f"{'='*55}")

    rows = []
    for img_path in tqdm(img_paths, desc=info.full_name):
        img_name = os.path.basename(img_path)
        image_source = classify_image_source(img_name)

        # Load ground truth
        label_path = os.path.join(
            test_labels_dir,
            os.path.splitext(img_name)[0] + ".txt"
        )
        gt_kpts, gt_vis = _load_gt_keypoints(label_path, img_path)

        # Run inference
        results = model.predict(img_path)
        detected = len(results) > 0

        mpe_val = np.nan
        pck_val = np.nan
        n_valid = 0

        if detected and gt_kpts is not None:
            pred = results[0]  # Top-1 detection
            # Pastikan shape sama dengan GT
            n_kpt = min(len(pred.keypoints), len(gt_kpts))
            mpe_val, _, n_valid = mean_pixel_error(
                pred.keypoints[:n_kpt],
                gt_kpts[:n_kpt],
                gt_vis[:n_kpt] if gt_vis is not None else None,
            )
            pck_val = pck(
                pred.keypoints[:n_kpt],
                gt_kpts[:n_kpt],
                gt_vis[:n_kpt] if gt_vis is not None else None,
                threshold=10.0,
            )

        rows.append({
            "image_name":       img_name,
            "image_source":     image_source,
            "model_family":     info.family,
            "model_variant":    info.variant,
            "model_full_name":  info.full_name,
            "instance_detected": int(detected),
            "mpe":              mpe_val,
            "n_valid_kpts":     n_valid,
            "pck_10":           pck_val,
        })

    df = pd.DataFrame(rows)

    # ── Latency measurement (subset gambar) ──────────────────────────────────
    latency_imgs = img_paths[:n_latency_samples]
    latency_result = benchmark_model(
        inference_fn=model.predict,
        sample_inputs=latency_imgs,
        model_path=model.weights_path,
        model_variant=info.full_name,
        n_warmup=config.get("latency_warmup_runs", 5),
        n_timed=min(100, len(latency_imgs)),
    )

    # Tambahkan latency sebagai kolom konstan di semua baris
    df["latency_mean_ms"] = latency_result["mean_ms"]
    df["latency_std_ms"]  = latency_result["std_ms"]
    df["fps"]             = latency_result["fps"]
    df["model_size_mb"]   = latency_result.get("model_size_mb", np.nan)

    _print_summary(df, info.full_name)
    return df


def _load_gt_keypoints(label_path: str, img_path: str):
    """
    Load ground truth keypoints dari YOLO label file.
    Return (keypoints_px, visibility) atau (None, None) jika file tidak ada.
    """
    if not os.path.exists(label_path):
        return None, None

    import cv2
    img = cv2.imread(img_path)
    if img is None:
        return None, None
    h, w = img.shape[:2]

    with open(label_path) as f:
        line = f.readline().strip()

    if not line:
        return None, None

    parts = list(map(float, line.split()))
    if len(parts) < 5:
        return None, None

    kpts_flat = parts[5:]
    n_kpt = len(kpts_flat) // 3
    keypoints  = np.zeros((n_kpt, 2), dtype=np.float32)
    visibility = np.zeros(n_kpt, dtype=int)

    for i in range(n_kpt):
        x_norm = kpts_flat[i * 3]
        y_norm = kpts_flat[i * 3 + 1]
        v      = int(kpts_flat[i * 3 + 2])
        keypoints[i] = [x_norm * w, y_norm * h]
        visibility[i] = v

    return keypoints, visibility


def _print_summary(df: pd.DataFrame, model_name: str) -> None:
    det_rate = df["instance_detected"].mean() * 100
    mpe_mean = df["mpe"].mean()
    mpe_std  = df["mpe"].std()
    fps      = df["fps"].iloc[0]
    size     = df["model_size_mb"].iloc[0]

    print(f"\n  ── {model_name} Summary ──")
    print(f"  Detection rate : {det_rate:.1f}%")
    print(f"  MPE            : {mpe_mean:.2f} ± {mpe_std:.2f} px")
    print(f"  FPS            : {fps:.1f}")
    if not np.isnan(size):
        print(f"  Model size     : {size:.1f} MB")


# ── Aggregate results ─────────────────────────────────────────────────────────

def aggregate_results(all_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Gabungkan hasil evaluasi semua model menjadi satu DataFrame summary.
    Ini yang jadi basis tabel utama di paper.

    Returns
    -------
    pd.DataFrame
        Satu baris per model. Kolom:
        model_full_name, model_family, model_variant,
        detection_rate, mpe_mean, mpe_std, pck_10_mean,
        latency_mean_ms, latency_std_ms, fps, model_size_mb
    """
    combined = pd.concat(all_dfs, ignore_index=True)

    summary = combined.groupby(
        ["model_full_name", "model_family", "model_variant"]
    ).agg(
        detection_rate=("instance_detected", "mean"),
        mpe_mean=("mpe", "mean"),
        mpe_std=("mpe", "std"),
        pck_10_mean=("pck_10", "mean"),
        n_images=("image_name", "count"),
        latency_mean_ms=("latency_mean_ms", "first"),
        latency_std_ms=("latency_std_ms", "first"),
        fps=("fps", "first"),
        model_size_mb=("model_size_mb", "first"),
    ).reset_index()

    # Urutkan berdasarkan MPE ascending (model terbaik di atas)
    summary = summary.sort_values("mpe_mean").reset_index(drop=True)
    summary.insert(0, "rank", range(1, len(summary) + 1))

    return summary


def save_results(
    per_image_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Simpan hasil evaluasi ke CSV."""
    os.makedirs(output_dir, exist_ok=True)

    per_image_path = os.path.join(output_dir, "results_per_image.csv")
    summary_path   = os.path.join(output_dir, "results_master.csv")

    per_image_df.to_csv(per_image_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"\n💾 Hasil tersimpan:")
    print(f"   Per-image : {per_image_path}")
    print(f"   Summary   : {summary_path}")
