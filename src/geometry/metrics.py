"""
src/geometry/metrics.py
========================
Semua metrik evaluasi untuk pose estimation dan homography.
Tidak ada metrik yang dihitung inline di notebook — semua import dari sini.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple


# ── Mean Pixel Error ──────────────────────────────────────────────────────────

def mean_pixel_error(
    pred_kpts: np.ndarray,
    gt_kpts: np.ndarray,
    visibility: Optional[np.ndarray] = None,
) -> Tuple[float, float, int]:
    """
    Hitung Mean Pixel Error (MPE) antara prediksi dan ground truth keypoints.
    Hanya menghitung untuk keypoint yang visible (visibility > 0).

    Parameters
    ----------
    pred_kpts : np.ndarray
        Shape (N, 2) — prediksi keypoints dalam pixel (x, y).
    gt_kpts : np.ndarray
        Shape (N, 2) — ground truth keypoints dalam pixel (x, y).
    visibility : np.ndarray atau None
        Shape (N,) — visibility flags (0=invisible, 1=occluded, 2=visible).
        Jika None, semua keypoint dianggap visible.

    Returns
    -------
    mpe : float
        Mean pixel error (rata-rata Euclidean distance).
    std : float
        Standard deviation pixel error.
    n_valid : int
        Jumlah keypoint yang dipakai dalam perhitungan.

    Notes
    -----
    Fix dari kode asli: kode lama tidak mengembalikan std.
    Std dibutuhkan untuk statistical significance testing.
    """
    assert pred_kpts.shape == gt_kpts.shape, \
        f"Shape mismatch: pred {pred_kpts.shape} vs gt {gt_kpts.shape}"

    if visibility is None:
        visibility = np.ones(len(pred_kpts))

    mask = visibility > 0
    if mask.sum() == 0:
        return np.nan, np.nan, 0

    pred_vis = pred_kpts[mask]
    gt_vis = gt_kpts[mask]

    errors = np.sqrt(np.sum((pred_vis - gt_vis) ** 2, axis=1))
    return float(errors.mean()), float(errors.std()), int(mask.sum())


def mean_pixel_error_per_keypoint(
    pred_kpts: np.ndarray,
    gt_kpts: np.ndarray,
    visibility: Optional[np.ndarray] = None,
    n_keypoints: int = 32,
) -> pd.DataFrame:
    """
    Hitung MPE per keypoint index (0..31).
    Berguna untuk analisis 'keypoint mana yang paling susah diprediksi'.

    Returns
    -------
    pd.DataFrame
        Columns: kpt_id, mpe, n_samples
    """
    records = []
    for i in range(n_keypoints):
        v = visibility[i] if visibility is not None else 1
        if v > 0:
            err = np.sqrt(np.sum((pred_kpts[i] - gt_kpts[i]) ** 2))
            records.append({"kpt_id": i, "error": err})

    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["kpt_id", "mpe", "n_samples"])

    return df.groupby("kpt_id")["error"].agg(
        mpe="mean", n_samples="count"
    ).reset_index()


# ── PCK (Percentage of Correct Keypoints) ────────────────────────────────────

def pck(
    pred_kpts: np.ndarray,
    gt_kpts: np.ndarray,
    visibility: Optional[np.ndarray] = None,
    threshold: float = 10.0,
) -> float:
    """
    Hitung PCK@threshold — persentase keypoint yang error-nya < threshold px.

    Parameters
    ----------
    threshold : float
        Error threshold dalam pixel. PCK@10 artinya threshold=10px.

    Returns
    -------
    float : PCK score (0..1)
    """
    if visibility is None:
        visibility = np.ones(len(pred_kpts))

    mask = visibility > 0
    if mask.sum() == 0:
        return np.nan

    errors = np.sqrt(np.sum((pred_kpts[mask] - gt_kpts[mask]) ** 2, axis=1))
    return float((errors < threshold).mean())


# ── Homography metrics ────────────────────────────────────────────────────────

def homography_validity_rate(results_df: pd.DataFrame) -> Dict[str, float]:
    """
    Hitung homography validity rate dari DataFrame hasil evaluasi.

    Parameters
    ----------
    results_df : pd.DataFrame
        Harus punya kolom: is_H_valid, is_four_points.
        Opsional: image_source (untuk breakdown per source).

    Returns
    -------
    dict dengan keys:
        overall_validity_rate   : float (0..1)
        four_point_rate         : float
        conditional_validity    : float (valid | four_points terpenuhi)
        per_source              : dict (jika kolom image_source ada)
    """
    total = len(results_df)
    if total == 0:
        return {}

    result = {
        "overall_validity_rate": results_df["is_H_valid"].mean(),
        "four_point_rate": results_df["is_four_points"].mean(),
        "conditional_validity": (
            results_df.loc[results_df["is_four_points"], "is_H_valid"].mean()
            if results_df["is_four_points"].any() else np.nan
        ),
        "n_total": total,
        "n_valid": results_df["is_H_valid"].sum(),
    }

    if "image_source" in results_df.columns:
        per_source = {}
        for source, grp in results_df.groupby("image_source"):
            per_source[source] = {
                "validity_rate": grp["is_H_valid"].mean(),
                "n": len(grp),
            }
        result["per_source"] = per_source

    return result


def reprojection_error_stats(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    H: np.ndarray,
) -> Dict[str, float]:
    """
    Hitung reprojection error stats dari H yang sudah dihitung.
    Berguna untuk analisis kualitas H beyond binary valid/invalid.

    Returns
    -------
    dict: mean_reproj_error, max_reproj_error, std_reproj_error
    """
    import cv2
    src = src_pts.astype(np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(src, H).reshape(-1, 2)
    errors = np.sqrt(np.sum((projected - dst_pts) ** 2, axis=1))
    return {
        "mean_reproj_error": float(errors.mean()),
        "std_reproj_error": float(errors.std()),
        "max_reproj_error": float(errors.max()),
        "median_reproj_error": float(np.median(errors)),
    }


# ── Image source classifier ───────────────────────────────────────────────────

def classify_image_source(
    filename: str,
    broadcast_keywords: Optional[List[str]] = None,
    scouting_keywords: Optional[List[str]] = None,
) -> str:
    """
    Klasifikasikan gambar sebagai 'broadcast', 'scouting', atau 'unknown'
    berdasarkan keyword dalam nama file.

    Parameters
    ----------
    filename : str
        Nama file gambar (basename, bukan full path).
    broadcast_keywords : list[str]
        Default: ["worldcup", "broadcast", "tv", "match"]
    scouting_keywords : list[str]
        Default: ["drone", "tactical", "scouting", "top"]

    Returns
    -------
    str : "broadcast" | "scouting" | "unknown"
    """
    if broadcast_keywords is None:
        broadcast_keywords = ["worldcup", "broadcast", "tv", "match"]
    if scouting_keywords is None:
        scouting_keywords = ["drone", "tactical", "scouting", "top"]

    fname_lower = filename.lower()
    if any(k in fname_lower for k in broadcast_keywords):
        return "broadcast"
    if any(k in fname_lower for k in scouting_keywords):
        return "scouting"
    return "unknown"
