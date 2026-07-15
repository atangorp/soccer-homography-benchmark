"""
src/geometry/homography.py
===========================
Utilitas komputasi homography: DLT via cv2.findHomography + RANSAC,
validasi matriks H, dan proyeksi titik.

Semua threshold diambil dari eval_global.yaml agar konsisten
di seluruh pipeline. Jangan hardcode nilai threshold di sini.
"""

from __future__ import annotations
import cv2
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
import yaml
import os


# ── Load global config ────────────────────────────────────────────────────────

def _load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    if config_path is None:
        # Cari dari root project (2 level di atas src/)
        base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        config_path = os.path.join(base, "configs", "eval_global.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Core homography computation ───────────────────────────────────────────────

def compute_homography(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    ransac_threshold: float = 5.0,
    method: int = cv2.RANSAC,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Hitung homography matrix H dari pasangan titik korespondensi
    menggunakan DLT + RANSAC.

    Parameters
    ----------
    src_pts : np.ndarray
        Shape (N, 2) — koordinat keypoint di image space (pixel).
    dst_pts : np.ndarray
        Shape (N, 2) — koordinat keypoint di canonical pitch space (120x80).
    ransac_threshold : float
        Reprojection error threshold untuk inlier classification.
        Ambil dari eval_global.yaml: ransac_threshold.
    method : int
        cv2.RANSAC (default) atau cv2.LMEDS.

    Returns
    -------
    H : np.ndarray atau None
        Matriks homography 3x3, atau None jika gagal.
    mask : np.ndarray atau None
        Boolean mask (N,) — True = inlier, False = outlier.

    Notes
    -----
    - Minimal 4 titik korespondensi diperlukan oleh DLT.
    - RANSAC secara iteratif memilih 4 titik acak, menghitung H,
      lalu menghitung berapa banyak titik lain yang konsisten
      (reprojection error < threshold). H terbaik disimpan.
    """
    if len(src_pts) < 4 or len(dst_pts) < 4:
        return None, None

    src = src_pts.astype(np.float32).reshape(-1, 1, 2)
    dst = dst_pts.astype(np.float32).reshape(-1, 1, 2)

    try:
        H, mask = cv2.findHomography(src, dst, method, ransacReprojThreshold=ransac_threshold)
        if H is None:
            return None, None
        return H, mask.flatten().astype(bool)
    except cv2.error:
        return None, None


def is_homography_valid(H: Optional[np.ndarray], max_condition_number: float = 1000.0) -> bool:
    """
    Validasi numerik matriks homography H.

    Checks:
    1. H tidak None
    2. H berukuran 3x3
    3. H tidak mengandung NaN atau Inf
    4. Condition number H di bawah threshold (matriks tidak ill-conditioned)

    Parameters
    ----------
    H : np.ndarray atau None
    max_condition_number : float
        Dari eval_global.yaml: max_condition_number. Default 1000.0.

    Returns
    -------
    bool

    Notes
    -----
    Condition number = sigma_max / sigma_min (dari SVD).
    Nilai besar → matriks numerically unstable → proyeksi tidak reliable.
    """
    if H is None:
        return False
    if H.shape != (3, 3):
        return False
    if not np.isfinite(H).all():
        return False

    try:
        cond = np.linalg.cond(H)
        return float(cond) < max_condition_number
    except np.linalg.LinAlgError:
        return False


def get_condition_number(H: Optional[np.ndarray]) -> float:
    """Return condition number dari H, atau np.inf jika invalid."""
    if H is None or not np.isfinite(H).all():
        return np.inf
    try:
        return float(np.linalg.cond(H))
    except np.linalg.LinAlgError:
        return np.inf


def project_points(
    points: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    """
    Proyeksikan titik-titik dari image space ke pitch space menggunakan H.

    Parameters
    ----------
    points : np.ndarray
        Shape (N, 2) dalam image coordinates.
    H : np.ndarray
        Homography matrix 3x3.

    Returns
    -------
    np.ndarray
        Shape (N, 2) dalam canonical pitch coordinates.
    """
    pts = points.astype(np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(pts, H)
    return projected.reshape(-1, 2)


# ── Homography dari DataFrame (interface utama pipeline) ─────────────────────

def compute_homography_from_df(
    df_src: pd.DataFrame,
    df_dst: pd.DataFrame,
    ransac_threshold: float = 5.0,
    min_keypoints: int = 6,
) -> Dict[str, Any]:
    """
    Hitung homography dari dua DataFrame yang sudah di-merge berdasarkan kpt_id.

    Parameters
    ----------
    df_src : pd.DataFrame
        Columns: kpt_id, x, y — keypoints di image space.
    df_dst : pd.DataFrame
        Columns: kpt_id, x, y — canonical pitch coordinates.
    ransac_threshold : float
        RANSAC inlier threshold (pixel).
    min_keypoints : int
        Minimum jumlah keypoint terdeteksi. Jika kurang → skip.

    Returns
    -------
    dict dengan keys:
        H               : np.ndarray atau None
        mask            : np.ndarray atau None (bool, shape N)
        is_four_points  : bool — apakah jumlah korespondensi >= min_keypoints
        is_H_valid      : bool — apakah H valid secara numerik
        condition_number: float
        n_inliers       : int
        n_outliers      : int
        n_correspondences : int
    """
    result = {
        "H": None,
        "mask": None,
        "is_four_points": False,
        "is_H_valid": False,
        "condition_number": np.inf,
        "n_inliers": 0,
        "n_outliers": 0,
        "n_correspondences": 0,
    }

    # Merge berdasarkan kpt_id untuk mendapat korespondensi yang valid
    merged = pd.merge(df_src, df_dst, on="kpt_id", suffixes=("_src", "_dst"))
    n = len(merged)
    result["n_correspondences"] = n
    result["is_four_points"] = n >= min_keypoints

    if not result["is_four_points"]:
        return result

    src_pts = merged[["x_src", "y_src"]].values.astype(np.float32)
    dst_pts = merged[["x_dst", "y_dst"]].values.astype(np.float32)

    H, mask = compute_homography(src_pts, dst_pts, ransac_threshold)
    result["H"] = H
    result["mask"] = mask

    if H is not None and mask is not None:
        result["is_H_valid"] = is_homography_valid(H)
        result["condition_number"] = get_condition_number(H)
        result["n_inliers"] = int(mask.sum())
        result["n_outliers"] = int((~mask).sum())

    return result
