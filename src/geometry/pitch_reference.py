"""
src/geometry/pitch_reference.py
================================
Single source of truth untuk koordinat 32 keypoint lapangan sepak bola
dalam canonical coordinate space (120 x 80 pixels).

PENTING: Jangan hardcode koordinat ini di notebook manapun.
         Selalu import dari modul ini.

Referensi koordinat diekstrak dari notebook asli:
    keypoint_detection_homography_mapping.ipynb (cell 24)
    dan divalidasi terhadap EDA.ipynb (cell 6).

Catatan diskrepansi:
    EDA.ipynb cell 10 memiliki urutan sedikit berbeda (kpt_id berbasis
    flip_idx, bukan sekuensial 0-31). Modul ini menggunakan koordinat
    dari homography notebook sebagai ground truth karena itulah yang
    dipakai dalam pipeline komputasi.

Pitch layout (120 x 80):
    ┌─────────────────────────────────────────────────────────────┐  y=0
    │  0          13    14                    24    25          0  │
    │  1    9 10  15    16    ...             26    17   0 1    18 │
    │  2   10 11                                   ...            │  y=30
    │       6  7                                   ...        30  │
    │  3   11 12                                        31    50  │
    │  4    ↓  ↓                                         ↓    62  │
    │  5          16    17                    27    28    5   80  │
    └─────────────────────────────────────────────────────────────┘
         x=0    x=18  x=60                x=102 x=120

"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


# ── Canonical pitch dimensions ───────────────────────────────────────────────

PITCH_WIDTH: int = 120
PITCH_HEIGHT: int = 80
N_KEYPOINTS: int = 32


# ── Keypoint index mapping (0-indexed) ───────────────────────────────────────
# Urutan ini harus konsisten dengan urutan keypoint di label YOLO.
# kpt_id[i] adalah label semantik dari keypoint ke-i di YOLO label file.
# Nilai ini berasal dari flip_idx di data.yaml Roboflow dataset.

KPT_SEQUENTIAL_IDS: List[int] = list(range(32))


# ── Canonical coordinates (dalam pixel space 120 x 80) ───────────────────────

_X_COORDS: List[int] = [
    0,   0,   0,   0,   0,
    0,   6,   6,   12,  18,
    18,  18,  18,  60,  60,
    60,  60,  102, 102, 102,
    102, 108, 114, 114, 120,
    120, 120, 120, 120, 120,
    50,  70,
]

_Y_COORDS: List[int] = [
    0,  18,  30,  50,  62,
    80, 30,  50,  40,  18,
    32, 48,  62,  0,   30,
    50, 80,  18,  32,  48,
    62, 40,  30,  50,  0,
    18, 30,  50,  62,  80,
    40, 40,
]

assert len(_X_COORDS) == N_KEYPOINTS, f"Expected {N_KEYPOINTS} x-coords"
assert len(_Y_COORDS) == N_KEYPOINTS, f"Expected {N_KEYPOINTS} y-coords"


# ── Public interface ──────────────────────────────────────────────────────────

def get_keypoint_coords() -> Dict[int, Tuple[int, int]]:
    """
    Return dict mapping kpt_id (0-31) → (x, y) canonical coords.

    Returns
    -------
    dict[int, tuple[int, int]]
        {0: (0, 0), 1: (0, 18), ..., 31: (70, 40)}
    """
    return {i: (_X_COORDS[i], _Y_COORDS[i]) for i in range(N_KEYPOINTS)}


def get_dst_dataframe() -> pd.DataFrame:
    """
    Return DataFrame keypoint referensi untuk dipakai dalam homography pipeline.
    Di-merge dengan src_pts berdasarkan kolom 'kpt_id'.

    Returns
    -------
    pd.DataFrame
        Columns: kpt_id (int), x (int), y (int)

    Example
    -------
    >>> df_dst = get_dst_dataframe()
    >>> df_dst.head(3)
       kpt_id   x   y
    0       0   0   0
    1       1   0  18
    2       2   0  30
    """
    return pd.DataFrame({
        "kpt_id": KPT_SEQUENTIAL_IDS,
        "x": _X_COORDS,
        "y": _Y_COORDS,
    })


def get_dst_numpy() -> np.ndarray:
    """
    Return array koordinat dst untuk cv2.findHomography.

    Returns
    -------
    np.ndarray
        Shape (32, 2), dtype float32. Urutan sesuai kpt_id 0..31.
    """
    return np.array(list(zip(_X_COORDS, _Y_COORDS)), dtype=np.float32)


def get_keypoint_names() -> Dict[int, str]:
    """
    Return dict mapping kpt_id → nama semantik keypoint.
    Nama disesuaikan dengan posisi di lapangan.

    Returns
    -------
    dict[int, str]
    """
    names = {
        0:  "top_left_corner",
        1:  "left_penalty_top",
        2:  "left_goal_top",
        3:  "left_goal_bottom",
        4:  "left_penalty_bottom",
        5:  "bottom_left_corner",
        6:  "left_penalty_left_top",
        7:  "left_penalty_left_bottom",
        8:  "center_left_circle",
        9:  "left_penalty_box_top",
        10: "left_penalty_spot_top",
        11: "left_penalty_spot_bottom",
        12: "left_penalty_box_bottom",
        13: "center_top",
        14: "center_circle_top",
        15: "center_circle_bottom",
        16: "center_bottom",
        17: "right_penalty_box_top",
        18: "right_penalty_spot_top",
        19: "right_penalty_spot_bottom",
        20: "right_penalty_box_bottom",
        21: "center_right_circle",
        22: "right_penalty_right_top",
        23: "right_penalty_right_bottom",
        24: "top_right_corner",
        25: "right_penalty_top",
        26: "right_goal_top",
        27: "right_goal_bottom",  # Note: y=50 not y=48 — see notebook cell 24
        28: "right_penalty_bottom",
        29: "bottom_right_corner",
        30: "center_spot",
        31: "center_spot_alt",
    }
    return names


def get_pitch_boundaries() -> Dict[str, int]:
    """Return batas koordinat lapangan canonical."""
    return {
        "x_min": 0,
        "x_max": PITCH_WIDTH,
        "y_min": 0,
        "y_max": PITCH_HEIGHT,
    }


def validate_keypoints(kpt_ids: List[int]) -> bool:
    """
    Validasi apakah list kpt_id valid (semua dalam range 0..31).

    Parameters
    ----------
    kpt_ids : list[int]

    Returns
    -------
    bool
    """
    return all(0 <= k < N_KEYPOINTS for k in kpt_ids)


if __name__ == "__main__":
    df = get_dst_dataframe()
    print(f"Pitch reference: {N_KEYPOINTS} keypoints on {PITCH_WIDTH}x{PITCH_HEIGHT} canvas")
    print(df.to_string())
