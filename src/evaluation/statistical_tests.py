"""
src/evaluation/statistical_tests.py
=====================================
Statistical significance testing untuk perbandingan multi-model.
Dibutuhkan oleh reviewer Scopus Q2 untuk memvalidasi bahwa perbedaan
performa antar model bukan noise statistik.

Tests yang diimplementasikan:
    1. Wilcoxon signed-rank test   — pairwise comparison (non-parametric)
    2. Kruskal-Wallis test         — multi-group comparison (non-parametric)
    3. Effect size (Cohen's d)     — besar praktis perbedaan

Mengapa non-parametric?
    Distribusi pixel error dan homography success rate tidak dijamin normal
    (Shapiro-Wilk test sering reject normalitas di dataset pose estimation).
    Non-parametric test lebih aman dan reviewer Q2 lebih jarang
    mempertanyakan asumsinya.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from itertools import combinations
from scipy import stats


# ── Pairwise Wilcoxon signed-rank test ───────────────────────────────────────

def pairwise_wilcoxon(
    data: Dict[str, np.ndarray],
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> pd.DataFrame:
    """
    Lakukan Wilcoxon signed-rank test untuk semua pasang model.

    Dipakai untuk: membandingkan distribusi MPE atau homography validity
    antara dua model secara pairwise.

    Parameters
    ----------
    data : dict
        {"model_name": array_of_errors_per_image}
        Semua array harus panjang sama (same test images).
    alpha : float
        Significance level. Default 0.05.
    alternative : str
        "two-sided", "less", atau "greater".

    Returns
    -------
    pd.DataFrame
        Columns: model_a, model_b, statistic, p_value, significant,
                 effect_size_d, interpretation

    Example
    -------
    >>> errors = {
    ...     "YOLO11-S":  np.array([5.2, 4.1, 6.3, ...]),
    ...     "HRNet-W48": np.array([3.1, 2.8, 4.2, ...]),
    ... }
    >>> df = pairwise_wilcoxon(errors)
    """
    model_names = list(data.keys())
    rows = []

    for m_a, m_b in combinations(model_names, 2):
        arr_a = np.array(data[m_a])
        arr_b = np.array(data[m_b])

        if len(arr_a) != len(arr_b):
            raise ValueError(
                f"Array length mismatch: {m_a} ({len(arr_a)}) vs {m_b} ({len(arr_b)}). "
                f"Pastikan evaluasi dilakukan pada test set yang sama."
            )

        stat, p_val = stats.wilcoxon(arr_a, arr_b, alternative=alternative)
        d = cohen_d(arr_a, arr_b)
        interp = _interpret_effect_size(d)

        rows.append({
            "model_a":        m_a,
            "model_b":        m_b,
            "statistic":      float(stat),
            "p_value":        float(p_val),
            "significant":    p_val < alpha,
            "effect_size_d":  float(d),
            "interpretation": interp,
            "alpha":          alpha,
        })

    return pd.DataFrame(rows)


# ── Kruskal-Wallis test ───────────────────────────────────────────────────────

def kruskal_wallis_test(
    data: Dict[str, np.ndarray],
    alpha: float = 0.05,
) -> Dict:
    """
    Kruskal-Wallis H-test untuk membandingkan semua model sekaligus.
    Jawab pertanyaan: "Apakah ada setidaknya satu model yang berbeda signifikan?"

    Parameters
    ----------
    data : dict
        {"model_name": array_of_values}
        Panjang array boleh berbeda (berbeda jumlah sampel valid).
    alpha : float
        Significance level.

    Returns
    -------
    dict:
        statistic    : float
        p_value      : float
        significant  : bool
        df           : int (degrees of freedom = n_models - 1)
        interpretation : str

    Notes
    -----
    Jika significant=True, lakukan pairwise_wilcoxon untuk post-hoc analysis.
    Jika significant=False, tidak perlu pairwise test.
    """
    groups = list(data.values())
    model_names = list(data.keys())

    stat, p_val = stats.kruskal(*groups)

    if p_val < alpha:
        interp = (
            f"Terdapat perbedaan signifikan antar {len(model_names)} model "
            f"(H={stat:.3f}, p={p_val:.4f} < {alpha}). "
            f"Lanjutkan dengan pairwise Wilcoxon untuk post-hoc analysis."
        )
    else:
        interp = (
            f"Tidak ada perbedaan signifikan antar model "
            f"(H={stat:.3f}, p={p_val:.4f} ≥ {alpha})."
        )

    return {
        "statistic":     float(stat),
        "p_value":       float(p_val),
        "significant":   bool(p_val < alpha),
        "df":            len(model_names) - 1,
        "n_models":      len(model_names),
        "models":        model_names,
        "interpretation": interp,
        "alpha":         alpha,
    }


# ── Effect size ───────────────────────────────────────────────────────────────

def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Hitung Cohen's d effect size antara dua distribusi.

    Interpretasi standar:
        |d| < 0.2  → negligible
        |d| < 0.5  → small
        |d| < 0.8  → medium
        |d| >= 0.8 → large

    Returns
    -------
    float : Cohen's d (bisa negatif jika mean(a) < mean(b))
    """
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return np.nan

    pooled_std = np.sqrt(
        ((n_a - 1) * np.var(a, ddof=1) + (n_b - 1) * np.var(b, ddof=1))
        / (n_a + n_b - 2)
    )
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def _interpret_effect_size(d: float) -> str:
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


# ── Summary table untuk paper ─────────────────────────────────────────────────

def generate_significance_table(
    results_df: pd.DataFrame,
    metric_col: str = "mpe_mean",
    model_col: str = "model_variant",
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate tabel signifikansi dari results_master.csv untuk paper.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame dari results_master.csv.
        Harus punya kolom: model_variant, mpe_mean (atau kolom lain).
    metric_col : str
        Kolom metrik yang mau dibandingkan.
    model_col : str
        Kolom nama model.
    group_col : str atau None
        Jika ada, buat tabel per group (misal: per image_source).

    Returns
    -------
    pd.DataFrame
        Tabel pairwise comparison siap dimasukkan ke paper.
    """
    # Pivot: satu kolom per model, satu baris per image
    if group_col:
        dfs = []
        for grp_val, grp in results_df.groupby(group_col):
            pivot = grp.pivot_table(
                index="image_id", columns=model_col, values=metric_col
            ).dropna()
            data_dict = {col: pivot[col].values for col in pivot.columns}
            df_wil = pairwise_wilcoxon(data_dict)
            df_wil.insert(0, group_col, grp_val)
            dfs.append(df_wil)
        return pd.concat(dfs, ignore_index=True)
    else:
        pivot = results_df.pivot_table(
            index="image_id", columns=model_col, values=metric_col
        ).dropna()
        data_dict = {col: pivot[col].values for col in pivot.columns}
        return pairwise_wilcoxon(data_dict)


# ── Normality check (opsional, untuk transparansi di paper) ──────────────────

def check_normality(data: Dict[str, np.ndarray], alpha: float = 0.05) -> pd.DataFrame:
    """
    Shapiro-Wilk test untuk setiap model.
    Gunakan ini untuk membenarkan pilihan non-parametric test di paper.

    Returns
    -------
    pd.DataFrame : Columns: model, statistic, p_value, is_normal
    """
    rows = []
    for name, arr in data.items():
        if len(arr) < 3:
            continue
        stat, p_val = stats.shapiro(arr[:5000])  # Shapiro max 5000 samples
        rows.append({
            "model":     name,
            "statistic": float(stat),
            "p_value":   float(p_val),
            "is_normal": p_val >= alpha,
        })
    df = pd.DataFrame(rows)
    n_normal = df["is_normal"].sum()
    n_total = len(df)
    print(f"\n📊 Normality check (Shapiro-Wilk, α={alpha}):")
    print(f"   Normal distribution: {n_normal}/{n_total} models")
    if n_normal < n_total:
        print(f"   → Rekomendasi: pakai non-parametric test (Wilcoxon / Kruskal-Wallis)")
    return df
