"""
src/evaluation/latency.py
==========================
Pengukuran latency GPU yang proper untuk benchmarking inferensi model.

Fix dari kode eksisting (keypoint_detection_homography_mapping.ipynb):
  ❌ Lama: wall-clock loop serial tanpa warmup, tanpa GPU sync
  ✅ Baru : 5 warmup runs + torch.cuda.synchronize() + 100 timed runs

Referensi metodologi:
  - MLPerf Inference: https://mlcommons.org/en/inference-tiny-07/
  - "Measuring latency of DNN inference on edge devices" (standard practice)
"""

from __future__ import annotations
import time
import numpy as np
from typing import Callable, List, Optional, Dict
import os


def measure_latency(
    inference_fn: Callable,
    sample_inputs: List,
    n_warmup: int = 5,
    n_timed: int = 100,
    use_cuda_sync: bool = True,
) -> Dict[str, float]:
    """
    Ukur latency inferensi model secara proper dengan warmup dan GPU sync.

    Parameters
    ----------
    inference_fn : callable
        Fungsi yang dipanggil dengan satu input. Signature: fn(input) -> output.
        Contoh: lambda img: model(img, verbose=False)
    sample_inputs : list
        List input yang akan dipakai secara siklik.
        Minimal 1 input. Lebih banyak lebih representatif.
    n_warmup : int
        Jumlah warmup runs (tidak di-timing).
        Tujuan: GPU cache warm, CUDA context init, PyTorch JIT compilation.
        Default: 5 (sesuai eval_global.yaml)
    n_timed : int
        Jumlah timed runs untuk menghitung statistik.
        Default: 100
    use_cuda_sync : bool
        Apakah pakai torch.cuda.synchronize() sebelum stop timer.
        WAJIB True untuk GPU inference agar timer tidak premature stop.

    Returns
    -------
    dict dengan keys:
        mean_ms     : float — mean latency dalam milidetik
        std_ms      : float — standard deviation
        min_ms      : float — minimum latency
        max_ms      : float — maximum latency
        p50_ms      : float — median
        p95_ms      : float — 95th percentile
        fps         : float — frames per second (1000 / mean_ms)
        n_runs      : int
        n_warmup    : int

    Notes
    -----
    Untuk model yang dijalankan di CPU (bukan GPU), set use_cuda_sync=False.
    CUDA sync tidak berpengaruh jika tidak ada GPU — aman dijalankan juga.
    """
    # Import opsional — tidak crash jika tidak ada torch
    _cuda_available = False
    try:
        import torch
        _cuda_available = torch.cuda.is_available() and use_cuda_sync
    except ImportError:
        pass

    def _sync():
        if _cuda_available:
            import torch
            torch.cuda.synchronize()

    if len(sample_inputs) == 0:
        raise ValueError("sample_inputs tidak boleh kosong")

    # ── Warmup runs ──────────────────────────────────────────────────────────
    for i in range(n_warmup):
        inp = sample_inputs[i % len(sample_inputs)]
        inference_fn(inp)
    _sync()

    # ── Timed runs ───────────────────────────────────────────────────────────
    times_ms = []
    for i in range(n_timed):
        inp = sample_inputs[i % len(sample_inputs)]

        _sync()
        t_start = time.perf_counter()
        inference_fn(inp)
        _sync()
        t_end = time.perf_counter()

        times_ms.append((t_end - t_start) * 1000.0)

    times = np.array(times_ms)

    return {
        "mean_ms":  float(np.mean(times)),
        "std_ms":   float(np.std(times)),
        "min_ms":   float(np.min(times)),
        "max_ms":   float(np.max(times)),
        "p50_ms":   float(np.percentile(times, 50)),
        "p95_ms":   float(np.percentile(times, 95)),
        "fps":      float(1000.0 / np.mean(times)),
        "n_runs":   n_timed,
        "n_warmup": n_warmup,
    }


def get_model_size_mb(model_path: str) -> float:
    """
    Dapatkan ukuran file model dalam megabytes.

    Parameters
    ----------
    model_path : str
        Path ke file .pt atau .pth

    Returns
    -------
    float : ukuran dalam MB
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")
    size_bytes = os.path.getsize(model_path)
    return size_bytes / (1024 ** 2)


def get_model_param_count(model) -> Dict[str, int]:
    """
    Hitung jumlah parameter model PyTorch.

    Parameters
    ----------
    model : torch.nn.Module

    Returns
    -------
    dict:
        total_params     : int
        trainable_params : int
        frozen_params    : int
    """
    try:
        import torch
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {
            "total_params": total,
            "trainable_params": trainable,
            "frozen_params": total - trainable,
        }
    except Exception as e:
        return {"error": str(e)}


def benchmark_model(
    inference_fn: Callable,
    sample_inputs: List,
    model_path: Optional[str] = None,
    model_variant: str = "unknown",
    n_warmup: int = 5,
    n_timed: int = 100,
) -> Dict:
    """
    One-stop benchmark function: latency + model size + metadata.

    Parameters
    ----------
    inference_fn : callable
    sample_inputs : list
    model_path : str atau None
        Jika diberikan, tambahkan model_size_mb ke output.
    model_variant : str
        Label untuk logging.

    Returns
    -------
    dict: semua metrik latency + model_size_mb + variant label
    """
    print(f"⏱️  Benchmarking {model_variant}...")
    print(f"   Warmup runs : {n_warmup}")
    print(f"   Timed runs  : {n_timed}")

    latency = measure_latency(inference_fn, sample_inputs, n_warmup, n_timed)

    result = {
        "model_variant": model_variant,
        **latency,
    }

    if model_path:
        try:
            result["model_size_mb"] = get_model_size_mb(model_path)
        except FileNotFoundError:
            result["model_size_mb"] = None

    print(f"   Mean latency: {latency['mean_ms']:.2f} ± {latency['std_ms']:.2f} ms")
    print(f"   FPS         : {latency['fps']:.1f}")
    if "model_size_mb" in result and result["model_size_mb"]:
        print(f"   Model size  : {result['model_size_mb']:.1f} MB")

    return result
