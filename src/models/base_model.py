"""
src/models/base_model.py
=========================
Abstract base class untuk semua model wrapper.

Dengan interface yang seragam, evaluation loop di 03_evaluation.ipynb
bisa memanggil model.predict() untuk semua 11 model tanpa if-else.

Setiap wrapper (YOLO11, HRNet, ViTPose, DETR) harus:
1. Inherit dari BasePoseModel
2. Implementasikan method predict() dan load_model()
3. Return format KeypointResult yang identik
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class KeypointResult:
    """
    Standar output dari semua model pose estimation.

    Attributes
    ----------
    keypoints : np.ndarray
        Shape (N_kpt, 2) — koordinat (x, y) dalam pixel image space.
    scores : np.ndarray
        Shape (N_kpt,) — confidence score per keypoint (0..1).
    visibility : np.ndarray
        Shape (N_kpt,) — visibility flag (0=invisible, 1=occluded, 2=visible).
        Untuk model yang tidak output visibility, di-derive dari score threshold.
    bbox : np.ndarray atau None
        Shape (4,) — [x1, y1, x2, y2] bounding box deteksi.
    instance_score : float
        Overall confidence score untuk deteksi instance ini.
    """
    keypoints:      np.ndarray
    scores:         np.ndarray
    visibility:     np.ndarray
    bbox:           Optional[np.ndarray] = None
    instance_score: float = 1.0

    def to_dataframe_rows(self) -> List[Dict]:
        """
        Convert ke list of dicts untuk merge dengan df_dst di homography pipeline.
        Format: [{"kpt_id": 0, "x": 100.2, "y": 200.5, "score": 0.92}, ...]
        """
        rows = []
        for i, (xy, score, vis) in enumerate(
            zip(self.keypoints, self.scores, self.visibility)
        ):
            if vis > 0 or score > 0:
                rows.append({
                    "kpt_id": i,
                    "x": float(xy[0]),
                    "y": float(xy[1]),
                    "score": float(score),
                    "visibility": int(vis),
                })
        return rows


@dataclass
class ModelInfo:
    """Metadata model untuk logging di results_master.csv."""
    family:    str   # "yolo11", "hrnet", "vitpose", "detr"
    variant:   str   # "small", "w18", "base", "r50", etc.
    full_name: str   # "YOLO11-Small", "HRNet-W18", etc.
    weights_path: str = ""
    model_size_mb: float = 0.0
    n_params: int = 0


# ── Abstract base class ───────────────────────────────────────────────────────

class BasePoseModel(ABC):
    """
    Abstract base class untuk semua pose estimation model wrappers.

    Usage pattern:
        model = YOLO11Wrapper("artifacts/weights/yolo11/small/best.pt")
        results = model.predict("path/to/image.jpg")
        # results: List[KeypointResult]
    """

    def __init__(self, weights_path: str, conf_threshold: float = 0.8):
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self._model = None
        self.info: Optional[ModelInfo] = None

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights ke memory. Dipanggil sekali sebelum inference loop."""
        pass

    @abstractmethod
    def predict(self, image_path: str) -> List[KeypointResult]:
        """
        Jalankan inference pada satu gambar.

        Parameters
        ----------
        image_path : str
            Path ke file gambar.

        Returns
        -------
        list[KeypointResult]
            List deteksi. Untuk soccer field biasanya hanya 1 instance.
            List kosong jika tidak ada deteksi yang memenuhi conf_threshold.
        """
        pass

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """Return metadata model."""
        pass

    def unload(self) -> None:
        """
        Hapus model dari memory. Panggil ini sebelum load model berikutnya
        untuk menghindari OOM di Colab.
        """
        self._model = None
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass

    def is_loaded(self) -> bool:
        return self._model is not None

    def predict_top1(self, image_path: str) -> Optional[KeypointResult]:
        """
        Convenience method: return hanya deteksi terbaik (highest instance_score).
        Cocok untuk soccer field yang selalu single-instance.
        """
        results = self.predict(image_path)
        if not results:
            return None
        return max(results, key=lambda r: r.instance_score)
