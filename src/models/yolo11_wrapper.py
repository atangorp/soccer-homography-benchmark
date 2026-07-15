"""
src/models/yolo11_wrapper.py
=============================
Wrapper untuk YOLO11-Pose menggunakan Ultralytics API.
Menyeragamkan output ke format KeypointResult standar.
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional
from .base_model import BasePoseModel, KeypointResult, ModelInfo
import os


class YOLO11Wrapper(BasePoseModel):
    """
    Wrapper untuk YOLO11-Pose (Small, Medium, XLarge).

    Parameters
    ----------
    weights_path : str
        Path ke file .pt hasil training. Contoh:
        "artifacts/weights/yolo11/small/best.pt"
    conf_threshold : float
        Minimum confidence untuk instance detection (default 0.8).
    n_keypoints : int
        Jumlah keypoint per instance (default 32).

    Example
    -------
    >>> model = YOLO11Wrapper("artifacts/weights/yolo11/small/best.pt")
    >>> model.load_model()
    >>> results = model.predict("data/converted/images/test_001.jpg")
    >>> print(results[0].keypoints.shape)  # (32, 2)
    """

    def __init__(
        self,
        weights_path: str,
        conf_threshold: float = 0.8,
        n_keypoints: int = 32,
        variant: str = "unknown",
    ):
        super().__init__(weights_path, conf_threshold)
        self.n_keypoints = n_keypoints
        self.variant = variant

    def load_model(self) -> None:
        from ultralytics import YOLO
        self._model = YOLO(self.weights_path)
        print(f"✅ YOLO11-{self.variant} loaded: {self.weights_path}")

    def predict(self, image_path: str) -> List[KeypointResult]:
        if self._model is None:
            raise RuntimeError("Model belum di-load. Panggil load_model() dulu.")

        results_raw = self._model(
            image_path,
            conf=self.conf_threshold,
            verbose=False,
        )

        output = []
        for r in results_raw:
            if r.keypoints is None or len(r.keypoints) == 0:
                continue

            kpts_data = r.keypoints.data.cpu().numpy()  # (N_inst, N_kpt, 3)
            boxes = r.boxes

            for i, kpt_inst in enumerate(kpts_data):
                # kpt_inst: shape (N_kpt, 3) — [x, y, conf]
                keypoints = kpt_inst[:, :2]          # (N_kpt, 2)
                scores    = kpt_inst[:, 2]            # (N_kpt,)

                # Derive visibility dari confidence score
                visibility = np.zeros(len(scores), dtype=int)
                visibility[scores > self.conf_threshold] = 2
                visibility[(scores > 0) & (scores <= self.conf_threshold)] = 1

                # Bounding box (jika tersedia)
                bbox = None
                if boxes is not None and i < len(boxes):
                    b = boxes[i].xyxy.cpu().numpy().flatten()
                    bbox = b

                instance_score = float(scores.mean())

                output.append(KeypointResult(
                    keypoints=keypoints,
                    scores=scores,
                    visibility=visibility,
                    bbox=bbox,
                    instance_score=instance_score,
                ))

        return output

    def get_model_info(self) -> ModelInfo:
        from src.evaluation.latency import get_model_size_mb
        size = get_model_size_mb(self.weights_path) if os.path.exists(self.weights_path) else 0.0
        return ModelInfo(
            family="yolo11",
            variant=self.variant,
            full_name=f"YOLO11-{self.variant.capitalize()}",
            weights_path=self.weights_path,
            model_size_mb=size,
        )
