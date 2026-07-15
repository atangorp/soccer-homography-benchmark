"""
src/models/vitpose_wrapper.py
==============================
Wrapper untuk ViTPose-Small/Base/Large menggunakan MMPose inference API.
Interface identik dengan HRNetWrapper — hanya berbeda di config dan checkpoint.
"""

from __future__ import annotations
import numpy as np
import os
from typing import List
from .base_model import BasePoseModel, KeypointResult, ModelInfo


class ViTPoseWrapper(BasePoseModel):
    """
    Wrapper untuk ViTPose (Small, Base, Large) via MMPose.

    ViTPose menggunakan Vision Transformer sebagai backbone.
    Pipeline-nya identik dengan HRNet (top-down, heatmap-based),
    sehingga wrapper-nya hampir sama persis.

    Parameters
    ----------
    weights_path : str
        Path ke .pth checkpoint ViTPose.
    config_path : str
        Path ke MMPose config .py file.
    variant : str
        "small", "base", atau "large".

    Notes
    -----
    ViTPose-Large membutuhkan minimal V100 16GB untuk inference yang stabil.
    Di T4 (16GB), ViTPose-Large bisa OOM jika batch > 1.
    Selalu gunakan batch=1 untuk inference di wrapper ini.
    """

    def __init__(
        self,
        weights_path: str,
        config_path: str,
        conf_threshold: float = 0.8,
        n_keypoints: int = 32,
        variant: str = "base",
        device: str = "cuda:0",
    ):
        super().__init__(weights_path, conf_threshold)
        self.config_path = config_path
        self.n_keypoints = n_keypoints
        self.variant = variant
        self.device = device

    def load_model(self) -> None:
        from mmpose.apis import init_model
        self._model = init_model(
            self.config_path,
            self.weights_path,
            device=self.device,
        )
        print(f"✅ ViTPose-{self.variant.capitalize()} loaded: {self.weights_path}")

    def predict(self, image_path: str) -> List[KeypointResult]:
        if self._model is None:
            raise RuntimeError("Model belum di-load. Panggil load_model() dulu.")

        import cv2
        from mmpose.apis import inference_topdown

        img = cv2.imread(image_path)
        if img is None:
            return []

        h, w = img.shape[:2]
        bboxes = np.array([[0, 0, w, h, 1.0]])
        results = inference_topdown(self._model, image_path, bboxes)

        output = []
        for res in results:
            if not hasattr(res, 'pred_instances'):
                continue
            pred = res.pred_instances
            kpts_all   = pred.keypoints.cpu().numpy()
            scores_all = pred.keypoint_scores.cpu().numpy()

            for kpts, scores in zip(kpts_all, scores_all):
                visibility = np.zeros(len(scores), dtype=int)
                visibility[scores > self.conf_threshold] = 2
                visibility[(scores > 0) & (scores <= self.conf_threshold)] = 1

                output.append(KeypointResult(
                    keypoints=kpts,
                    scores=scores,
                    visibility=visibility,
                    bbox=np.array([0, 0, w, h]),
                    instance_score=float(scores.mean()),
                ))
        return output

    def get_model_info(self) -> ModelInfo:
        from src.evaluation.latency import get_model_size_mb
        size = get_model_size_mb(self.weights_path) if os.path.exists(self.weights_path) else 0.0
        return ModelInfo(
            family="vitpose",
            variant=self.variant,
            full_name=f"ViTPose-{self.variant.capitalize()}",
            weights_path=self.weights_path,
            model_size_mb=size,
        )
