"""
src/models/hrnet_wrapper.py
============================
Wrapper untuk HRNet-W18/W32/W48 menggunakan MMPose inference API.
Menyeragamkan output ke format KeypointResult standar.
"""

from __future__ import annotations
import numpy as np
import os
from typing import List, Optional
from .base_model import BasePoseModel, KeypointResult, ModelInfo


class HRNetWrapper(BasePoseModel):
    """
    Wrapper untuk HRNet via MMPose.

    Parameters
    ----------
    weights_path : str
        Path ke .pth checkpoint.
        Contoh: "artifacts/weights/hrnet/w48/best_coco_AP.pth"
    config_path : str
        Path ke MMPose config .py file.
        Contoh: "artifacts/weights/hrnet/w48/hrnet_w48_config.py"
    conf_threshold : float
        Keypoint visibility threshold.
    variant : str
        "w18", "w32", atau "w48".

    Notes
    -----
    MMPose menggunakan top-down pipeline:
        1. Deteksi bounding box seluruh lapangan (atau gunakan full-image bbox)
        2. Crop + resize ke imgsz (256x192 atau 256x256)
        3. Prediksi heatmap per keypoint
        4. Argmax dari heatmap → koordinat keypoint

    Karena soccer field adalah whole-image object, kita pakai full-image
    bounding box sebagai input ke MMPose top-down pipeline.
    """

    def __init__(
        self,
        weights_path: str,
        config_path: str,
        conf_threshold: float = 0.8,
        n_keypoints: int = 32,
        variant: str = "w48",
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
        print(f"✅ HRNet-{self.variant.upper()} loaded: {self.weights_path}")

    def predict(self, image_path: str) -> List[KeypointResult]:
        if self._model is None:
            raise RuntimeError("Model belum di-load. Panggil load_model() dulu.")

        import cv2
        from mmpose.apis import inference_topdown

        img = cv2.imread(image_path)
        if img is None:
            return []

        h, w = img.shape[:2]

        # Full-image bounding box sebagai "person" detection
        # Format MMPose: [x1, y1, x2, y2, score]
        bboxes = np.array([[0, 0, w, h, 1.0]])

        # Run inference
        results = inference_topdown(self._model, image_path, bboxes)

        output = []
        for res in results:
            # MMPose output: pred_instances.keypoints (N, K, 2)
            #                pred_instances.keypoint_scores (N, K)
            if not hasattr(res, 'pred_instances'):
                continue

            pred = res.pred_instances
            kpts_all  = pred.keypoints.cpu().numpy()      # (N_inst, K, 2)
            scores_all = pred.keypoint_scores.cpu().numpy() # (N_inst, K)

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
            family="hrnet",
            variant=self.variant,
            full_name=f"HRNet-{self.variant.upper()}",
            weights_path=self.weights_path,
            model_size_mb=size,
        )
