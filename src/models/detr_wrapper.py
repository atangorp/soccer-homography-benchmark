"""
src/models/detr_wrapper.py
===========================
Wrapper untuk Deformable DETR-R50/R101.

Berbeda dari YOLO/HRNet/ViTPose: DETR menggunakan set-prediction paradigm.
Tidak ada NMS post-processing — model langsung output N query predictions,
lalu kita ambil prediction dengan confidence tertinggi.
"""

from __future__ import annotations
import numpy as np
import torch
import os
from typing import List, Optional
from .base_model import BasePoseModel, KeypointResult, ModelInfo


class DeformableDETRWrapper(BasePoseModel):
    """
    Wrapper untuk Deformable DETR Pose (R50, R101).

    Parameters
    ----------
    weights_path : str
        Path ke checkpoint .pth.
    model_config : dict
        Config dict: num_queries, n_keypoints, num_feature_levels, dll.
    variant : str
        "r50" atau "r101".

    Notes
    -----
    Set-prediction: DETR output langsung N set keypoint predictions.
    Untuk soccer field (single instance), kita ambil prediction
    dengan logit tertinggi sebagai "top-1 detection".

    Hungarian matching loss saat training memastikan setiap query
    bertanggung jawab pada satu instance — tidak ada duplikasi.

    DETR butuh CUDA ops untuk deformable attention dikompilasi dulu:
        cd third_party/deformable_detr
        cd models/ops && python setup.py build install
    """

    def __init__(
        self,
        weights_path: str,
        model_config: Optional[dict] = None,
        conf_threshold: float = 0.8,
        n_keypoints: int = 32,
        variant: str = "r50",
        device: str = "cuda",
    ):
        super().__init__(weights_path, conf_threshold)
        self.model_config = model_config or {
            "num_queries": 1,
            "num_feature_levels": 4,
            "n_keypoints": n_keypoints,
        }
        self.n_keypoints = n_keypoints
        self.variant = variant
        self.device = device
        self._transform = None

    def load_model(self) -> None:
        """
        Load Deformable DETR model dari checkpoint.

        Asumsi: model didefinisikan di third_party/deformable_detr/
        dan sudah di-build CUDA ops-nya.
        """
        import sys
        # Tambahkan path ke Deformable DETR jika belum ada
        detr_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "third_party", "deformable_detr"
        )
        if detr_path not in sys.path and os.path.exists(detr_path):
            sys.path.insert(0, detr_path)

        checkpoint = torch.load(self.weights_path, map_location=self.device)

        # Jika checkpoint menyimpan config
        if "args" in checkpoint:
            saved_args = checkpoint["args"]

        # Load model dari state dict
        # (implementasi spesifik tergantung training script yang dipakai)
        # Pattern umum Deformable DETR:
        try:
            from main import build_model_main
            self._model, _, _ = build_model_main(saved_args)
            self._model.load_state_dict(checkpoint["model"])
            self._model.to(self.device)
            self._model.eval()
        except Exception as e:
            # Fallback: load state dict langsung
            print(f"⚠️  Standard load gagal ({e}), mencoba fallback...")
            self._model = checkpoint.get("model_state", checkpoint)

        # Transform untuk preprocessing input
        from torchvision import transforms
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        print(f"✅ DETR-{self.variant.upper()} loaded: {self.weights_path}")

    def predict(self, image_path: str) -> List[KeypointResult]:
        if self._model is None:
            raise RuntimeError("Model belum di-load. Panggil load_model() dulu.")

        import cv2
        from PIL import Image

        img_pil = Image.open(image_path).convert("RGB")
        img_cv  = cv2.imread(image_path)
        h, w    = img_cv.shape[:2]

        # Preprocess
        img_tensor = self._transform(img_pil).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self._model(img_tensor)

        # DETR output: pred_logits (B, N_query, 2), pred_keypoints (B, N_query, K*2)
        pred_logits  = outputs["pred_logits"][0]  # (N_query, 2)
        pred_kpts    = outputs["pred_keypoints"][0]  # (N_query, K*2) normalized

        # Ambil top-1 query (softmax logit tertinggi untuk class "field")
        probs = pred_logits.softmax(-1)[:, 0]  # class 0 = soccer field
        best_idx = probs.argmax().item()

        best_kpts_norm = pred_kpts[best_idx].cpu().numpy()  # (K*2,) normalized
        best_conf      = float(probs[best_idx].cpu())

        if best_conf < self.conf_threshold:
            return []

        # Denormalize ke pixel coords
        kpts = best_kpts_norm.reshape(self.n_keypoints, 2)
        kpts[:, 0] *= w
        kpts[:, 1] *= h

        # DETR tidak output per-keypoint score — pakai instance score sebagai proxy
        scores = np.full(self.n_keypoints, best_conf, dtype=np.float32)
        visibility = np.full(self.n_keypoints, 2, dtype=int)  # Assume visible

        return [KeypointResult(
            keypoints=kpts,
            scores=scores,
            visibility=visibility,
            bbox=np.array([0, 0, w, h]),
            instance_score=best_conf,
        )]

    def get_model_info(self) -> ModelInfo:
        from src.evaluation.latency import get_model_size_mb
        size = get_model_size_mb(self.weights_path) if os.path.exists(self.weights_path) else 0.0
        return ModelInfo(
            family="detr",
            variant=self.variant,
            full_name=f"Deformable-DETR-{self.variant.upper()}",
            weights_path=self.weights_path,
            model_size_mb=size,
        )
