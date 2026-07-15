# src/models/__init__.py
from .base_model      import BasePoseModel, KeypointResult, ModelInfo
from .yolo11_wrapper  import YOLO11Wrapper
from .hrnet_wrapper   import HRNetWrapper
from .vitpose_wrapper import ViTPoseWrapper
from .detr_wrapper    import DeformableDETRWrapper

__all__ = [
    "BasePoseModel", "KeypointResult", "ModelInfo",
    "YOLO11Wrapper", "HRNetWrapper", "ViTPoseWrapper", "DeformableDETRWrapper",
]
