"""
setup_colab.py
==============
Jalankan script ini PERTAMA KALI di Google Colab untuk membuat
seluruh folder hierarchy proyek di Google Drive.

Usage:
    !python scripts/setup_colab.py --drive_root "/content/drive/MyDrive/Colab Notebooks"
"""

import os
import argparse
import json
from datetime import datetime


FOLDERS = [
    "configs",
    "data/raw",
    "data/converted/annotations",
    "data/converted/images",
    "notebooks",
    "src/data",
    "src/geometry",
    "src/models",
    "src/evaluation",
    "artifacts/weights/yolo11/small",
    "artifacts/weights/yolo11/medium",
    "artifacts/weights/yolo11/xlarge",
    "artifacts/weights/hrnet/w18",
    "artifacts/weights/hrnet/w32",
    "artifacts/weights/hrnet/w48",
    "artifacts/weights/vitpose/small",
    "artifacts/weights/vitpose/base",
    "artifacts/weights/vitpose/large",
    "artifacts/weights/detr/r50",
    "artifacts/weights/detr/r101",
    "artifacts/logs/training",
    "artifacts/logs/evaluation",
    "artifacts/results/figures",
    "artifacts/results/tables",
    "scripts",
    "paper/figures",
    "paper/tables",
]

GITKEEP_DIRS = [
    "data/raw",
    "data/converted/annotations",
    "artifacts/weights/yolo11/small",
    "artifacts/weights/yolo11/medium",
    "artifacts/weights/yolo11/xlarge",
    "artifacts/weights/hrnet/w18",
    "artifacts/weights/hrnet/w32",
    "artifacts/weights/hrnet/w48",
    "artifacts/weights/vitpose/small",
    "artifacts/weights/vitpose/base",
    "artifacts/weights/vitpose/large",
    "artifacts/weights/detr/r50",
    "artifacts/weights/detr/r101",
    "artifacts/logs/training",
    "artifacts/logs/evaluation",
]


def setup(drive_root: str):
    project_root = os.path.join(drive_root, "soccer-homography-benchmark")
    print(f"\n📁 Creating project at: {project_root}\n")

    created, skipped = 0, 0
    for folder in FOLDERS:
        full_path = os.path.join(project_root, folder)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            print(f"  ✅ Created: {folder}")
            created += 1
        else:
            print(f"  ⏩ Exists:  {folder}")
            skipped += 1

    # .gitkeep so empty folders are tracked
    for folder in GITKEEP_DIRS:
        gk = os.path.join(project_root, folder, ".gitkeep")
        if not os.path.exists(gk):
            open(gk, "w").close()

    # Write project metadata
    meta = {
        "project": "soccer-homography-benchmark",
        "created_at": datetime.now().isoformat(),
        "description": "Benchmarking 11 pose estimation architectures for soccer field homography prediction",
        "models": {
            "yolo11": ["small", "medium", "xlarge"],
            "hrnet": ["w18", "w32", "w48"],
            "vitpose": ["small", "base", "large"],
            "detr": ["r50", "r101"],
        },
        "dataset": "soccer-field-localization (Roboflow)",
        "n_keypoints": 32,
        "pitch_size": {"width": 120, "height": 80},
    }
    with open(os.path.join(project_root, "project_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*50}")
    print(f"✨ Setup complete!")
    print(f"   Folders created : {created}")
    print(f"   Already existed : {skipped}")
    print(f"   Project root    : {project_root}")
    print(f"{'='*50}\n")
    print("📌 Next steps:")
    print("   1. Copy src/ dan configs/ ke project_root")
    print("   2. Upload dataset Roboflow ke data/raw/")
    print("   3. Jalankan notebooks/01_dataset_conversion.ipynb")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--drive_root",
        default="/content/drive/MyDrive/Colab Notebooks",
        help="Root Google Drive path"
    )
    args = parser.parse_args()
    setup(args.drive_root)
