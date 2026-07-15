"""
src/data/yolo_to_coco.py
=========================
Konversi dataset YOLO Pose format (.txt labels + data.yaml)
ke format COCO JSON yang dibutuhkan oleh MMPose (HRNet, ViTPose, DETR).

Usage (dari notebook 01_dataset_conversion.ipynb):
    from src.data.yolo_to_coco import convert_split, verify_coco_json

    convert_split(
        yolo_dir="/content/drive/MyDrive/.../soccer-field-localization.v9i.yolov8",
        output_dir="data/converted/annotations",
        split="train",
        n_keypoints=32,
    )

Perbedaan format YOLO Pose vs COCO JSON:
    YOLO:
        - Koordinat: normalized (0..1)
        - Bbox: cx cy w h (normalized, center-format)
        - Label: flat text file, satu line per instance
        - Keypoints: [x_n, y_n, v, x_n, y_n, v, ...]

    COCO:
        - Koordinat: pixel absolute
        - Bbox: [x_topleft, y_topleft, w, h] (pixel)
        - Label: JSON dengan struktur images + annotations + categories
        - Keypoints: [x1, y1, v1, x2, y2, v2, ...] flat array dalam annotation
"""

from __future__ import annotations
import os
import json
import glob
import yaml
import cv2
import numpy as np
from typing import Optional
from tqdm import tqdm


# ── Main conversion function ──────────────────────────────────────────────────

def convert_split(
    yolo_dir: str,
    output_dir: str,
    split: str = "train",
    n_keypoints: int = 32,
    data_yaml: Optional[str] = None,
    verbose: bool = True,
) -> str:
    """
    Konversi satu split (train/val/test) dari YOLO Pose ke COCO JSON.

    Parameters
    ----------
    yolo_dir : str
        Root direktori dataset YOLO. Harus punya subfolder:
        {split}/images/ dan {split}/labels/
    output_dir : str
        Direktori output untuk file JSON.
    split : str
        "train", "val", atau "test"
    n_keypoints : int
        Jumlah keypoint per instance (32 untuk soccer field).
    data_yaml : str atau None
        Path ke data.yaml. Jika None, cari di yolo_dir/data.yaml.
    verbose : bool
        Print progress.

    Returns
    -------
    str : Path ke output JSON file.

    Raises
    ------
    FileNotFoundError : Jika direktori image/label tidak ditemukan.
    ValueError : Jika jumlah keypoint tidak konsisten.
    """
    images_dir = os.path.join(yolo_dir, split, "images")
    labels_dir = os.path.join(yolo_dir, split, "labels")

    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images dir tidak ditemukan: {images_dir}")
    if not os.path.exists(labels_dir):
        raise FileNotFoundError(f"Labels dir tidak ditemukan: {labels_dir}")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{split}.json")

    # Load data.yaml untuk metadata
    if data_yaml is None:
        data_yaml = os.path.join(yolo_dir, "data.yaml")
    with open(data_yaml) as f:
        yaml_data = yaml.safe_load(f)

    kpt_ids = yaml_data.get("flip_idx", list(range(n_keypoints)))
    if verbose:
        print(f"\n🔄 Converting {split} split...")
        print(f"   Source: {images_dir}")
        print(f"   Output: {output_path}")
        print(f"   Keypoints: {n_keypoints}")

    # Kumpulkan semua image files
    image_files = sorted(
        glob.glob(os.path.join(images_dir, "*.jpg")) +
        glob.glob(os.path.join(images_dir, "*.png")) +
        glob.glob(os.path.join(images_dir, "*.jpeg"))
    )

    if len(image_files) == 0:
        raise FileNotFoundError(f"Tidak ada gambar di: {images_dir}")

    coco_images = []
    coco_annotations = []
    ann_id = 1
    skipped = 0

    for img_id, img_path in enumerate(tqdm(image_files, desc=split, disable=not verbose)):
        img = cv2.imread(img_path)
        if img is None:
            if verbose:
                print(f"  ⚠️  Tidak bisa baca: {img_path}")
            skipped += 1
            continue

        img_h, img_w = img.shape[:2]
        img_filename = os.path.basename(img_path)

        coco_images.append({
            "id": img_id,
            "file_name": img_filename,
            "height": img_h,
            "width": img_w,
        })

        # Cari label file yang sesuai
        label_name = os.path.splitext(img_filename)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_name)

        if not os.path.exists(label_path):
            # Gambar tanpa label — valid (background frame)
            continue

        with open(label_path) as f:
            lines = [l.strip() for l in f if l.strip()]

        for line in lines:
            parts = list(map(float, line.split()))
            if len(parts) < 5 + n_keypoints * 3:
                if verbose:
                    print(f"  ⚠️  Label pendek di {label_name}: {len(parts)} fields")
                continue

            # Parse YOLO format
            cls_id = int(parts[0])
            cx_n, cy_n, bw_n, bh_n = parts[1:5]
            kpts_flat_norm = parts[5: 5 + n_keypoints * 3]

            # Validasi jumlah keypoint
            if len(kpts_flat_norm) != n_keypoints * 3:
                raise ValueError(
                    f"Keypoint count mismatch di {label_name}: "
                    f"expected {n_keypoints*3}, got {len(kpts_flat_norm)}"
                )

            # Denormalisasi bbox: YOLO cx,cy,w,h → COCO x_tl, y_tl, w, h (pixel)
            bw_px = bw_n * img_w
            bh_px = bh_n * img_h
            x_tl = (cx_n - bw_n / 2) * img_w
            y_tl = (cy_n - bh_n / 2) * img_h

            # Denormalisasi keypoints
            kpts_coco = []
            n_visible = 0
            for i in range(n_keypoints):
                x_norm = kpts_flat_norm[i * 3]
                y_norm = kpts_flat_norm[i * 3 + 1]
                v = int(kpts_flat_norm[i * 3 + 2])

                x_px = x_norm * img_w
                y_px = y_norm * img_h

                kpts_coco.extend([x_px, y_px, v])
                if v > 0:
                    n_visible += 1

            coco_annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [float(x_tl), float(y_tl), float(bw_px), float(bh_px)],
                "area": float(bw_px * bh_px),
                "keypoints": kpts_coco,
                "num_keypoints": n_visible,
                "iscrowd": 0,
            })
            ann_id += 1

    # Build category (sesuai COCO format untuk pose estimation)
    category = {
        "id": 1,
        "name": "soccer_field",
        "supercategory": "field",
        "num_keypoints": n_keypoints,
        "keypoints": [str(i) for i in range(n_keypoints)],
        "skeleton": [],  # Opsional: tambahkan koneksi antar keypoint
    }

    coco_output = {
        "info": {
            "description": "Soccer Field Localization — COCO format",
            "version": "1.0",
            "source": "Converted from Roboflow YOLO Pose format",
            "split": split,
        },
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [category],
    }

    with open(output_path, "w") as f:
        json.dump(coco_output, f, indent=2)

    if verbose:
        print(f"\n✅ Selesai!")
        print(f"   Images     : {len(coco_images)}")
        print(f"   Annotations: {len(coco_annotations)}")
        print(f"   Skipped    : {skipped}")
        print(f"   Output     : {output_path}\n")

    return output_path


def convert_all_splits(
    yolo_dir: str,
    output_dir: str,
    splits: Optional[list] = None,
    n_keypoints: int = 32,
) -> dict:
    """
    Konversi semua splits sekaligus.

    Returns
    -------
    dict : {"train": path, "val": path, "test": path}
    """
    if splits is None:
        splits = ["train", "val", "test"]

    results = {}
    for split in splits:
        split_img_dir = os.path.join(yolo_dir, split, "images")
        if not os.path.exists(split_img_dir):
            print(f"⏩ Skip {split} — tidak ada di {split_img_dir}")
            continue
        results[split] = convert_split(yolo_dir, output_dir, split, n_keypoints)

    return results


# ── Verification ──────────────────────────────────────────────────────────────

def verify_coco_json(json_path: str, verbose: bool = True) -> bool:
    """
    Verifikasi COCO JSON dengan pycocotools.
    Jalankan ini WAJIB setelah konversi sebelum training HRNet/ViTPose/DETR.

    Parameters
    ----------
    json_path : str
        Path ke file COCO JSON.

    Returns
    -------
    bool : True jika valid.
    """
    try:
        from pycocotools.coco import COCO
    except ImportError:
        print("⚠️  pycocotools tidak terinstall. Jalankan: pip install pycocotools")
        return False

    try:
        coco = COCO(json_path)
        ann_ids = coco.getAnnIds()
        img_ids = coco.getImgIds()
        cat_ids = coco.getCatIds()

        if verbose:
            print(f"\n✅ COCO JSON valid: {json_path}")
            print(f"   Images      : {len(img_ids)}")
            print(f"   Annotations : {len(ann_ids)}")
            print(f"   Categories  : {len(cat_ids)}")

        # Spot-check: ambil satu annotation, pastikan keypoints ada
        if ann_ids:
            sample = coco.loadAnns(ann_ids[0])[0]
            assert "keypoints" in sample, "Annotation tidak punya field 'keypoints'"
            assert len(sample["keypoints"]) % 3 == 0, "Keypoints tidak kelipatan 3"

        return True

    except Exception as e:
        print(f"❌ Verifikasi gagal: {e}")
        return False


def compare_stats(yolo_dir: str, coco_json_path: str, split: str = "train") -> None:
    """
    Bandingkan statistik keypoint antara YOLO asli dan hasil konversi COCO.
    Gunakan ini untuk memastikan tidak ada data yang hilang atau bergeser.
    """
    # Hitung dari YOLO
    labels_dir = os.path.join(yolo_dir, split, "labels")
    yolo_vis = {0: 0, 1: 0, 2: 0}
    yolo_total_kpts = 0

    for fname in glob.glob(os.path.join(labels_dir, "*.txt")):
        with open(fname) as f:
            for line in f:
                parts = list(map(float, line.split()))
                kpts = parts[5:]
                for i in range(2, len(kpts), 3):
                    v = int(kpts[i])
                    yolo_vis[v] = yolo_vis.get(v, 0) + 1
                    yolo_total_kpts += 1

    # Hitung dari COCO
    with open(coco_json_path) as f:
        coco = json.load(f)

    coco_vis = {0: 0, 1: 0, 2: 0}
    coco_total_kpts = 0

    for ann in coco["annotations"]:
        kpts = ann["keypoints"]
        for i in range(2, len(kpts), 3):
            v = int(kpts[i])
            coco_vis[v] = coco_vis.get(v, 0) + 1
            coco_total_kpts += 1

    print(f"\n📊 Perbandingan statistik {split}:")
    print(f"{'Metric':<30} {'YOLO':>10} {'COCO JSON':>12} {'Match':>8}")
    print("-" * 62)
    print(f"{'Total keypoints':<30} {yolo_total_kpts:>10} {coco_total_kpts:>12} "
          f"{'✅' if yolo_total_kpts == coco_total_kpts else '❌':>8}")
    for v in [0, 1, 2]:
        label = {0: "Invisible (v=0)", 1: "Occluded (v=1)", 2: "Visible (v=2)"}[v]
        match = yolo_vis[v] == coco_vis[v]
        print(f"{label:<30} {yolo_vis[v]:>10} {coco_vis[v]:>12} {'✅' if match else '❌':>8}")
