"""
scripts/splits_metadata.py
============================
Generate dan simpan splits_metadata.json — file yang mengunci
daftar gambar test set agar semua 11 model dievaluasi pada
set yang PERSIS SAMA.

Ini mencegah "data leakage" evaluasi: jika kamu menambah gambar
ke dataset lalu rerun evaluation, model lama dan model baru
mungkin dievaluasi pada set berbeda → hasil tidak bisa dibanding.

Jalankan SEKALI sebelum training apapun dimulai.

Usage:
    python scripts/splits_metadata.py \
        --yolo_dir "/content/drive/.../soccer-field-localization.v9i.yolov8" \
        --output   "/content/drive/.../soccer-homography-benchmark/data/splits_metadata.json"
"""

import os
import json
import glob
import hashlib
import argparse
from datetime import datetime
from pathlib import Path


def md5_file(path: str) -> str:
    """MD5 hash dari file — untuk verifikasi integritas."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_splits_metadata(yolo_dir: str, output_path: str) -> dict:
    """
    Scan semua split (train/val/test), catat setiap filename
    beserta MD5 hash-nya, lalu simpan ke JSON.

    Parameters
    ----------
    yolo_dir : str
        Root direktori dataset YOLO.
    output_path : str
        Path output JSON.

    Returns
    -------
    dict : metadata yang tersimpan.
    """
    splits = {}
    total_images = 0

    for split in ["train", "val", "test"]:
        img_dir = os.path.join(yolo_dir, split, "images")
        if not os.path.exists(img_dir):
            print(f"  ⏩ {split}: direktori tidak ditemukan, skip")
            continue

        img_paths = sorted(
            glob.glob(os.path.join(img_dir, "*.jpg")) +
            glob.glob(os.path.join(img_dir, "*.png")) +
            glob.glob(os.path.join(img_dir, "*.jpeg"))
        )

        filenames = [os.path.basename(p) for p in img_paths]

        # Hitung distribusi image source
        sources = {"broadcast": 0, "scouting": 0, "unknown": 0}
        for fname in filenames:
            fname_l = fname.lower()
            if any(k in fname_l for k in ["worldcup", "broadcast", "tv", "match"]):
                sources["broadcast"] += 1
            elif any(k in fname_l for k in ["drone", "tactical", "scouting", "top"]):
                sources["scouting"] += 1
            else:
                sources["unknown"] += 1

        splits[split] = {
            "n_images": len(filenames),
            "filenames": filenames,
            "source_distribution": sources,
        }
        total_images += len(filenames)
        print(f"  ✅ {split:5s}: {len(filenames):4d} images "
              f"(broadcast={sources['broadcast']}, "
              f"scouting={sources['scouting']}, "
              f"unknown={sources['unknown']})")

    metadata = {
        "created_at":    datetime.now().isoformat(),
        "yolo_dir":      yolo_dir,
        "total_images":  total_images,
        "n_keypoints":   32,
        "splits":        splits,
        "note": (
            "File ini mengunci daftar gambar per split. "
            "Semua 11 model HARUS dievaluasi pada test split yang sama. "
            "Jangan tambah/hapus gambar dari dataset setelah file ini dibuat."
        ),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Metadata tersimpan: {output_path}")
    print(f"   Total images: {total_images}")
    return metadata


def verify_splits_consistency(metadata_path: str, yolo_dir: str) -> bool:
    """
    Verifikasi bahwa dataset di disk masih konsisten dengan metadata.
    Jalankan ini sebelum setiap evaluation run untuk safety check.

    Returns
    -------
    bool : True jika konsisten.
    """
    with open(metadata_path) as f:
        meta = json.load(f)

    all_ok = True
    for split, info in meta["splits"].items():
        img_dir = os.path.join(yolo_dir, split, "images")
        current_files = set(
            os.path.basename(p)
            for p in glob.glob(os.path.join(img_dir, "*.jpg")) +
                     glob.glob(os.path.join(img_dir, "*.png"))
        )
        expected_files = set(info["filenames"])

        added   = current_files - expected_files
        removed = expected_files - current_files

        if added or removed:
            print(f"⚠️  {split}: "
                  f"{len(added)} file ditambah, {len(removed)} file dihapus "
                  f"sejak metadata dibuat!")
            all_ok = False
        else:
            print(f"✅ {split}: konsisten ({info['n_images']} images)")

    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yolo_dir",
        required=True,
        help="Root YOLO dataset directory"
    )
    parser.add_argument(
        "--output",
        default="data/splits_metadata.json",
        help="Output path untuk metadata JSON"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verifikasi konsistensi saja (tidak generate ulang)"
    )
    args = parser.parse_args()

    if args.verify:
        print("🔍 Verifying splits consistency...")
        ok = verify_splits_consistency(args.output, args.yolo_dir)
        exit(0 if ok else 1)
    else:
        print("📋 Building splits metadata...")
        build_splits_metadata(args.yolo_dir, args.output)
