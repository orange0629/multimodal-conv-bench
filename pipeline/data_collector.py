#!/usr/bin/env python3
"""
Seed Image Data Collector
==========================
Downloads and indexes image datasets to use as seed images for the synthesis pipeline.

Supported datasets
------------------
  coco       MS-COCO 2017 validation split (5k images, rich captions)
  vg         Visual Genome (108k images, dense scene graph annotations)

Each dataset is saved to --data-dir as:
  <dataset>/images/          raw image files
  <dataset>/metadata.jsonl   one record per image: {id, path, captions/objects, split}

The metadata.jsonl can then be passed to synthesize.py via --seed-image-dir.

Usage examples
--------------
  # Download COCO val split
  python pipeline/data_collector.py --dataset coco --data-dir data/

  # Download a 1000-image subset of Visual Genome
  python pipeline/data_collector.py --dataset vg --data-dir data/ --max-images 1000

  # List what's already downloaded
  python pipeline/data_collector.py --list --data-dir data/
"""

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# ── Download helpers ──────────────────────────────────────────────────────────

def download_file(url: str, dest: Path, desc: str = "") -> Path:
    """Stream-download url → dest, with a tqdm progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] already exists: {dest}", file=sys.stderr)
        return dest

    print(f"  [download] {desc or url} → {dest}", file=sys.stderr)
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                      desc=dest.name, file=sys.stderr) as bar:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            bar.update(len(chunk))
    return dest


def extract_zip(zip_path: Path, dest_dir: Path):
    print(f"  [extract] {zip_path.name} → {dest_dir}", file=sys.stderr)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)

# ── COCO ──────────────────────────────────────────────────────────────────────

COCO_URLS = {
    "val_images": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


def collect_coco(data_dir: Path, max_images: int | None = None):
    """Download COCO 2017 val split and write metadata.jsonl."""
    coco_dir  = data_dir / "coco"
    img_dir   = coco_dir / "images"
    ann_dir   = coco_dir / "annotations"
    meta_file = coco_dir / "metadata.jsonl"

    coco_dir.mkdir(parents=True, exist_ok=True)

    # Download images
    zip_path = coco_dir / "val2017.zip"
    download_file(COCO_URLS["val_images"], zip_path, desc="COCO val2017 images")
    if not img_dir.exists():
        extract_zip(zip_path, coco_dir)          # extracts to coco_dir/val2017/
        (coco_dir / "val2017").rename(img_dir)

    # Download annotations
    ann_zip = coco_dir / "annotations.zip"
    download_file(COCO_URLS["annotations"], ann_zip, desc="COCO annotations")
    if not ann_dir.exists():
        extract_zip(ann_zip, coco_dir)

    # Parse captions
    cap_file = ann_dir / "captions_val2017.json"
    with open(cap_file) as f:
        raw = json.load(f)

    # image_id → list of caption strings
    caps: dict[int, list[str]] = {}
    for ann in raw["annotations"]:
        caps.setdefault(ann["image_id"], []).append(ann["caption"])

    images = raw["images"]
    if max_images:
        images = images[:max_images]

    print(f"  [index] Writing {len(images)} records → {meta_file}", file=sys.stderr)
    with open(meta_file, "w") as f:
        for img in tqdm(images, desc="indexing", file=sys.stderr):
            record = {
                "id":       img["id"],
                "dataset":  "coco",
                "split":    "val2017",
                "path":     str(img_dir / img["file_name"]),
                "filename": img["file_name"],
                "captions": caps.get(img["id"], []),
                "width":    img["width"],
                "height":   img["height"],
            }
            f.write(json.dumps(record) + "\n")

    print(f"[coco] Done. {len(images)} images at {coco_dir}", file=sys.stderr)

# ── Visual Genome ─────────────────────────────────────────────────────────────

VG_URLS = {
    "images_part1": "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip",
    "images_part2": "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip",
    "region_desc":  "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/region_descriptions.json.zip",
    "objects":      "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects.json.zip",
}


def collect_vg(data_dir: Path, max_images: int | None = None):
    """Download Visual Genome images + region descriptions and write metadata.jsonl."""
    vg_dir    = data_dir / "vg"
    img_dir   = vg_dir / "images"
    meta_file = vg_dir / "metadata.jsonl"

    vg_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    # Download image zips (two parts)
    for key in ("images_part1", "images_part2"):
        zip_path = vg_dir / f"{key}.zip"
        download_file(VG_URLS[key], zip_path, desc=f"VG {key}")
        if not any(img_dir.glob("*.jpg")):
            extract_zip(zip_path, img_dir)

    # Download region descriptions for captions
    rd_zip = vg_dir / "region_descriptions.zip"
    rd_json = vg_dir / "region_descriptions.json"
    download_file(VG_URLS["region_desc"], rd_zip, desc="VG region descriptions")
    if not rd_json.exists():
        extract_zip(rd_zip, vg_dir)

    # Parse: image_id → top region descriptions (use as captions)
    print("  [parse] Loading region descriptions (this may take a moment)...", file=sys.stderr)
    with open(rd_json) as f:
        rd_data = json.load(f)

    if max_images:
        rd_data = rd_data[:max_images]

    print(f"  [index] Writing {len(rd_data)} records → {meta_file}", file=sys.stderr)
    with open(meta_file, "w") as f:
        for entry in tqdm(rd_data, desc="indexing", file=sys.stderr):
            image_id = entry["id"]
            # Each region has a phrase; take up to 5 as description snippets
            phrases = [r["phrase"] for r in entry.get("regions", [])[:5]]

            # VG images can be in either VG_100K or VG_100K_2 subfolder
            img_path = img_dir / f"{image_id}.jpg"

            record = {
                "id":          image_id,
                "dataset":     "vg",
                "path":        str(img_path),
                "filename":    f"{image_id}.jpg",
                "captions":    phrases,   # region descriptions as loose captions
            }
            f.write(json.dumps(record) + "\n")

    print(f"[vg] Done. {len(rd_data)} images at {vg_dir}", file=sys.stderr)

# ── List downloaded datasets ──────────────────────────────────────────────────

def list_datasets(data_dir: Path):
    print(f"Datasets in {data_dir}:")
    for meta in sorted(data_dir.rglob("metadata.jsonl")):
        count = sum(1 for _ in open(meta))
        print(f"  {meta.parent.name:10s}  {count:6d} images  ({meta})")

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Download and index seed image datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset", choices=["coco", "vg"], default=None,
                   help="Dataset to download.")
    p.add_argument("--data-dir", default="data",
                   help="Root directory to save datasets. Default: data/")
    p.add_argument("--max-images", type=int, default=None,
                   help="Limit number of images (useful for quick tests).")
    p.add_argument("--list", action="store_true",
                   help="List already-downloaded datasets and exit.")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    if args.list:
        list_datasets(data_dir)
        return

    if not args.dataset:
        print("Error: specify --dataset coco or --dataset vg (or --list)", file=sys.stderr)
        sys.exit(1)

    if args.dataset == "coco":
        collect_coco(data_dir, args.max_images)
    elif args.dataset == "vg":
        collect_vg(data_dir, args.max_images)


if __name__ == "__main__":
    main()
