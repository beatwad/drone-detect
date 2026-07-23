#!/usr/bin/env python3
"""Collapse Roboflow augmentation copies in a dataset by keeping the least-rotated
original and deleting the rest.

Roboflow exports each source image up to 6x as `<stem>_jpg.rf.<hash>.jpg`, several
of which are ROTATED with black-padded corners. Within a base-stem group every
member is the same underlying image, so natural darkness cancels and the member
with the smallest near-black pixel ratio is the least-rotated (cleanest) copy.
Keep that one; delete the other copies and their label .txt.

Dry-run by default; pass --apply to delete.
"""
import argparse, re
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
RF_AUG = re.compile(r"_(?:jpe?g|png|bmp)\.rf\.[0-9a-f]+$", re.I)
BLACK = 8          # pixel <= this (0-255 grayscale) counts as black padding


def black_ratio(path: Path) -> float:
    try:
        im = np.asarray(Image.open(path).convert("L").resize((80, 80)))
    except Exception:
        return 1.0                      # unreadable => never prefer it
    return float((im <= BLACK).mean())


def label_for(img: Path) -> Path:
    return img.parent.parent / "labels" / (img.stem + ".txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data/raw/Drone.v1i.yolov5pytorch"))
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    src = Path(args.src)

    # group all image copies by base stem, across train/valid/test
    groups = defaultdict(list)
    for f in src.rglob("*"):
        if f.is_file() and f.suffix.lower() in IMG_EXTS and f.parent.name == "images":
            groups[RF_AUG.sub("", f.stem)].append(f)

    cross = {b: {f.parent.parent.name for f in v} for b, v in groups.items()
             if len({f.parent.parent.name for f in v}) > 1}

    to_delete = []
    for base, members in groups.items():
        if len(members) == 1:
            continue
        keep = min(sorted(members), key=black_ratio)     # sorted => deterministic ties
        to_delete += [m for m in members if m != keep]

    n_img = sum(len(v) for v in groups.values())
    print(f"src: {src}")
    print(f"base stems: {len(groups)}   total images: {n_img}")
    print(f"multi-copy groups: {sum(1 for v in groups.values() if len(v) > 1)}")
    print(f"to delete: {len(to_delete)} images (+ their labels)   -> keep {n_img - len(to_delete)}")
    if cross:
        print(f"WARNING: {len(cross)} base stems span multiple splits (grouped across): "
              f"{dict(list(cross.items())[:5])}")

    if not args.apply:
        print("\n(dry-run) sample deletions:")
        for m in to_delete[:8]:
            print("   ", m.relative_to(src), " black=%.3f" % black_ratio(m))
        return

    miss_lbl = 0
    for m in to_delete:
        lbl = label_for(m)
        m.unlink()
        if lbl.exists():
            lbl.unlink()
        else:
            miss_lbl += 1
    print(f"\ndeleted {len(to_delete)} images ({miss_lbl} had no label file).")


if __name__ == "__main__":
    main()
