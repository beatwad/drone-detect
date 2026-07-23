#!/usr/bin/env python3
"""Collapse Roboflow augmentation copies by keeping the least-rotated original,
detecting rotation from PADDED CORNERS (black OR grey).

Roboflow rotation augmentation fills the exposed corners with a solid achromatic
color -- black (~0) or grey (~40-120). A corner 2x2 patch is "padding" when its
mean is achromatic (max-min of RGB < CHROMA) and dark/grey (brightness < GREY_MAX,
so white product-shot backgrounds are NOT mistaken for padding). Within a base-stem
group every member is the same source image, so the copy with the FEWEST padding
corners is the least-rotated (cleanest) one -- keep it, delete the rest.

Supersedes dedup_roboflow_black.py, which only counted pure-black pixels and so
kept grey-padded rotations. Dry-run unless --apply.
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
CHROMA = 12         # max-min of mean RGB below this => achromatic (grey/black)
GREY_MAX = 150      # corner brightness below this => padding (excludes white bg)


def pad_corners(path: Path) -> int:
    """Number of the 4 corners (2x2) that look like rotation padding."""
    try:
        im = np.asarray(Image.open(path).convert("RGB")).astype(np.int16)
    except Exception:
        return 4                                   # unreadable => never prefer it
    h, w, _ = im.shape
    patches = [im[0:2, 0:2], im[0:2, w-2:w], im[h-2:h, 0:2], im[h-2:h, w-2:w]]
    n = 0
    for c in patches:
        m = c.reshape(-1, 3).mean(0)
        if (m.max() - m.min()) < CHROMA and m.mean() < GREY_MAX:
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data/raw/Drone.v1i.yolov5pytorch"))
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    src = Path(args.src)

    groups = defaultdict(list)
    for f in src.rglob("*"):
        if f.is_file() and f.suffix.lower() in IMG_EXTS and f.parent.name == "images":
            groups[RF_AUG.sub("", f.stem)].append(f)

    to_delete = []
    for base, members in groups.items():
        if len(members) == 1:
            continue
        # keep fewest padding corners; deterministic name tie-break
        ranked = sorted(members, key=lambda m: (pad_corners(m), m.name))
        to_delete += ranked[1:]

    n_img = sum(len(v) for v in groups.values())
    print(f"src: {src}")
    print(f"base stems: {len(groups)}   total images: {n_img}")
    print(f"multi-copy groups: {sum(1 for v in groups.values() if len(v) > 1)}")
    print(f"to delete: {len(to_delete)} images (+ labels)   -> keep {n_img - len(to_delete)}")

    if not args.apply:
        shown = 0
        print("\n(dry-run) sample groups (pad-corner count; * = kept):")
        for base, members in groups.items():
            if len(members) >= 3 and shown < 4:
                ranked = sorted(members, key=lambda m: (pad_corners(m), m.name))
                print(f"  base {base}")
                for i, m in enumerate(ranked):
                    print(f"     {'*' if i == 0 else ' '} pad={pad_corners(m)}  {m.name[:55]}")
                shown += 1
        return

    miss = 0
    for img in to_delete:
        lbl = img.parent.parent / "labels" / (img.stem + ".txt")
        img.unlink()
        if lbl.exists():
            lbl.unlink()
        else:
            miss += 1
    print(f"\ndeleted {len(to_delete)} images ({miss} had no label).")


if __name__ == "__main__":
    main()
