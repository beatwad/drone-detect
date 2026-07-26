#!/usr/bin/env python3
"""Delete frames whose every box is below the model's resolvable size.

YOLOv5's finest detection head is P3 at stride 8, so an object under ~12px at the
640 input has about one cell of spatial support and is effectively undetectable.
Measured on val: miss rate 33% at 8-12px and 37% at 4-8px, against 2.5% above
64px. Those frames cost training capacity and drag the headline metric without
being reachable, and they are long-range anyway -- not this project's <=10m regime.

Deletes an image+label pair only when EVERY box in it is below MIN_PX. Frames that
mix a resolvable box with a tiny one are left ALONE rather than having the tiny box
stripped: removing a box while its drone stays in frame teaches the model that a
drone is background, which is the mislabelling that poisoned Database1 (see the
`empty-label-contamination` note). Empty frames are kept -- they are negatives, not
long-range targets.

Sizes are read as sqrt(w*h) scaled to REF_PX, matching how training letterboxes to
640. Dry-run unless --apply.
"""
import argparse
from pathlib import Path
from collections import Counter

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
MIN_PX = 12.0              # below this, P3/stride-8 has ~1 cell of support
REF_PX = 640               # training imgsz


def box_sides(lbl: Path, w: int, h: int):
    """sqrt(area) of each box, in pixels at the REF_PX letterboxed input."""
    scale = REF_PX / max(w, h)
    out = []
    for row in lbl.read_text().split("\n"):
        p = row.split()
        if len(p) == 5:
            out.append(((float(p[3]) * w) * (float(p[4]) * h)) ** 0.5 * scale)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data/raw/Drone Detection.v5i.yolov5pytorch"))
    ap.add_argument("--min-px", type=float, default=MIN_PX)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    src = Path(args.src)

    to_delete, tally = [], Counter()
    for img in sorted(f for f in src.rglob("*")
                      if f.is_file() and f.suffix.lower() in IMG_EXTS):
        lbl = img.parent.parent / "labels" / f"{img.stem}.txt"
        if not lbl.exists():
            tally["no label (skipped)"] += 1
            continue
        w, h = Image.open(img).size
        sides = box_sides(lbl, w, h)
        if not sides:
            tally["empty (kept as negative)"] += 1
        elif all(s < args.min_px for s in sides):
            tally["all boxes tiny -> DELETE"] += 1
            to_delete.append((img, lbl))
        elif any(s < args.min_px for s in sides):
            tally["mixed (kept intact)"] += 1
        else:
            tally["all boxes resolvable (kept)"] += 1

    print(f"src: {src}   threshold: {args.min_px}px at {REF_PX}")
    for k, v in tally.most_common():
        print(f"  {k:32} {v:6d}")
    per_split = Counter(img.relative_to(src).parts[0] for img, _ in to_delete)
    print(f"\n{'DELETING' if args.apply else 'would delete'} {len(to_delete)} image+label pairs"
          f"   ({', '.join(f'{k}={v}' for k, v in sorted(per_split.items()))})")

    if not args.apply:
        print("\ndry run -- pass --apply to delete")
        return
    for img, lbl in to_delete:
        img.unlink()
        lbl.unlink()
    print(f"deleted {len(to_delete)} image+label pairs")


if __name__ == "__main__":
    main()
