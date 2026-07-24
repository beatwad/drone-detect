#!/usr/bin/env python3
"""Collapse Roboflow augmentation copies by keeping the alphabetically-first image
per base stem and deleting the rest (image + label).

For datasets whose augmentation is photometric only (exposure/blur, NO rotation),
every copy in a base group is equivalent, so there's nothing to "select" -- keep
the first by name, drop the rest. Operates on one split subdir by default.
Dry-run unless --apply.
"""
import argparse, re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
RF_AUG = re.compile(r"_(?:jpe?g|png|bmp)\.rf\.[0-9a-f]+$", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="images dir to dedup (e.g. .../train/images)")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    src = Path(args.src)

    groups = defaultdict(list)
    for f in src.iterdir():
        if f.is_file() and f.suffix.lower() in IMG_EXTS:
            groups[RF_AUG.sub("", f.stem)].append(f)

    to_delete = []
    for base, members in groups.items():
        keep, *rest = sorted(members, key=lambda p: p.name)   # alphabetically first kept
        to_delete += rest

    n = sum(len(v) for v in groups.values())
    print(f"src: {src}")
    print(f"images: {n}   groups: {len(groups)}   "
          f"{'DELETING' if args.apply else 'would delete'}: {len(to_delete)}   keep: {n - len(to_delete)}")

    if not args.apply:
        print("\n(dry-run) sample group:")
        b, members = max(groups.items(), key=lambda kv: len(kv[1]))
        for i, m in enumerate(sorted(members, key=lambda p: p.name)):
            print(f"   {'KEEP  ' if i == 0 else 'delete'} {m.name[:60]}")
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
