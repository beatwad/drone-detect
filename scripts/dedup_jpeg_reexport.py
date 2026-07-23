#!/usr/bin/env python3
"""In UAVs.v2i (D), Roboflow re-exported some `.jpeg` originals twice:
    single:  <base>_jpeg.rf.<hashA>.jpg
    double:  <base>_jpeg_jpg.rf.<hashB>.jpg   (a re-export of the single)
For any <base> that has BOTH forms, delete the single(s) and keep the double(s).
Deletes the image and its paired label .txt. Dry-run unless --apply.
"""
import argparse, re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
DOUBLE = re.compile(r"_jpeg_jpg\.rf\.[0-9a-f]+$", re.I)   # <base>_jpeg_jpg.rf.<hash>
SINGLE = re.compile(r"_jpeg\.rf\.[0-9a-f]+$", re.I)       # <base>_jpeg.rf.<hash>


def classify(stem: str):
    """Return ('double'|'single', base) or None. Order matters: test DOUBLE first
    (it also contains a '_jpeg' but not '_jpeg.rf.')."""
    if DOUBLE.search(stem):
        return "double", DOUBLE.sub("", stem)
    if SINGLE.search(stem):
        return "single", SINGLE.sub("", stem)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data/raw/UAVs.v2i.yolov5pytorch"))
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    src = Path(args.src)

    groups = defaultdict(lambda: {"single": [], "double": []})
    for f in src.rglob("*"):
        if f.is_file() and f.suffix.lower() in IMG_EXTS and f.parent.name == "images":
            c = classify(f.stem)
            if c:
                kind, base = c
                groups[base][kind].append(f)

    paired_bases = {b: g for b, g in groups.items() if g["single"] and g["double"]}
    to_delete = [img for g in paired_bases.values() for img in g["single"]]

    print(f"src: {src}")
    print(f"bases with a _jpeg single: {sum(1 for g in groups.values() if g['single'])}")
    print(f"bases with a _jpeg_jpg double: {sum(1 for g in groups.values() if g['double'])}")
    print(f"bases having BOTH (pairs): {len(paired_bases)}")
    print(f"single images to delete: {len(to_delete)} (+ their labels)")

    if not args.apply:
        print("\n(dry-run) sample pairs:")
        for b, g in list(paired_bases.items())[:5]:
            print(f"  base={b}")
            for s in g["single"]:
                print(f"     DELETE {s.name}")
            for d in g["double"]:
                print(f"     keep   {d.name}")
        return

    miss = 0
    for img in to_delete:
        lbl = img.parent.parent / "labels" / (img.stem + ".txt")
        img.unlink()
        if lbl.exists():
            lbl.unlink()
        else:
            miss += 1
    print(f"\ndeleted {len(to_delete)} single images ({miss} had no label).")


if __name__ == "__main__":
    main()
