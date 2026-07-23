#!/usr/bin/env python3
"""Delete false-empty frames in Database1 range 1..350.

For each numbered frame N in [1, 350], if its YOLO label has no box (no line
with >=5 fields -- i.e. empty or whitespace-only), delete BOTH the image and
the label. These are frames where a drone is present but was never annotated.

Dry-run by default; pass --apply to actually delete.
"""
import argparse
from pathlib import Path

DIR = Path(__file__).resolve().parents[1] / "data/raw/Database1/Database1"
IMG_EXTS = (".JPEG", ".jpeg", ".JPG", ".jpg", ".png")
LO, HI = 1, 350


def has_box(txt: Path) -> bool:
    for line in txt.read_text().splitlines():
        if len(line.split()) >= 5:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry-run)")
    args = ap.parse_args()

    to_del = []          # (img, txt)
    missing_txt = []
    for n in range(LO, HI + 1):
        img = next((DIR / f"{n}{e}" for e in IMG_EXTS if (DIR / f"{n}{e}").exists()), None)
        if img is None:
            continue
        txt = DIR / f"{n}.txt"
        if not txt.exists():
            missing_txt.append(img.name)
            continue
        if not has_box(txt):
            to_del.append((img, txt))

    print(f"range {LO}..{HI}: {len(to_del)} empty-label frames "
          f"({'DELETING' if args.apply else 'dry-run, would delete'})")
    for img, txt in to_del:
        print(f"  {img.name}  +  {txt.name}")
        if args.apply:
            img.unlink()
            txt.unlink()
    if missing_txt:
        print(f"\n{len(missing_txt)} image(s) had NO .txt at all (left untouched): "
              f"{', '.join(missing_txt[:10])}{' ...' if len(missing_txt) > 10 else ''}")
    if args.apply:
        print(f"\ndeleted {len(to_del)} image+label pairs.")


if __name__ == "__main__":
    main()
