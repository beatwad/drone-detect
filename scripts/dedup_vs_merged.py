#!/usr/bin/env python3
"""Remove images from a NEW raw dataset that already exist in the merged set.

Two stages, against an index of every image in data/drone/{train,val}/images:
  1. candidates -- 64-bit perceptual hash within PHASH_DIST bits, found by
     4x16-bit LSH banding.
  2. verification -- two pixel correlations on 64x64/96x96 grey thumbnails:
       ncc  = normalized cross-correlation of intensity
       gncc = NCC of the gradient magnitude (suppresses flat regions)
     accepted as a duplicate when  ncc >= NCC_STRONG  or  (ncc >= NCC_MIN and
     gncc >= GNCC_MIN).

Stage 2 is not optional, and it needs both metrics, because most of this data is a
small object on flat sky:
  - the pHash is near-degenerate -- unrelated sky images collide at Hamming 0-6
    (measured: at distance 6, ~90% of pHash matches are different images), so the
    hash can only supply recall.
  - ncc alone is inflated by the matching sky gradient: unrelated pairs were
    confirmed by eye up to ncc 0.93.
  - gncc alone rejects real duplicates whose only high-frequency content is the
    small object itself, e.g. two frames of one static scene where a bird moved
    (ncc 0.998, gncc 0.03). Hence the NCC_STRONG escape hatch.
Thresholds were placed against ~60 pairs checked by eye; the confirmed boundary
for gncc is between 0.33 (different) and 0.41 (duplicate).

Known limit: a duplicate that was re-CROPPED lands around ncc 0.80-0.87 and
survives. Raising NCC_MIN to catch those admits more sky false positives than the
handful of crops it recovers, so they are left in.

Deletes the matched image and its paired label .txt, and writes a CSV of every
candidate (accepted or not) for review. Dry-run unless --apply.

NOTE this matches per image, NOT per Roboflow base stem. It is only correct for a
source with no rotation/flip augmentation copies (check its README.roboflow.txt):
augmentation moves the pHash past any sane threshold, so a rotated copy of a
duplicate would survive. Conversely, base stems can collide between the
sub-datasets a Roboflow project was assembled from (`00004_jpg.rf.*` appearing in
train, valid and test as three unrelated images), so grouping by base stem would
delete unrelated images.
"""
import argparse, csv
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool

import imagehash
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
PHASH_DIST = 12            # candidate net; wider than merge_dataset.py's 6 since pixels decide
NCC_MIN = 0.90             # with GNCC_MIN
NCC_STRONG = 0.98          # on its own
GNCC_MIN = 0.40
NCC_SIZE, GNCC_SIZE = 64, 96
BANDS, BAND_BITS = 4, 16


def _norm(a):
    """Zero-mean, unit-variance, so a correlation of two of these is invariant to
    brightness/contrast (JPEG re-encode, exposure shifts)."""
    a = a - a.mean()
    s = a.std()
    return a / s if s > 1e-6 else a


def grey(path: Path, size: int):
    return np.asarray(Image.open(path).convert("L").resize((size, size), Image.BILINEAR),
                      dtype=np.float32)


def thumb(path: Path):
    """(intensity, gradient-magnitude) thumbnails, both normalized."""
    a = grey(path, GNCC_SIZE)
    g = np.hypot(np.diff(a, axis=1)[:-1, :], np.diff(a, axis=0)[:, :-1])
    return _norm(grey(path, NCC_SIZE)), _norm(g)


def phash64(path: Path):
    """64-bit perceptual hash, or None if the image can't be read."""
    try:
        h = imagehash.phash(Image.open(path).convert("RGB"))
    except Exception:
        return None
    v = 0
    for bit in h.hash.flatten():
        v = (v << 1) | int(bit)
    return v


def images_under(d: Path):
    return sorted(f for f in d.rglob("*")
                  if f.is_file() and f.suffix.lower() in IMG_EXTS)


def hash_all(paths, workers, tag):
    print(f"hashing {len(paths)} {tag} images ...", flush=True)
    with Pool(workers) as pool:
        hashes = pool.map(phash64, paths, chunksize=64)
    bad = sum(h is None for h in hashes)
    if bad:
        print(f"  {bad} unreadable image(s) skipped")
    return [(p, h) for p, h in zip(paths, hashes) if h is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="new dataset root to clean")
    ap.add_argument("--ref", default=str(ROOT / "data/drone"),
                    help="merged set root (scanned recursively)")
    ap.add_argument("--dist", type=int, default=PHASH_DIST)
    ap.add_argument("--ncc-min", type=float, default=NCC_MIN)
    ap.add_argument("--ncc-strong", type=float, default=NCC_STRONG)
    ap.add_argument("--gncc-min", type=float, default=GNCC_MIN)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--report", default=None, help="CSV of candidates to write")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    src, ref = Path(args.src), Path(args.ref)

    ref_hashes = hash_all(images_under(ref), args.workers, "reference")
    # LSH: an image is a candidate if it shares any 16-bit band with a ref image
    bands = [defaultdict(list) for _ in range(BANDS)]
    for p, hv in ref_hashes:
        for b in range(BANDS):
            bands[b][(hv >> (b * BAND_BITS)) & 0xFFFF].append((p, hv))

    src_hashes = hash_all(images_under(src), args.workers, "source")

    # stage 1: pHash candidates
    candidates = defaultdict(list)  # src path -> [(ref path, distance)]
    for p, hv in src_hashes:
        seen = set()
        for b in range(BANDS):
            for rp, rhv in bands[b].get((hv >> (b * BAND_BITS)) & 0xFFFF, ()):
                if rp in seen:
                    continue
                d = bin(hv ^ rhv).count("1")
                if d <= args.dist:
                    seen.add(rp)
                    candidates[p].append((rp, d))
    n_pairs = sum(len(v) for v in candidates.values())
    print(f"\npHash candidates: {n_pairs} pairs over {len(candidates)} src images"
          f" (dist <= {args.dist})")

    # stage 2: verify each candidate pair on pixels, keep the best ref per src
    print(f"verifying (ncc >= {args.ncc_strong}, or ncc >= {args.ncc_min} "
          f"and gncc >= {args.gncc_min}) ...", flush=True)
    thumbs = {}
    def get(p):
        if p not in thumbs:
            thumbs[p] = thumb(p)
        return thumbs[p]

    def is_dup(ncc, gncc):
        return ncc >= args.ncc_strong or (ncc >= args.ncc_min and gncc >= args.gncc_min)

    scored, matches = [], {}        # matches: src path -> (ref path, distance, ncc, gncc)
    for p, cands in candidates.items():
        best = None
        for rp, d in cands:
            (a, ga), (b, gb) = get(p), get(rp)
            ncc, gncc = float((a * b).mean()), float((ga * gb).mean())
            scored.append((ncc, gncc, d, p, rp))
            if is_dup(ncc, gncc) and (best is None or ncc > best[2]):
                best = (rp, d, ncc, gncc)
        if best:
            matches[p] = best
    to_delete = sorted(matches)

    by_dist, by_src_split, by_ref_split = (defaultdict(int) for _ in range(3))
    for p, (rp, d, _, _) in matches.items():
        by_dist[d] += 1
        by_src_split[p.relative_to(src).parts[0]] += 1
        by_ref_split[rp.relative_to(ref).parts[0]] += 1
    print(f"\nsrc images: {len(src_hashes)}")
    print(f"{'DELETING' if args.apply else 'would delete'} {len(matches)} duplicates"
          f"   keep: {len(src_hashes) - len(matches)}")
    for name, tally in (("by pHash distance", by_dist), ("by src split", by_src_split),
                       ("matched ref split", by_ref_split)):
        print(f"  {name}: " + ", ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    # pairs the gncc gate decides narrowly -- if this grows, re-check by eye
    near = sum(1 for ncc, gncc, _, _, _ in scored
               if args.ncc_min <= ncc < args.ncc_strong
               and abs(gncc - args.gncc_min) <= 0.08)
    print(f"  pairs within +-0.08 of the gncc gate: {near} (want few; verify by eye)")

    if args.report:
        with open(args.report, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ncc", "gncc", "phash_distance", "deleted", "src_image", "ref_image"])
            for ncc, gncc, d, p, rp in sorted(scored, reverse=True):
                w.writerow([f"{ncc:.4f}", f"{gncc:.4f}", d,
                            int(matches.get(p, (None,))[0] == rp), p, rp])
        print(f"  wrote {args.report}")

    if not args.apply:
        print("\ndry run -- pass --apply to delete")
        return
    n_lbl = 0
    for p in to_delete:
        lbl = p.parent.parent / "labels" / f"{p.stem}.txt"
        p.unlink()
        if lbl.exists():
            lbl.unlink()
            n_lbl += 1
    print(f"\ndeleted {len(to_delete)} images + {n_lbl} labels")


if __name__ == "__main__":
    main()
