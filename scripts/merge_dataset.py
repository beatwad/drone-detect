#!/usr/bin/env python3
"""Merge the two Kaggle drone datasets into one clean single-class YOLO set.

Sources (both single-class `0: drone`):
  A = data/raw/drone_dataset   (muki2003, ~1359 imgs, CLOSE-range framing)
  B = data/raw/Database1       (sshikamaru, ~4010 imgs, LONG-range, video frames)

Steps: pair image<->label by stem, drop unpaired, drop exact-hash duplicates,
prefix filenames by source (avoids stem collisions), cluster NEAR-duplicates by
perceptual hash (video frames differ slightly frame-to-frame), then a seeded
80/20 split STRATIFIED by source where whole pHash-clusters go to one side only
(prevents near-dup train/val leakage). Copy into
data/drone/{train,val}/{images,labels}, write configs/drone.yaml + a manifest.
"""
import argparse, csv, hashlib, shutil, random
from pathlib import Path
from collections import defaultdict

import imagehash
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
ROOT = Path(__file__).resolve().parents[1]
PHASH_HAMMING = 6          # <= this many differing bits => near-duplicate
BIG_BUCKET = 200           # LSH bucket bigger than this => merge whole (safe over-merge)

# source dir -> (prefix, regime tag)
SOURCES = {
    ROOT / "data/raw/drone_dataset": ("A", "close"),
    ROOT / "data/raw/Database1": ("B", "long"),
}


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def collect(src_dir: Path):
    imgs, lbls = {}, {}
    for f in src_dir.rglob("*"):
        if not f.is_file():
            continue
        stem, ext = f.stem, f.suffix.lower()
        if ext in IMG_EXTS:
            imgs[stem] = f
        elif ext == ".txt" and f.name.lower() != "classes.txt":
            lbls[stem] = f
    paired = {s: (imgs[s], lbls[s]) for s in (imgs.keys() & lbls.keys())}
    return paired, len(imgs), len(lbls)


class UF:
    def __init__(self, n): self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[ra] = rb


def phash64(path: Path):
    """64-bit perceptual hash, or None if the image can't be read (corrupt)."""
    try:
        h = imagehash.phash(Image.open(path).convert("RGB"))  # 8x8 => 64 bits
    except Exception:
        return None
    v = 0
    for bit in h.hash.flatten():
        v = (v << 1) | int(bit)
    return v


def cluster_near_dups(items):
    """items: list of (idx, phash int). Returns UF over indices via LSH banding."""
    n = len(items)
    uf = UF(n)
    # 4 bands of 16 bits; images sharing any band are near-dup candidates
    for band in range(4):
        shift = band * 16
        buckets = defaultdict(list)
        for i, (_, hv) in enumerate(items):
            buckets[(hv >> shift) & 0xFFFF].append(i)
        for members in buckets.values():
            if len(members) < 2:
                continue
            if len(members) > BIG_BUCKET:
                # too big to pair up; safe to over-merge (only reduces split
                # independence slightly, never causes leakage)
                for k in members[1:]:
                    uf.union(members[0], k)
                continue
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    ia, ib = members[a], members[b]
                    if bin(items[ia][1] ^ items[ib][1]).count("1") <= PHASH_HAMMING:
                        uf.union(ia, ib)
    return uf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data/drone"))
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            (out / split / kind).mkdir(parents=True, exist_ok=True)

    # 1) collect + exact-dedup per source
    records = []  # (new_stem, prefix, regime, img_path, lbl_path)
    for src_dir, (prefix, regime) in SOURCES.items():
        paired, n_img, n_lbl = collect(src_dir)
        seen = set(); kept = 0
        for stem in sorted(paired):
            img, lbl = paired[stem]
            h = md5(img)
            if h in seen:
                continue
            seen.add(h)
            records.append((f"{prefix}_{stem}", prefix, regime, img, lbl))
            kept += 1
        print(f"[{prefix}] imgs={n_img} lbls={n_lbl} paired={len(paired)} "
              f"dropped_exactdup={len(paired)-kept} -> kept={kept}")

    # 2) perceptual-hash near-duplicate clustering (drop unreadable images)
    print(f"computing pHash for {len(records)} images ...")
    good, dropped_corrupt = [], 0
    for r in records:
        hv = phash64(r[3])
        if hv is None:
            dropped_corrupt += 1
            continue
        good.append((r, hv))
    if dropped_corrupt:
        print(f"dropped {dropped_corrupt} unreadable/corrupt image(s)")
    records = [g[0] for g in good]
    hashes = [(i, g[1]) for i, g in enumerate(good)]
    uf = cluster_near_dups(hashes)
    groups = defaultdict(list)
    for i in range(len(records)):
        groups[uf.find(i)].append(i)
    n_clusters = len(groups)
    n_dup_frames = len(records) - n_clusters
    print(f"pHash clusters: {n_clusters} (grouped {n_dup_frames} near-dup frames)")

    # 3) split by cluster, stratified by source (whole cluster -> one side)
    src_clusters = defaultdict(list)  # prefix -> list of cluster (list of idx)
    for members in groups.values():
        prefix = records[members[0]][1]  # clusters are ~single-source
        src_clusters[prefix].append(members)

    assign = {}  # idx -> split
    for prefix, clusters in src_clusters.items():
        rng.shuffle(clusters)
        n_total = sum(len(c) for c in clusters)
        target_val = round(n_total * args.val_frac)
        n_val = 0
        for c in clusters:
            split = "val" if n_val < target_val else "train"
            if split == "val":
                n_val += len(c)
            for idx in c:
                assign[idx] = split

    # 4) copy files + manifest
    manifest = []
    for i, (new_stem, prefix, regime, img, lbl) in enumerate(records):
        split = assign[i]
        shutil.copy2(img, out / split / "images" / f"{new_stem}{img.suffix.lower()}")
        shutil.copy2(lbl, out / split / "labels" / f"{new_stem}.txt")
        manifest.append((new_stem, prefix, regime, split, uf.find(i), str(img)))

    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stem", "source", "regime", "split", "cluster", "orig_image"])
        w.writerows(manifest)

    (ROOT / "configs/drone.yaml").write_text(
        f"path: {out}\ntrain: {out / 'train/images'}\nval: {out / 'val/images'}\n"
        f"nc: 1\nnames:\n  0: drone\n"
    )

    tr = sum(1 for m in manifest if m[3] == "train")
    print(f"\nTOTAL kept={len(manifest)}  train={tr} val={len(manifest)-tr}")
    for reg in ("close", "long"):
        t = sum(1 for m in manifest if m[2] == reg and m[3] == "train")
        v = sum(1 for m in manifest if m[2] == reg and m[3] == "val")
        print(f"  regime {reg:5}: train={t} val={v}")
    print(f"wrote configs/drone.yaml and {out / 'manifest.csv'}")


if __name__ == "__main__":
    main()
