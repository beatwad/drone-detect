#!/usr/bin/env python3
"""Merge the drone datasets into one clean single-class YOLO set.

Sources (all single-class `0: drone`):
  A = data/raw/drone_dataset            (muki2003, ~1359 imgs, CLOSE-range framing)
  B = data/raw/Database1                (sshikamaru, ~4010 imgs, LONG-range, video frames)
  C = data/Drone.v1i.yolov5pytorch      (Roboflow, ~17.7k imgs, mixed range)
  D = data/UAVs.v2i.yolov5pytorch       (Roboflow, ~9.3k imgs, mixed range)

Steps: pair image<->label by stem, drop unpaired, collapse Roboflow augmentation
copies, drop exact-hash duplicates, prefix filenames by source (avoids stem
collisions), cluster NEAR-duplicates by perceptual hash (video frames differ
slightly frame-to-frame), then a seeded 80/20 split STRATIFIED by source AND by
regime, where whole pHash-clusters go to one side only (prevents near-dup
train/val leakage).
Copy into data/drone/{train,val}/{images,labels}, write configs/drone.yaml + a
manifest.

The Roboflow exports ship pre-augmented: the same source image appears up to 6x
as `<stem>_jpg.rf.<hash>.jpg`. Their flips/rotations move the pHash well past
PHASH_HAMMING (measured: 99% of same-source pairs), so clustering would NOT group
them and copies could land on both sides of the split. We therefore keep one copy
per base stem -- augmentation belongs in the training loop, not the dataset.

The Roboflow sources are also range-bimodal, so `regime` is derived per-image from
the largest box rather than being a per-source constant.
"""
import argparse, csv, hashlib, re, shutil, random
from pathlib import Path
from collections import Counter, defaultdict

import imagehash
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
ROOT = Path(__file__).resolve().parents[1]
PHASH_HAMMING = 6          # <= this many differing bits => near-duplicate
BIG_BUCKET = 200           # LSH bucket bigger than this => merge whole (safe over-merge)

# Roboflow augmentation suffix: `0002_jpg.rf.<md5>` -> base stem `0002`
RF_AUG = re.compile(r"_(?:jpe?g|png|bmp)\.rf\.[0-9a-f]+$", re.I)

# regime thresholds on largest-box area as a fraction of frame
CLOSE_AREA = 0.10          # >= 10% of frame => close range (our target regime)
LONG_AREA = 0.01           # <  1% of frame  => long range

# source dir -> (prefix, collapse Roboflow augmentation copies?)
SOURCES = {
    ROOT / "data/raw/drone_dataset": ("A", False),
    ROOT / "data/raw/Database1": ("B", False),
    ROOT / "data/Drone.v1i.yolov5pytorch": ("C", True),
    ROOT / "data/UAVs.v2i.yolov5pytorch": ("D", True),
}
REGIMES = ("close", "mid", "long", "empty")


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


def regime_of(lbl_path: Path) -> str:
    """close/mid/long from the largest box in a YOLO label file."""
    best = 0.0
    for row in lbl_path.read_text().split("\n"):
        p = row.split()
        if len(p) == 5:
            best = max(best, float(p[3]) * float(p[4]))
    if best == 0.0:
        return "empty"          # no boxes: background negative
    if best >= CLOSE_AREA:
        return "close"
    return "long" if best < LONG_AREA else "mid"


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
            # wipe first: a re-run reshuffles the split, and leftovers from a
            # previous split would leave the same image in BOTH train and val
            shutil.rmtree(out / split / kind, ignore_errors=True)
            (out / split / kind).mkdir(parents=True, exist_ok=True)

    # 1) collect + exact-dedup per source
    records = []  # (new_stem, prefix, regime, img_path, lbl_path)
    for src_dir, (prefix, collapse) in SOURCES.items():
        paired, n_img, n_lbl = collect(src_dir)
        stems = sorted(paired)
        n_aug = 0
        if collapse:
            by_base = {}
            for stem in stems:
                by_base.setdefault(RF_AUG.sub("", stem), stem)  # first wins, deterministic
            n_aug = len(stems) - len(by_base)
            stems = sorted(by_base.values())
        seen = set(); kept = 0
        for stem in stems:
            img, lbl = paired[stem]
            h = md5(img)
            if h in seen:
                continue
            seen.add(h)
            records.append((f"{prefix}_{stem}", prefix, regime_of(lbl), img, lbl))
            kept += 1
        print(f"[{prefix}] imgs={n_img} lbls={n_lbl} paired={len(paired)} "
              f"dropped_augcopy={n_aug} dropped_exactdup={len(stems)-kept} -> kept={kept}")

    # 2) perceptual-hash near-duplicate clustering (drop unreadable images)
    #    all near-duplicate videos are grouped into clusters (e.g. all frames from one video)
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

    # 3) split by cluster (whole cluster contains all near-dup images: this is
    #    what prevents near-dup train/val leakage). So we take a cluster into val 
    #    only while every group (source, regime) it touches is still under
    #    its own val target. Largest clusters first, so one big video-frame
    #    cluster can't overshoot the target the way first-fit did.
    def budget_keys(i):
        """
        Groups an image counts against: its source AND its regime. Tagged so
        the two families share one Counter without colliding (e.g. 'C' the source vs
        'close' the regime). So the result split is roughly stratified against 
        both source and regime.
        """
        return ("src", records[i][1]), ("reg", records[i][2])

    per_group_total = Counter(g for i in range(len(records)) for g in budget_keys(i))
    target_val = {g: round(n * args.val_frac) for g, n in per_group_total.items()}
    n_val = Counter()

    clusters = list(groups.values())
    rng.shuffle(clusters)                      # tie-break randomly, then pack big-first
    clusters.sort(key=len, reverse=True)

    assign = {}  # idx -> split
    for c in clusters:
        comp = Counter(g for i in c for g in budget_keys(i))
        if all(n_val[g] + k <= target_val[g] for g, k in comp.items()):
            split = "val"
            n_val.update(comp)
        else:
            split = "train"
        for idx in c:
            assign[idx] = split

    # 4) copy files + manifest
    manifest = []
    val_by_regime = defaultdict(list)  # regime -> copied val image paths
    for i, (new_stem, prefix, regime, img, lbl) in enumerate(records):
        split = assign[i]
        dst = out / split / "images" / f"{new_stem}{img.suffix.lower()}"
        shutil.copy2(img, dst)
        shutil.copy2(lbl, out / split / "labels" / f"{new_stem}.txt")
        if split == "val":
            val_by_regime[regime].append(dst)
        manifest.append((new_stem, prefix, regime, split, uf.find(i), str(img)))

    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stem", "source", "regime", "split", "cluster", "orig_image"])
        w.writerows(manifest)

    cfg_dir = ROOT / "configs"
    (cfg_dir / "drone.yaml").write_text(
        f"path: {out}\ntrain: {out / 'train/images'}\nval: {out / 'val/images'}\n"
        f"nc: 1\nnames:\n  0: drone\n"
    )

    # 5) per-regime val subsets. Emitted here, by the same run that decides the split,
    #    so they can never drift out of sync with it
    for stale in cfg_dir.glob("val_*.cache"):
        stale.unlink()                      # keyed to the old list; force a rescan
    for reg in REGIMES:
        lst = cfg_dir / f"val_{reg}.txt"
        yml = cfg_dir / f"drone_val_{reg}.yaml"
        paths = val_by_regime.get(reg, [])
        if not paths:
            lst.unlink(missing_ok=True)     # don't leave a stale list behind
            yml.unlink(missing_ok=True)
            continue
        lst.write_text("".join(f"{p}\n" for p in sorted(paths)))
        yml.write_text(
            f"path: {out}\ntrain: {out / 'train/images'}\nval: {lst}\n"
            f"nc: 1\nnames:\n  0: drone\n"
        )

    tr = sum(1 for m in manifest if m[3] == "train")
    print(f"\nTOTAL kept={len(manifest)}  train={tr} val={len(manifest)-tr}")
    for reg in REGIMES:
        t = sum(1 for m in manifest if m[2] == reg and m[3] == "train")
        v = sum(1 for m in manifest if m[2] == reg and m[3] == "val")
        print(f"  regime {reg:5}: train={t} val={v}")
    print(f"wrote configs/drone.yaml and {out / 'manifest.csv'}")
    print("wrote per-regime val subsets: " +
          ", ".join(f"val_{r}.txt({len(val_by_regime[r])})"
                    for r in REGIMES if val_by_regime.get(r)))


if __name__ == "__main__":
    main()
