#!/usr/bin/env python3
"""
split_no_leak.py — build a leakage-free split of the CHVG / Roboflow PPE dataset.

Why this exists
---------------
The PPE-EYE paper (Computers 2026, 15, 45) augments 1,699 source images up to
28,000+ instances and *then* splits 80:20. If that is what happened, near-duplicate
copies of the same source photo land on both sides of the split and mAP50 is
inflated. This script splits by SOURCE IMAGE GROUP, so every augmented sibling of
an image stays in the same split.

Two grouping signals are used:
  1. Filename stem. Roboflow exports as  <stem>_jpg.rf.<hash>.jpg  — all augmented
     children of one source share <stem>.
  2. Perceptual hash (dHash, 64-bit). Catches siblings whose stems were mangled,
     and catches genuine near-duplicate photos in the raw dataset (burst frames
     from the same CCTV clip are a real problem in construction datasets).

Groups are merged with union-find, then greedily assigned to train/val/test while
balancing per-class object counts.

Usage
-----
  python split_no_leak.py --src /data/chvg_raw --dst /data/chvg_clean \
      --val 0.15 --test 0.15 --hamming 6

--src expects YOLO layout, either flat (images/, labels/) or already split
(train/images, valid/images, test/images ...). Everything is pooled and re-split.

Outputs --dst/{train,val,test}/{images,labels} plus data.yaml, and prints a
leakage audit you should paste into your logbook.
"""

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ----------------------------------------------------------------------------- hashing
def dhash(path: Path, size: int = 8) -> int:
    """64-bit difference hash. Robust to brightness/exposure jitter, sensitive to content."""
    try:
        im = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    except Exception:
        return -1
    a = np.asarray(im, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ----------------------------------------------------------------------------- union-find
class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


# ----------------------------------------------------------------------------- helpers
def roboflow_stem(name: str) -> str:
    """'foo_jpg.rf.deadbeef.jpg' -> 'foo'   |   'foo.rf.beef.jpg' -> 'foo'"""
    for marker in ("_jpg.rf.", "_png.rf.", "_jpeg.rf.", ".rf."):
        if marker in name:
            return name.split(marker)[0]
    return Path(name).stem


def label_path_for(img: Path) -> Path:
    # .../images/x.jpg -> .../labels/x.txt
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def read_classes(img: Path):
    lp = label_path_for(img)
    if not lp.exists():
        return []
    out = []
    for line in lp.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(int(float(line.split()[0])))
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--hamming", type=int, default=6,
                    help="dHash distance below which two images are treated as duplicates")
    ap.add_argument("--names", type=str, default="",
                    help="comma-separated class names; else read from src data.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--copy", action="store_true", help="copy instead of symlink")
    args = ap.parse_args()
    random.seed(args.seed)

    imgs = sorted(p for p in args.src.rglob("*") if p.suffix.lower() in IMG_EXT)
    imgs = [p for p in imgs if "images" in p.parts]
    if not imgs:
        imgs = sorted(p for p in args.src.rglob("*") if p.suffix.lower() in IMG_EXT)
    print(f"[scan] {len(imgs)} images under {args.src}")
    if not imgs:
        raise SystemExit("no images found")

    # ---- pass 1: group by filename stem
    uf = UF(len(imgs))
    by_stem = defaultdict(list)
    for i, p in enumerate(imgs):
        by_stem[roboflow_stem(p.name)].append(i)
    for idxs in by_stem.values():
        for j in idxs[1:]:
            uf.union(idxs[0], j)
    print(f"[group] {len(by_stem)} distinct filename stems")

    # ---- pass 2: merge by perceptual hash
    print("[hash] computing dHash ...")
    hashes = [dhash(p) for p in imgs]
    buckets = defaultdict(list)
    for i, h in enumerate(hashes):
        if h < 0:
            continue
        # bucket on high 16 bits so we only compare plausible candidates
        buckets[h >> 48].append(i)
    merged = 0
    for idxs in buckets.values():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                if uf.find(i) != uf.find(j) and hamming(hashes[i], hashes[j]) <= args.hamming:
                    uf.union(i, j)
                    merged += 1
    print(f"[hash] merged {merged} pairs across different stems (near-duplicates)")

    groups = defaultdict(list)
    for i in range(len(imgs)):
        groups[uf.find(i)].append(i)
    glist = list(groups.values())
    print(f"[group] {len(glist)} independent groups from {len(imgs)} images "
          f"(mean {len(imgs)/len(glist):.1f} images/group)")
    if len(imgs) / len(glist) > 3:
        print("       ^ high expansion factor. A naive random split here would leak badly.")

    # ---- class inventory per group
    gcls = []
    for g in glist:
        c = defaultdict(int)
        for i in g:
            for k in read_classes(imgs[i]):
                c[k] += 1
        gcls.append(c)
    all_cls = sorted({k for c in gcls for k in c})
    total = {k: sum(c.get(k, 0) for c in gcls) for k in all_cls}

    # ---- greedy balanced assignment: rarest-class-first, largest group first
    target = {"train": 1 - args.val - args.test, "val": args.val, "test": args.test}
    have = {s: defaultdict(int) for s in target}
    assign = {}
    order = sorted(range(len(glist)),
                   key=lambda gi: (-sum(gcls[gi].values()), random.random()))
    for gi in order:
        best, best_cost = None, None
        for s in target:
            cost = 0.0
            for k in all_cls:
                want = total[k] * target[s]
                cost += (((have[s][k] + gcls[gi].get(k, 0)) - want) / max(want, 1)) ** 2
            if best_cost is None or cost < best_cost:
                best, best_cost = s, cost
        assign[gi] = best
        for k, v in gcls[gi].items():
            have[best][k] += v

    # ---- materialise
    for s in target:
        for sub in ("images", "labels"):
            (args.dst / s / sub).mkdir(parents=True, exist_ok=True)
    counts = defaultdict(int)
    for gi, g in enumerate(glist):
        s = assign[gi]
        for i in g:
            src_i = imgs[i]
            src_l = label_path_for(src_i)
            dst_i = args.dst / s / "images" / src_i.name
            dst_l = args.dst / s / "labels" / (src_i.stem + ".txt")
            if args.copy:
                shutil.copy2(src_i, dst_i)
            else:
                if dst_i.exists():
                    dst_i.unlink()
                dst_i.symlink_to(src_i.resolve())
            if src_l.exists():
                shutil.copy2(src_l, dst_l)
            else:
                dst_l.write_text("")
            counts[s] += 1

    # ---- data.yaml
    names = [n.strip() for n in args.names.split(",") if n.strip()]
    if not names:
        y = list(args.src.rglob("data.yaml"))
        if y:
            txt = y[0].read_text()
            if "names:" in txt:
                seg = txt.split("names:")[1]
                if "[" in seg:
                    names = [x.strip().strip("'\"") for x in
                             seg[seg.index("[") + 1: seg.index("]")].split(",")]
    if not names:
        names = [f"class{i}" for i in range(max(all_cls) + 1)]
    (args.dst / "data.yaml").write_text(
        f"path: {args.dst.resolve()}\ntrain: train/images\nval: val/images\n"
        f"test: test/images\nnc: {len(names)}\nnames: {names}\n"
    )

    # ---- audit
    print("\n=== SPLIT AUDIT ===")
    for s in target:
        print(f"{s:<6} images={counts[s]:<6} groups="
              f"{sum(1 for gi in assign if assign[gi]==s):<5} "
              f"objects={sum(have[s].values())}")
    print("\nper-class object counts")
    print(f"{'cls':<6}{'name':<12}{'train':>8}{'val':>8}{'test':>8}")
    for k in all_cls:
        nm = names[k] if k < len(names) else str(k)
        print(f"{k:<6}{nm:<12}{have['train'][k]:>8}{have['val'][k]:>8}{have['test'][k]:>8}")

    # cross-split duplicate check (should be zero by construction — verifies the logic)
    split_of = {}
    for gi, g in enumerate(glist):
        for i in g:
            split_of[i] = assign[gi]
    bad = 0
    for idxs in buckets.values():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                if hamming(hashes[i], hashes[j]) <= args.hamming and split_of[i] != split_of[j]:
                    bad += 1
    print(f"\ncross-split near-duplicate pairs: {bad}  (must be 0)")
    (args.dst / "split_audit.json").write_text(json.dumps({
        "images": len(imgs), "groups": len(glist),
        "expansion_factor": round(len(imgs) / len(glist), 2),
        "cross_split_dupes": bad,
        "counts": {s: counts[s] for s in target},
    }, indent=2))
    print(f"\nwrote {args.dst/'data.yaml'}")


if __name__ == "__main__":
    main()
