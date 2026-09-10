#!/usr/bin/env python3
"""
sh17_map.py — map the SH17 17-class object-detection dataset onto our
person-level compliance ontology W / WH / WHV / WV.

SH17 gives separate boxes for person / helmet / safety-vest (and 14 other
classes we ignore). We need one box per person, class = what they are wearing:

    W    person, no helmet, no vest
    WH   person + helmet
    WHV  person + helmet + vest
    WV   person + vest

Association (deliberately simple and auditable — the same kind of heuristic
experiments/honest_eval.py uses):
    helmet worn  : helmet-box centre in the top HEAD_BAND of the person box AND
                   >= CONTAIN of the helmet box inside the person box
    vest worn    : vest-box centre in the person torso band AND >= CONTAIN of the
                   vest box inside the person box
    greedy 1-to-1: each helmet / vest consumed by at most one person (best
                   containment first) so one helmet cannot dress a whole crowd.

Output: a YOLO dataset at data/sh17_compliance/ with images resized to
RESIZE_LONG px (SH17 originals are ~5760x3840 — decoding those every epoch is
the bottleneck), mapped labels, data.yaml, and SH17's own train/val split.

Run once:
    PPE/Scripts/python.exe experiments/sh17_map.py \
        --src "C:/Users/user/Desktop/khizar/amsar-AI/research/sh17 PPE dataset"
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# SH17 ids derived by matching this export's YOLO labels to its VOC xml
PERSON, HELMET, VEST = 0, 10, 16
NAMES = ["W", "WH", "WHV", "WV"]
VERDICT = {(0, 0): 0, (1, 0): 1, (1, 1): 2, (0, 1): 3}  # (helmet, vest) -> class id

HEAD_BAND = 0.40      # helmet centre must sit in the top 40% of the person box
TORSO_LO, TORSO_HI = 0.15, 0.80
CONTAIN = 0.45
RESIZE_LONG = 1280


def _read(label_file: Path):
    rows = []
    for ln in label_file.read_text().splitlines():
        t = ln.split()
        if len(t) >= 5:
            rows.append((int(float(t[0])), *(float(x) for x in t[1:5])))
    return rows


def _contain(inner, outer):
    ix1 = max(inner[0], outer[0]); iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2]); iy2 = min(inner[3], outer[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    a = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return (iw * ih) / a if a > 0 else 0.0


def _xyxy(b):  # b = (cx, cy, w, h)
    return [b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2]


def map_one(rows):
    persons = [r[1:] for r in rows if r[0] == PERSON]
    helmets = [r[1:] for r in rows if r[0] == HELMET]
    vests = [r[1:] for r in rows if r[0] == VEST]
    p_xy = [_xyxy(p) for p in persons]
    worn = [[0, 0] for _ in persons]  # [helmet, vest]

    def assign(items, slot, lo, hi):
        pairs = []
        for j, it in enumerate(items):
            ib = _xyxy(it)
            cy = it[1]
            for i, pb in enumerate(p_xy):
                c = _contain(ib, pb)
                if c < CONTAIN:
                    continue
                ph = pb[3] - pb[1]
                rel = (cy - pb[1]) / ph if ph > 0 else 1.0
                if lo <= rel <= hi:
                    pairs.append((c, j, i))
        pairs.sort(reverse=True)
        uj, ui = set(), set()
        for c, j, i in pairs:
            if j in uj or i in ui:
                continue
            worn[i][slot] = 1
            uj.add(j); ui.add(i)

    assign(helmets, 0, 0.0, HEAD_BAND)
    assign(vests, 1, TORSO_LO, TORSO_HI)

    out = []
    for pb, (hh, vv) in zip(persons, worn):
        out.append((VERDICT[(hh, vv)], *pb))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="SH17 dataset root (has images/ labels/)")
    ap.add_argument("--dst", default="data/sh17_compliance")
    ap.add_argument("--long", type=int, default=RESIZE_LONG)
    ap.add_argument("--all-images", action="store_true",
                    help="keep every image with a person. Default keeps only images with "
                         ">=1 PPE-bearing person: SH17 annotates PPE sparsely, so a "
                         "person in a no-PPE image is 'no helmet/vest ANNOTATED', not "
                         "'no PPE present' — 92.7%% of boxes would become a noisy W.")
    ap.add_argument("--neg", type=int, default=0,
                    help="additionally keep N no-PPE images as hard negatives")
    args = ap.parse_args()

    src = Path(args.src)
    root = Path(__file__).resolve().parent.parent
    dst = (root / args.dst) if not Path(args.dst).is_absolute() else Path(args.dst)
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)

    def split_stems(name):
        f = src / name
        if not f.exists():
            return None
        return [Path(x.strip()).stem for x in f.read_text().splitlines() if x.strip()]

    train_stems = split_stems("train_files.txt")
    val_stems = split_stems("val_files.txt")
    want = None
    if train_stems is not None:
        want = set(train_stems) | set(val_stems or [])

    label_files = sorted((src / "labels").glob("*.txt"))
    val_set = set(val_stems or [])
    cls_hist = Counter()
    n_persons = n_imgs = 0
    written = {"train": [], "val": []}
    img_lookup = {p.stem: p for p in (src / "images").iterdir() if p.is_file()}

    # first pass: decide which stems to keep (cheap — labels only)
    keep, negatives = [], []
    for lf in label_files:
        stem = lf.stem
        if want is not None and stem not in want:
            continue
        if stem not in img_lookup:
            continue
        mapped = map_one(_read(lf))
        if not mapped:
            continue
        has_ppe = any(c != 0 for c, *_ in mapped)
        if has_ppe or args.all_images:
            keep.append((stem, mapped))
        elif args.neg:
            negatives.append((stem, mapped))
    if args.neg and negatives:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(negatives), size=min(args.neg, len(negatives)), replace=False)
        keep.extend(negatives[i] for i in idx)
    keep.sort()
    print(f"keeping {len(keep)} images "
          f"({'all' if args.all_images else 'PPE-bearing'}"
          f"{f' + {min(args.neg, len(negatives))} negatives' if args.neg else ''})")

    for i, (stem, mapped) in enumerate(keep):
        img = img_lookup[stem]
        im = cv2.imread(str(img))
        if im is None:
            continue
        h, w = im.shape[:2]
        scale = args.long / max(h, w)
        if scale < 1.0:
            im = cv2.resize(im, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        out_img = dst / "images" / f"{stem}.jpg"
        cv2.imwrite(str(out_img), im, [cv2.IMWRITE_JPEG_QUALITY, 92])

        lines = []
        for c, cx, cy, bw, bh in mapped:
            cls_hist[NAMES[c]] += 1
            n_persons += 1
            lines.append(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (dst / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        n_imgs += 1

        sp = "val" if (val_stems and stem in val_set) else "train"
        written[sp].append(str(out_img))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(keep)} …")

    (dst / "train.txt").write_text("\n".join(written["train"]) + "\n")
    (dst / "val.txt").write_text("\n".join(written["val"]) + "\n")
    (dst / "data.yaml").write_text(
        f"path: {dst.as_posix()}\n"
        f"train: train.txt\nval: val.txt\ntest: val.txt\n"
        f"nc: 4\nnames: {NAMES}\n")

    print(f"\n{n_imgs} images, {n_persons} person boxes")
    print("class balance:", dict(cls_hist))
    print(f"train {len(written['train'])}  val {len(written['val'])}")
    print(f"wrote {dst}/data.yaml")


if __name__ == "__main__":
    main()
