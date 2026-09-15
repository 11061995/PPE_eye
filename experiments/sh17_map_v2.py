#!/usr/bin/env python3
"""
sh17_map_v2.py - SH17 -> person-level compliance (W/WH/WHV/WV), version 2.

What changed vs sh17_map.py
---------------------------
v1 kept only the 498 images containing at least one helmet/vest box, because in
an image with no PPE annotation a person means "no helmet *annotated*", not
"no helmet *worn*" - labelling those W would have invented violations.

v2 removes that guess. SH17 annotates a **bare head** as its own class
(id 12, `head`). A person box whose head region contains a `head` box and no
`helmet` box is a **confirmed** no-helmet worker, not an annotation gap. So the
helmet axis is resolvable for far more of the dataset: 5787 of 7617
person-bearing images have *every* person resolvable.

Two guards keep that from backfiring:

1. **All-or-nothing per image.** An image is kept only if every person in it is
   resolvable (helmet box or head box). YOLO has no "ignore" label, so a single
   unresolved person would become an unlabelled object the model is punished for
   detecting.
2. **Geometry filter on the head-only images.** The 5358 head-only images are
   close-up stock portraits - 23.9% pooled median person-box area against
   Pictor test's 1.6%. Importing them wholesale would both swamp the class
   balance with W and pull the model toward close-ups. Only those under
   `--area-max` are kept (default 4% -> 319 images, 2.68 persons/img at 1.21%
   median area, which matches the deployment distribution).

The vest axis is unchanged and still one-sided: a `safety-vest` box means vest,
its absence means "no vest annotated". SH17 has no bare-torso class, so this
cannot be fixed the same way. All reported metrics use the `helmet` rule, which
is the axis this file makes sound.

Outputs a YOLO dataset with a **group-disjoint** train/val/test split (SH17
filenames run in shoot-consecutive blocks, e.g. pexels-photo-10341095..122, so a
random split would leak near-duplicates across it), plus a separate
`holdout` set - the head-only images rejected by the geometry filter. Those are
never trained on and form a real distribution-shift test.

Run:
    PPE/Scripts/python.exe experiments/sh17_map_v2.py --src "../DATASETS/sh17 PPE dataset"
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import cv2
import numpy as np

PERSON, HELMET, HEAD, VEST = 0, 10, 12, 16
NAMES = ["W", "WH", "WHV", "WV"]
VERDICT = {(0, 0): 0, (1, 0): 1, (1, 1): 2, (0, 1): 3}

HEAD_BAND = 0.40           # helmet / head centre must sit in the top 40% of the person
TORSO_LO, TORSO_HI = 0.15, 0.80
CONTAIN = 0.45
RESIZE_LONG = 1280


def _read(p: Path):
    rows = []
    for ln in p.read_text().splitlines():
        t = ln.split()
        if len(t) >= 5:
            rows.append((int(float(t[0])), *(float(x) for x in t[1:5])))
    return rows


def _xyxy(b):
    return (b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2)


def _contain(inner, outer):
    ix1 = max(inner[0], outer[0]); iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2]); iy2 = min(inner[3], outer[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    a = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return (iw * ih) / a if a > 0 else 0.0


def _assign(items, person_boxes, lo, hi):
    """Greedy 1-to-1: each item worn by at most one person. Returns person indices."""
    pairs = []
    for j, it in enumerate(items):
        ib = _xyxy(it)
        for i, pb in enumerate(person_boxes):
            c = _contain(ib, pb)
            if c < CONTAIN:
                continue
            ph = pb[3] - pb[1]
            rel = (it[1] - pb[1]) / ph if ph > 0 else 1.0
            if lo <= rel <= hi:
                pairs.append((c, j, i))
    pairs.sort(reverse=True)
    used_item, used_person = set(), set()
    for c, j, i in pairs:
        if j in used_item or i in used_person:
            continue
        used_item.add(j); used_person.add(i)
    return used_person


def map_one(rows):
    """-> (labels, resolvable, has_ppe_annotation); labels = [(cls, cx, cy, w, h)]"""
    P = [r[1:] for r in rows if r[0] == PERSON]
    if not P:
        return [], False, False
    H = [r[1:] for r in rows if r[0] == HELMET]
    D = [r[1:] for r in rows if r[0] == HEAD]
    V = [r[1:] for r in rows if r[0] == VEST]
    pb = [_xyxy(p) for p in P]

    helmeted = _assign(H, pb, 0.0, HEAD_BAND)
    bareheaded = _assign(D, pb, 0.0, HEAD_BAND)
    vested = _assign(V, pb, TORSO_LO, TORSO_HI)

    resolvable = len(helmeted | bareheaded) == len(P)
    out = [(VERDICT[(1 if i in helmeted else 0, 1 if i in vested else 0)], *P[i])
           for i in range(len(P))]
    return out, resolvable, bool(H or V)


def group_key(stem: str) -> str:
    """SH17 shoots run in consecutive id blocks; bucket them so near-duplicate
    frames cannot straddle the train/val/test boundary."""
    m = re.match(r"^(.*?)(\d+)$", stem)
    if not m:
        return stem
    prefix, num = m.group(1), int(m.group(2))
    return f"{prefix}{num // 50}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="data/sh17c_v2")
    ap.add_argument("--long", type=int, default=RESIZE_LONG)
    ap.add_argument("--area-max", type=float, default=0.04,
                    help="head-only images are kept only if their median person-box "
                         "area is below this (deployment-geometry filter)")
    ap.add_argument("--holdout-n", type=int, default=400,
                    help="close-up head-only images to set aside as a shift test")
    ap.add_argument("--reuse", default="data/sh17_compliance/images",
                    help="already-resized jpgs to copy instead of re-decoding")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    src = Path(args.src)
    if not src.is_absolute():
        src = (root / src).resolve()
    dst = root / args.dst
    for sub in ("images", "labels", "holdout/images", "holdout/labels"):
        (dst / sub).mkdir(parents=True, exist_ok=True)

    img_lookup = {p.stem: p for p in (src / "images").iterdir() if p.is_file()}
    reuse_dir = root / args.reuse

    # ---- pass 1: labels only (cheap) --------------------------------------
    ppe_imgs, head_imgs, holdout_imgs = [], [], []
    n_person_imgs = n_unresolvable = 0
    for lf in sorted((src / "labels").glob("*.txt")):
        if lf.stem not in img_lookup:
            continue
        labels, resolvable, has_ppe = map_one(_read(lf))
        if not labels:
            continue
        n_person_imgs += 1
        if not resolvable:
            n_unresolvable += 1
            continue
        med_area = median([l[3] * l[4] for l in labels])
        rec = (lf.stem, labels, med_area)
        if has_ppe:
            ppe_imgs.append(rec)
        elif med_area < args.area_max:
            head_imgs.append(rec)
        else:
            holdout_imgs.append(rec)

    rng = np.random.default_rng(0)
    holdout_imgs.sort()
    if len(holdout_imgs) > args.holdout_n:
        idx = rng.choice(len(holdout_imgs), size=args.holdout_n, replace=False)
        holdout_imgs = [holdout_imgs[i] for i in sorted(idx)]

    keep = sorted(ppe_imgs + head_imgs)
    print(f"person-bearing images      : {n_person_imgs}")
    print(f"  dropped, unresolvable    : {n_unresolvable}")
    print(f"  kept, PPE-annotated      : {len(ppe_imgs)}")
    print(f"  kept, bare-head confirmed: {len(head_imgs)} (median area < {args.area_max:.0%})")
    print(f"  close-up shift holdout   : {len(holdout_imgs)}")

    # ---- group-disjoint split ---------------------------------------------
    by_group = defaultdict(list)
    for rec in keep:
        by_group[group_key(rec[0])].append(rec)
    groups = sorted(by_group)
    rng2 = np.random.default_rng(1)
    rng2.shuffle(groups)
    n = len(groups)
    n_val = max(1, int(n * args.val_frac))
    n_test = max(1, int(n * args.test_frac))
    split_of = {}
    for g in groups[:n_test]:
        split_of[g] = "test"
    for g in groups[n_test:n_test + n_val]:
        split_of[g] = "val"
    for g in groups[n_test + n_val:]:
        split_of[g] = "train"

    # ---- pass 2: write images + labels ------------------------------------
    def emit(rec, out_img_dir, out_lbl_dir):
        stem, labels, _ = rec
        out_img = out_img_dir / f"{stem}.jpg"
        if not out_img.exists():
            cached = reuse_dir / f"{stem}.jpg"
            if cached.exists():
                out_img.write_bytes(cached.read_bytes())
            else:
                im = cv2.imread(str(img_lookup[stem]))
                if im is None:
                    return None
                h, w = im.shape[:2]
                s = args.long / max(h, w)
                if s < 1.0:
                    im = cv2.resize(im, (round(w * s), round(h * s)),
                                    interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(out_img), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out_lbl_dir / f"{stem}.txt").write_text(
            "\n".join(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                      for c, cx, cy, bw, bh in labels) + "\n")
        return str(out_img)

    written = defaultdict(list)
    cls_hist = defaultdict(Counter)
    for i, rec in enumerate(keep):
        sp = split_of[group_key(rec[0])]
        p = emit(rec, dst / "images", dst / "labels")
        if p is None:
            continue
        written[sp].append(p)
        for c, *_ in rec[1]:
            cls_hist[sp][NAMES[c]] += 1
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(keep)} ...", flush=True)

    hold_paths = []
    for i, rec in enumerate(holdout_imgs):
        p = emit(rec, dst / "holdout/images", dst / "holdout/labels")
        if p:
            hold_paths.append(p)
            for c, *_ in rec[1]:
                cls_hist["holdout"][NAMES[c]] += 1
        if (i + 1) % 200 == 0:
            print(f"  holdout {i + 1}/{len(holdout_imgs)} ...", flush=True)

    for sp in ("train", "val", "test"):
        (dst / f"{sp}.txt").write_text("\n".join(written[sp]) + "\n")
    (dst / "holdout.txt").write_text("\n".join(hold_paths) + "\n")

    (dst / "data.yaml").write_text(
        f"# SH17 -> compliance, v2 (bare-head negatives, group-disjoint split)\n"
        f"path: {dst.as_posix()}\ntrain: train.txt\nval: val.txt\ntest: test.txt\n"
        f"nc: 4\nnames: {NAMES}\n")
    (dst / "data_holdout.yaml").write_text(
        f"# Close-up head-only images REJECTED by the geometry filter. Never trained\n"
        f"# on by any model built from data.yaml -> a real distribution-shift test.\n"
        f"path: {dst.as_posix()}\ntrain: holdout.txt\nval: holdout.txt\ntest: holdout.txt\n"
        f"nc: 4\nnames: {NAMES}\n")

    summary = {
        "person_bearing_images": n_person_imgs,
        "dropped_unresolvable": n_unresolvable,
        "kept_ppe_annotated": len(ppe_imgs),
        "kept_barehead_confirmed": len(head_imgs),
        "closeup_holdout": len(hold_paths),
        "area_max": args.area_max,
        "groups": n,
        "counts": {sp: len(written[sp]) for sp in ("train", "val", "test")},
        "class_balance": {sp: dict(c) for sp, c in cls_hist.items()},
    }
    (dst / "map_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {dst}/data.yaml and {dst}/data_holdout.yaml")


if __name__ == "__main__":
    main()
