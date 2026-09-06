#!/usr/bin/env python3
"""
audit_split.py — interrogate an EXISTING Roboflow train/valid/test export.

Run this before you train anything. No GPU, no torch, ~1-3 min on 1,700 images.
Windows-friendly (no symlinks, no POSIX assumptions).

Expects the layout Roboflow gives you:

    <root>/
      data.yaml          (optional but preferred - gives real class names)
      train/images  train/labels
      valid/images  valid/labels
      test/images   test/labels

What it answers
---------------
1. LEAKAGE. Does any source photo appear on both sides of a split boundary,
   either as an augmented sibling (shared filename stem) or as a perceptual
   near-duplicate? This is the question that decides whether PPE-EYE's
   mAP50 = 0.969 is believable.

   Roboflow's DEFAULT generate flow splits first, then augments train only.
   If that is what happened, cross-split collisions will be ZERO and the
   leakage hypothesis is dead - which is a useful thing to know before you
   spend a week on it.

2. EXPANSION. images-per-source-group, per split. Train >> valid/test means
   augmentation was applied to train only (healthy). Roughly equal expansion
   across all three splits means augmentation happened BEFORE splitting
   (leaky), because valid/test should never contain augmented copies.

3. LABEL SCHEME. Which classes exist, so you know now whether you are on the
   CHVG scheme (person/vest/head/white/yellow/blue/red -> comparable to the
   paper, compliance must be inferred) or the explicit-negative scheme
   (Hardhat/NO-Hardhat/... -> easier, NOT comparable to the paper).

4. CLASS BALANCE per split, plus classes that are absent or near-absent from
   test. An AP for a class with 4 test instances is noise, not a measurement.

5. BROKEN PAIRS. Images with no label file, labels with no image, empty labels,
   out-of-range class ids, malformed coordinates.

Usage
-----
    python audit_split.py --root "C:/data/ppe"
    python audit_split.py --root /data/ppe --hamming 6 --json audit.json

Reading the verdict
-------------------
    cross_split_stem_collisions > 0   -> definite leakage, re-split required
    cross_split_near_duplicates > 0   -> probable leakage (or burst frames from
                                         the same clip, which is nearly as bad)
    both zero                          -> the split is honest. If mAP50 is still
                                         ~0.97, the benchmark is just easy, and
                                         the real story is domain shift on your
                                         own cameras.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:
    raise SystemExit("need Pillow:  pip install pillow numpy")

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")
ALIASES = {"valid": ("valid", "val", "validation")}

CHVG_HINT = {"person", "vest", "glass", "head", "white", "yellow", "blue", "red"}
NEG_HINT = {"no-hardhat", "no-safety vest", "no-mask", "no-vest", "no-helmet"}


# ------------------------------------------------------------------ helpers
def dhash(path, size=8):
    """64-bit difference hash: stable under brightness/exposure jitter."""
    try:
        im = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    except Exception:
        return None
    a = np.asarray(im, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def popcount(x):
    return bin(x).count("1")


def stem_of(name):
    """'foo_jpg.rf.deadbeef.jpg' -> 'foo'  (Roboflow augmented-sibling key)"""
    for mk in ("_jpg.rf.", "_jpeg.rf.", "_png.rf.", ".rf."):
        if mk in name:
            return name.split(mk)[0]
    return Path(name).stem


def find_split_dir(root, split):
    for cand in ALIASES.get(split, (split,)):
        d = root / cand / "images"
        if d.is_dir():
            return d
    return None


def read_names(root):
    y = root / "data.yaml"
    if not y.exists():
        return None
    txt = y.read_text(encoding="utf-8", errors="ignore")
    if "names:" not in txt:
        return None
    seg = txt.split("names:", 1)[1]
    if "[" in seg and "]" in seg.split("\n\n")[0] + "]":
        try:
            inner = seg[seg.index("[") + 1: seg.index("]")]
            return [x.strip().strip("'\"") for x in inner.split(",") if x.strip()]
        except ValueError:
            pass
    # block style:  names:\n  - person\n  - vest
    out = []
    for line in seg.splitlines()[1:]:
        s = line.strip()
        if s.startswith("- "):
            out.append(s[2:].strip().strip("'\""))
        elif s and not s.startswith("#"):
            break
    return out or None


def label_for(img):
    return img.parent.parent / "labels" / (img.stem + ".txt")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--hamming", type=int, default=6,
                    help="dHash distance at or below which two images are 'the same photo'")
    ap.add_argument("--json", default="")
    ap.add_argument("--list-collisions", type=int, default=8,
                    help="how many colliding examples to print")
    args = ap.parse_args()

    root = args.root
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    names = read_names(root)
    print(f"root: {root.resolve()}")
    print(f"data.yaml names: {names if names else 'NOT FOUND (class ids only)'}\n")

    # ---- inventory
    imgs, split_of = [], {}
    for sp in SPLITS:
        d = find_split_dir(root, sp)
        if d is None:
            print(f"[warn] no {sp}/images directory")
            continue
        found = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)
        for p in found:
            split_of[len(imgs)] = sp
            imgs.append(p)
        print(f"[scan] {sp:<6} {len(found):>6} images   ({d})")
    if not imgs:
        raise SystemExit("no images found - check --root")
    print(f"[scan] {'TOTAL':<6} {len(imgs):>6}\n")

    # ---- labels: integrity + class counts
    cls_count = {sp: defaultdict(int) for sp in SPLITS}
    obj_total = {sp: 0 for sp in SPLITS}
    missing_label, empty_label, bad_line = [], [], []
    max_cls = -1
    for i, p in enumerate(imgs):
        sp = split_of[i]
        lp = label_for(p)
        if not lp.exists():
            missing_label.append(p)
            continue
        try:
            lines = [l for l in lp.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]
        except Exception:
            bad_line.append(lp)
            continue
        if not lines:
            empty_label.append(p)
            continue
        for l in lines:
            t = l.split()
            if len(t) < 5:
                bad_line.append(lp)
                continue
            try:
                c = int(float(t[0]))
                vals = [float(x) for x in t[1:5]]
            except ValueError:
                bad_line.append(lp)
                continue
            if not all(-0.01 <= v <= 1.01 for v in vals):
                bad_line.append(lp)
            cls_count[sp][c] += 1
            obj_total[sp] += 1
            max_cls = max(max_cls, c)

    orphan_labels = []
    for sp in SPLITS:
        d = find_split_dir(root, sp)
        if d is None:
            continue
        ld = d.parent / "labels"
        if not ld.is_dir():
            continue
        stems = {p.stem for p in d.iterdir() if p.suffix.lower() in IMG_EXT}
        orphan_labels += [p for p in ld.glob("*.txt") if p.stem not in stems]

    # ---- label scheme
    if names:
        low = {n.lower() for n in names}
        if low & NEG_HINT:
            scheme = "EXPLICIT-NEGATIVE (Hardhat / NO-Hardhat style)"
            note = ("live_check.py -> MODE A. Model outputs compliance directly. "
                    "EASIER, but NOT the CHVG scheme PPE-EYE used, so your mAP is "
                    "not comparable to their 0.969.")
        elif len(low & CHVG_HINT) >= 5:
            scheme = "CHVG (person/vest/glass/head + 4 helmet colours)"
            note = ("live_check.py -> MODE B. Comparable to the paper, but compliance "
                    "must be INFERRED by helmet->person association, which the paper "
                    "never evaluates. Four helmet-colour classes split your data 4 ways "
                    "for zero safety value: consider collapsing them to one 'helmet'.")
        else:
            scheme = "OTHER / custom"
            note = "Check which of MODE A / MODE B live_check.py reports at startup."
    else:
        scheme, note = "UNKNOWN (no data.yaml)", "Add data.yaml or pass --names to the other scripts."

    # ---- grouping by filename stem
    stem_splits = defaultdict(set)
    stem_members = defaultdict(list)
    for i, p in enumerate(imgs):
        s = stem_of(p.name)
        stem_splits[s].add(split_of[i])
        stem_members[s].append(i)
    stem_collide = {s: sorted(v) for s, v in stem_splits.items() if len(v) > 1}

    per_split_groups = {sp: len({stem_of(imgs[i].name)
                                 for i in split_of if split_of[i] == sp}) for sp in SPLITS}

    # ---- perceptual near-duplicate scan
    print("[hash] computing perceptual hashes ...")
    hashes = [dhash(p) for p in imgs]
    n_hash_fail = sum(1 for h in hashes if h is None)
    buckets = defaultdict(list)
    for i, h in enumerate(hashes):
        if h is not None:
            buckets[h >> 48].append(i)   # compare only plausible candidates
    dup_cross, dup_within = [], 0
    for idxs in buckets.values():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                if popcount(hashes[i] ^ hashes[j]) <= args.hamming:
                    if split_of[i] != split_of[j]:
                        dup_cross.append((i, j))
                    else:
                        dup_within += 1

    # ================================================================ report
    print("\n" + "=" * 72)
    print("LABEL SCHEME")
    print("=" * 72)
    print(f"  {scheme}")
    print(f"  -> {note}")

    print("\n" + "=" * 72)
    print("EXPANSION FACTOR  (images per source group)")
    print("=" * 72)
    print(f"  {'split':<8}{'images':>8}{'groups':>8}{'expansion':>12}")
    exp = {}
    for sp in SPLITS:
        n = sum(1 for i in split_of if split_of[i] == sp)
        g = per_split_groups[sp] or 1
        exp[sp] = n / g
        print(f"  {sp:<8}{n:>8}{per_split_groups[sp]:>8}{n/g:>12.2f}")
    tr_exp = exp.get("train", 0)
    tv = max(exp.get("valid", 0), exp.get("test", 0))
    print()
    if tr_exp <= 1.05 and tv <= 1.05:
        print("  READ: no augmentation anywhere (raw export). You must generate")
        print("        augmentations yourself - AFTER splitting, and on train only.")
    elif tr_exp > 1.15 and tv <= 1.05:
        print("  READ: train augmented, valid/test clean. This is Roboflow's DEFAULT")
        print("        and it is the CORRECT setup. No augmentation leakage here.")
    elif tv > 1.05:
        print(f"  READ: valid/test hold >1 image per source group (max {tv:.2f}).")
        print("        Augmented copies are sitting in the eval sets, which means")
        print("        augmentation ran BEFORE the split. LEAKY - see next section.")
    else:
        print(f"  READ: ambiguous (train {tr_exp:.2f}, eval {tv:.2f}). Trust the")
        print("        LEAKAGE section below over this heuristic - it inspects")
        print("        actual cross-split collisions rather than ratios.")

    print("\n" + "=" * 72)
    print("LEAKAGE")
    print("=" * 72)
    print(f"  cross-split stem collisions   {len(stem_collide)}")
    print(f"  cross-split near-duplicates   {len(dup_cross)}  (dHash <= {args.hamming})")
    print(f"  within-split near-duplicates  {dup_within}  (not leakage, but inflates n)")
    if n_hash_fail:
        print(f"  unreadable images             {n_hash_fail}")

    if stem_collide:
        print(f"\n  examples (same source photo, different splits):")
        for s, v in list(stem_collide.items())[:args.list_collisions]:
            print(f"    {s[:52]:<52} {v}")
    if dup_cross:
        print(f"\n  examples (visually near-identical, different splits):")
        for i, j in dup_cross[:args.list_collisions]:
            print(f"    [{split_of[i]:<5}] {imgs[i].name[:38]}")
            print(f"    [{split_of[j]:<5}] {imgs[j].name[:38]}")
            print(f"      hamming={popcount(hashes[i]^hashes[j])}")

    leak = len(stem_collide) + len(dup_cross)
    print("\n  VERDICT: ", end="")
    if leak == 0:
        print("no cross-split leakage detected.")
        print("    The existing split is honest. Train on it as-is; skip the")
        print("    re-split. If mAP50 still lands near 0.97, the benchmark is")
        print("    simply easy, and the interesting number is what happens on")
        print("    YOUR cameras. My leakage hypothesis about the paper was wrong.")
    else:
        print(f"{leak} leaking pairs/groups. The existing split OVERSTATES accuracy.")
        print("    Re-split before you believe any number:")
        print(f"      python split_no_leak.py --src \"{root}\" --dst \"{root}_clean\" --copy")

    print("\n" + "=" * 72)
    print("CLASS BALANCE  (object counts)")
    print("=" * 72)
    allc = sorted({c for sp in SPLITS for c in cls_count[sp]})
    print(f"  {'id':<4}{'name':<18}{'train':>9}{'valid':>9}{'test':>9}{'test %':>9}")
    thin = []
    for c in allc:
        nm = names[c] if names and c < len(names) else f"class{c}"
        tr, va, te = cls_count['train'][c], cls_count['valid'][c], cls_count['test'][c]
        tot = tr + va + te
        print(f"  {c:<4}{nm:<18}{tr:>9}{va:>9}{te:>9}{100*te/max(tot,1):>8.1f}%")
        if te < 30:
            thin.append((nm, te))
    print(f"  {'':<4}{'TOTAL':<18}{obj_total['train']:>9}{obj_total['valid']:>9}{obj_total['test']:>9}")
    if names and max_cls >= len(names):
        print(f"\n  ERROR: label uses class id {max_cls} but data.yaml lists {len(names)} names.")
    if thin:
        print("\n  THIN TEST CLASSES (<30 objects) - per-class AP here is noise:")
        for nm, n in thin:
            print(f"    {nm:<18}{n:>5} test objects")
        print("    The paper's weakest class (`glass`) had 51 source instances total.")
        print("    Any AP you compute for such a class carries a huge error bar, and")
        print("    the paper reports no confidence intervals at all.")

    print("\n" + "=" * 72)
    print("INTEGRITY")
    print("=" * 72)
    print(f"  images with no label file     {len(missing_label)}")
    print(f"  images with EMPTY label       {len(empty_label)}  (valid as background negatives)")
    print(f"  labels with no image          {len(orphan_labels)}")
    print(f"  malformed label lines in      {len(set(bad_line))} files")
    for lbl, lst in (("no-label", missing_label), ("orphan-label", orphan_labels),
                     ("malformed", sorted(set(bad_line)))):
        for p in lst[:3]:
            print(f"    {lbl}: {p.name}")

    if args.json:
        Path(args.json).write_text(json.dumps(dict(
            root=str(root), scheme=scheme, names=names,
            images={sp: sum(1 for i in split_of if split_of[i] == sp) for sp in SPLITS},
            groups=per_split_groups,
            expansion={k: round(v, 3) for k, v in exp.items()},
            cross_split_stem_collisions=len(stem_collide),
            cross_split_near_duplicates=len(dup_cross),
            within_split_near_duplicates=dup_within,
            objects=obj_total,
            class_counts={sp: dict(cls_count[sp]) for sp in SPLITS},
            missing_label=len(missing_label), empty_label=len(empty_label),
            orphan_labels=len(orphan_labels), malformed_files=len(set(bad_line)),
            leaking_pairs=leak,
        ), indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")

    print("\nNEXT")
    print("  leakage == 0 -> train on this split directly:")
    print("     yolo detect train model=yolo11s.pt data=<root>/data.yaml epochs=50 imgsz=640 batch=16")
    print("  leakage  > 0 -> re-split first with split_no_leak.py --copy, then train both")
    print("                  and report the delta. That delta is a publishable result.")


if __name__ == "__main__":
    main()
