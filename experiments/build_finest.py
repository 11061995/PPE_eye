#!/usr/bin/env python3
"""
build_finest.py - assemble the training set for the final PPE-EYE checkpoint.

The recipe sweeps of Weeks 1-3 and the architecture ablations of Week 6 all came
back flat; Week 4 found the only lever that moved the deployment metric was the
training data. So the "finest version" of PPE-EYE is defined by its data, and
this script is where that definition lives.

Composition
-----------
  Pictor train              727 imgs  - in-domain for the fixed test set
  SH17-v2 train             528 imgs  - adds confirmed bare-head violations and
                                        the WHV/WV instances Pictor barely has

val  = Pictor val   (154 imgs, 6.49 persons/img)
test = Pictor test  (316 imgs, 3.16 persons/img)

val and test stay **pure Pictor and unchanged from Week 1**, so every number the
final model produces is directly comparable to all 21 earlier rows. Nothing from
SH17-v2's val or test split enters training - those are held out as gates.

Why SH17-v2 rather than the Week-4 mix
--------------------------------------
The Week-4 mix used sh17_map.py's 498 PPE-annotated images. sh17_map_v2.py adds
319 images whose workers are confirmed bare-headed by SH17's own `head` class,
filtered to deployment geometry (2.68 persons/img at 1.21% median box area,
against Pictor test's 3.16 at 1.6%). That matters because the measured failure
mode of every PPE-EYE checkpoint so far is **under-alarming** out of domain
(HEAD_TO_HEAD.md: 0.151 violation recall on SH17), and a confirmed bare head is
exactly the evidence a model needs to say "no helmet" instead of staying silent.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

NAMES = ["W", "WH", "WHV", "WV"]
ROOT = Path(__file__).resolve().parent.parent


def image_list(entry: Path) -> list[Path]:
    if entry.suffix.lower() == ".txt":
        return [Path(x.strip()) for x in entry.read_text().splitlines() if x.strip()]
    return sorted(p for p in entry.rglob("*")
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})


def label_for(p: Path) -> Path:
    parts = list(p.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def tally(imgs: list[Path]) -> tuple[Counter, int]:
    c, n = Counter(), 0
    for im in imgs:
        lf = label_for(im)
        if not lf.exists():
            continue
        for ln in lf.read_text().splitlines():
            t = ln.split()
            if len(t) >= 5:
                c[NAMES[int(float(t[0]))]] += 1
                n += 1
    return c, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", default="data/finest")
    ap.add_argument("--pictor", default="data")
    ap.add_argument("--sh17", default="data/sh17c_v2")
    args = ap.parse_args()

    dst = ROOT / args.dst
    dst.mkdir(parents=True, exist_ok=True)
    pictor = ROOT / args.pictor
    sh17 = ROOT / args.sh17

    parts = {
        "pictor_train": image_list(pictor / "train" / "images"),
        "sh17v2_train": image_list(sh17 / "train.txt"),
    }
    train = [p for v in parts.values() for p in v]
    (dst / "train.txt").write_text("\n".join(str(p) for p in train) + "\n")

    (dst / "data.yaml").write_text(
        "# Final PPE-EYE training set - see experiments/build_finest.py\n"
        "# val/test are pure Pictor and identical to Week 1, so results stay\n"
        "# comparable with every earlier row.\n"
        f"path: {dst.as_posix()}\n"
        f"train: train.txt\n"
        f"val: {(pictor / 'val' / 'images').as_posix()}\n"
        f"test: {(pictor / 'test' / 'images').as_posix()}\n"
        f"nc: 4\nnames: {NAMES}\n")

    # A second config that differs ONLY in the validation set.
    #
    # Selecting on pure-Pictor val while training on a mix picks the checkpoint
    # that is best for one corner of the training distribution: final_s_base's
    # best.pt came from epoch 8 of 38, and its fitness fell monotonically after.
    # Pictor val is 6.49 persons/image against test's 3.16 and the mixed train
    # set's 1.77, so it is the least representative split in the repo. Adding
    # SH17-v2's val makes the early-stopping signal match what the model is
    # actually being asked to learn. `test` is untouched, so the reported number
    # stays comparable.
    val_mixed = [*image_list(pictor / "val" / "images"),
                 *image_list(sh17 / "val.txt")]
    (dst / "val_mixed.txt").write_text("\n".join(str(p) for p in val_mixed) + "\n")
    (dst / "data_mixval.yaml").write_text(
        "# Same train and test as data.yaml; val is Pictor val + SH17-v2 val so\n"
        "# checkpoint selection is not driven by the least representative split.\n"
        f"path: {dst.as_posix()}\n"
        f"train: train.txt\n"
        f"val: val_mixed.txt\n"
        f"test: {(pictor / 'test' / 'images').as_posix()}\n"
        f"nc: 4\nnames: {NAMES}\n")

    print(f"{'set':16s} {'imgs':>6s} {'boxes':>6s} {'p/img':>6s}  class balance")
    for name, imgs in [*parts.items(),
                       ("TRAIN total", train),
                       ("val (pictor)", image_list(pictor / "val" / "images")),
                       ("test (pictor)", image_list(pictor / "test" / "images")),
                       ("gate sh17v2test", image_list(sh17 / "test.txt")),
                       ("gate closeup", image_list(sh17 / "holdout.txt"))]:
        c, n = tally(imgs)
        print(f"{name:16s} {len(imgs):6d} {n:6d} {n / max(len(imgs), 1):6.2f}  "
              + " ".join(f"{k}={c.get(k, 0)}" for k in NAMES))
    print(f"\nwrote {dst / 'data.yaml'}")


if __name__ == "__main__":
    main()
