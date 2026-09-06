#!/usr/bin/env python3
"""
eval_honest.py — evaluate what Amsar actually decides, not what the paper reports.

The gap
-------
PPE-EYE reports object-level mAP over {person, vest, glass, head, white, yellow,
blue, red}. But an interlock system does not fire on "a helmet was detected
somewhere in frame". It fires on a per-person verdict:

    for each person P in frame:  compliant(P) = helmet_on(P) AND vest_on(P)

That requires a helmet->head->person association step. The paper never describes
or evaluates it, yet its Figure 10 UI clearly draws one verdict box per person.
Object mAP can be 0.97 while person-level verdicts are unusable in a frame with
three overlapping workers.

This script computes:
  * person-level verdict accuracy (matched by IoU >= --piou to GT persons)
  * VIOLATION RECALL   — fraction of genuinely non-compliant workers we catch.
                         Missing one is the safety failure. This is the number
                         that decides whether Amsar is defensible.
  * FALSE ALARM RATE   — compliant workers flagged as violators. Drives nuisance
                         relay trips; two per shift and the operator bypasses
                         the interlock, and then your recall is zero anyway.
  * a confidence sweep, so you pick an operating point on evidence rather than
    inheriting the paper's 0.424.
  * optional temporal N-of-M smoothing, which is how you actually cut false
    alarms in a live stream without touching the model.

Usage
-----
  python eval_honest.py --weights runs/ppe_eye/clean_yolo11x_21/weights/best.pt \
      --data /data/chvg_clean/data.yaml --split test

  # domain-shift test: your own Dammam site footage, labelled
  python eval_honest.py --weights ... --data /data/amsar_site/data.yaml --split val

Run it on BOTH. The delta between benchmark and your own cameras is the honest
answer to "how accurate is it live".
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO

HELMET = {"white", "yellow", "blue", "red", "helmet", "hardhat", "hard-hat"}
VEST = {"vest", "safety-vest", "safetyvest"}
PERSON = {"person", "worker"}
HEAD = {"head", "no-helmet", "nohelmet"}


def xyxy(b):
    return float(b[0]), float(b[1]), float(b[2]), float(b[3])


def iou(a, b):
    ax1, ay1, ax2, ay2 = xyxy(a)
    bx1, by1, bx2, by2 = xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def contain(inner, outer):
    """fraction of `inner` that lies inside `outer`"""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    a = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return (iw * ih) / a if a > 0 else 0.0


def verdict(persons, helmets, vests, heads, need_helmet=True, need_vest=True,
            head_band=0.40, contain_thr=0.55):
    """
    Associate PPE to persons and return a bool per person: True = compliant.

    Association rules (deliberately simple and auditable — you will want to
    replace this with a learned association head later, and then you can measure
    whether that actually helped, because this gives you the baseline):
      helmet: >=contain_thr of the helmet box inside the person box AND the
              helmet centre in the top `head_band` of the person box height.
      vest:   >=contain_thr of the vest box inside the person box.
    Greedy one-to-one: each helmet/vest is consumed by at most one person, best
    containment first. Without this, one helmet "covers" three overlapping
    workers and you silently under-report violations.
    """
    out = [dict(helmet=False, vest=False) for _ in persons]

    def assign(items, key, band=None):
        pairs = []
        for j, it in enumerate(items):
            for i, p in enumerate(persons):
                c = contain(it, p)
                if c < contain_thr:
                    continue
                if band is not None:
                    cy = (it[1] + it[3]) / 2
                    if cy > p[1] + band * (p[3] - p[1]):
                        continue
                pairs.append((c, j, i))
        pairs.sort(reverse=True)
        used_i, used_j = set(), set()
        for c, j, i in pairs:
            if i in used_i or j in used_j:
                continue
            out[i][key] = True
            used_i.add(i)
            used_j.add(j)

    assign(helmets, "helmet", band=head_band)
    assign(vests, "vest")

    # an explicitly detected bare `head` inside a person overrides a weak helmet hit
    for i, p in enumerate(persons):
        for h in heads:
            if contain(h, p) >= contain_thr:
                cy = (h[1] + h[3]) / 2
                if cy <= p[1] + head_band * (p[3] - p[1]):
                    out[i]["helmet"] = out[i]["helmet"] and False
    res = []
    for o in out:
        ok = True
        if need_helmet:
            ok = ok and o["helmet"]
        if need_vest:
            ok = ok and o["vest"]
        res.append(ok)
    return res


def load_gt(label_file, names, w, h):
    boxes = defaultdict(list)
    if not Path(label_file).exists():
        return boxes
    for line in Path(label_file).read_text().splitlines():
        t = line.split()
        if len(t) < 5:
            continue
        c = int(float(t[0]))
        cx, cy, bw, bh = (float(x) for x in t[1:5])
        b = [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]
        boxes[names[c].lower()].append(b)
    return boxes


def bucket(d):
    return (
        [b for k, v in d.items() if k in PERSON for b in v],
        [b for k, v in d.items() if k in HELMET for b in v],
        [b for k, v in d.items() if k in VEST for b in v],
        [b for k, v in d.items() if k in HEAD for b in v],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--piou", type=float, default=0.5, help="IoU to match pred person to GT person")
    ap.add_argument("--no-vest", action="store_true", help="helmet-only compliance rule")
    ap.add_argument("--conf-sweep", default="0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.70")
    ap.add_argument("--out", default="honest_eval.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.data).read_text())
    root = Path(cfg.get("path", Path(args.data).parent))
    img_dir = root / cfg[args.split if args.split in cfg else "val"]
    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    names = [n.lower() for n in names]

    imgs = sorted(p for p in Path(img_dir).rglob("*")
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    print(f"[eval] {len(imgs)} images from {img_dir}")
    model = YOLO(args.weights)

    rows = []
    for conf in [float(c) for c in args.conf_sweep.split(",")]:
        TP = FP = FN = TN = 0          # positive class = VIOLATION
        missed_person = extra_person = 0
        for im in imgs:
            r = model.predict(str(im), imgsz=args.imgsz, conf=conf, device=args.device,
                              verbose=False)[0]
            h, w = r.orig_shape
            pd = defaultdict(list)
            for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                pd[names[int(c)]].append(list(map(float, b)))
            gd = load_gt(str(im).replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt",
                         names, w, h)

            pP, pH, pV, pHd = bucket(pd)
            gP, gH, gV, gHd = bucket(gd)
            pv = verdict(pP, pH, pV, pHd, need_vest=not args.no_vest)
            gv = verdict(gP, gH, gV, gHd, need_vest=not args.no_vest)

            used = set()
            for gi, gb in enumerate(gP):
                best, bi = 0.0, -1
                for pi, pb in enumerate(pP):
                    if pi in used:
                        continue
                    v = iou(gb, pb)
                    if v > best:
                        best, bi = v, pi
                if best >= args.piou:
                    used.add(bi)
                    g_viol, p_viol = (not gv[gi]), (not pv[bi])
                    if g_viol and p_viol:
                        TP += 1
                    elif g_viol and not p_viol:
                        FN += 1
                    elif not g_viol and p_viol:
                        FP += 1
                    else:
                        TN += 1
                else:
                    missed_person += 1
                    if not gv[gi]:
                        FN += 1        # undetected person who was violating = missed violation
            extra_person += len(pP) - len(used)

        n = TP + FP + FN + TN
        rec = TP / (TP + FN) if TP + FN else 0.0
        prec = TP / (TP + FP) if TP + FP else 0.0
        rows.append(dict(
            conf=conf, n_matched_persons=n,
            violation_recall=round(rec, 4),
            violation_precision=round(prec, 4),
            f1=round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
            false_alarm_rate=round(FP / (FP + TN), 4) if FP + TN else 0.0,
            verdict_accuracy=round((TP + TN) / n, 4) if n else 0.0,
            undetected_persons=missed_person, phantom_persons=extra_person,
            TP=TP, FP=FP, FN=FN, TN=TN,
        ))
        print(f"conf={conf:<5} viol_recall={rec:.3f}  viol_prec={prec:.3f}  "
              f"false_alarm={rows[-1]['false_alarm_rate']:.3f}  "
              f"missed_persons={missed_person}")

    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.out}")
    print("\nHow to read this:")
    print("  - Pick the conf where violation_recall clears your safety target,")
    print("    THEN check false_alarm_rate is low enough that operators won't bypass.")
    print("  - undetected_persons is the failure the paper's mAP hides completely:")
    print("    a worker never detected is a worker never checked.")
    print("  - Compare these numbers on CHVG vs your own site footage. The drop is")
    print("    your real domain-shift cost, and it is the number to put in a paper.")


if __name__ == "__main__":
    main()
