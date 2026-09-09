#!/usr/bin/env python3
"""
honest_eval.py — person-level compliance verdicts for the W/WH/WHV/WV ontology.

Unlike the repo-root eval_honest.py (which is built for the CHVG 8-class scheme
and needs a helmet->head->person association step), this dataset's classes ARE
the per-person verdict:

    W    worker, no PPE resolved
    WH   worker + helmet
    WHV  worker + helmet + vest   <- the only fully-compliant state
    WV   worker + vest (no helmet)

So the "association" problem disappears: every GT box already carries a verdict.
This script matches predicted boxes to GT boxes by IoU and scores the verdict.

Compliance rule
---------------
  strict  : compliant iff class == WHV            (helmet AND vest)
  helmet  : compliant iff class in {WH, WHV}      (helmet only; vest ignored)

Report (positive class = VIOLATION, i.e. non-compliant worker):
  violation_recall      TP / (TP + FN)  — non-compliant workers caught. Safety metric.
  violation_precision   TP / (TP + FP)
  false_alarm_rate      FP / (FP + TN)  — compliant workers flagged. Nuisance-trip metric.
  verdict_accuracy      (TP + TN) / matched
  undetected_persons    GT persons never detected (never checked) at this conf
  phantom_persons       predicted persons with no GT match

Caveat baked into this dataset: only 20 test boxes are WHV and 6 are WV, so the
false_alarm_rate (which depends on compliant-worker count) is estimated from a
tiny sample under the strict rule. The helmet rule is far better supported
(~537 helmet-on vs ~462 helmet-off test boxes).

Usage
-----
  PPE/Scripts/python.exe experiments/honest_eval.py \
      --weights runs/ppe_enh/w1_s_adamw_1e4/weights/best.pt \
      --data data/data.yaml --split test --rule strict
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml
from ultralytics import YOLO

RULES = {"strict": {"WHV"}, "helmet": {"WH", "WHV"}}
DEFAULT_SWEEP = "0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.70"


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt(label_file: Path, w: int, h: int):
    out = []
    if not label_file.exists():
        return out
    for line in label_file.read_text().splitlines():
        t = line.split()
        if len(t) < 5:
            continue
        c = int(float(t[0]))
        cx, cy, bw, bh = (float(x) for x in t[1:5])
        out.append((c, [(cx - bw / 2) * w, (cy - bh / 2) * h,
                        (cx + bw / 2) * w, (cy + bh / 2) * h]))
    return out


def run(weights: str, data: str, split: str, rule: str, imgsz: int,
        device: str, iou_thr: float, sweep: list[float],
        label_dir_name: str = "labels", predictor=None) -> dict:
    """
    predictor: optional callable(image_path) -> list[(cls:int, [x1,y1,x2,y2], score:float)].
    Default = plain full-frame YOLO. Pass a SAHI sliced predictor for the
    upper-bound row. Predictions are gathered once per image at the lowest sweep
    conf, then filtered per threshold.
    """
    cfg = yaml.safe_load(Path(data).read_text())
    root = Path(cfg.get("path", Path(data).parent))
    img_dir = root / cfg[split if split in cfg else "val"]
    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    compliant_ids = {i for i, n in enumerate(names) if n in RULES[rule]}

    img_dir = Path(img_dir)
    label_dir = img_dir.parent / label_dir_name
    imgs = sorted(p for p in img_dir.rglob("*")
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    conf_floor = min(sweep)

    if predictor is None:
        model = YOLO(weights)

        def predictor(im):
            r = model.predict(str(im), imgsz=imgsz, conf=conf_floor, device=device,
                              verbose=False)[0]
            return [(int(c), list(map(float, b)), float(s)) for b, c, s in zip(
                r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(),
                r.boxes.conf.cpu().numpy())]

    import cv2
    cache = {im: predictor(im) for im in imgs}           # im -> [(cls, box, score)]
    dims = {im: cv2.imread(str(im)).shape[:2] for im in imgs}  # im -> (h, w)

    rows = []
    for conf in sweep:
        TP = FP = FN = TN = 0
        undetected = phantom = 0
        for im in imgs:
            h, w = dims[im]
            preds = [(c, b) for (c, b, s) in cache[im] if s >= conf]
            gts = load_gt(label_dir / (im.stem + ".txt"), w, h)

            used = set()
            for gc, gb in gts:
                best, bi = 0.0, -1
                for pi, (pc, pb) in enumerate(preds):
                    if pi in used:
                        continue
                    v = iou(gb, pb)
                    if v > best:
                        best, bi = v, pi
                g_viol = gc not in compliant_ids
                if best >= iou_thr:
                    used.add(bi)
                    p_viol = preds[bi][0] not in compliant_ids
                    if g_viol and p_viol:
                        TP += 1
                    elif g_viol and not p_viol:
                        FN += 1
                    elif not g_viol and p_viol:
                        FP += 1
                    else:
                        TN += 1
                else:
                    undetected += 1
                    if g_viol:
                        FN += 1  # missed violation: never detected, never checked
            phantom += len(preds) - len(used)

        n = TP + FP + FN + TN
        rec = TP / (TP + FN) if TP + FN else 0.0
        prec = TP / (TP + FP) if TP + FP else 0.0
        rows.append(dict(
            conf=round(conf, 2), rule=rule, matched=n,
            violation_recall=round(rec, 4),
            violation_precision=round(prec, 4),
            f1=round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
            false_alarm_rate=round(FP / (FP + TN), 4) if FP + TN else 0.0,
            verdict_accuracy=round((TP + TN) / n, 4) if n else 0.0,
            undetected_persons=undetected, phantom_persons=phantom,
            TP=TP, FP=FP, FN=FN, TN=TN,
        ))
        print(f"conf={conf:<5} rule={rule:<6} viol_recall={rec:.3f} "
              f"viol_prec={prec:.3f} false_alarm={rows[-1]['false_alarm_rate']:.3f} "
              f"undetected={undetected}")

    best = max(rows, key=lambda r: r["f1"])
    return {"weights": weights, "split": split, "rule": rule, "iou_thr": iou_thr,
            "n_images": len(imgs), "sweep": rows, "best_f1_row": best}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="data/data.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--rule", default="strict", choices=list(RULES))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--conf-sweep", default=DEFAULT_SWEEP)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    sweep = [float(c) for c in args.conf_sweep.split(",")]
    res = run(args.weights, args.data, args.split, args.rule, args.imgsz,
              args.device, args.iou, sweep)
    out = args.out or f"honest_eval_{args.rule}_{args.split}.json"
    Path(out).write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out}")
    b = res["best_f1_row"]
    print(f"best F1 @ conf={b['conf']}: viol_recall={b['violation_recall']} "
          f"false_alarm={b['false_alarm_rate']} verdict_acc={b['verdict_accuracy']}")


if __name__ == "__main__":
    main()
