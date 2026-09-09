#!/usr/bin/env python3
"""
audit.py — dataset composition + model-assisted label flags for W/WH/WHV/WV.

Week 1 already answered the headline question ("is WV=0.995 leakage?") with a
count: WV has 6 test objects, WHV has 20 — both below the <30 "AP is noise"
threshold. This is the deeper pass.

Produces:
  * composition   — per-split class counts, objects/image, box-size percentiles,
                    thin-class flags.
  * flags         — per GT/pred box: tiny, edge, aspect, no_model_support,
                    unlabeled_pred, class_disagree. Written to
                    results/w2_label_audit_flags.csv and a top-N review queue.
  * auto-clean    — conservative pass: drop GT boxes that are BOTH `tiny` AND
                    `no_model_support` (annotation dust the model cannot see).
                    Written to data/<split>/labels_clean/ for w2_eval_clean_vs_noisy.

Not a substitute for human review — the CSV is the review queue. Auto-clean only
removes boxes under a rule conservative enough to run unattended.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO

TINY_FRAC = 0.003      # box area < 0.3% of image
EDGE_PX = 2
ASPECT = 4.0
SUPPORT_IOU = 0.2      # GT considered "supported" if some low-conf pred overlaps this much
PRED_CONF = 0.05       # conf floor for support check
UNLABELED_CONF = 0.5   # high-conf pred with no GT => possible missing label


def _splits(cfg):
    return [s for s in ("train", "val", "test") if s in cfg]


def composition(data: str) -> dict:
    cfg = yaml.safe_load(Path(data).read_text())
    root = Path(cfg.get("path", Path(data).parent))
    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    out = {"names": names, "splits": {}}
    for s in _splits(cfg):
        ldir = root / cfg[s]
        ldir = (ldir.parent / "labels")
        objs, per_img, areas = Counter(), [], []
        files = sorted(ldir.glob("*.txt"))
        for f in files:
            n = 0
            for ln in f.read_text().splitlines():
                t = ln.split()
                if len(t) < 5:
                    continue
                c = int(float(t[0]))
                objs[names[c]] += 1
                areas.append(float(t[3]) * float(t[4]))
                n += 1
            per_img.append(n)
        a = np.array(areas) if areas else np.array([0.0])
        out["splits"][s] = {
            "images": len(files),
            "objects": int(sum(objs.values())),
            "per_class": {k: objs.get(k, 0) for k in names},
            "objects_per_image_mean": round(float(np.mean(per_img)), 2) if per_img else 0,
            "box_area_frac_p10_p50_p90": [round(float(x), 5) for x in np.percentile(a, [10, 50, 90])],
            "thin_classes_lt30": [k for k in names if objs.get(k, 0) < 30],
        }
    return out


def _gt_boxes(label_file: Path, w: int, h: int):
    out = []
    if not label_file.exists():
        return out
    for ln in label_file.read_text().splitlines():
        t = ln.split()
        if len(t) < 5:
            continue
        c = int(float(t[0]))
        cx, cy, bw, bh = (float(x) for x in t[1:5])
        out.append((c, [(cx - bw / 2) * w, (cy - bh / 2) * h,
                        (cx + bw / 2) * w, (cy + bh / 2) * h], (bw, bh)))
    return out


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def flag_split(weights: str, data: str, split: str, results_dir: Path) -> dict:
    cfg = yaml.safe_load(Path(data).read_text())
    root = Path(cfg.get("path", Path(data).parent))
    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    img_dir = Path(root / cfg[split])
    label_dir = img_dir.parent / "labels"
    clean_dir = img_dir.parent / "labels_clean"
    clean_dir.mkdir(exist_ok=True)

    model = YOLO(weights)
    imgs = sorted(p for p in img_dir.rglob("*")
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})

    flags, counts = [], Counter()
    dropped = kept = 0
    for im in imgs:
        r = model.predict(str(im), imgsz=640, conf=PRED_CONF, device="0", verbose=False)[0]
        h, w = r.orig_shape
        preds = [(int(c), list(map(float, b)), float(s)) for b, c, s in zip(
            r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(),
            r.boxes.conf.cpu().numpy())]
        lf = label_dir / (im.stem + ".txt")
        gts = _gt_boxes(lf, w, h)

        keep_lines = []
        raw_lines = lf.read_text().splitlines() if lf.exists() else []
        for gi, (gc, gb, (bw, bh)) in enumerate(gts):
            fl = []
            area_frac = bw * bh
            if area_frac < TINY_FRAC:
                fl.append("tiny")
            if gb[0] <= EDGE_PX or gb[1] <= EDGE_PX or gb[2] >= w - EDGE_PX or gb[3] >= h - EDGE_PX:
                fl.append("edge")
            ar = max(bw / bh, bh / bw) if bw > 0 and bh > 0 else 99
            if ar > ASPECT:
                fl.append("aspect")
            best_iou, best_pc = 0.0, None
            for pc, pb, _ in preds:
                v = _iou(gb, pb)
                if v > best_iou:
                    best_iou, best_pc = v, pc
            if best_iou < SUPPORT_IOU:
                fl.append("no_model_support")
            elif best_iou >= 0.5 and best_pc != gc:
                fl.append(f"class_disagree(gt={names[gc]},pred={names[best_pc]})")
            for f in fl:
                counts[f.split("(")[0]] += 1
                flags.append([im.name, "gt", gi, names[gc], round(area_frac, 5),
                              round(best_iou, 3), f])
            # conservative auto-clean: dust the model can't see
            if "tiny" in fl and "no_model_support" in fl:
                dropped += 1
            elif gi < len(raw_lines):
                keep_lines.append(raw_lines[gi])
                kept += 1
        (clean_dir / (im.stem + ".txt")).write_text("\n".join(keep_lines) + ("\n" if keep_lines else ""))

        for pc, pb, sc in preds:
            if sc < UNLABELED_CONF:
                continue
            if max((_iou(pb, gb) for _, gb, _ in gts), default=0.0) < SUPPORT_IOU:
                counts["unlabeled_pred"] += 1
                flags.append([im.name, "pred", -1, names[pc], "", round(sc, 3), "unlabeled_pred"])

    csv_path = results_dir / "w2_label_audit_flags.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["image", "src", "box_idx", "class", "area_frac", "iou_or_conf", "flag"])
        wtr.writerows(sorted(flags, key=lambda r: r[0]))

    return {
        "split": split,
        "flagged_boxes": len(flags),
        "flag_counts": dict(counts),
        "auto_clean": {"dropped": dropped, "kept": kept,
                       "rule": "tiny AND no_model_support",
                       "clean_label_dir": str(clean_dir)},
        "flags_csv": str(csv_path),
        "review_queue_images": sorted({r[0] for r in flags})[:200],
    }


def run(weights: str, data: str, results_dir: Path) -> dict:
    comp = composition(data)
    fl = flag_split(weights, data, "test", results_dir)
    (results_dir / "w2_label_audit_composition.json").write_text(
        json.dumps(comp, indent=2), encoding="utf-8")
    return {"eval": "label_audit", "weights": weights,
            "composition": comp, "flags": fl}
