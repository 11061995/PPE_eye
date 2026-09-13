#!/usr/bin/env python3
"""
calibrate.py — detection confidence calibration (item #12).

A detection is "correct" if it matches a ground-truth box of the SAME class at
IoU >= thr, assigned greedily in descending score order (Kuppers et al. 2020).
Calibration asks whether the score predicts that correctness: among boxes scored
0.7, are 70% of them right?

Temperature scaling is fitted on the VAL split and reported on TEST. It is a
monotone map, so it CANNOT change violation recall or the recall/false-alarm
tradeoff -- ranking is untouched. What it changes is what a threshold *means*,
which is what lets an operator pick one by policy instead of by sweeping.

OPERATING REGIME: ~65% of detections above the 0.01 collection floor score below
0.067, and they are trivially calibrated (they are nearly all wrong, and say so).
An n-weighted ECE over all detections is therefore dominated by boxes no operator
would ever see. Every metric here is reported twice: over all detections, and
restricted to conf >= FIT_FLOOR, which is the regime the deployment actually
thresholds in. T is fitted on the restricted regime for the same reason.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import honest_eval as he  # noqa: E402

EPS = 1e-6
CONF_FLOOR = 0.01      # detection collection floor
FIT_FLOOR = 0.20      # operating regime: fit and report calibration here
N_BINS = 15


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def apply_T(p, T):
    """Temperature-scale a probability: sigmoid(logit(p) / T)."""
    return _sigmoid(_logit(p) / T)


def collect(weights, data, split, imgsz, device, iou_thr=0.5):
    """Return (scores, correct, cls) arrays over every detection in the split."""
    import cv2
    from ultralytics import YOLO

    cfg = yaml.safe_load(Path(data).read_text())
    root = Path(cfg.get("path", Path(data).parent))
    img_dir = Path(root / cfg[split if split in cfg else "val"])
    label_dir = img_dir.parent / "labels"
    imgs = sorted(p for p in img_dir.rglob("*")
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})

    model = YOLO(weights)
    S, Y, C = [], [], []
    for im in imgs:
        h, w = cv2.imread(str(im)).shape[:2]
        r = model.predict(str(im), imgsz=imgsz, conf=CONF_FLOOR, device=device,
                          verbose=False)[0]
        preds = sorted(
            [(float(s), int(c), list(map(float, b))) for b, c, s in zip(
                r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(),
                r.boxes.conf.cpu().numpy())],
            key=lambda t: -t[0])
        gts = he.load_gt(label_dir / (im.stem + ".txt"), w, h)
        used = set()
        for s, c, b in preds:
            best, bi = 0.0, -1
            for gi, (gc, gb) in enumerate(gts):
                if gi in used or gc != c:
                    continue
                v = he.iou(gb, b)
                if v > best:
                    best, bi = v, gi
            hit = best >= iou_thr
            if hit:
                used.add(bi)
            S.append(s)
            Y.append(1 if hit else 0)
            C.append(c)
    return np.array(S), np.array(Y), np.array(C)


def ece(scores, correct, n_bins=N_BINS):
    """Expected / maximum calibration error, equal-width bins."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    e = m = 0.0
    bins = []
    n = len(scores)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (scores > lo) & (scores <= hi) if lo > 0 else (scores >= lo) & (scores <= hi)
        k = int(sel.sum())
        if k == 0:
            bins.append({"lo": round(float(lo), 3), "hi": round(float(hi), 3),
                         "n": 0, "conf": None, "acc": None})
            continue
        conf, acc = float(scores[sel].mean()), float(correct[sel].mean())
        gap = abs(acc - conf)
        e += (k / n) * gap
        m = max(m, gap)
        bins.append({"lo": round(float(lo), 3), "hi": round(float(hi), 3), "n": k,
                     "conf": round(conf, 4), "acc": round(acc, 4),
                     "gap": round(acc - conf, 4)})
    return round(e, 4), round(m, 4), bins


def regime(scores, correct, floor=FIT_FLOOR):
    """ECE and signed bias restricted to conf >= floor (the operating regime)."""
    sel = scores >= floor
    if sel.sum() < 30:
        return {"floor": floor, "n": int(sel.sum()),
                "note": "n<30, not reported"}
    s2, y2 = scores[sel], correct[sel]
    e, m, _ = ece(s2, y2)
    return {"floor": floor, "n": int(sel.sum()), "ece": e, "mce": m,
            "mean_conf": round(float(s2.mean()), 4),
            "precision": round(float(y2.mean()), 4),
            "signed_gap": round(float(y2.mean() - s2.mean()), 4)}


def nll(scores, correct, T):
    p = np.clip(apply_T(scores, T), EPS, 1 - EPS)
    return float(-(correct * np.log(p) + (1 - correct) * np.log(1 - p)).mean())


def fit_T(scores, correct, lo=0.20, hi=6.0, iters=60):
    """Golden-section search on NLL. No scipy dependency."""
    phi = (5 ** 0.5 - 1) / 2
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = nll(scores, correct, c), nll(scores, correct, d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = nll(scores, correct, c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = nll(scores, correct, d)
    return round((a + b) / 2, 4)


def brier(scores, correct):
    return round(float(((scores - correct) ** 2).mean()), 4)


def _summary(scores, correct, names, cls, tag):
    e, m, bins = ece(scores, correct)
    per_class = {}
    for i, nm in enumerate(names):
        sel = cls == i
        if sel.sum() >= 30:          # thin-class guard: below this, ECE is noise
            ce, _, _ = ece(scores[sel], correct[sel])
            per_class[nm] = {"n": int(sel.sum()), "ece": ce,
                             "mean_conf": round(float(scores[sel].mean()), 4),
                             "precision": round(float(correct[sel].mean()), 4)}
        else:
            per_class[nm] = {"n": int(sel.sum()), "ece": None,
                             "note": "n<30, ECE not reported"}
    return {"tag": tag, "n_detections": int(len(scores)), "ece": e, "mce": m,
            "brier": brier(scores, correct), "nll": round(nll(scores, correct, 1.0), 4),
            "mean_conf": round(float(scores.mean()), 4),
            "precision": round(float(correct.mean()), 4),
            "bins": bins, "per_class": per_class}


def run(models: dict[str, str], data: str, imgsz: int, device: str,
        iou_thr: float = 0.5) -> dict:
    cfg = yaml.safe_load(Path(data).read_text())
    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]

    cache_dir = Path(__file__).resolve().parent / "results" / "calib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    out = {"data": data, "iou_thr": iou_thr, "n_bins": N_BINS,
           "conf_floor": CONF_FLOOR, "fit_floor": FIT_FLOOR,
           "names": names, "models": {}}

    for tag, w in models.items():
        print(f"\n=== {tag}: collecting val detections ===")
        vs, vy, vc = collect(w, data, "val", imgsz, device, iou_thr)
        print(f"=== {tag}: collecting test detections ===")
        ts, ty, tc = collect(w, data, "test", imgsz, device, iou_thr)

        np.savez_compressed(cache_dir / f"{tag}_dets.npz",
                            val_s=vs, val_y=vy, val_c=vc,
                            test_s=ts, test_y=ty, test_c=tc)

        # T_all: naive fit over every detection. T_op: fit on the operating
        # regime only. They differ because the sub-0.067 mass dominates T_all.
        T_all = fit_T(vs, vy)
        sel = vs >= FIT_FLOOR
        T_op = fit_T(vs[sel], vy[sel]) if sel.sum() >= 30 else None

        before = _summary(ts, ty, names, tc, "test_uncalibrated")
        after_all = _summary(apply_T(ts, T_all), ty, names, tc, "test_T_all")
        after_op = (_summary(apply_T(ts, T_op), ty, names, tc, "test_T_operating")
                    if T_op else None)

        out["models"][tag] = {
            "weights": w,
            "temperature_all_detections": T_all,
            "temperature_operating_regime": T_op,
            "fitted_on": {"split": "val", "n_detections": int(len(vs)),
                          "n_in_regime": int(sel.sum()),
                          "val_ece_before": ece(vs, vy)[0],
                          "val_ece_after_T_all": ece(apply_T(vs, T_all), vy)[0]},
            "test_before": before,
            "test_after_T_all": after_all,
            "test_after_T_operating": after_op,
            "regime": {
                "uncalibrated": regime(ts, ty),
                "T_all": regime(apply_T(ts, T_all), ty),
                "T_operating": regime(apply_T(ts, T_op), ty) if T_op else None,
            },
            "delta_ece_all": round(after_all["ece"] - before["ece"], 4),
        }
        r = out["models"][tag]["regime"]
        print(f"{tag}: T_all={T_all} T_op={T_op} | global ECE {before['ece']} -> "
              f"{after_all['ece']} | regime ECE {r['uncalibrated']['ece']} -> "
              f"{r['T_operating']['ece'] if T_op else 'n/a'}")
    return out
