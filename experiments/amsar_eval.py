#!/usr/bin/env python3
"""
amsar_eval.py - run AMSAR's actual detection layer over the Pictor test split
and score it with the SAME person-level harness used for PPE-EYE.

Why this exists
---------------
PPE-EYE and Amsar solve the same problem with different algorithms:

  PPE-EYE   one model, 4-class person-level ontology (W/WH/WHV/WV).
            The class IS the compliance verdict.

  Amsar     two-model cascade:
              1. stock COCO yolov8n.pt  -> person boxes            (CONF = 0.4)
              2. ppe.pt (6 PPE classes) -> helmet/vest/glove boxes (PPE_HAT_CONF = 0.40)
              3. `_associate_ppe()`     -> pure geometry maps items to persons
            Compliance is INFERRED from the presence/absence of a PPE box.
            ppe.pt has NO negative class, so "no helmet" and "helmet missed"
            are the same signal.

Comparing them needs both to emit the same thing. This script wraps Amsar's
cascade in a predictor that outputs W/WH/WHV/WV person boxes, then hands it to
honest_eval.run() - identical images, identical IoU matching, identical rule,
identical metrics. Any difference in the numbers is a difference in the
algorithm, not the yardstick.

Fidelity notes (read before quoting any number)
-----------------------------------------------
* `_associate_ppe` and `_derive_ppe_classes` below are COPIED VERBATIM from
  Amsar's cameras.py / ppe_models.py. Diff them if you doubt it.
* The conf sweep sweeps the PERSON stage only. Amsar's PPE stage has its own
  fixed constant (PPE_HAT_CONF = 0.40) and is held there throughout, which is
  what the deployed code does.
* No temporal smoothing. PPESmoother is a video-only mechanism and the test
  split is stills. This is not a handicap: a freshly-spawned track in
  PPESmoother starts at items=[None,0], and _step(None, 0, False) returns
  False immediately - so for any worker ENTERING frame, Amsar's live verdict
  is exactly the un-smoothed verdict measured here. The smoother only protects
  workers already confirmed compliant on a previous frame.
* `--capture amsar` reproduces the FrameReader's 800x448 decode before
  inference (boxes are scaled back to native coords for scoring). `native`
  skips it, which is the generous reading.

Usage
-----
  PPE/Scripts/python.exe experiments/amsar_eval.py --rule helmet
  PPE/Scripts/python.exe experiments/amsar_eval.py --rule helmet --capture amsar
  PPE/Scripts/python.exe experiments/amsar_eval.py --rule helmet --assoc greedy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
import honest_eval  # noqa: E402

AMSAR_DIR = Path(r"C:\Users\user\Desktop\khizar\amsar-AI\Amsar-AI-")

# -- Amsar constants, copied from cameras.py -----------------------
AMSAR_W, AMSAR_H = 800, 448   # FrameReader decode size
PPE_HAT_CONF = 0.40           # Amsar holds the PPE stage here regardless of person conf
VEST_COVERAGE = 0.12
_VEST_RANGES = [
    (np.array([18, 80, 80]), np.array([35, 255, 255])),   # yellow
    (np.array([5, 80, 80]), np.array([18, 255, 255])),    # orange
    (np.array([36, 80, 80]), np.array([90, 255, 255])),   # lime / hi-vis green
]

# -- VERBATIM from Amsar ppe_models.py -----------------------------
_NO_HAT_KW = ("no-hardhat", "no_hardhat", "no-helmet", "no_helmet",
              "nohardhat", "nohelmet", "no-hat", "no_hat", "head")
_HAT_KW = ("hardhat", "helmet", "hat")
_VEST_KW = ("vest", "hi-vis", "hivis")
_NO_VEST_KW = ("no-vest", "no_vest", "novest")
_GLOVE_KW = ("glove",)
_NO_GLOVE_KW = ("no-glove", "no_glove", "noglove", "bare")


def _derive_ppe_classes(names: dict):
    """VERBATIM from Amsar ppe_models.py."""
    hat, no_hat, vest, glove = set(), set(), set(), set()
    for cid, nm in names.items():
        n, cid = str(nm).lower(), int(cid)
        neg = n.startswith(("no-", "no_", "no ")) or any(k in n for k in _NO_VEST_KW + _NO_GLOVE_KW)
        if any(k in n for k in _NO_HAT_KW):
            no_hat.add(cid)
        elif any(k in n for k in _HAT_KW):
            hat.add(cid)
        if neg:
            continue
        if any(k in n for k in _VEST_KW):
            vest.add(cid)
        if any(k in n for k in _GLOVE_KW):
            glove.add(cid)
    return hat, no_hat, vest, glove


# -- VERBATIM from Amsar cameras.py (_associate_ppe) ----------------
def _associate_ppe(frame, person_boxes, ppe_boxes_by_role):
    """Amsar's association, unchanged: first PPE box whose CENTRE falls in the
    person's sub-region wins, and the box is NOT consumed - so one helmet can
    satisfy several overlapping persons."""
    hat_boxes, vest_boxes, glove_boxes, vest_cls_present = ppe_boxes_by_role
    results = []
    for x1, y1, x2, y2 in person_boxes:
        ph = y2 - y1

        # hat: hat-box centre in person's top 45 %
        has_hat = False
        head_y2 = y1 + ph * 0.45
        for hx1, hy1, hx2, hy2 in hat_boxes:
            cx, cy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
            if x1 <= cx <= x2 and y1 <= cy <= head_y2:
                has_hat = True
                break

        # vest: model detection (combined) OR HSV colour fallback
        has_vest = False
        if vest_cls_present:
            mid_y1, mid_y2 = y1 + ph * 0.25, y1 + ph * 0.75
            for vx1, vy1, vx2, vy2 in vest_boxes:
                vcx, vcy = (vx1 + vx2) / 2, (vy1 + vy2) / 2
                if x1 <= vcx <= x2 and mid_y1 <= vcy <= mid_y2:
                    has_vest = True
                    break
        else:
            tx1, tx2 = int(x1), int(x2)
            ty1 = int(y1 + ph * 0.25)
            ty2 = int(y1 + ph * 0.70)
            crop = frame[ty1:ty2, tx1:tx2]
            if crop.size > 0:
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                mask = np.zeros(hsv.shape[:2], np.uint8)
                for lo, hi in _VEST_RANGES:
                    mask |= cv2.inRange(hsv, lo, hi)
                has_vest = cv2.countNonZero(mask) / mask.size > VEST_COVERAGE

        # gloves: glove-box centre anywhere inside the person box
        has_gloves = False
        for gx1, gy1, gx2, gy2 in glove_boxes:
            gcx, gcy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
            if x1 <= gcx <= x2 and y1 <= gcy <= y2:
                has_gloves = True
                break

        results.append((has_hat, has_vest, has_gloves))
    return results


# -- FIXED variant: greedy 1-to-1, same predicate -------------------
def _sub_iou(item_box, region):
    ix1, iy1 = max(item_box[0], region[0]), max(item_box[1], region[1])
    ix2, iy2 = min(item_box[2], region[2]), min(item_box[3], region[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = ((item_box[2] - item_box[0]) * (item_box[3] - item_box[1])
          + (region[2] - region[0]) * (region[3] - region[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def _assign_greedy(person_boxes, item_boxes, region_of):
    """One item box serves at most ONE person. Same geometric predicate as
    Amsar (centre inside the sub-region); ties broken best-overlap-first.
    This is the fix ported from sh17_map.py's containment matching."""
    n = len(person_boxes)
    pairs = []
    for pi, pb in enumerate(person_boxes):
        reg = region_of(pb)
        for ii, ib in enumerate(item_boxes):
            cx, cy = (ib[0] + ib[2]) / 2, (ib[1] + ib[3]) / 2
            if reg[0] <= cx <= reg[2] and reg[1] <= cy <= reg[3]:
                pairs.append((_sub_iou(ib, reg), pi, ii))
    pairs.sort(reverse=True)
    got, used_p, used_i = [False] * n, set(), set()
    for _, pi, ii in pairs:
        if pi in used_p or ii in used_i:
            continue
        got[pi] = True
        used_p.add(pi)
        used_i.add(ii)
    return got


def _associate_ppe_greedy(frame, person_boxes, ppe_boxes_by_role):
    hat_boxes, vest_boxes, glove_boxes, vest_cls_present = ppe_boxes_by_role
    hats = _assign_greedy(person_boxes, hat_boxes,
                          lambda b: (b[0], b[1], b[2], b[1] + (b[3] - b[1]) * 0.45))
    if vest_cls_present:
        vests = _assign_greedy(person_boxes, vest_boxes,
                               lambda b: (b[0], b[1] + (b[3] - b[1]) * 0.25,
                                          b[2], b[1] + (b[3] - b[1]) * 0.75))
    else:
        vests = [False] * len(person_boxes)
    gloves = _assign_greedy(person_boxes, glove_boxes, lambda b: tuple(b))
    return list(zip(hats, vests, gloves))


# -- verdict mapping: (hat, vest) -> W/WH/WHV/WV --------------------
def _verdict(has_hat: bool, has_vest: bool) -> int:
    #  names: 0=W  1=WH  2=WHV  3=WV
    if has_hat and has_vest:
        return 2
    if has_hat:
        return 1
    if has_vest:
        return 3
    return 0


def build_predictor(person_w, ppe_w, capture, assoc, conf_floor, device, imgsz, ppe_conf=PPE_HAT_CONF):
    person_model = YOLO(str(person_w))
    ppe_model = YOLO(str(ppe_w))
    hat_cls, no_hat_cls, vest_cls, glove_cls = _derive_ppe_classes(ppe_model.names)
    print(f"[amsar] ppe.pt classes : {ppe_model.names}")
    print(f"[amsar] derived roles  : HAT={hat_cls} NO_HAT={no_hat_cls or 'EMPTY'} "
          f"VEST={vest_cls or 'HSV-fallback'} GLOVES={glove_cls or 'skipped'}")
    print(f"[amsar] capture={capture} assoc={assoc} person_conf_floor={conf_floor} "
          f"ppe_conf={ppe_conf} (fixed)")

    associate = _associate_ppe if assoc == "firstmatch" else _associate_ppe_greedy
    diag = {"person_dets": 0, "hat_boxes": 0, "vest_boxes": 0, "images": 0}

    def predictor(im_path):
        img = cv2.imread(str(im_path))
        oh, ow = img.shape[:2]
        if capture == "amsar":
            frame = cv2.resize(img, (AMSAR_W, AMSAR_H))
            sx, sy = ow / AMSAR_W, oh / AMSAR_H
        else:
            frame, sx, sy = img, 1.0, 1.0

        pr = person_model.predict(frame, imgsz=imgsz, classes=[0], conf=conf_floor,
                                  device=device, verbose=False)[0]
        pboxes = pr.boxes.xyxy.cpu().numpy()
        pscores = pr.boxes.conf.cpu().numpy()

        qr = ppe_model.predict(frame, imgsz=imgsz, conf=ppe_conf,
                               device=device, verbose=False)[0]
        hat_b, vest_b, glove_b = [], [], []
        for box, cls in zip(qr.boxes.xyxy.cpu().numpy(), qr.boxes.cls.cpu().numpy()):
            c = int(cls)
            if c in hat_cls:
                hat_b.append(box)
            elif c in vest_cls:
                vest_b.append(box)
            elif c in glove_cls:
                glove_b.append(box)

        diag["person_dets"] += len(pboxes)
        diag["hat_boxes"] += len(hat_b)
        diag["vest_boxes"] += len(vest_b)
        diag["images"] += 1

        res = associate(frame, pboxes, (hat_b, vest_b, glove_b, bool(vest_cls)))
        out = []
        for (x1, y1, x2, y2), s, (hh, hv, _hg) in zip(pboxes, pscores, res):
            out.append((_verdict(bool(hh), bool(hv)),
                        [x1 * sx, y1 * sy, x2 * sx, y2 * sy], float(s)))
        return out

    return predictor, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person-weights", default=str(AMSAR_DIR / "yolov8n.pt"))
    ap.add_argument("--ppe-weights", default=str(AMSAR_DIR / "ppe.pt"))
    ap.add_argument("--data", default="data/data.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--rule", default="helmet", choices=list(honest_eval.RULES))
    ap.add_argument("--capture", default="native", choices=["native", "amsar"])
    ap.add_argument("--assoc", default="firstmatch", choices=["firstmatch", "greedy"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--conf-sweep", default=honest_eval.DEFAULT_SWEEP)
    ap.add_argument("--ppe-conf", type=float, default=PPE_HAT_CONF)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    sweep = [float(c) for c in args.conf_sweep.split(",")]
    predictor, diag = build_predictor(args.person_weights, args.ppe_weights,
                                      args.capture, args.assoc, min(sweep),
                                      args.device, args.imgsz, args.ppe_conf)

    res = honest_eval.run(weights=f"AMSAR[{Path(args.person_weights).name}+"
                                  f"{Path(args.ppe_weights).name}]",
                          data=args.data, split=args.split, rule=args.rule,
                          imgsz=args.imgsz, device=args.device, iou_thr=args.iou,
                          sweep=sweep, predictor=predictor)
    res["system"] = "amsar_cascade"
    res["capture"] = args.capture
    res["assoc"] = args.assoc
    res["ppe_conf_fixed"] = args.ppe_conf
    res["diagnostics"] = diag
    res["note"] = ("No temporal smoothing (stills). Person conf swept; PPE stage "
                   "held at Amsar's PPE_HAT_CONF.")

    out = args.out or f"experiments/results/amsar_{args.capture}_{args.assoc}_{args.rule}{args.tag}.json"
    Path(out).write_text(json.dumps(res, indent=2))
    print(f"\n[amsar] diagnostics: {diag}")
    print(f"wrote {out}")
    b = res["best_f1_row"]
    print(f"best F1 @ conf={b['conf']}: viol_recall={b['violation_recall']} "
          f"false_alarm={b['false_alarm_rate']} verdict_acc={b['verdict_accuracy']}")


if __name__ == "__main__":
    main()
