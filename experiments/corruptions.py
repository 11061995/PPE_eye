#!/usr/bin/env python3
"""
corruptions.py — RTSP-realistic corruption benchmark, scored at PERSON level.

Six corruptions x five severities, applied in-memory to the test images. We
score violation recall / false alarm / undetected persons, NOT object mAP:
Weeks 1-4 established that object mAP does not track deployment quality here
(w4_mixed moved violation recall +0.11 while mAP50 stayed flat at 0.49).

Only the `helmet` rule is swept. The `strict` rule's false-alarm denominator on
this test set is 4-9 persons (Week 4) — it cannot resolve a corruption effect.
"""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import honest_eval as he  # noqa: E402

# severity 1..5, index 0 unused
SEV = {
    "motion_blur":  [3, 5, 9, 13, 17],        # kernel length, px
    "defocus":      [1, 2, 3, 4, 6],          # disk radius, px
    "jpeg":         [50, 35, 25, 15, 8],      # encoder quality
    "low_light":    [0.60, 0.45, 0.30, 0.20, 0.12],   # luminance gain
    "overexposure": [1.3, 1.5, 1.8, 2.2, 2.8],        # gain before clipping
    "dust_haze":    [0.90, 0.80, 0.70, 0.55, 0.40],   # transmission t
}
NAMES = list(SEV)


def _motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle_deg, 1.0)
    k = cv2.warpAffine(k, M, (length, length))
    s = k.sum()
    return k / s if s > 0 else k


def _disk_kernel(radius: int) -> np.ndarray:
    d = 2 * radius + 1
    yy, xx = np.mgrid[:d, :d] - radius
    k = ((xx ** 2 + yy ** 2) <= radius ** 2).astype(np.float32)
    return k / k.sum()


def corrupt(img: np.ndarray, name: str, severity: int, rng: np.random.Generator) -> np.ndarray:
    """img: BGR uint8. severity 1-5. Returns BGR uint8, same shape."""
    v = SEV[name][severity - 1]
    f = img.astype(np.float32)

    if name == "motion_blur":
        # camera pan/worker motion: direction varies per frame, mostly horizontal
        angle = float(rng.uniform(-25, 25))
        return cv2.filter2D(img, -1, _motion_kernel(int(v), angle))

    if name == "defocus":
        return cv2.filter2D(img, -1, _disk_kernel(int(v)))

    if name == "jpeg":
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(v)])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img

    if name == "low_light":
        # scale luminance, then add sensor noise that grows as the signal shrinks
        out = f * v
        sigma = 3.0 + 12.0 * (1.0 - v)
        out += rng.normal(0.0, sigma, out.shape)
        return np.clip(out, 0, 255).astype(np.uint8)

    if name == "overexposure":
        return np.clip(f * v, 0, 255).astype(np.uint8)

    if name == "dust_haze":
        # atmospheric scattering: I*t + A*(1-t), A = bright airlight
        A = 200.0
        return np.clip(f * v + A * (1.0 - v), 0, 255).astype(np.uint8)

    raise ValueError(name)


def make_predictor(model, imgsz: int, device: str, conf_floor: float,
                   name: str | None, severity: int):
    """honest_eval predictor that corrupts the frame before the forward pass."""
    def predictor(im_path):
        img = cv2.imread(str(im_path))
        if name is not None:
            # seed per image path so every model sees the identical corruption
            # crc32, not hash(): str hashing is salted per process, we need
            # the same corruption every run and for every model
            seed = zlib.crc32(f"{im_path.name}|{name}|{severity}".encode())
            rng = np.random.default_rng(seed)
            img = corrupt(img, name, severity, rng)
        r = model.predict(img, imgsz=imgsz, conf=conf_floor, device=device,
                          verbose=False)[0]
        return [(int(c), list(map(float, b)), float(s)) for b, c, s in zip(
            r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(),
            r.boxes.conf.cpu().numpy())]
    return predictor


def run(models: dict[str, str], data: str, split: str, imgsz: int, device: str,
        sweep: list[float], severities: int = 5, rule: str = "helmet",
        iou_thr: float = 0.5, corruptions: list[str] | None = None) -> dict:
    """models: {tag: weights_path}. Returns nested {tag: {corruption: {sev: row}}}."""
    from ultralytics import YOLO

    corruptions = corruptions or NAMES
    conf_floor = min(sweep)
    out: dict = {"rule": rule, "split": split, "imgsz": imgsz,
                 "severities": severities, "corruptions": corruptions,
                 "models": {}}

    for tag, weights in models.items():
        m = YOLO(weights)
        print(f"\n=== {tag} : clean baseline ===")
        clean = he.run(weights, data, split, rule, imgsz, device, iou_thr, sweep,
                       predictor=make_predictor(m, imgsz, device, conf_floor, None, 0))
        entry = {"weights": weights, "clean": clean["best_f1_row"],
                 "clean_sweep": clean["sweep"], "corrupted": {}}

        for cname in corruptions:
            entry["corrupted"][cname] = {}
            for sev in range(1, severities + 1):
                print(f"\n=== {tag} : {cname} severity {sev} ===")
                res = he.run(weights, data, split, rule, imgsz, device, iou_thr,
                             sweep,
                             predictor=make_predictor(m, imgsz, device,
                                                      conf_floor, cname, sev))
                b = res["best_f1_row"]
                b["delta_violation_recall"] = round(
                    b["violation_recall"] - entry["clean"]["violation_recall"], 4)
                b["delta_undetected_persons"] = (
                    b["undetected_persons"] - entry["clean"]["undetected_persons"])
                entry["corrupted"][cname][str(sev)] = b
        out["models"][tag] = entry

    return out
