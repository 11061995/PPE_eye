#!/usr/bin/env python3
"""
sahi_eval.py — sliced-inference upper bound (plan item #4).

Week 2's honest eval showed ~27-58% of test workers are never detected, driven
by the train(close-up) vs test(crowded/distant) mismatch. SAHI slices each frame
and runs the detector per tile, which is the standard recovery for small distant
objects. This measures how much of that undetected-person gap slicing closes —
an upper-bound row, not a deployable config (too slow for 8 streams continuously).

Reuses experiments/honest_eval.run() via its `predictor` hook, so the person-
level metrics are computed identically to the full-frame baseline.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import honest_eval as he


def make_predictor(weights: str, conf_floor: float, device: str,
                   slice_px: int, overlap: float):
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    dev = "cuda:0" if device in ("0", "cuda", "cuda:0") else device
    model = None
    for mt in ("ultralytics", "yolov8"):
        try:
            model = AutoDetectionModel.from_pretrained(
                model_type=mt, model_path=weights,
                confidence_threshold=conf_floor, device=dev)
            break
        except Exception:  # noqa: BLE001
            continue
    if model is None:
        raise RuntimeError("SAHI could not load the model (tried ultralytics, yolov8)")

    def predict(im):
        res = get_sliced_prediction(
            str(im), model, slice_height=slice_px, slice_width=slice_px,
            overlap_height_ratio=overlap, overlap_width_ratio=overlap, verbose=0)
        out = []
        for op in res.object_prediction_list:
            x1, y1, x2, y2 = op.bbox.to_xyxy()
            out.append((int(op.category.id), [float(x1), float(y1), float(x2), float(y2)],
                        float(op.score.value)))
        return out

    return predict


def run(weights: str, data: str, split: str, device: str, sweep: list[float],
        results_dir: Path, slice_px: int = 320, overlap: float = 0.2,
        rules=("strict", "helmet")) -> dict:
    conf_floor = min(sweep)
    predictor = make_predictor(weights, conf_floor, device, slice_px, overlap)

    out = {"eval": "sahi_upperbound", "weights": weights, "split": split,
           "slice_px": slice_px, "overlap": overlap, "rules": {}}
    t0 = time.time()
    for rule in rules:
        res = he.run(weights, data, split, rule, 640, device, 0.5, sweep,
                     predictor=predictor)
        (results_dir / f"w2_sahi_upperbound_{rule}.json").write_text(
            json.dumps(res, indent=2), encoding="utf-8")
        b = res["best_f1_row"]
        out["rules"][rule] = {
            "conf": b["conf"],
            "violation_recall": b["violation_recall"],
            "false_alarm_rate": b["false_alarm_rate"],
            "verdict_accuracy": b["verdict_accuracy"],
            "undetected_persons": b["undetected_persons"],
            "min_undetected": min(r["undetected_persons"] for r in res["sweep"]),
            "max_recall": max(r["violation_recall"] for r in res["sweep"]),
        }
    out["minutes"] = round((time.time() - t0) / 60, 1)
    return out
