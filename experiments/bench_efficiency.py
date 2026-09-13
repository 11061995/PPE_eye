#!/usr/bin/env python3
"""
bench_efficiency.py - cost side of the Amsar vs PPE-EYE comparison.

AMSAR_COMPARISON.md measures which system is more ACCURATE. It says nothing
about which is CHEAPER, and that is half the plug-in decision: Amsar runs 8
cameras off one edge box, so per-frame cost sets how many cameras fit.

The structural difference:

  Amsar     TWO forward passes per frame (yolov8n person + ppe.pt) plus a
            Python-level association loop that is O(persons x items).
  PPE-EYE   ONE forward pass. The class IS the verdict, so there is no second
            model and no association step.

Everything is timed on identical images, identical imgsz, same device, same
warmup, so the only difference is the pipeline. Wall-clock per frame is what
the camera loop actually experiences, so that is the headline number; the
per-stage split is reported to show WHERE the cost sits.

Usage
-----
  PPE/Scripts/python.exe experiments/bench_efficiency.py
  PPE/Scripts/python.exe experiments/bench_efficiency.py --imgsz 640 --n 200
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amsar_eval  # reuse the VERBATIM association code  # noqa: E402

AMSAR_DIR = Path(r"C:\Users\user\Desktop\khizar\amsar-AI\Amsar-AI-")


def model_stats(weights: str) -> dict:
    """Params / GFLOPs / on-disk size for one checkpoint."""
    m = YOLO(weights)
    n_p = sum(p.numel() for p in m.model.parameters())
    try:
        from ultralytics.utils.torch_utils import get_flops
        gflops = round(get_flops(m.model, imgsz=640), 2)
    except Exception:
        gflops = None
    return {
        "weights": Path(weights).name,
        "params_M": round(n_p / 1e6, 2),
        "gflops_640": gflops,
        "file_MB": round(Path(weights).stat().st_size / 1e6, 2),
    }


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench_amsar(imgs, imgsz, device, capture, warmup, ppe_conf, person_conf):
    """Amsar's real pipeline: decode -> (optional 800x448 resize) -> person
    model -> ppe model -> _associate_ppe. Timed end to end AND per stage."""
    person = YOLO(str(AMSAR_DIR / "yolov8n.pt"))
    ppe = YOLO(str(AMSAR_DIR / "ppe.pt"))
    hat_cls, _no_hat, vest_cls, glove_cls = amsar_eval._derive_ppe_classes(ppe.names)

    frames = []
    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            continue
        if capture == "amsar":
            img = cv2.resize(img, (amsar_eval.AMSAR_W, amsar_eval.AMSAR_H))
        frames.append(img)

    def one(frame):
        t0 = time.perf_counter()
        pr = person.predict(frame, imgsz=imgsz, classes=[0], conf=person_conf,
                            device=device, verbose=False)[0]
        _sync()
        t1 = time.perf_counter()
        qr = ppe.predict(frame, imgsz=imgsz, conf=ppe_conf, device=device,
                         verbose=False)[0]
        _sync()
        t2 = time.perf_counter()

        pboxes = pr.boxes.xyxy.cpu().numpy()
        hat_b, vest_b, glove_b = [], [], []
        for box, cls in zip(qr.boxes.xyxy.cpu().numpy(), qr.boxes.cls.cpu().numpy()):
            c = int(cls)
            if c in hat_cls:
                hat_b.append(box)
            elif c in vest_cls:
                vest_b.append(box)
            elif c in glove_cls:
                glove_b.append(box)
        amsar_eval._associate_ppe(frame, pboxes, (hat_b, vest_b, glove_b, bool(vest_cls)))
        t3 = time.perf_counter()
        return (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, (t3 - t0) * 1e3, len(pboxes)

    for f in frames[:warmup]:
        one(f)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    stage_p, stage_q, stage_a, total, npers = [], [], [], [], []
    for f in frames:
        a, b, c, t, n = one(f)
        stage_p.append(a); stage_q.append(b); stage_a.append(c)
        total.append(t); npers.append(n)

    return {
        "system": "amsar_cascade",
        "capture": capture,
        "forward_passes_per_frame": 2,
        "person_ms_median": round(statistics.median(stage_p), 2),
        "ppe_ms_median": round(statistics.median(stage_q), 2),
        "assoc_ms_median": round(statistics.median(stage_a), 3),
        "total_ms_median": round(statistics.median(total), 2),
        "total_ms_mean": round(statistics.mean(total), 2),
        "total_ms_p95": round(sorted(total)[int(0.95 * len(total)) - 1], 2),
        "fps": round(1000 / statistics.median(total), 1),
        "peak_vram_MB": round(torch.cuda.max_memory_allocated() / 1e6, 1)
        if torch.cuda.is_available() else None,
        "mean_persons_per_frame": round(statistics.mean(npers), 2),
        "n_frames": len(frames),
    }


def bench_ppeeye(weights, imgs, imgsz, device, warmup, conf):
    """PPE-EYE: one model, one pass, verdict comes straight out of the class."""
    model = YOLO(str(weights))
    frames = [cv2.imread(str(p)) for p in imgs]
    frames = [f for f in frames if f is not None]

    def one(frame):
        t0 = time.perf_counter()
        r = model.predict(frame, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
        _sync()
        n = len(r.boxes)
        t1 = time.perf_counter()
        return (t1 - t0) * 1e3, n

    for f in frames[:warmup]:
        one(f)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    total, ndet = [], []
    for f in frames:
        t, n = one(f)
        total.append(t); ndet.append(n)

    return {
        "system": "ppe_eye_single",
        "weights": str(weights),
        "forward_passes_per_frame": 1,
        "total_ms_median": round(statistics.median(total), 2),
        "total_ms_mean": round(statistics.mean(total), 2),
        "total_ms_p95": round(sorted(total)[int(0.95 * len(total)) - 1], 2),
        "fps": round(1000 / statistics.median(total), 1),
        "peak_vram_MB": round(torch.cuda.max_memory_allocated() / 1e6, 1)
        if torch.cuda.is_available() else None,
        "mean_dets_per_frame": round(statistics.mean(ndet), 2),
        "n_frames": len(frames),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/test/images")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--person-conf", type=float, default=0.4)  # Amsar's shipped CONF
    ap.add_argument("--ppe-conf", type=float, default=0.40)    # Amsar's PPE_HAT_CONF
    ap.add_argument("--cameras", type=int, default=8)          # Amsar's deployment target
    ap.add_argument("--out", default="experiments/results/efficiency_bench.json")
    args = ap.parse_args()

    imgs = sorted(p for p in Path(args.images).rglob("*")
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})[:args.n]
    print(f"[bench] {len(imgs)} images  imgsz={args.imgsz}  device={args.device}")
    print(f"[bench] gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    ppeeye_weights = {
        "ppe_eye_yolo11s_w1baseline": "runs/ppe_enh/w1_s_baseline_repro/weights/best.pt",
        "ppe_eye_yolo11n_w1": "runs/ppe_enh/w1_n_adamw_1e3/weights/best.pt",
        "ppe_eye_yolo11s_w4mixed": "runs/ppe_enh/w4_mixed_train/weights/best.pt",
    }

    out = {"imgsz": args.imgsz, "n_images": len(imgs), "cameras": args.cameras,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
           "model_stats": {}, "runs": {}}

    out["model_stats"]["amsar_yolov8n_person"] = model_stats(str(AMSAR_DIR / "yolov8n.pt"))
    out["model_stats"]["amsar_ppe"] = model_stats(str(AMSAR_DIR / "ppe.pt"))
    for k, w in ppeeye_weights.items():
        if Path(w).exists():
            out["model_stats"][k] = model_stats(w)

    print("\n[bench] amsar cascade (native res)...")
    out["runs"]["amsar_native"] = bench_amsar(imgs, args.imgsz, args.device, "native",
                                              args.warmup, args.ppe_conf, args.person_conf)
    print("[bench] amsar cascade (deployed 800x448 capture)...")
    out["runs"]["amsar_deployed"] = bench_amsar(imgs, args.imgsz, args.device, "amsar",
                                                args.warmup, args.ppe_conf, args.person_conf)
    for k, w in ppeeye_weights.items():
        if Path(w).exists():
            print(f"[bench] {k}...")
            out["runs"][k] = bench_ppeeye(w, imgs, args.imgsz, args.device,
                                          args.warmup, 0.1)

    # camera-capacity view: how many 8-camera streams fit in real time
    for k, r in out["runs"].items():
        ms = r["total_ms_median"]
        r["max_fps_per_cam_at_N"] = round(1000 / (ms * args.cameras), 2)
        r["gpu_ms_for_N_cams_at_5fps"] = round(ms * args.cameras * 5, 1)

    Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"\n{'system':<32} {'ms/frame':>9} {'fps':>7} {'passes':>7} "
          f"{'VRAM MB':>8} {'fps/cam@8':>10}")
    for k, r in out["runs"].items():
        print(f"{k:<32} {r['total_ms_median']:>9.2f} {r['fps']:>7.1f} "
              f"{r['forward_passes_per_frame']:>7} {str(r['peak_vram_MB']):>8} "
              f"{r['max_fps_per_cam_at_N']:>10.2f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
