#!/usr/bin/env python3
"""
bench_jetson.py — measure real latency on the Orin Nano. Do not trust 7.3 ms.

The paper reports 7.3 ms inference for YOLO11x, benchmarked on a Dell XPS 9320
with Intel Iris Xe integrated graphics. YOLO11x is ~57M params / ~195 GFLOPs at
640x640. 7.3 ms implies ~137 FPS for the largest model in the family on an iGPU.
Note also that Table 3 lists exactly 7.3 ms for a *different* model from a
*different* paper (ref [20]). That number was almost certainly copied.

Your constraint is harder than theirs: Orin Nano 8GB, 8 simultaneous RTSP streams.
What matters is not single-image latency but sustained aggregate throughput at
batch=8, with NVENC and FFmpeg already competing for the same silicon.

This measures, per model size and precision:
  * p50 / p95 / p99 latency at batch 1..8
  * aggregate FPS and the implied per-camera FPS for 8 cameras
  * whether you clear your required per-camera rate

Run it with your HLS/NVENC pipeline ALREADY RUNNING. Benchmarking on an idle
Jetson gives a number you will never see in production.

Usage
-----
  # once, to build engines (slow: INT8 calibration can take 20+ min for x)
  python bench_jetson.py --weights best.pt --export --half
  python bench_jetson.py --weights best.pt --export --int8 --calib /data/chvg_clean/data.yaml

  # then
  python bench_jetson.py --engine best.engine --batches 1,2,4,8 --target-fps 5
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np


def power_mode():
    try:
        out = subprocess.check_output(["nvpmodel", "-q"], text=True, stderr=subprocess.DEVNULL)
        return " ".join(out.split())
    except Exception:
        return "unknown (run: sudo nvpmodel -q). Use MAXN and `sudo jetson_clocks`."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="")
    ap.add_argument("--engine", default="")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--calib", default="", help="data.yaml for INT8 calibration")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batches", default="1,2,4,8")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--cameras", type=int, default=8)
    ap.add_argument("--target-fps", type=float, default=5.0,
                    help="per-camera FPS your zone logic needs")
    ap.add_argument("--out", default="bench.json")
    args = ap.parse_args()

    from ultralytics import YOLO

    print(f"[env] power mode: {power_mode()}")
    print(f"[env] benchmark with FFmpeg/NVENC running, or the numbers are fiction.\n")

    batches = [int(b) for b in args.batches.split(",")]

    if args.export:
        if not args.weights:
            raise SystemExit("--export needs --weights")
        m = YOLO(args.weights)
        kw = dict(format="engine", imgsz=args.imgsz, device=0,
                  batch=max(batches), dynamic=True, workspace=4)
        if args.int8:
            if not args.calib:
                raise SystemExit("--int8 needs --calib pointing at a data.yaml")
            kw.update(int8=True, data=args.calib)
        elif args.half:
            kw.update(half=True)
        print(f"[export] building engine {kw} ...")
        p = m.export(**kw)
        print(f"[export] wrote {p}")
        print("[export] IMPORTANT: re-run eval_honest.py on the ENGINE, not the .pt.")
        print("         INT8 quantisation costs accuracy, especially on the small")
        print("         classes (glass, distant helmets). Measure that cost, do not")
        print("         assume it away.")
        args.engine = str(p)

    src = args.engine or args.weights
    if not src:
        raise SystemExit("need --engine or --weights")
    model = YOLO(src, task="detect")

    rows = []
    for b in batches:
        imgs = [np.random.randint(0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8)
                for _ in range(b)]
        for _ in range(args.warmup):
            model.predict(imgs, imgsz=args.imgsz, device=0, verbose=False)
        lat = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            model.predict(imgs, imgsz=args.imgsz, device=0, verbose=False)
            lat.append((time.perf_counter() - t0) * 1000)
        lat = np.array(lat)
        agg = b / (lat.mean() / 1000)
        per_cam = agg / args.cameras
        rows.append(dict(
            batch=b,
            p50_ms=round(float(np.percentile(lat, 50)), 2),
            p95_ms=round(float(np.percentile(lat, 95)), 2),
            p99_ms=round(float(np.percentile(lat, 99)), 2),
            ms_per_image=round(float(lat.mean() / b), 2),
            aggregate_fps=round(agg, 1),
            per_camera_fps=round(per_cam, 2),
            meets_target=bool(per_cam >= args.target_fps),
        ))
        r = rows[-1]
        flag = "OK " if r["meets_target"] else "MISS"
        print(f"[{flag}] batch={b:<2} p50={r['p50_ms']:>7.2f}ms  p95={r['p95_ms']:>7.2f}ms  "
              f"{r['ms_per_image']:>6.2f} ms/img  agg={r['aggregate_fps']:>6.1f} FPS  "
              f"per-cam={r['per_camera_fps']:>5.2f} FPS")

    Path(args.out).write_text(json.dumps(
        dict(source=src, imgsz=args.imgsz, cameras=args.cameras,
             target_fps=args.target_fps, rows=rows), indent=2))
    print(f"\nwrote {args.out}")

    best = max(rows, key=lambda r: r["per_camera_fps"])
    print(f"\nBest per-camera rate: {best['per_camera_fps']:.2f} FPS at batch {best['batch']}.")
    if best["per_camera_fps"] < args.target_fps:
        print("This model does not fit your deployment. Options, in order of cost:")
        print("  1. Smaller backbone (yolo11s / yolo11n). Then re-run eval_honest.py")
        print("     and report the accuracy you traded away — that trade IS the result.")
        print("  2. Drop to 480x480 input. Cheap, but small-object recall falls hard;")
        print("     the paper's own confusion matrix already loses 20% of `glass`.")
        print("  3. Stagger cameras: motion-gate idle streams, full rate only on")
        print("     zone-adjacent cameras. Highest engineering cost, best accuracy.")
    print("\nAlso record: is the interlock latency budget end-to-end? RTSP jitter +")
    print("decode + inference + relay GPIO. Inference is usually not the bottleneck,")
    print("which is exactly why the paper's 7.3 ms was never the interesting number.")


if __name__ == "__main__":
    main()
