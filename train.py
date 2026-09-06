#!/usr/bin/env python3
"""
train.py — reproduce PPE-EYE (Computers 2026, 15, 45) Table 4, with a control.

The paper's Table 4 grid, reproduced verbatim:

  model      epochs  batch  optimizer  lr        momentum  cos_lr   claimed mAP50
  yolo10n      25     32    auto       0.01      0.937     False    0.770
  yolo10s      75     16    auto       0.01      0.937     False    0.873
  yolo11l     100     28    NAdam      0.00001   0.937     True     0.907
  yolo11x      87     20    NAdam      0.00001   0.5       True     0.910
  yolo11x      21     16    NAdam      0.00001   0.937     True     0.969   <- headline

Note yolo11x@21ep beating yolo11x@87ep on the same lr is the single most
suspicious cell in the paper. Reproducing both under one clean split is the
cheapest way to find out whether it is real.

The control that matters
------------------------
Run every config TWICE: once on a group-aware split (split_no_leak.py) and once
on a naive random split of the same augmented pool. The delta between the two is
your estimate of how much of the 96.9% is leakage rather than skill. Report that
delta — it is a publishable result on its own.

Usage
-----
  python train.py --data /data/chvg_clean/data.yaml --tag clean
  python train.py --data /data/chvg_naive/data.yaml --tag naive
  python train.py --data ... --only yolo11x_21 --deploy     # + deployable sizes

Note: Ultralytics exposes NAdam as optimizer='NAdam'. lr0=1e-5 with a pretrained
backbone is extremely low; expect the 21-epoch run to barely move off the COCO
initialisation. That is itself informative.
"""

import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO

PAPER_GRID = {
    "yolo10n_25":  dict(model="yolov10n.pt", epochs=25,  batch=32, optimizer="auto",  lr0=0.01,    momentum=0.937, cos_lr=False, claimed=0.770),
    "yolo10s_75":  dict(model="yolov10s.pt", epochs=75,  batch=16, optimizer="auto",  lr0=0.01,    momentum=0.937, cos_lr=False, claimed=0.873),
    "yolo11l_100": dict(model="yolo11l.pt",  epochs=100, batch=28, optimizer="NAdam", lr0=0.00001, momentum=0.937, cos_lr=True,  claimed=0.907),
    "yolo11x_87":  dict(model="yolo11x.pt",  epochs=87,  batch=20, optimizer="NAdam", lr0=0.00001, momentum=0.5,   cos_lr=True,  claimed=0.910),
    "yolo11x_21":  dict(model="yolo11x.pt",  epochs=21,  batch=16, optimizer="NAdam", lr0=0.00001, momentum=0.937, cos_lr=True,  claimed=0.969),
}

# What you would actually ship on an Orin Nano driving 8 cameras.
DEPLOY_GRID = {
    "yolo11n_100": dict(model="yolo11n.pt", epochs=100, batch=32, optimizer="AdamW", lr0=0.001, momentum=0.937, cos_lr=True, claimed=None),
    "yolo11s_100": dict(model="yolo11s.pt", epochs=100, batch=24, optimizer="AdamW", lr0=0.001, momentum=0.937, cos_lr=True, claimed=None),
    "yolo11m_100": dict(model="yolo11m.pt", epochs=100, batch=16, optimizer="AdamW", lr0=0.001, momentum=0.937, cos_lr=True, claimed=None),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="runs/ppe_eye")
    ap.add_argument("--only", default="", help="comma list of config keys")
    ap.add_argument("--deploy", action="store_true", help="also run the deployable sizes")
    ap.add_argument("--no-aug", action="store_true",
                    help="disable Ultralytics online aug (the dataset is already augmented; "
                         "stacking both is a confound the paper never controls for)")
    args = ap.parse_args()

    grid = dict(PAPER_GRID)
    if args.deploy:
        grid.update(DEPLOY_GRID)
    if args.only:
        keys = [k.strip() for k in args.only.split(",")]
        grid = {k: v for k, v in grid.items() if k in keys}

    off = dict(hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0,
               scale=0.0, shear=0.0, fliplr=0.0, flipud=0.0, mosaic=0.0,
               mixup=0.0, erasing=0.0) if args.no_aug else {}

    results = []
    for key, cfg in grid.items():
        claimed = cfg.pop("claimed", None)
        name = f"{args.tag}_{key}"
        print(f"\n{'='*70}\n{name}  ({cfg['model']}, {cfg['epochs']}ep, batch {cfg['batch']})\n{'='*70}")
        t0 = time.time()
        m = YOLO(cfg.pop("model"))
        m.train(data=args.data, imgsz=args.imgsz, seed=args.seed, device=args.device,
                project=args.project, name=name, exist_ok=True, deterministic=True,
                val=True, plots=True, **cfg, **off)
        train_min = (time.time() - t0) / 60

        # held-out TEST split, never seen during training or model selection
        mt = m.val(data=args.data, split="test", imgsz=args.imgsz, device=args.device,
                   project=args.project, name=f"{name}_test", exist_ok=True)
        row = dict(
            config=key, tag=args.tag, train_minutes=round(train_min, 1),
            claimed_mAP50=claimed,
            test_mAP50=round(float(mt.box.map50), 4),
            test_mAP50_95=round(float(mt.box.map), 4),
            test_precision=round(float(mt.box.mp), 4),
            test_recall=round(float(mt.box.mr), 4),
            per_class_AP50={mt.names[i]: round(float(v), 4)
                            for i, v in zip(mt.box.ap_class_index, mt.box.ap50)},
        )
        if claimed:
            row["delta_vs_paper"] = round(row["test_mAP50"] - claimed, 4)
        results.append(row)
        print(json.dumps(row, indent=2))

    out = Path(args.project) / f"summary_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    print(f"\n{'config':<14}{'claimed':>9}{'test mAP50':>12}{'delta':>9}{'mAP50-95':>10}")
    for r in results:
        c = f"{r['claimed_mAP50']:.3f}" if r["claimed_mAP50"] else "-"
        d = f"{r.get('delta_vs_paper', 0):+.3f}" if r["claimed_mAP50"] else "-"
        print(f"{r['config']:<14}{c:>9}{r['test_mAP50']:>12.3f}{d:>9}{r['test_mAP50_95']:>10.3f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
