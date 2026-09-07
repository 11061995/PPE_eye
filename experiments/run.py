#!/usr/bin/env python3
"""
run.py — run one enhancement experiment and write its result row.

  python experiments/run.py --next        # loop mode: next pending, deps-ready
  python experiments/run.py w1_s_adamw_1e3 # run a specific experiment
  python experiments/run.py --status       # print STATUS.md contents and exit

Only `kind: train` is implemented. When --next hits a non-train kind it prints
which handler is missing and exits 3 — that is the review gate between weeks.

After a successful run it regenerates the report (report.py).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402

TRAIN_KEYS = {
    "data", "imgsz", "epochs", "patience", "batch", "seed", "device",
    "optimizer", "lr0", "cos_lr", "warmup_epochs", "close_mosaic", "momentum",
    "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
    "perspective", "flipud", "fliplr", "mosaic", "mixup", "copy_paste", "erasing",
}


def _epochs_trained(run_dir: Path) -> int | None:
    csv = run_dir / "results.csv"
    if not csv.exists():
        return None
    lines = [ln for ln in csv.read_text().splitlines() if ln.strip()]
    return max(len(lines) - 1, 0)  # minus header


def handle_train(exp: dict) -> dict:
    from ultralytics import YOLO

    p = exp["params"]
    model_name = p["model"] if "model" in p else "yolo11s.pt"
    kw = {k: v for k, v in p.items() if k in TRAIN_KEYS}
    kw["data"] = str(lib.ROOT / kw["data"]) if not Path(kw["data"]).is_absolute() else kw["data"]

    run_dir = lib.ROOT / "runs" / "ppe_enh" / exp["id"]
    t0 = time.time()
    m = YOLO(model_name)
    m.train(project=str(lib.ROOT / "runs" / "ppe_enh"), name=exp["id"],
            exist_ok=True, deterministic=True, val=True, plots=True, **kw)
    train_min = round((time.time() - t0) / 60, 1)

    mt = m.val(data=kw["data"], split="test", imgsz=kw["imgsz"],
               device=kw.get("device", "0"), project=str(lib.ROOT / "runs" / "ppe_enh"),
               name=f"{exp['id']}_test", exist_ok=True, plots=True)

    per_class = {mt.names[i]: round(float(v), 4)
                 for i, v in zip(mt.box.ap_class_index, mt.box.ap50)}
    lat = lib.latency_ms(m, kw["imgsz"], kw.get("device", "0"))

    return {
        "model": model_name,
        "imgsz": kw["imgsz"],
        "optimizer": kw.get("optimizer"),
        "lr0": kw.get("lr0"),
        "epochs_cfg": kw.get("epochs"),
        "epochs_trained": _epochs_trained(run_dir),
        "train_minutes": train_min,
        "test_mAP50": round(float(mt.box.map50), 4),
        "test_mAP50_95": round(float(mt.box.map), 4),
        "test_precision": round(float(mt.box.mp), 4),
        "test_recall": round(float(mt.box.mr), 4),
        "per_class_AP50": per_class,
        **lat,
        "weights": str(run_dir / "weights" / "best.pt"),
        "run_dir": str(run_dir),
    }


HANDLERS = {"train": handle_train}


def run_one(exp: dict) -> int:
    lib.mark_running(exp)
    base = {
        "id": exp["id"], "week": exp["week"], "tier": exp["tier"],
        "item": exp.get("item"), "kind": exp["kind"], "desc": exp["desc"],
        "params": exp["params"], "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        metrics = HANDLERS[exp["kind"]](exp)
        row = {**base, **metrics, "status": "done",
               "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    except Exception as e:  # noqa: BLE001
        row = {**base, "status": "failed", "error": str(e),
               "traceback": traceback.format_exc(),
               "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    lib.save_result(row)
    print(json.dumps({k: row[k] for k in row if k not in ("traceback", "params")}, indent=2))
    return 0 if row["status"] == "done" else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", help="experiment id")
    ap.add_argument("--next", action="store_true", help="run next pending, deps-ready")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()

    if args.status:
        subprocess.run([sys.executable, str(lib.EXP_DIR / "report.py")], check=False)
        stat = lib.RESULTS_DIR / "STATUS.md"
        if stat.exists():
            print(stat.read_text(encoding="utf-8"))
        return 0

    _, exps = lib.load_registry()
    by_id = {e["id"]: e for e in exps}

    if args.next:
        running = lib.running_experiment()
        if running:
            print(f"[run] {running} already marked running — not starting another")
            return 2
        exp = lib.next_experiment()
        if exp is None:
            print("[run] nothing to do — all experiments terminal or deps unmet")
            return 0
        if exp["kind"] not in HANDLERS:
            print(f"[run] next is {exp['id']} (kind={exp['kind']}) — handler not "
                  f"implemented. Review gate: implement Week {exp['week']} handlers.")
            return 3
        print(f"[run] starting {exp['id']}: {exp['desc']}")
        rc = run_one(exp)
    elif args.target:
        if args.target not in by_id:
            print(f"[run] unknown experiment id: {args.target}")
            return 1
        rc = run_one(by_id[args.target])
    else:
        ap.print_help()
        return 1

    if not args.no_report:
        subprocess.run([sys.executable, str(lib.EXP_DIR / "report.py")], check=False)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
