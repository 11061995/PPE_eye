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

    # optional cross-dataset evaluation on other data.yaml files
    cross = {}
    for tag_path in p.get("also_eval_data", []):
        tag, dpath = (tag_path.split("=", 1) if "=" in tag_path else (Path(tag_path).parent.name, tag_path))
        dpath = dpath if Path(dpath).is_absolute() else str(lib.ROOT / dpath)
        try:
            xv = m.val(data=dpath, split="test", imgsz=kw["imgsz"],
                       device=kw.get("device", "0"),
                       project=str(lib.ROOT / "runs" / "ppe_enh"),
                       name=f"{exp['id']}_x_{tag}", exist_ok=True, plots=False)
            cross[tag] = {
                "mAP50": round(float(xv.box.map50), 4),
                "mAP50_95": round(float(xv.box.map), 4),
                "precision": round(float(xv.box.mp), 4),
                "recall": round(float(xv.box.mr), 4),
                "per_class_AP50": {xv.names[i]: round(float(v), 4)
                                   for i, v in zip(xv.box.ap_class_index, xv.box.ap50)},
            }
        except Exception as e:  # noqa: BLE001
            cross[tag] = {"error": str(e)}

    return {
        **({"cross_dataset": cross} if cross else {}),
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


def _best_week1_weights() -> str:
    """Path to the best Week-1 train run by test_mAP50_95 (fallback: adamw_1e4)."""
    best, best_v = None, -1.0
    for e_id in [f.stem for f in lib.RESULTS_DIR.glob("w1_*.json")]:
        r = lib.load_result(e_id) or {}
        if r.get("status") == "done" and isinstance(r.get("test_mAP50_95"), (int, float)):
            if r["test_mAP50_95"] > best_v:
                best, best_v = r.get("weights"), r["test_mAP50_95"]
    return best or str(lib.ROOT / "runs/ppe_enh/w1_s_adamw_1e4/weights/best.pt")


def handle_eval(exp: dict) -> dict:
    """Person-level honest eval (both compliance rules) on a chosen checkpoint."""
    import honest_eval as he

    p = exp["params"]
    ek = p.get("eval_kind", "honest")
    if ek not in ("honest", "clean_vs_noisy", "sahi"):
        raise NotImplementedError(
            f"eval_kind={ek!r} not implemented yet (experiment {exp['id']}). "
            f"Build it in the week it is scheduled.")
    weights = p.get("weights") or _best_week1_weights()
    weights = weights if Path(weights).is_absolute() else str(lib.ROOT / weights)
    data = str(lib.ROOT / p.get("data", "data/data.yaml"))
    split = p.get("split", "test")
    imgsz = p.get("imgsz", 640)
    sweep = [float(c) for c in str(p.get("conf_sweep", he.DEFAULT_SWEEP)).split(",")]
    rules = p.get("rules", ["strict", "helmet"])

    def eval_set(label_dir_name, tag):
        out = {}
        for rule in rules:
            res = he.run(weights, data, split, rule, imgsz, p.get("device", "0"),
                         p.get("iou", 0.5), sweep, label_dir_name=label_dir_name)
            (lib.RESULTS_DIR / f"{exp['id']}_{tag}_{rule}.json").write_text(
                json.dumps(res, indent=2), encoding="utf-8")
            b = res["best_f1_row"]
            out[rule] = {
                "best_conf": b["conf"],
                "violation_recall": b["violation_recall"],
                "false_alarm_rate": b["false_alarm_rate"],
                "verdict_accuracy": b["verdict_accuracy"],
                "undetected_persons": b["undetected_persons"],
                "max_recall": max(r["violation_recall"] for r in res["sweep"]),
                "min_false_alarm": min(r["false_alarm_rate"] for r in res["sweep"]),
            }
        return out

    if ek == "sahi":
        import sahi_eval
        return sahi_eval.run(weights, data, split, p.get("device", "0"), sweep,
                             lib.RESULTS_DIR, slice_px=p.get("slice_px", 320),
                             overlap=p.get("overlap", 0.2))

    if ek == "honest":
        # keep the simple <id>_<rule>.json names for the report figure
        out = {}
        for rule in rules:
            res = he.run(weights, data, split, rule, imgsz, p.get("device", "0"),
                         p.get("iou", 0.5), sweep)
            (lib.RESULTS_DIR / f"{exp['id']}_{rule}.json").write_text(
                json.dumps(res, indent=2), encoding="utf-8")
            b = res["best_f1_row"]
            out[rule] = {k: b[k] for k in ("conf", "violation_recall",
                        "false_alarm_rate", "verdict_accuracy", "undetected_persons")}
            out[rule]["max_recall"] = max(r["violation_recall"] for r in res["sweep"])
        return {"weights": weights, "split": split, "eval": "honest_person_level",
                "rules": out}

    noisy = eval_set("labels", "noisy")
    clean = eval_set("labels_clean", "clean")
    delta = {r: {k: round(clean[r][k] - noisy[r][k], 4)
                 for k in ("violation_recall", "false_alarm_rate", "verdict_accuracy")}
             for r in rules}
    return {"weights": weights, "split": split, "eval": "clean_vs_noisy",
            "noisy": noisy, "clean": clean, "delta_clean_minus_noisy": delta}


def handle_audit(exp: dict) -> dict:
    import audit

    p = exp["params"]
    weights = p.get("weights") or _best_week1_weights()
    weights = weights if Path(weights).is_absolute() else str(lib.ROOT / weights)
    data = str(lib.ROOT / p.get("data", "data/data.yaml"))
    return audit.run(weights, data, lib.RESULTS_DIR)


def handle_corrupt(exp: dict) -> dict:
    """RTSP-realistic corruption sweep, scored person-level (see corruptions.py)."""
    import corruptions

    p = exp["params"]
    data = str(lib.ROOT / p.get("data", "data/data.yaml"))
    sweep = [float(c) for c in str(p.get("conf_sweep", "0.10,0.25,0.50")).split(",")]

    models = p.get("models") or {"w4_mixed": "runs/ppe_enh/w4_mixed_train/weights/best.pt"}
    models = {t: (w if Path(w).is_absolute() else str(lib.ROOT / w))
              for t, w in models.items()}
    missing = [t for t, w in models.items() if not Path(w).exists()]
    if missing:
        raise FileNotFoundError(f"missing checkpoints for {missing}")

    return corruptions.run(models, data, p.get("split", "test"), p.get("imgsz", 640),
                           p.get("device", "0"), sweep,
                           severities=int(p.get("severities", 5)),
                           rule=p.get("rule", "helmet"),
                           iou_thr=float(p.get("iou", 0.5)),
                           corruptions=p.get("corruptions"))


def handle_calib(exp: dict) -> dict:
    """Confidence calibration: reliability + ECE, temperature fitted on val."""
    import calibrate

    p = exp["params"]
    data = str(lib.ROOT / p.get("data", "data/data.yaml"))
    models = p.get("models") or {"w4_mixed": "runs/ppe_enh/w4_mixed_train/weights/best.pt"}
    models = {t: (w if Path(w).is_absolute() else str(lib.ROOT / w))
              for t, w in models.items()}
    missing = [t for t, w in models.items() if not Path(w).exists()]
    if missing:
        raise FileNotFoundError(f"missing checkpoints for {missing}")
    return calibrate.run(models, data, p.get("imgsz", 640), p.get("device", "0"),
                         float(p.get("iou", 0.5)))


HANDLERS = {"train": handle_train, "eval": handle_eval,
            "audit": handle_audit, "corrupt": handle_corrupt,
            "calib": handle_calib}


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
