#!/usr/bin/env python3
"""
gate.py - the acceptance test a checkpoint must pass before it can replace
Amsar's shipped `ppe.pt`.

HEAD_TO_HEAD.md closed by asking for exactly this: "gate the swap on an OOD
acceptance number ... and don't ship until a checkpoint clears it. Today nothing
does." This is that gate, made runnable.

Three test beds, all scored person-level under the `helmet` rule:

  pictor    Pictor test, 316 imgs - in-domain, and identical to Week 1, so the
            number is comparable to all 21 earlier rows
  sh17v2    SH17-v2 test, group-disjoint from training - held out, same source
  sh17v2_strict
            the subset of that bed no checkpoint in this repo has trained on -
            the only bed where the incumbents and the final model are on equal
            footing. It is all-violation (see below).
  closeup   SH17 close-up holdout - images the geometry filter rejected, so a
            genuine distribution shift. Every worker in it is bare-headed, which
            means FP and TN are both structurally 0: **false-alarm rate and
            verdict accuracy are undefined here** and only violation recall and
            undetected-person count are reported. The same is true of
            sh17v2_strict.

Every test bed prints its own trivial baselines, because "verdict accuracy 0.70"
means nothing until you know what a classifier that ignores the image scores on
that same set. On a set with a 0.67 violation base rate, always-say-violation
already scores 0.67.

Usage
-----
  PPE/Scripts/python.exe experiments/gate.py --weights runs/.../best.pt
  PPE/Scripts/python.exe experiments/gate.py --weights a.pt b.pt --tag base thin
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

import honest_eval as he

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results"

BEDS = {
    "pictor": ("data/data.yaml", "test"),
    "sh17v2": ("data/sh17c_v2/data.yaml", "test"),
    "sh17v2_strict": ("data/sh17c_v2/data_strict.yaml", "test"),
    "closeup": ("data/sh17c_v2/data_holdout.yaml", "test"),
}

# Beds on which false-alarm rate and verdict accuracy are undefined because every
# ground-truth worker is a violation, so FP and TN are structurally zero.
ALL_VIOLATION_BEDS = {"sh17v2_strict", "closeup"}

# Which checkpoints have seen which bed. The w4_* runs trained on the Week-4 v1
# SH17 mapping, which shares 59 of the 125 sh17v2 test images; the final runs
# train on sh17c_v2's own train split and share none. So `sh17v2` flatters the
# incumbents and not the final model - if the final model wins there, the margin
# is a lower bound.
CONTAMINATION = {
    "sh17v2": ("59 of 125 images were in the Week-4 v1 SH17 training set, so this "
               "bed is held out for the final runs but NOT for w4_sh17_train / "
               "w4_mixed_train / w3_darkaug"),
}


def baselines(data: str, split: str, rule: str) -> dict:
    """What a classifier that never looks at the image scores on this test bed."""
    cfg = yaml.safe_load(Path(data).read_text())
    root = Path(cfg.get("path", Path(data).parent))
    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    compliant = {i for i, n in enumerate(names) if n in he.RULES[rule]}
    imgs, label_of = he._resolve_split(root / cfg[split if split in cfg else "val"])
    c = Counter()
    for im in imgs:
        lf = label_of(im)
        if not lf.exists():
            continue
        for ln in lf.read_text().splitlines():
            t = ln.split()
            if len(t) >= 5:
                c["compliant" if int(float(t[0])) in compliant else "violation"] += 1
    n = sum(c.values())
    rate = c["violation"] / n if n else 0.0
    return {
        "n_images": len(imgs), "n_persons": n,
        "violation_base_rate": round(rate, 4),
        "always_violation_accuracy": round(rate, 4),
        "always_compliant_accuracy": round(1 - rate, 4),
        "trivial_best_accuracy": round(max(rate, 1 - rate), 4),
    }


def run_bed(weights: str, bed: str, imgsz: int, device: str, rule: str,
            sweep: list[float], iou: float) -> dict:
    rel, split = BEDS[bed]
    data = str(ROOT / rel)
    base = baselines(data, split, rule)
    res = he.run(weights, data, split, rule, imgsz, device, iou, sweep)
    rows = [{k: r[k] for k in ("conf", "violation_recall", "violation_precision",
                               "false_alarm_rate", "verdict_accuracy",
                               "undetected_persons", "TP", "FP", "FN", "TN")}
            for r in res["sweep"]]
    out = {"bed": bed, "data": data, "baselines": base, "sweep": rows,
           "best_f1": res["best_f1_row"]}
    if bed in ALL_VIOLATION_BEDS:
        out["note"] = ("every ground-truth worker is a violation, so FP = TN = 0 "
                       "by construction: false_alarm_rate and verdict_accuracy "
                       "are undefined and must not be quoted from this bed. Read "
                       "violation_recall and undetected_persons only.")
    if bed in CONTAMINATION:
        out["contamination"] = CONTAMINATION[bed]
    return out


def verdict(report: dict, min_acc: float, min_recall: float,
            max_fa: float, conf: float = 0.10) -> dict:
    """Pass/fail.

    Five checks, and every one of them exists because a real checkpoint failed
    it while passing the others:

    * held-out accuracy, absolute AND against that bed's trivial baseline — a
      set with a 0.67 violation base rate hands 0.67 to a classifier that never
      looks at the image;
    * in-domain violation recall, so out-of-domain gains cannot be bought by
      going quiet on the footage we do have;
    * in-domain false-alarm ceiling and a trivial-baseline check on in-domain
      accuracy. `final_n_deploy` passed every other check with an in-domain
      false-alarm rate of 0.503 and verdict accuracy of 0.492 — *below* the
      always-say-compliant baseline of 0.537. A system that flags half the
      compliant workforce gets switched off in a week, which is the failure
      HEAD_TO_HEAD.md measured on Amsar's own cascade (FA 0.886).

    In-domain numbers are read at `conf`, the threshold Amsar actually runs
    (`cameras.PPE_EYE_CONF = 0.10`), not at the best-F1 row — a model is
    deployed at a fixed threshold, so that is where it must be judged.
    """
    pb = report["beds"]["pictor"]
    p = next((r for r in pb["sweep"] if abs(r["conf"] - conf) < 1e-6),
             pb["best_f1"])
    p_trivial = pb["baselines"]["trivial_best_accuracy"]
    s = report["beds"]["sh17v2"]
    s_best = max(s["sweep"], key=lambda r: r["verdict_accuracy"])
    trivial = s["baselines"]["trivial_best_accuracy"]
    checks = {
        "heldout_verdict_accuracy>=target": (s_best["verdict_accuracy"], min_acc,
                                             s_best["verdict_accuracy"] >= min_acc),
        "heldout_beats_trivial_baseline": (s_best["verdict_accuracy"], trivial,
                                           s_best["verdict_accuracy"] > trivial),
        "indomain_violation_recall>=target": (p["violation_recall"], min_recall,
                                              p["violation_recall"] >= min_recall),
        "indomain_false_alarm<=ceiling": (p["false_alarm_rate"], max_fa,
                                          p["false_alarm_rate"] <= max_fa),
        "indomain_beats_trivial_baseline": (p["verdict_accuracy"], p_trivial,
                                            p["verdict_accuracy"] > p_trivial),
    }
    return {
        "operating_conf": conf,
        "checks": {k: {"value": v, "target": t, "pass": bool(ok)}
                   for k, (v, t, ok) in checks.items()},
        "indomain_row": p,
        "heldout_best_row": s_best,
        "iso_false_alarm": iso_fa(pb["sweep"]),
        "passed": all(ok for _, _, ok in checks.values()),
    }


def iso_fa(sweep: list[dict], budgets=(0.15, 0.20, 0.25, 0.30)) -> dict:
    """Best violation recall reachable within a false-alarm budget.

    Comparing checkpoints at a matched *threshold* flatters whichever one happens
    to be more conservative at that number; a site does not care what the
    threshold is, it cares how many violations get caught per nuisance alarm it
    tolerates. WEEK4_FINDINGS.md made the same point about matched recall. So for
    each budget this reports the best recall available at or under it, and the
    threshold that delivers it.
    """
    out = {}
    for b in budgets:
        ok = [r for r in sweep if r["false_alarm_rate"] <= b]
        if not ok:
            out[f"FA<={b:.2f}"] = None
            continue
        best = max(ok, key=lambda r: r["violation_recall"])
        out[f"FA<={b:.2f}"] = {
            "conf": best["conf"],
            "violation_recall": best["violation_recall"],
            "false_alarm_rate": best["false_alarm_rate"],
            "verdict_accuracy": best["verdict_accuracy"],
            "undetected_persons": best["undetected_persons"],
        }
    return out



def print_summary(report: dict) -> None:
    """The three tables a deployment decision is actually made from."""
    rows = {t: r for t, r in report.items() if "gate" in r}
    if not rows:
        print("no gated checkpoints in this report")
        return
    any_r = next(iter(rows.values()))
    conf = any_r["gate"]["operating_conf"]

    print()
    print(f"=== 1. at the shipped operating point (conf {conf}) ===")
    print()
    head = (f"{'checkpoint':18s}{'heldout acc':>12s}{'in-dom R':>10s}"
            f"{'in-dom FA':>11s}{'in-dom acc':>12s}{'checks':>8s}")
    print(head)
    print("-" * len(head))
    for t, r in rows.items():
        g = r["gate"]
        ind = g["indomain_row"]
        npass = sum(c["pass"] for c in g["checks"].values())
        checks = f"{npass}/{len(g['checks'])}"
        print(f"{t:18s}{g['heldout_best_row']['verdict_accuracy']:12.3f}"
              f"{ind['violation_recall']:10.3f}{ind['false_alarm_rate']:11.3f}"
              f"{ind['verdict_accuracy']:12.3f}{checks:>8s}")
    pb = any_r["beds"]["pictor"]["baselines"]
    hb = any_r["beds"]["sh17v2"]["baselines"]
    print(f"{'trivial baseline':18s}{hb['trivial_best_accuracy']:12.3f}"
          f"{'-':>10s}{'-':>11s}{pb['trivial_best_accuracy']:12.3f}")

    print()
    print("=== 2. in-domain violation recall within a false-alarm budget ===")
    print("    (a site tolerates a nuisance rate, not a threshold)")
    print()
    budgets = list(any_r["gate"]["iso_false_alarm"])
    head = f"{'checkpoint':18s}" + "".join(f"{b:>20s}" for b in budgets)
    print(head)
    print("-" * len(head))
    for t, r in rows.items():
        line = f"{t:18s}"
        for b in budgets:
            v = r["gate"]["iso_false_alarm"].get(b)
            cell = f"{v['violation_recall']:.3f} @{v['conf']}" if v else "cannot reach"
            line += f"{cell:>20s}"
        print(line)

    print()
    print("=== 3. all-violation beds (recall only; FA and accuracy undefined) ===")
    print()
    beds = [b for b in ("sh17v2_strict", "closeup") if b in any_r["beds"]]
    head = f"{'checkpoint':18s}" + "".join(f"{b + ' recall':>22s}" for b in beds)
    print(head)
    print("-" * len(head))
    for t, r in rows.items():
        line = f"{t:18s}"
        for b in beds:
            best = max(r["beds"][b]["sweep"], key=lambda x: x["violation_recall"])
            line += f"{best['violation_recall']:22.3f}"
        print(line)
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="*", default=[])
    ap.add_argument("--tag", nargs="*", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--rule", default="helmet")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--conf-sweep",
                    default="0.03,0.05,0.07,0.10,0.15,0.20,0.25,0.30,0.40,0.50",
                    help="the sweep starts below Amsar's current PPE_EYE_CONF=0.10 "
                         "on purpose: a checkpoint with a lower false-alarm rate "
                         "can be run at a lower threshold for the same nuisance "
                         "budget, which is where its extra recall comes from "
                         "(WEEK5_FINDINGS.md). Which threshold to ship is a "
                         "deployment choice; --operating-conf states it.")
    ap.add_argument("--beds", nargs="*", default=list(BEDS))
    ap.add_argument("--min-accuracy", type=float, default=0.70,
                    help="held-out verdict accuracy the checkpoint must reach")
    ap.add_argument("--min-recall", type=float, default=0.45,
                    help="in-domain violation recall it must not regress below "
                         "(w4_mixed, the incumbent best, scores 0.457)")
    ap.add_argument("--max-false-alarm", type=float, default=0.30,
                    help="in-domain false-alarm ceiling. The prior best scores "
                         "0.253 and the Week-1 baseline 0.204, so anything above "
                         "0.30 is a nuisance regression on footage we already have")
    ap.add_argument("--operating-conf", type=float, default=0.10,
                    help="the threshold the product runs at (cameras.PPE_EYE_CONF); "
                         "in-domain checks are read here, not at best-F1")
    ap.add_argument("--summary", action="store_true",
                    help="print the decision tables from an existing GATE.json")
    ap.add_argument("--rescore", action="store_true",
                    help="recompute pass/fail from an existing GATE.json without "
                         "re-running inference")
    ap.add_argument("--out", default="experiments/results/GATE.json")
    args = ap.parse_args()

    sweep = [float(c) for c in args.conf_sweep.split(",")]
    tags = args.tag or [Path(w).parent.parent.name for w in args.weights]
    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    # Resume-friendly: a GPU fault on model 3 must not discard models 1 and 2,
    # and is not a reason to re-measure them.
    report = json.loads(out.read_text()) if out.exists() else {}

    if args.summary:
        print_summary(report)
        return

    if args.rescore:
        for tag, r in report.items():
            if "beds" not in r:
                continue
            r["gate"] = verdict(r, args.min_accuracy, args.min_recall,
                                args.max_false_alarm, args.operating_conf)
            g = r["gate"]
            print(f"{tag:16s} {'PASS' if g['passed'] else 'FAIL'}  " + "  ".join(
                f"{'x' if c['pass'] else ' '}{k.split('_')[0]}:{c['value']}"
                for k, c in g["checks"].items()))
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nrescored {out}")
        return

    for tag, w in zip(tags, args.weights):
        w = w if Path(w).is_absolute() else str(ROOT / w)
        try:
            beds = {b: run_bed(w, b, args.imgsz, args.device, args.rule, sweep, args.iou)
                    for b in args.beds}
        except Exception as e:  # noqa: BLE001
            print(f"\n===== {tag} ===== FAILED: {type(e).__name__}: {e}")
            report[tag] = {"weights": w, "error": f"{type(e).__name__}: {e}"}
            out.write_text(json.dumps(report, indent=2), encoding="utf-8")
            continue
        r = {"weights": w, "rule": args.rule, "beds": beds}
        if {"pictor", "sh17v2"} <= set(beds):
            r["gate"] = verdict(r, args.min_accuracy, args.min_recall,
                                args.max_false_alarm, args.operating_conf)
        report[tag] = r
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n===== {tag} =====")
        for b, v in beds.items():
            best = v["best_f1"]
            print(f"  {b:8s} n={v['baselines']['n_persons']:5d} "
                  f"trivial_acc={v['baselines']['trivial_best_accuracy']:.3f} | "
                  f"conf={best['conf']} recall={best['violation_recall']:.3f} "
                  f"FA={best['false_alarm_rate']:.3f} "
                  f"acc={best['verdict_accuracy']:.3f} "
                  f"undetected={best['undetected_persons']}")
        if "gate" in r:
            print(f"  GATE: {'PASS' if r['gate']['passed'] else 'FAIL'}")
            for k, c in r["gate"]["checks"].items():
                print(f"    [{'x' if c['pass'] else ' '}] {k}: "
                      f"{c['value']} vs {c['target']}")

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
