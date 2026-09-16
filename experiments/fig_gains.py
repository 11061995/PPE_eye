#!/usr/bin/env python3
"""
fig_gains.py - the one picture this project needs: where each checkpoint sits on
the recall / false-alarm plane, in domain and out.

Object mAP is not plotted anywhere here, deliberately. Weeks 1-4 established it
does not track deployment quality on this dataset (w4_mixed moved violation
recall +11 points while mAP50 stayed flat at 0.495), so a figure in mAP would
contradict the project's own headline.

Reads the GATE_*.json files written by gate.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results"
FIGS = RESULTS / "figs"

STYLE = {  # tag -> (colour, marker, label)
    "w1_baseline": ("#8c8c8c", "o", "W1 Pictor-only"),
    "w4_sh17": ("#d1495b", "s", "W4 SH17-only (shipped today)"),
    "w4_mixed": ("#edae49", "^", "W4 mixed (prior best)"),
}
FINAL_STYLE = [("#00798c", "D"), ("#003d5b", "P"), ("#30638e", "X"), ("#2f6d3c", "v")]


def load() -> dict:
    out = {}
    for f in sorted(RESULTS.glob("GATE*.json")):
        try:
            out.update(json.loads(f.read_text()))
        except json.JSONDecodeError:
            continue
    return out


def main():
    data = load()
    if not data:
        print("[fig_gains] no GATE*.json yet")
        return
    FIGS.mkdir(parents=True, exist_ok=True)

    beds = [("pictor", "In domain — Pictor test (999 persons)"),
            ("sh17v2", "Held out — SH17-v2 test (305 persons)")]
    fig, axes = plt.subplots(1, len(beds), figsize=(13, 5.6))

    # Assign one style per checkpoint up front. Doing it inside the panel loop
    # gave the same checkpoint a different colour in each panel.
    spare = iter(FINAL_STYLE)
    style = {}
    for tag in data:
        if tag in STYLE:
            style[tag] = STYLE[tag]
        else:
            colour, marker = next(spare, ("#444444", "*"))
            style[tag] = (colour, marker, tag)

    for ax, (bed, title) in zip(axes, beds):
        for tag, rep in data.items():
            b = rep.get("beds", {}).get(bed)
            if not b:
                continue
            colour, marker, label = style[tag]
            xs = [r["false_alarm_rate"] for r in b["sweep"]]
            ys = [r["violation_recall"] for r in b["sweep"]]
            ax.plot(xs, ys, "-", color=colour, alpha=0.45, lw=1.2, zorder=2)
            ax.scatter(xs, ys, s=22, color=colour, marker=marker, zorder=3)
            # mark the operating point Amsar actually uses (PPE_EYE_CONF = 0.10)
            op = next((r for r in b["sweep"] if abs(r["conf"] - 0.10) < 1e-6), None)
            if op:
                ax.scatter([op["false_alarm_rate"]], [op["violation_recall"]],
                           s=170, facecolor="none", edgecolor=colour, lw=2.2,
                           marker=marker, zorder=4, label=label)
        base = next((rep["beds"][bed]["baselines"] for rep in data.values()
                     if bed in rep.get("beds", {})), None)
        if base:
            ax.plot([0, 1], [0, 1], ":", color="#bbbbbb", lw=1,
                    label="flag-everyone trade-off", zorder=1)
            ax.annotate(f"violation base rate {base['violation_base_rate']:.2f}\n"
                        f"n = {base['n_persons']} persons",
                        xy=(0.98, 0.03), xycoords="axes fraction",
                        ha="right", va="bottom", fontsize=8, color="#666666")
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("false-alarm rate  (compliant workers wrongly flagged)")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25, lw=0.6)
    axes[0].set_ylabel("violation recall  (real violations caught)")
    # upper-left is empty in domain but occupied out of domain, where the best
    # checkpoints sit high and left - which is the whole point of the figure.
    axes[0].legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    axes[1].legend(fontsize=7.5, loc="lower right", framealpha=0.9)

    fig.suptitle("PPE-EYE checkpoints on the recall / false-alarm plane\n"
                 "hollow ring = the conf 0.10 operating point Amsar runs at · "
                 "up and to the left is better", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = FIGS / "gains_recall_vs_falsealarm.png"
    fig.savefig(out, dpi=150)
    print(f"[fig_gains] wrote {out}")


if __name__ == "__main__":
    main()
