#!/usr/bin/env python3
"""
report.py — aggregate experiments/results/*.json into REPORT.md, STATUS.md and
figures. Safe to run any time; plots only what exists.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METRIC_COLS = [
    ("test_mAP50", "mAP50"),
    ("test_mAP50_95", "mAP50-95"),
    ("test_precision", "P"),
    ("test_recall", "R"),
    ("latency_ms_mean", "lat ms"),
    ("fps_mean", "fps"),
    ("train_minutes", "train min"),
]
PRIOR_BEST_MAP5095 = 0.14  # lightaug_yolo11s, for the go/no-go column


def _rows():
    _, exps = lib.load_registry()
    out = []
    for e in exps:
        r = lib.load_result(e["id"]) or {}
        out.append({**e, "result": r, "status": r.get("status", "pending")})
    return out


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 100 else f"{v:.1f}"
    return "-" if v is None else str(v)


def write_status(rows):
    buckets = {"done": [], "running": [], "pending": [], "failed": [], "skipped": [], "blocked": []}
    for r in rows:
        buckets.setdefault(r["status"], []).append(r["id"])
    lines = ["# Experiment status", ""]
    for k in ["running", "done", "pending", "failed", "blocked", "skipped"]:
        ids = buckets.get(k, [])
        lines.append(f"**{k}** ({len(ids)}): {', '.join(ids) if ids else '—'}")
        lines.append("")
    (lib.RESULTS_DIR / "STATUS.md").write_text("\n".join(lines), encoding="utf-8")


def write_report(rows):
    lib.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    md = ["# PPE-EYE enhancement — results", "",
          f"Prior best (lightaug_yolo11s): **mAP50-95 ≈ {PRIOR_BEST_MAP5095}**. "
          "`Δ` below is vs that.", ""]
    weeks = sorted({r["week"] for r in rows})
    for w in weeks:
        wr = [r for r in rows if r["week"] == w]
        md.append(f"## Week {w}")
        md.append("")
        hdr = ["id", "item", "status"] + [h for _, h in METRIC_COLS] + ["Δ mAP50-95", "desc"]
        md.append("| " + " | ".join(hdr) + " |")
        md.append("|" + "|".join(["---"] * len(hdr)) + "|")
        for r in wr:
            res = r["result"]
            cells = [r["id"], str(r.get("item", "-")), r["status"]]
            for key, _ in METRIC_COLS:
                cells.append(_fmt(res.get(key)))
            d = res.get("test_mAP50_95")
            cells.append(f"{d - PRIOR_BEST_MAP5095:+.4f}" if isinstance(d, (int, float)) else "-")
            cells.append(r["desc"])
            md.append("| " + " | ".join(cells) + " |")
        md.append("")

        # per-class table for the week, if any done
        done = [r for r in wr if r["status"] == "done" and r["result"].get("per_class_AP50")]
        if done:
            classes = sorted({c for r in done for c in r["result"]["per_class_AP50"]})
            md.append(f"### Week {w} per-class AP50")
            md.append("")
            md.append("| id | " + " | ".join(classes) + " |")
            md.append("|" + "|".join(["---"] * (len(classes) + 1)) + "|")
            for r in done:
                pc = r["result"]["per_class_AP50"]
                md.append("| " + r["id"] + " | " + " | ".join(_fmt(pc.get(c)) for c in classes) + " |")
            md.append("")

    figs = sorted(lib.FIGS_DIR.glob("*.png")) if lib.FIGS_DIR.exists() else []
    if figs:
        md.append("## Figures")
        md.append("")
        for f in figs:
            md.append(f"![{f.stem}](figs/{f.name})")
            md.append("")
    (lib.RESULTS_DIR / "REPORT.md").write_text("\n".join(md), encoding="utf-8")


def _done(rows, exp_id):
    for r in rows:
        if r["id"] == exp_id and r["status"] == "done":
            return r["result"]
    return None


def fig_lr_sweep(rows):
    ids = ["w1_s_baseline_repro", "w1_s_sgd_1e2", "w1_s_adamw_1e3", "w1_s_adamw_1e4"]
    data = [(i, _done(rows, i)) for i in ids]
    data = [(i, r) for i, r in data if r]
    if len(data) < 2:
        return
    labels = [i.replace("w1_s_", "") for i, _ in data]
    vals = [r["test_mAP50_95"] for _, r in data]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, vals, color="#4C72B0")
    ax.axhline(PRIOR_BEST_MAP5095, ls="--", c="grey", label=f"prior best {PRIOR_BEST_MAP5095}")
    ax.set_ylabel("test mAP50-95")
    ax.set_title("Week 1 — learning-rate / optimizer sweep (yolo11s)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(lib.FIGS_DIR / "w1_lr_sweep.png", dpi=120)
    plt.close(fig)


def fig_resolution(rows):
    pts = []
    for exp_id, sz in [("w1_s_adamw_1e3", 640), ("w1_s_res_960", 960), ("w1_s_res_1280", 1280)]:
        r = _done(rows, exp_id)
        if r:
            pts.append((sz, r["test_mAP50_95"], r.get("latency_ms_mean")))
    if len(pts) < 2:
        return
    pts.sort()
    xs = [p[0] for p in pts]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(xs, [p[1] for p in pts], "o-", color="#4C72B0", label="mAP50-95")
    ax1.set_xlabel("input resolution (imgsz)")
    ax1.set_ylabel("test mAP50-95", color="#4C72B0")
    ax2 = ax1.twinx()
    ax2.plot(xs, [p[2] for p in pts], "s--", color="#C44E52", label="latency ms")
    ax2.set_ylabel("latency ms (1 img, 4060 Ti)", color="#C44E52")
    ax1.set_title("Week 1 — resolution vs accuracy vs latency")
    ax1.set_xticks(xs)
    fig.tight_layout()
    fig.savefig(lib.FIGS_DIR / "w1_resolution.png", dpi=120)
    plt.close(fig)


def fig_aug_ablation(rows):
    ids = ["w1_s_noaug", "w1_s_adamw_1e3", "w1_s_domainaug", "w1_s_imbalance"]
    data = [(i.replace("w1_s_", ""), _done(rows, i)) for i in ids]
    data = [(l, r) for l, r in data if r]
    if len(data) < 2:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([l for l, _ in data], [r["test_mAP50_95"] for _, r in data], color="#55A868")
    ax.set_ylabel("test mAP50-95")
    ax.set_title("Week 1 — augmentation ablation (base = adamw_1e3)")
    fig.tight_layout()
    fig.savefig(lib.FIGS_DIR / "w1_aug_ablation.png", dpi=120)
    plt.close(fig)


def fig_size_latency(rows):
    pts = []
    for exp_id, name in [("w1_n_adamw_1e3", "n"), ("w1_s_adamw_1e3", "s"), ("w1_m_adamw_1e3", "m")]:
        r = _done(rows, exp_id)
        if r and r.get("latency_ms_mean"):
            pts.append((name, r["latency_ms_mean"], r["test_mAP50_95"]))
    if len(pts) < 2:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([p[1] for p in pts], [p[2] for p in pts], "o-", color="#8172B2")
    for name, x, y in pts:
        ax.annotate(f"yolo11{name}", (x, y), textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel("latency ms (1 img, 4060 Ti)")
    ax.set_ylabel("test mAP50-95")
    ax.set_title("Week 1 — size / accuracy / latency")
    fig.tight_layout()
    fig.savefig(lib.FIGS_DIR / "w1_size_latency.png", dpi=120)
    plt.close(fig)


def main():
    lib.FIGS_DIR.mkdir(parents=True, exist_ok=True)
    rows = _rows()
    write_status(rows)
    for fn in (fig_lr_sweep, fig_resolution, fig_aug_ablation, fig_size_latency):
        try:
            fn(rows)
        except Exception as e:  # noqa: BLE001
            print(f"[report] {fn.__name__} failed: {e}")
    write_report(rows)
    print(f"[report] wrote {lib.RESULTS_DIR / 'REPORT.md'} and STATUS.md")


if __name__ == "__main__":
    main()
