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


DATA_LABEL = {
    "data/data.yaml": "pictor",
    "data/sh17_compliance/data.yaml": "sh17-v1",
    "data/sh17_compliance/data_ood.yaml": "sh17-v1-ood",
    "data/sh17c_v2/data.yaml": "sh17-v2",
    "data/sh17c_v2/data_holdout.yaml": "sh17-closeup",
    "data/mixed_pictor_sh17/data.yaml": "pictor(test)",
    "data/finest/data.yaml": "pictor(test)",
}
_SUPPORT_CACHE: dict = {}


def _testset(row) -> str:
    """Which data the row's test_* metrics were scored on. Without this the
    report silently compares w4_sh17_train's SH17 mAP against everyone else's
    Pictor mAP."""
    d = (row.get("result", {}).get("params") or row.get("params") or {}).get("data")
    if not d:
        return "-"
    d = str(d).replace("\\", "/")
    for k, v in DATA_LABEL.items():
        if d.endswith(k):
            return v
    return Path(d).parent.name


def _support(done_rows, cls) -> int:
    """Test-set instance count for a class, so a mean AP over classes with 456
    and 6 instances is not read as if both were equally trustworthy. Counted on
    Pictor test, which is the shared test bed for every comparable row."""
    if not _SUPPORT_CACHE:
        names = ["W", "WH", "WHV", "WV"]
        lbl = lib.ROOT / "data" / "test" / "labels"
        for f in lbl.glob("*.txt"):
            for line in f.read_text().splitlines():
                t = line.split()
                if len(t) >= 5:
                    i = int(float(t[0]))
                    if i < len(names):
                        _SUPPORT_CACHE[names[i]] = _SUPPORT_CACHE.get(names[i], 0) + 1
    return _SUPPORT_CACHE.get(cls, 0)


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
        hdr = (["id", "item", "status", "test set"] + [h for _, h in METRIC_COLS]
               + ["Δ mAP50-95", "desc"])
        md.append("| " + " | ".join(hdr) + " |")
        md.append("|" + "|".join(["---"] * len(hdr)) + "|")
        for r in wr:
            res = r["result"]
            if res.get("eval") and r["status"] == "done":
                continue  # eval rows get their own block below
            cells = [r["id"], str(r.get("item", "-")), r["status"],
                     _testset(r)]
            for key, _ in METRIC_COLS:
                cells.append(_fmt(res.get(key)))
            d = res.get("test_mAP50_95")
            cells.append(f"{d - PRIOR_BEST_MAP5095:+.4f}" if isinstance(d, (int, float)) else "-")
            cells.append(r["desc"])
            md.append("| " + " | ".join(cells) + " |")
        md.append("")

        _week_evals(md, wr, w)
        _week_week6(md, wr, w)

        # per-class table for the week, if any done
        done = [r for r in wr if r["status"] == "done" and r["result"].get("per_class_AP50")]
        if done:
            classes = sorted({c for r in done for c in r["result"]["per_class_AP50"]})
            md.append(f"### Week {w} per-class AP50")
            md.append("")
            md.append("| id | " + " | ".join(
                f"{c} (n={_support(done, c)})" if _support(done, c) else c
                for c in classes) + " |")
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


def _week_week6(md, wr, w):
    """Render the kinds whose results are not a single metric row: crossdata,
    the architecture ablations, distillation and the temporal-vote proxy."""
    special = [r for r in wr if r["status"] == "done"
               and r["kind"] in ("crossdata", "arch", "track")]
    if not special:
        return
    for r in special:
        res, k = r["result"], r["kind"]
        md.append(f"**{r['id']}** ({k})")
        md.append("")
        if k == "crossdata":
            md.append(f"`{Path(res['weights']).parent.parent.name}` on **{res['target']}** "
                      f"(never trained on it) — mAP50 {res['test_mAP50']}, "
                      f"mAP50-95 {res['test_mAP50_95']}")
            md.append("")
            for rule, v in res.get("person_level", {}).items():
                md.append(f"| conf ({rule}) | violation recall | false-alarm | verdict acc | undetected |")
                md.append("|---|---|---|---|---|")
                for row in v["sweep"]:
                    md.append(f"| {row['conf']} | {row['violation_recall']} | "
                              f"{row['false_alarm_rate']} | {row['verdict_accuracy']} | "
                              f"{row['undetected_persons']} |")
                md.append("")
        elif k == "arch" and res.get("variants"):
            md.append("| variant | params M | GFLOPs | mAP50 | mAP50-95 | lat ms | fps | train min |")
            md.append("|---|---|---|---|---|---|---|---|")
            for v, x in res["variants"].items():
                md.append(f"| {v} | {x.get('params_M')} | {x.get('GFLOPs')} | "
                          f"{x.get('test_mAP50')} | {x.get('test_mAP50_95')} | "
                          f"{x.get('latency_ms_mean')} | {x.get('fps_mean')} | "
                          f"{x.get('train_minutes')} |")
            md.append("")
            md.append(f"_{res.get('note', '')}_")
            md.append("")
        elif k == "arch" and res.get("curve"):
            md.append(f"single pass {res['single_pass']['latency_ms_mean']} ms · "
                      f"person stage {res['person_stage']['latency_ms_mean']} ms · "
                      f"per-crop classifier {res['per_crop_classifier_ms']} ms")
            md.append("")
            md.append("| workers in frame | single-pass ms | cascade ms | cascade / single |")
            md.append("|---|---|---|---|")
            for row in res["curve"]:
                md.append(f"| {row['workers']} | {row['single_pass_ms']} | "
                          f"{row['cascade_ms']} | {row['cascade_over_single']}x |")
            md.append("")
            md.append(f"_{res.get('note', '')}_")
            md.append("")
        elif k == "arch" and res.get("formats"):
            md.append("| format | mAP50 | mAP50-95 | retention | lat ms | fps | file MB |")
            md.append("|---|---|---|---|---|---|---|")
            for f, x in res["formats"].items():
                if "error" in x:
                    md.append(f"| {f} | — | — | — | — | — | _{x['error'][:70]}_ |")
                    continue
                md.append(f"| {f} | {x.get('test_mAP50')} | {x.get('test_mAP50_95')} | "
                          f"{x.get('mAP50_retention')} | {x.get('latency_ms_mean')} | "
                          f"{x.get('fps_mean')} | {x.get('file_MB')} |")
            md.append("")
        elif k == "arch" and res.get("models"):
            md.append("| model | mAP50 | mAP50-95 | P | R | lat ms | fps | train min |")
            md.append("|---|---|---|---|---|---|---|---|")
            for m, x in res["models"].items():
                if "error" in x:
                    md.append(f"| {m} | — | — | — | — | — | — | _{x['error'][:60]}_ |")
                    continue
                md.append(f"| {m} | {x.get('test_mAP50')} | {x.get('test_mAP50_95')} | "
                          f"{x.get('test_precision')} | {x.get('test_recall')} | "
                          f"{x.get('latency_ms_mean')} | {x.get('fps_mean')} | "
                          f"{x.get('train_minutes')} |")
            md.append("")
        elif k == "track":
            md.append("Confidence-weighted vote over N noisy re-observations "
                      f"(rule = {res['rule']}). **{res.get('note', '')}**")
            md.append("")
            md.append("| N | conf | violation recall | violation precision | false-alarm | verdict acc | undetected |")
            md.append("|---|---|---|---|---|---|---|")
            for n, v in res["by_n"].items():
                b = v["best_f1"]
                md.append(f"| {n} | {b['conf']} | {b['violation_recall']} | "
                          f"{b['violation_precision']} | {b['false_alarm_rate']} | "
                          f"{b['verdict_accuracy']} | {b['undetected_persons']} |")
            md.append("")


def _week_evals(md, wr, w):
    evals = [r for r in wr if r["status"] == "done" and r["result"].get("eval")]
    if not evals:
        return
    md.append(f"### Week {w} — person-level / audit results")
    md.append("")
    for r in evals:
        res = r["result"]
        e = res["eval"]
        md.append(f"**{r['id']}** ({e})")
        md.append("")
        if e == "honest_person_level":
            md.append("| rule | conf | violation recall | false-alarm | verdict acc | undetected |")
            md.append("|---|---|---|---|---|---|")
            for rule, v in res["rules"].items():
                md.append(f"| {rule} | {v.get('conf')} | {v['violation_recall']} | "
                          f"{v['false_alarm_rate']} | {v['verdict_accuracy']} | "
                          f"{v['undetected_persons']} |")
        elif e == "clean_vs_noisy":
            md.append("| rule | recall noisy→clean | false-alarm | verdict acc noisy→clean | Δ recall |")
            md.append("|---|---|---|---|---|")
            for rule in res["noisy"]:
                nz, cl = res["noisy"][rule], res["clean"][rule]
                dl = res["delta_clean_minus_noisy"][rule]
                md.append(f"| {rule} | {nz['violation_recall']} → {cl['violation_recall']} | "
                          f"{nz['false_alarm_rate']} | {nz['verdict_accuracy']} → "
                          f"{cl['verdict_accuracy']} | {dl['violation_recall']:+} |")
        elif e == "sahi_upperbound":
            ff = next((x for x in wr if x["id"] == "w2_honest_eval"), None)
            ff = ff["result"]["rules"] if ff and ff["result"].get("rules") else {}
            md.append(f"slice {res['slice_px']}px / overlap {res['overlap']} · "
                      f"{res.get('minutes', '?')} min")
            md.append("")
            md.append("| rule | recall full → SAHI | undetected full → SAHI | false-alarm SAHI |")
            md.append("|---|---|---|---|")
            for rule, v in res["rules"].items():
                f = ff.get(rule, {})
                md.append(f"| {rule} | {f.get('violation_recall', '?')} → {v['violation_recall']} "
                          f"| {f.get('undetected_persons', '?')} → {v['undetected_persons']} "
                          f"| {v['false_alarm_rate']} |")
        elif e == "label_audit":
            comp = res["composition"]["splits"]
            fl = res["flags"]
            md.append("| split | images | objects | obj/img | box area p10/p50/p90 | thin (<30) |")
            md.append("|---|---|---|---|---|---|")
            for s, x in comp.items():
                md.append(f"| {s} | {x['images']} | {x['objects']} | "
                          f"{x['objects_per_image_mean']} | "
                          f"{'/'.join(str(v) for v in x['box_area_frac_p10_p50_p90'])} | "
                          f"{', '.join(x['thin_classes_lt30']) or '—'} |")
            md.append("")
            md.append(f"Flags ({fl['flagged_boxes']} on {comp['test']['objects']} test boxes): "
                      + ", ".join(f"{k}={v}" for k, v in fl["flag_counts"].items()))
            md.append("")
            ac = fl["auto_clean"]
            md.append(f"Auto-clean ({ac['rule']}): dropped {ac['dropped']}, kept {ac['kept']} "
                      f"→ `{Path(ac['clean_label_dir']).name}/`. Review queue: `{Path(fl['flags_csv']).name}`.")
        md.append("")


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


def fig_honest_eval(rows):
    import json
    curves = {}
    for rule in ("strict", "helmet"):
        f = lib.RESULTS_DIR / f"w2_honest_eval_{rule}.json"
        if f.exists():
            curves[rule] = json.loads(f.read_text())["sweep"]
    if not curves:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for rule, sweep in curves.items():
        conf = [r["conf"] for r in sweep]
        ax1.plot(conf, [r["violation_recall"] for r in sweep], "o-", label=f"{rule} recall")
        ax1.plot(conf, [r["false_alarm_rate"] for r in sweep], "s--", label=f"{rule} false-alarm")
        ax2.plot([r["false_alarm_rate"] for r in sweep],
                 [r["violation_recall"] for r in sweep], "o-", label=rule)
    ax1.set_xlabel("confidence threshold")
    ax1.set_ylabel("rate")
    ax1.set_title("Week 2 — violation recall & false-alarm vs conf")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax2.set_xlabel("false-alarm rate")
    ax2.set_ylabel("violation recall")
    ax2.set_title("recall / false-alarm trade-off")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(lib.FIGS_DIR / "w2_honest_eval.png", dpi=120)
    plt.close(fig)


def main():
    lib.FIGS_DIR.mkdir(parents=True, exist_ok=True)
    rows = _rows()
    write_status(rows)
    for fn in (fig_lr_sweep, fig_resolution, fig_aug_ablation, fig_size_latency,
               fig_honest_eval):
        try:
            fn(rows)
        except Exception as e:  # noqa: BLE001
            print(f"[report] {fn.__name__} failed: {e}")
    write_report(rows)
    print(f"[report] wrote {lib.RESULTS_DIR / 'REPORT.md'} and STATUS.md")


if __name__ == "__main__":
    main()
