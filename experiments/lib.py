"""Shared helpers for the enhancement experiment loop.

No pandas / sklearn — stdlib + numpy + yaml only, so it runs in the PPE/ venv
as shipped.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / "experiments"
RESULTS_DIR = EXP_DIR / "results"
FIGS_DIR = RESULTS_DIR / "figs"
REGISTRY = EXP_DIR / "registry.yaml"
JSONL = RESULTS_DIR / "results.jsonl"

TERMINAL = {"done", "failed", "skipped", "blocked"}


def load_registry() -> tuple[dict, list[dict]]:
    """Return (defaults, [experiment dicts]) with params merged over defaults."""
    doc = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    defaults = doc.get("defaults", {}) or {}
    exps = []
    for e in doc["experiments"]:
        merged = dict(defaults)
        merged.update(e.get("params", {}) or {})
        e = dict(e)
        e["params"] = merged
        e.setdefault("deps", [])
        exps.append(e)
    return defaults, exps


def result_path(exp_id: str) -> Path:
    return RESULTS_DIR / f"{exp_id}.json"


def load_result(exp_id: str) -> dict | None:
    p = result_path(exp_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def save_result(row: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = result_path(row["id"])
    p.write_text(json.dumps(row, indent=2), encoding="utf-8")
    if row.get("status") in TERMINAL:
        with JSONL.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
    return p


def status_of(exp_id: str) -> str:
    r = load_result(exp_id)
    if r is None:
        return "pending"
    return r.get("status", "pending")


def deps_ready(exp: dict) -> bool:
    return all(status_of(d) == "done" for d in exp.get("deps", []))


def next_experiment() -> dict | None:
    """First experiment with no terminal result whose deps are all done."""
    _, exps = load_registry()
    for e in exps:
        st = status_of(e["id"])
        if st in TERMINAL:
            continue
        if st == "running":
            return None  # something is already in flight
        if deps_ready(e):
            return e
    return None


def running_experiment() -> str | None:
    _, exps = load_registry()
    for e in exps:
        if status_of(e["id"]) == "running":
            return e["id"]
    return None


def mark_running(exp: dict) -> None:
    save_result({
        "id": exp["id"], "week": exp["week"], "tier": exp["tier"],
        "item": exp.get("item"), "kind": exp["kind"], "desc": exp["desc"],
        "status": "running", "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": exp["params"],
    })


def latency_ms(model, imgsz: int, device: str = "0", n: int = 60, warmup: int = 10):
    """Single-image forward latency, mean over n runs after warmup."""
    import numpy as np

    dummy = (np.random.rand(imgsz, imgsz, 3) * 255).astype("uint8")
    for _ in range(warmup):
        model.predict(dummy, imgsz=imgsz, device=device, verbose=False)
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        model.predict(dummy, imgsz=imgsz, device=device, verbose=False)
        t.append((time.perf_counter() - t0) * 1000.0)
    t.sort()
    return {
        "latency_ms_mean": round(sum(t) / len(t), 2),
        "latency_ms_p50": round(t[len(t) // 2], 2),
        "latency_ms_p95": round(t[int(len(t) * 0.95)], 2),
        "fps_mean": round(1000.0 / (sum(t) / len(t)), 1),
    }
