#!/usr/bin/env python3
"""
week6.py - handlers for the experiment kinds that were declared in
registry.yaml but had no implementation: `crossdata`, `arch`, `distill`, `track`.

Each one answers a schedule item and writes a report row. They are kept out of
run.py so run.py stays a dispatcher.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import lib


# ------------------------------------------------------------- crossdata ----

def crossdata(exp: dict) -> dict:
    """Item #9 - evaluate a fixed checkpoint on a dataset it never trained on,
    at both object level (mAP) and person level (violation recall / false alarm).

    The person-level half is the one that matters: Week 4 showed object mAP does
    not track deployment quality here, and the whole point of a generalisation
    test is to predict deployment behaviour on a new site.
    """
    from ultralytics import YOLO
    import honest_eval as he

    p = exp["params"]
    target = p.get("target", "sh17")
    data = p.get("data")
    if not data:
        raise FileNotFoundError(
            f"crossdata target {target!r} has no dataset on this machine. "
            f"Set params.data to a data.yaml, or mark the experiment blocked.")
    data = data if Path(data).is_absolute() else str(lib.ROOT / data)
    if not Path(data).exists():
        raise FileNotFoundError(f"{data} does not exist (target={target})")

    weights = p["weights"]
    weights = weights if Path(weights).is_absolute() else str(lib.ROOT / weights)
    imgsz = int(p.get("imgsz", 640))
    device = p.get("device", "0")
    split = p.get("split", "test")

    m = YOLO(weights)
    v = m.val(data=data, split=split, imgsz=imgsz, device=device,
              project=str(lib.ROOT / "runs" / "ppe_enh"),
              name=f"{exp['id']}_val", exist_ok=True, plots=False)

    sweep = [float(c) for c in str(p.get("conf_sweep", "0.10,0.20,0.30,0.50")).split(",")]
    person = {}
    for rule in p.get("rules", ["helmet"]):
        res = he.run(weights, data, split, rule, imgsz, device,
                     float(p.get("iou", 0.5)), sweep)
        (lib.RESULTS_DIR / f"{exp['id']}_{rule}.json").write_text(
            json.dumps(res, indent=2), encoding="utf-8")
        person[rule] = {
            "sweep": [{k: r[k] for k in ("conf", "violation_recall",
                                         "false_alarm_rate", "verdict_accuracy",
                                         "undetected_persons")}
                      for r in res["sweep"]],
            "best_f1": res["best_f1_row"],
        }

    return {
        "target": target, "weights": weights, "data": data, "split": split,
        "test_mAP50": round(float(v.box.map50), 4),
        "test_mAP50_95": round(float(v.box.map), 4),
        "test_precision": round(float(v.box.mp), 4),
        "test_recall": round(float(v.box.mr), 4),
        "per_class_AP50": {v.names[i]: round(float(x), 4)
                           for i, x in zip(v.box.ap_class_index, v.box.ap50)},
        "person_level": person,
    }


# ------------------------------------------------------------------ arch ----

def arch(exp: dict) -> dict:
    """Items #13-#17. `variant` selects what is being ablated."""
    p = exp["params"]
    variant = p.get("variant")
    if variant == "two_stage":
        return _two_stage(exp)
    if variant == "quant":
        return _quant(exp)
    if variant == "baselines":
        return _baselines(exp)
    return _train_variants(exp)


def _train_variants(exp: dict) -> dict:
    """Train one or more yolo11 neck/head variants on the same data + recipe as
    the reference run, so the only thing that differs is the architecture."""
    from ultralytics import YOLO
    import arch as archmod

    archmod.register_modules()
    p = exp["params"]
    variants = p.get("variants") or [p["variant"]]
    scale = p.get("scale", "s")
    data = str(lib.ROOT / p["data"]) if not Path(p["data"]).is_absolute() else p["data"]
    cfg_dir = lib.EXP_DIR / "cfg"
    imgsz = int(p.get("imgsz", 640))
    device = p.get("device", "0")

    train_keys = {"epochs", "patience", "batch", "seed", "optimizer", "lr0",
                  "cos_lr", "warmup_epochs", "close_mosaic", "momentum"}
    kw = {k: v for k, v in p.items() if k in train_keys}

    rows = {}
    for v in variants:
        name = f"{exp['id']}_{v}"
        yaml_path = archmod.build_yaml(v, scale, int(p.get("nc", 4)), cfg_dir)
        t0 = time.time()
        m = YOLO(str(yaml_path))
        stats = archmod.model_stats(m)
        # Transfer COCO weights for every layer whose shape still matches, so the
        # variants start from the same backbone as the Week-1/4 runs and the only
        # randomly-initialised parameters are the blocks being ablated.
        init = p.get("init_weights", f"yolo11{scale}.pt")
        m.load(str(lib.ROOT / init) if (lib.ROOT / init).exists() else init)
        m.train(data=data, imgsz=imgsz, device=device,
                project=str(lib.ROOT / "runs" / "ppe_enh"), name=name,
                exist_ok=True, deterministic=True, val=True, plots=False, **kw)
        train_min = round((time.time() - t0) / 60, 1)
        mt = m.val(data=data, split="test", imgsz=imgsz, device=device,
                   project=str(lib.ROOT / "runs" / "ppe_enh"),
                   name=f"{name}_test", exist_ok=True, plots=False)
        rows[v] = {
            **stats,
            **lib.latency_ms(m, imgsz, device),
            "train_minutes": train_min,
            "test_mAP50": round(float(mt.box.map50), 4),
            "test_mAP50_95": round(float(mt.box.map), 4),
            "test_precision": round(float(mt.box.mp), 4),
            "test_recall": round(float(mt.box.mr), 4),
            "per_class_AP50": {mt.names[i]: round(float(x), 4)
                               for i, x in zip(mt.box.ap_class_index, mt.box.ap50)},
            "weights": str(lib.ROOT / "runs" / "ppe_enh" / name / "weights" / "best.pt"),
        }
        print(f"[arch] {v}: mAP50={rows[v]['test_mAP50']} "
              f"params={stats['params_M']}M GFLOPs={stats['GFLOPs']}", flush=True)

    return {"scale": scale, "data": data, "variants": rows,
            "note": "COCO weights transferred for every shape-matching layer; "
                    "only the ablated blocks start random. The `none` variant is "
                    "the control - same code path, stock yolo11 neck."}


def _two_stage(exp: dict) -> dict:
    """Item #15 - the cascade Amsar ships, costed properly.

    A single-pass model is O(1) in worker count. A detector -> crop -> classifier
    cascade is O(n): every worker adds a classifier forward. This measures both
    and reports the crossover, which is the real argument against the cascade for
    crowded footage (Pictor test averages 3.16 workers/image, worst case 20+).
    """
    from ultralytics import YOLO

    p = exp["params"]
    imgsz = int(p.get("imgsz", 640))
    device = p.get("device", "0")
    crop_sz = int(p.get("crop_size", 128))
    single_w = p["single_weights"]
    single_w = single_w if Path(single_w).is_absolute() else str(lib.ROOT / single_w)

    single = YOLO(single_w)
    person = YOLO(str(lib.ROOT / p.get("person_model", "yolo11n.pt")))

    single_lat = lib.latency_ms(single, imgsz, device)
    person_lat = lib.latency_ms(person, imgsz, device)

    # per-crop classifier cost, measured as a batch-1 classifier forward
    import torch
    from ultralytics.nn.modules import Conv

    dev = torch.device(f"cuda:{device}" if str(device).isdigit() else str(device))
    clf = torch.nn.Sequential(
        Conv(3, 32, 3, 2), Conv(32, 64, 3, 2), Conv(64, 128, 3, 2),
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
        torch.nn.Linear(128, 4)).to(dev).eval()
    x = torch.rand(1, 3, crop_sz, crop_sz, device=dev)
    with torch.no_grad():
        for _ in range(20):
            clf(x)
        torch.cuda.synchronize() if dev.type == "cuda" else None
        ts = []
        for _ in range(100):
            t0 = time.perf_counter()
            clf(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
    per_crop_ms = round(float(np.median(ts)), 3)

    workers = p.get("worker_counts", [1, 2, 3, 5, 10, 20, 40])
    curve = []
    for n in workers:
        cascade = person_lat["latency_ms_mean"] + n * per_crop_ms
        curve.append({
            "workers": n,
            "single_pass_ms": single_lat["latency_ms_mean"],
            "cascade_ms": round(cascade, 2),
            "cascade_over_single": round(cascade / single_lat["latency_ms_mean"], 2),
        })
    crossover = next((r["workers"] for r in curve
                      if r["cascade_ms"] > r["single_pass_ms"]), None)

    return {
        "single_pass": {"weights": single_w, **single_lat},
        "person_stage": {"model": p.get("person_model", "yolo11n.pt"), **person_lat},
        "per_crop_classifier_ms": per_crop_ms,
        "crop_size": crop_sz,
        "curve": curve,
        "cascade_cheaper_below_workers": crossover,
        "note": "classifier is a 3-conv stub - a deliberate lower bound on the "
                "cascade's stage-2 cost. A real PPE classifier is slower, so the "
                "crossover reported here is optimistic for the cascade.",
    }


def _quant(exp: dict) -> dict:
    """Item #17 - post-training quantisation. What FP16 / INT8 cost in accuracy.

    QAT proper is not run: it needs a training loop with fake-quant observers,
    and Week 6's premise (architecture has headroom) was not supported by the
    variants above, so a QAT recovery experiment has nothing to recover *to*.
    PTQ is the number that actually gates deployment.
    """
    from ultralytics import YOLO

    p = exp["params"]
    weights = p["weights"]
    weights = weights if Path(weights).is_absolute() else str(lib.ROOT / weights)
    data = str(lib.ROOT / p["data"]) if not Path(p["data"]).is_absolute() else p["data"]
    imgsz = int(p.get("imgsz", 640))
    device = p.get("device", "0")

    rows = {}
    m = YOLO(weights)
    v = m.val(data=data, split="test", imgsz=imgsz, device=device,
              project=str(lib.ROOT / "runs" / "ppe_enh"),
              name=f"{exp['id']}_fp32", exist_ok=True, plots=False)
    rows["fp32"] = {"test_mAP50": round(float(v.box.map50), 4),
                    "test_mAP50_95": round(float(v.box.map), 4),
                    **lib.latency_ms(m, imgsz, device),
                    "file_MB": round(Path(weights).stat().st_size / 1e6, 2)}

    # FP16 needs no export - same graph at half precision, and it is what a
    # Jetson TensorRT FP16 engine approximates. Both accuracy and latency go
    # through ultralytics' own `half=` flag: hand-calling `model.half()` and then
    # predicting on a uint8 array raises a dtype mismatch, since the preprocessor
    # still produces float32.
    try:
        import time as _t
        import numpy as _np
        hv = m.val(data=data, split="test", imgsz=imgsz, device=device, half=True,
                   project=str(lib.ROOT / "runs" / "ppe_enh"),
                   name=f"{exp['id']}_fp16", exist_ok=True, plots=False)
        dummy = (_np.random.rand(imgsz, imgsz, 3) * 255).astype("uint8")
        for _ in range(10):
            m.predict(dummy, imgsz=imgsz, device=device, half=True, verbose=False)
        ts = []
        for _ in range(60):
            t0 = _t.perf_counter()
            m.predict(dummy, imgsz=imgsz, device=device, half=True, verbose=False)
            ts.append((_t.perf_counter() - t0) * 1000.0)
        ts.sort()
        rows["fp16_torch"] = {
            "test_mAP50": round(float(hv.box.map50), 4),
            "test_mAP50_95": round(float(hv.box.map), 4),
            "latency_ms_mean": round(sum(ts) / len(ts), 2),
            "latency_ms_p50": round(ts[len(ts) // 2], 2),
            "latency_ms_p95": round(ts[int(len(ts) * 0.95)], 2),
            "fps_mean": round(1000.0 / (sum(ts) / len(ts)), 1),
            "file_MB": round(rows["fp32"]["file_MB"] / 2, 2),
        }
    except Exception as e:  # noqa: BLE001
        rows["fp16_torch"] = {"error": f"{type(e).__name__}: {e}"}

    for fmt, kw, tag, dev in [("engine", {"half": True}, "fp16_tensorrt", device),
                              ("engine", {"int8": True, "data": data}, "int8_tensorrt", device),
                              ("onnx", {}, "onnx_cpu", "cpu"),
                              ("torchscript", {}, "torchscript", device)]:
        try:
            t0 = time.time()
            out = m.export(format=fmt, imgsz=imgsz, device=device, verbose=False, **kw)
            em = YOLO(str(out))
            ev = em.val(data=data, split="test", imgsz=imgsz, device=dev,
                        project=str(lib.ROOT / "runs" / "ppe_enh"),
                        name=f"{exp['id']}_{tag}", exist_ok=True, plots=False)
            rows[tag] = {
                "test_mAP50": round(float(ev.box.map50), 4),
                "test_mAP50_95": round(float(ev.box.map), 4),
                **lib.latency_ms(em, imgsz, dev, n=20 if dev == "cpu" else 60),
                "device": dev,
                "file_MB": round(Path(out).stat().st_size / 1e6, 2),
                "export_minutes": round((time.time() - t0) / 60, 1),
                "path": str(out),
            }
        except Exception as e:  # noqa: BLE001
            rows[tag] = {"error": f"{type(e).__name__}: {e}"}
        print(f"[quant] {tag}: {rows[tag]}", flush=True)

    base = rows["fp32"]["test_mAP50"]
    for k, r in rows.items():
        if "test_mAP50" in r and base:
            r["mAP50_retention"] = round(r["test_mAP50"] / base, 4)
    return {"weights": weights, "formats": rows,
            "note": "QAT is not run: it needs fake-quant observers in the training "
                    "loop, and the Week-6 variants found no architectural headroom "
                    "for it to recover. PTQ is what gates deployment anyway. "
                    "TensorRT rows depend on the `tensorrt` package being present "
                    "on the target; on a Jetson they are the numbers that matter."}


def _baselines(exp: dict) -> dict:
    """Item #16 - non-YOLO detectors on the same data and recipe."""
    from ultralytics import RTDETR, YOLO

    p = exp["params"]
    data = str(lib.ROOT / p["data"]) if not Path(p["data"]).is_absolute() else p["data"]
    imgsz = int(p.get("imgsz", 640))
    device = p.get("device", "0")
    train_keys = {"epochs", "patience", "batch", "seed", "cos_lr",
                  "warmup_epochs", "close_mosaic"}
    kw = {k: v for k, v in p.items() if k in train_keys}

    rows = {}
    for name in p.get("models", ["rtdetr-l.pt"]):
        tag = Path(name).stem
        try:
            Cls = RTDETR if "rtdetr" in name.lower() else YOLO
            m = Cls(name)
            t0 = time.time()
            m.train(data=data, imgsz=imgsz, device=device,
                    project=str(lib.ROOT / "runs" / "ppe_enh"),
                    name=f"{exp['id']}_{tag}", exist_ok=True, plots=False, **kw)
            mt = m.val(data=data, split="test", imgsz=imgsz, device=device,
                       project=str(lib.ROOT / "runs" / "ppe_enh"),
                       name=f"{exp['id']}_{tag}_test", exist_ok=True, plots=False)
            rows[tag] = {
                "train_minutes": round((time.time() - t0) / 60, 1),
                "test_mAP50": round(float(mt.box.map50), 4),
                "test_mAP50_95": round(float(mt.box.map), 4),
                "test_precision": round(float(mt.box.mp), 4),
                "test_recall": round(float(mt.box.mr), 4),
                "per_class_AP50": {mt.names[i]: round(float(x), 4)
                                   for i, x in zip(mt.box.ap_class_index, mt.box.ap50)},
                **lib.latency_ms(m, imgsz, device),
                "weights": str(lib.ROOT / "runs" / "ppe_enh" /
                               f"{exp['id']}_{tag}" / "weights" / "best.pt"),
            }
        except Exception as e:  # noqa: BLE001
            rows[tag] = {"error": f"{type(e).__name__}: {e}"}
        print(f"[baseline] {tag}: {rows[tag].get('test_mAP50', rows[tag])}", flush=True)
    return {"data": data, "models": rows}


# --------------------------------------------------------------- distill ----

def distill(exp: dict) -> dict:
    """Item #11 - knowledge distillation, done as teacher pseudo-labelling.

    Feature/logit KD needs a custom trainer hooking both networks' necks.
    Rather than fork ultralytics' loss, this runs the cheap, auditable variant:
    the teacher labels the training images, its boxes are merged into the ground
    truth (confident teacher boxes only), and the student trains on the union.
    That transfers the teacher's decision boundary on exactly the images where
    the human labels are thin - which Week 2 showed is this dataset's problem.
    """
    from ultralytics import YOLO

    p = exp["params"]
    teacher_w = p["teacher"]
    teacher_w = teacher_w if Path(teacher_w).is_absolute() else str(lib.ROOT / teacher_w)
    student = p.get("student", "yolo11n.pt")
    data = str(lib.ROOT / p["data"]) if not Path(p["data"]).is_absolute() else p["data"]
    imgsz = int(p.get("imgsz", 640))
    device = p.get("device", "0")
    conf = float(p.get("teacher_conf", 0.5))
    iou_dup = float(p.get("dedup_iou", 0.5))

    import yaml as _yaml
    dcfg = _yaml.safe_load(Path(data).read_text())
    base = Path(dcfg.get("path", Path(data).parent))
    train_entry = dcfg["train"]

    def list_images(entry):
        q = base / entry if not Path(entry).is_absolute() else Path(entry)
        if q.suffix == ".txt":
            return [Path(x.strip()) for x in q.read_text().splitlines() if x.strip()]
        return sorted(x for x in q.iterdir() if x.suffix.lower() in
                      (".jpg", ".jpeg", ".png", ".bmp"))

    imgs = list_images(train_entry)
    out_root = lib.ROOT / "data" / f"{exp['id']}_pseudo"
    (out_root / "labels").mkdir(parents=True, exist_ok=True)

    teacher = YOLO(teacher_w)
    added = kept = 0
    listing = []
    for i in range(0, len(imgs), 32):
        batch = imgs[i:i + 32]
        preds = teacher.predict([str(x) for x in batch], imgsz=imgsz, device=device,
                                conf=conf, verbose=False)
        for img, pr in zip(batch, preds):
            gt_file = Path(str(img).replace("/images/", "/labels/")
                           .replace("\\images\\", "\\labels\\")).with_suffix(".txt")
            gt = []
            if gt_file.exists():
                for ln in gt_file.read_text().splitlines():
                    t = ln.split()
                    if len(t) >= 5:
                        gt.append([int(float(t[0])), *[float(x) for x in t[1:5]]])
            kept += len(gt)
            for b, c in zip(pr.boxes.xywhn.tolist(), pr.boxes.cls.tolist()):
                if not any(_iou_xywhn(b, g[1:]) > iou_dup for g in gt):
                    gt.append([int(c), *b])
                    added += 1
            dst = out_root / "labels" / (img.stem + ".txt")
            dst.write_text("\n".join(f"{g[0]} {g[1]:.6f} {g[2]:.6f} {g[3]:.6f} {g[4]:.6f}"
                                     for g in gt) + "\n")
            listing.append(str(img))
    (out_root / "train.txt").write_text("\n".join(listing) + "\n")

    # a data.yaml whose labels/ resolves to the pseudo-labelled copy
    imgdir = out_root / "images"
    imgdir.mkdir(exist_ok=True)
    for src in imgs:
        link = imgdir / src.name
        if not link.exists():
            try:
                link.symlink_to(src)
            except OSError:
                link.write_bytes(Path(src).read_bytes())
    (out_root / "train_local.txt").write_text(
        "\n".join(str(imgdir / Path(x).name) for x in listing) + "\n")
    dy = out_root / "data.yaml"
    dy.write_text(
        f"path: {out_root.as_posix()}\ntrain: train_local.txt\n"
        f"val: {dcfg['val'] if str(dcfg['val']).startswith(('/', 'C:')) else (base / dcfg['val']).as_posix()}\n"
        f"test: {dcfg.get('test', dcfg['val']) if str(dcfg.get('test', dcfg['val'])).startswith(('/', 'C:')) else (base / dcfg.get('test', dcfg['val'])).as_posix()}\n"
        f"nc: {dcfg['nc']}\nnames: {dcfg['names']}\n")

    train_keys = {"epochs", "patience", "batch", "seed", "optimizer", "lr0",
                  "cos_lr", "warmup_epochs", "close_mosaic"}
    kw = {k: v for k, v in p.items() if k in train_keys}
    t0 = time.time()
    s = YOLO(student)
    s.train(data=str(dy), imgsz=imgsz, device=device,
            project=str(lib.ROOT / "runs" / "ppe_enh"), name=exp["id"],
            exist_ok=True, deterministic=True, val=True, plots=False, **kw)
    mt = s.val(data=data, split="test", imgsz=imgsz, device=device,
               project=str(lib.ROOT / "runs" / "ppe_enh"),
               name=f"{exp['id']}_test", exist_ok=True, plots=False)
    return {
        "teacher": teacher_w, "student": student,
        "teacher_conf": conf,
        "gt_boxes": kept, "pseudo_boxes_added": added,
        "train_minutes": round((time.time() - t0) / 60, 1),
        "test_mAP50": round(float(mt.box.map50), 4),
        "test_mAP50_95": round(float(mt.box.map), 4),
        "test_precision": round(float(mt.box.mp), 4),
        "test_recall": round(float(mt.box.mr), 4),
        "per_class_AP50": {mt.names[i]: round(float(x), 4)
                           for i, x in zip(mt.box.ap_class_index, mt.box.ap50)},
        **lib.latency_ms(s, imgsz, device),
        "weights": str(lib.ROOT / "runs" / "ppe_enh" / exp["id"] / "weights" / "best.pt"),
    }


def _iou_xywhn(a, b):
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    u = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / u if u > 0 else 0.0


# ----------------------------------------------------------------- track ----

def track(exp: dict) -> dict:
    """Item #10 - N-frame majority voting.

    There is no video in this repo, so a tracker cannot be run end to end. What
    *can* be measured honestly is the part of temporal voting that actually does
    the work: aggregating N noisy observations of the same worker. Each test
    image is re-inferred N times under independent mild capture noise (the
    severity-1 corruptions from Week 3 - what consecutive RTSP frames differ by),
    the per-person verdicts are pooled by confidence-weighted vote, and the
    person-level metrics are recomputed.

    This is a **proxy**: it models observation noise but not viewpoint change or
    tracker identity errors, so it is an upper bound on what Amsar's PPESmoother
    can buy. Labelled as such wherever it is reported.
    """
    import cv2
    import honest_eval as he
    from ultralytics import YOLO

    p = exp["params"]
    weights = p["weights"]
    weights = weights if Path(weights).is_absolute() else str(lib.ROOT / weights)
    data = str(lib.ROOT / p["data"]) if not Path(p["data"]).is_absolute() else p["data"]
    imgsz = int(p.get("imgsz", 640))
    device = p.get("device", "0")
    iou_thr = float(p.get("iou", 0.5))
    rule = p.get("rule", "helmet")
    n_sweep = p.get("n_sweep", [1, 3, 5, 7, 9])
    sweep = [float(c) for c in str(p.get("conf_sweep", "0.10,0.20,0.30,0.50")).split(",")]
    conf_floor = min(sweep)

    model = YOLO(weights)

    def make_predictor(n: int):
        """Emit one verdict per person from N noisy re-observations of the still.

        Frame 0 supplies the person boxes; frames 1..N-1 vote on the class of the
        box they overlap. The returned score is the mean confidence the winning
        class accumulated, so a verdict only the odd frame supports is damped -
        which is the whole point of temporal smoothing.
        """
        rng = np.random.default_rng(0)

        def jitter(im, k):
            if k == 0:
                return im
            out = im.astype(np.float32)
            out *= float(rng.uniform(0.92, 1.08))            # exposure flicker
            out += rng.normal(0, 3.0, out.shape)             # sensor noise
            out = np.clip(out, 0, 255).astype(np.uint8)
            if rng.random() < 0.5:
                out = cv2.GaussianBlur(out, (3, 3), 0.6)     # focus breathing
            return out

        def predict(im_path):
            im = cv2.imread(str(im_path))
            if im is None:
                return []
            frames = []
            for k in range(n):
                r = model.predict(jitter(im, k), imgsz=imgsz, device=device,
                                  conf=conf_floor, verbose=False)[0]
                frames.append((r.boxes.xyxy.cpu().numpy(),
                               r.boxes.cls.cpu().numpy().astype(int),
                               r.boxes.conf.cpu().numpy()))
            base_b, base_c, base_s = frames[0]
            votes = [{int(base_c[i]): float(base_s[i])} for i in range(len(base_b))]
            for fb, fc, fs in frames[1:]:
                for j in range(len(fb)):
                    best, bi = 0.0, -1
                    for i in range(len(base_b)):
                        v = _iou_xyxy(fb[j], base_b[i])
                        if v > best:
                            best, bi = v, i
                    if bi >= 0 and best >= 0.5:
                        votes[bi][int(fc[j])] = votes[bi].get(int(fc[j]), 0.0) + float(fs[j])
            out = []
            for i, vt in enumerate(votes):
                cls = max(vt, key=vt.get)
                out.append((cls, list(map(float, base_b[i])), vt[cls] / n))
            return out

        return predict

    results = {}
    for n in n_sweep:
        res = he.run(weights, data, p.get("split", "test"), rule, imgsz, device,
                     iou_thr, sweep, predictor=make_predictor(n))
        (lib.RESULTS_DIR / f"{exp['id']}_n{n}.json").write_text(
            json.dumps(res, indent=2), encoding="utf-8")
        results[n] = {"sweep": [{k: r[k] for k in
                                 ("conf", "violation_recall", "violation_precision",
                                  "false_alarm_rate", "verdict_accuracy",
                                  "undetected_persons", "TP", "FP", "TN", "FN")}
                                for r in res["sweep"]],
                      "best_f1": res["best_f1_row"]}
        print(f"[track] N={n}: best {results[n]['best_f1']}", flush=True)

    return {"weights": weights, "rule": rule, "n_sweep": n_sweep,
            "by_n": results, "proxy": True,
            "note": "no video in repo - N independent noisy re-observations of "
                    "each still, not true tracking. Models observation noise but "
                    "not viewpoint change or tracker identity error, so it is an "
                    "upper bound on what Amsar's PPESmoother can buy."}


def _iou_xyxy(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


HANDLERS = {"crossdata": crossdata, "arch": arch, "distill": distill, "track": track}
