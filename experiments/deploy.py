#!/usr/bin/env python3
"""
deploy.py - install a trained checkpoint into Amsar as `ppe_eye.pt`.

Amsar loads its single-pass engine from `ppe_models.PPE_EYE_MODEL_PATH`
("ppe_eye.pt", beside ppe_models.py) and refuses any model whose labels are not
the person-level compliance ontology - `ppe_eye.verdict_map()` returns {} for
PPE-object labels, because mistaking `ppe.pt` for a compliance model would
silently clear every worker on site.

This script runs that same check *before* copying, so a bad checkpoint is
rejected here rather than at 3am on a site. It also refuses to install a
checkpoint that has not passed experiments/gate.py, unless --force is given.

  PPE/Scripts/python.exe experiments/deploy.py --weights runs/ppe_enh/final_s_base/weights/best.pt
  PPE/Scripts/python.exe experiments/deploy.py --weights ... --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESEARCH = ROOT.parent
AMSAR = RESEARCH / "Amsar-AI--main" / "Amsar-AI--main"

# Amsar's own ontology check, duplicated so this script does not need to import
# from the Amsar tree. Kept byte-identical in behaviour to ppe_eye.verdict_map.
_CODES = {"W": (False, False), "WH": (True, False),
          "WV": (False, True), "WHV": (True, True)}


def verdict_map(names: dict) -> dict:
    out = {}
    for cid, nm in (names or {}).items():
        code = "".join(ch for ch in str(nm).upper() if ch.isalpha())
        if code not in _CODES:
            return {}
        out[int(cid)] = _CODES[code]
    return out if len(set(out.values())) > 1 else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dest", default=str(AMSAR / "ppe_eye.pt"))
    ap.add_argument("--gate", default="experiments/results/GATE.json")
    ap.add_argument("--tag", default=None, help="which entry in GATE.json this is")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="install even though the gate did not pass")
    args = ap.parse_args()

    src = Path(args.weights)
    if not src.is_absolute():
        src = ROOT / src
    if not src.exists():
        sys.exit(f"no such checkpoint: {src}")
    dest = Path(args.dest)

    from ultralytics import YOLO
    m = YOLO(str(src))
    names = m.names
    vmap = verdict_map(names)
    print(f"checkpoint : {src}")
    print(f"labels     : {names}")
    if not vmap:
        sys.exit("REFUSED: these labels are not the W/WH/WHV/WV compliance "
                 "ontology. Amsar would reject this model at load time "
                 "(ppe_eye.verdict_map), so installing it would only disable "
                 "the single-pass engine and fall back to nothing.")
    print(f"verdict map: {vmap}  OK")

    # gate status
    gate_path = ROOT / args.gate if not Path(args.gate).is_absolute() else Path(args.gate)
    gate_info = None
    if gate_path.exists():
        report = json.loads(gate_path.read_text())
        for tag, r in report.items():
            if args.tag and tag != args.tag:
                continue
            if Path(r["weights"]).resolve() == src.resolve():
                gate_info = (tag, r.get("gate"))
                break
    if gate_info and gate_info[1]:
        tag, g = gate_info
        print(f"gate       : {tag} -> {'PASS' if g['passed'] else 'FAIL'}")
        for k, c in g["checks"].items():
            print(f"             [{'x' if c['pass'] else ' '}] {k}: "
                  f"{c['value']} (target {c['target']})")
        if not g["passed"] and not args.force:
            sys.exit("REFUSED: this checkpoint has not passed the acceptance gate "
                     "(experiments/gate.py). Re-run the gate, or pass --force and "
                     "record in the deploy note why shipping it anyway is right.")
    else:
        print("gate       : no gate record for this checkpoint")
        if not args.force:
            sys.exit("REFUSED: run experiments/gate.py on this checkpoint first, "
                     "or pass --force.")

    if args.dry_run:
        print(f"\n[dry-run] would copy -> {dest}")
        return

    if dest.exists():
        backup = dest.with_name(f"ppe_eye.prev-{time.strftime('%Y%m%d-%H%M%S')}.pt")
        shutil.copy2(dest, backup)
        print(f"backed up existing weights -> {backup.name}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    size = dest.stat().st_size / 1e6

    note = {
        "installed": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_checkpoint": str(src),
        "labels": {int(k): v for k, v in names.items()},
        "file_MB": round(size, 2),
        "gate": gate_info[1] if gate_info else None,
    }
    (dest.with_suffix(".provenance.json")).write_text(json.dumps(note, indent=2))
    print(f"installed  : {dest}  ({size:.1f} MB)")
    print(f"provenance : {dest.with_suffix('.provenance.json').name}")
    print("\nA prebuilt TensorRT engine next to the .pt takes precedence "
          "(ppe_models.USE_TENSORRT). If ppe_eye.engine exists it is now stale - "
          "rebuild it with export_trt.py or delete it.")


if __name__ == "__main__":
    main()
