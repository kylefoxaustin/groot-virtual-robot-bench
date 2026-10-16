#!/usr/bin/env python3
"""Summarise your full-cost runs the way the published numbers were computed, next to ours.

  python3 drone/score_fullcost.py RUN_DIR [RUN_DIR ...]      (each RUN_DIR holds a fullcost.json from drone_fulltrt.py fullcost)
  python3 drone/score_fullcost.py --compare rtx5090|thor|orin RUN_DIR ...

Rate = 1000 / (median over runs of each run's median decision time). Run spread = (max - min) / median of the run medians.
Each decision is timed from the raw 224 px frame + prompt + state to the 16 setpoints, host work included.
Needs python3 + numpy."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
args = sys.argv[1:]; cmp = None
if args[:1] == ["--compare"]: cmp, args = args[1], args[2:]
if not args: sys.exit(__doc__)
runs = [json.load(open(os.path.join(d, "fullcost.json"))) for d in args]
med = [float(np.median([x["total_ms"] for x in r["rows"]])) for r in runs]; mm = float(np.median(med))
st = {s: float(np.median([np.median([x[s] for x in r["rows"]]) for r in runs])) for s in ("transform_ms", "gpu_policy_ms", "head_ms")}
print(f"runs {len(runs)} x {runs[0]['frames']} decisions; run medians {[round(x, 3) for x in med]} ms")
print(f"full cost {mm:.3f} ms = {1e3 / mm:.1f} Hz; run spread {100 * (max(med) - min(med)) / mm:.2f}%")
print("stages (median of run medians): " + ", ".join(f"{k[:-3]} {v:.3f} ms" for k, v in st.items()))
print(f"software: {runs[0]['software']}")
if cmp:
    T = json.load(open(os.path.join(HERE, "..", "results", "drone", "drone_table_2026-10-15.json")))["full_cost"][cmp]
    print(f"\nours ({cmp}): {T['median_of_run_medians_ms']} ms = {T['hz']} Hz, run spread {T['run_spread_pct']}%, "
          f"stages {T['stage_ms_median_of_run_medians']}, software {T['software']}")
    print(f"yours / ours: {mm / T['median_of_run_medians_ms']:.3f} x the time")
