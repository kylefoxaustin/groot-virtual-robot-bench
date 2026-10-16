#!/usr/bin/env python3
"""Summarise your IQ-9075 full-cost runs (drone_bframe_board.py fullcost) the way drone/score_fullcost.py summarises the GPU
runs, and check that the decisions are right.

  python3 drone/iq9/policy/score_fullcost_iq9.py RUN_DIR [RUN_DIR ...]          single-stream runs (one fullcost.json each)
  python3 drone/iq9/policy/score_fullcost_iq9.py --dual D0 D1 [D0 D1 ...]       dual-NSP repeats, as pairs of run dirs

Rate = 1000 / (median over runs of each run's median decision time). Run spread = (max - min) / median of the run medians.
Each decision is timed from the raw 224 px frame + prompt + state to the 16 setpoints, host work included. Stages: GR00T's
transforms, the raw write, the NPU call (BFRAME, including its host glue) and the head.
Output check: every decision's executed setpoints vs the RTX 5090 bf16 reference on the same gate frame, in the gate's
units (drone/score_gate.py); the full-cost path runs GR00T's transforms on the board, so this also checks them.
Dual: aggregate decisions per second of both streams inside the window where both ran, per repeat.
Needs python3 + numpy."""
import gzip, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); DRONE = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, DRONE)
import drone_metrics as M

args = sys.argv[1:]; dual = args[:1] == ["--dual"]
if dual: args = args[1:]
if not args or (dual and len(args) % 2): sys.exit(__doc__)
ref = json.load(gzip.open(os.path.join(DRONE, "..", "results", "drone", "raw", "gate", "reference_actions.json.gz"), "rt"))
A_B, ysd = np.array(ref["bf16_5090_eager"]), np.array(ref["ysd_gated_dims"])
L = lambda d: json.load(open(os.path.join(d, "fullcost.json")))
STAGES = ("transform_ms", "rawwrite_ms", "npu_policy_ms", "head_ms")


def gate(d):
    g = json.load(gzip.open(os.path.join(d, "fullcost_actions.json.gz"), "rt"))
    return M.dev_sigma(np.array(g["exec"]), A_B[np.array(g["j"])], ysd)[1]


def summary(dirs, label):
    runs = [L(d) for d in dirs]
    med = [float(np.median([x["total_ms"] for x in r["rows"]])) for r in runs]; mm = float(np.median(med))
    st = {s: float(np.median([np.median([x[s] for x in r["rows"]]) for r in runs])) for s in STAGES}
    print(f"{label}: {len(runs)} runs x {runs[0]['frames']} decisions; run medians {[round(x, 3) for x in med]} ms")
    print(f"  full cost {mm:.3f} ms = {1e3 / mm:.2f} decisions/s; run spread {100 * (max(med) - min(med)) / mm:.2f} %")
    print("  stages (median of run medians): " + ", ".join(f"{k[:-3]} {v:.3f} ms" for k, v in st.items()))
    gs = [gate(d) for d in dirs]
    print(f"  output check vs RTX 5090 bf16: {[round(x, 4) for x in gs]} sigma per run "
          f"({'PASS' if max(gs) <= M.GATE_SIGMA else 'FAIL'} at <= {M.GATE_SIGMA})")
    print(f"  arm {runs[0].get('arm')} device {runs[0].get('device')}, software {runs[0]['software']}, valid tokens {runs[0]['nvalid_unique']}")


if not dual:
    groups = {}
    for d in args:
        r = L(d); groups.setdefault((r.get("arm"), r.get("device")), []).append(d)
    for (arm, dev), ds in groups.items(): summary(ds, f"arm {arm}, device {dev}")
else:
    for i in range(0, len(args), 2):
        r0, r1 = L(args[i]), L(args[i + 1])
        a = max(r0["rows"][0]["t_start"], r1["rows"][0]["t_start"]); b = min(r0["rows"][-1]["t_end"], r1["rows"][-1]["t_end"])
        n = sum(1 for r in (r0, r1) for x in r["rows"] if a <= x["t_end"] <= b)
        meds = [float(np.median([x["total_ms"] for x in r["rows"]])) for r in (r0, r1)]
        print(f"dual repeat {i // 2 + 1}: stream medians {[round(m, 3) for m in meds]} ms (devices {r0['device']}, {r1['device']}); "
              f"both running for {b - a:.2f} s: {n} decisions = {n / (b - a):.2f} decisions/s aggregate; "
              f"output check {round(gate(args[i]), 4)}, {round(gate(args[i + 1]), 4)} sigma")
