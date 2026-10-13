#!/usr/bin/env python3
"""Views for "does quantization change closed-loop behaviour?" on the IQ-9075, built from YOUR runs (numpy only).

Inputs (all produced by the iq9/work/closed_loop_m7 scripts, see iq9/README.md section 4):
  --ref   the RTX 5090 bf16 reference run dir            (closed-loop client output)
  --fp16  the IQ-9075 fp16 run dir
  --arm   label=dir  one per quantized run (e.g. int8_head=..., int16_head=...)
  --gate  label=gate.json  optional per-graph gate outputs from the closed_loop_m7 gate_*.py scripts (cosine vs fp16)
Outputs (JSON to --out, summary to stdout):
  - closed-loop success per run (from rollout_summary.json)
  - call-0 action error per joint group: first planned chunk, identical first scene, quantized vs fp16 board
    (rel_mae over left arm 0:7, right arm 7:14, left hand 14:20, right hand 20:26; seeds 0-4, arms also over all seeds)
  - one closed-loop joint trajectory per run for a seed the reference solved (the right-arm joint that moves most),
    every 4th control step, for plotting
  - the per-graph cosines from any --gate files, passed through unchanged
Each episode_XX_seedS_actions.npy is (calls x 16 steps x 29 DoF)."""
import argparse, json, os
import numpy as np

GROUPS = {"left_arm": (0, 7), "right_arm": (7, 14), "left_hand": (14, 20), "right_hand": (20, 26)}


def summ(d): return json.load(open(os.path.join(d, "rollout_summary.json")))
def epi(d): return {e["seed"]: e["episode"] for e in summ(d)["episodes"]}
def load(d, s): return np.load(os.path.join(d, f"episode_{epi(d)[s]:02d}_seed{s}_actions.npy")).astype(np.float64)


def rel_mae(a, b, lo, hi):       # identical to closed_loop_m7/finalize_m7.rel_mae
    a, b = a[..., lo:hi], b[..., lo:hi]
    return float(np.mean(np.abs(a - b)) / (np.mean(np.abs(b)) + 1e-12))


ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True); ap.add_argument("--fp16", required=True)
ap.add_argument("--arm", action="append", required=True, help="label=dir"); ap.add_argument("--gate", action="append", default=[])
ap.add_argument("--out", default="quant_views.json"); ap.add_argument("--seed", type=int, default=None)
a = ap.parse_args()
arms = dict(x.split("=", 1) for x in a.arm)
seeds = sorted(epi(a.fp16))
out = {"tag": "DERIVED from MEASURED closed-loop traces", "reference": a.ref, "fp16": a.fp16, "arms": arms}

out["closed_loop_success"] = {lab: f"{sum(e['success'] for e in summ(d)['episodes'])}/{len(summ(d)['episodes'])}"
                              for lab, d in {"reference": a.ref, "fp16": a.fp16, **arms}.items()}
c0 = {}
for lab, d in arms.items():
    g = {k: round(float(np.mean([rel_mae(load(d, s)[0], load(a.fp16, s)[0], lo, hi) for s in seeds[:5]])), 3) for k, (lo, hi) in GROUPS.items()}
    g["arms_0_14_all_seeds"] = round(float(np.mean([rel_mae(load(d, s)[0], load(a.fp16, s)[0], 0, 14) for s in seeds])), 3)
    c0[lab] = g
out["call0_action_error_vs_fp16"] = c0

rs = {e["seed"]: e["success"] for e in summ(a.ref)["episodes"]}
seed = a.seed if a.seed is not None else next(s for s in seeds if rs.get(s))
ref = load(a.ref, seed).reshape(-1, 29); j = 7 + int(np.argmax(ref[:, 7:14].std(axis=0)))
out["trajectory"] = {"seed": seed, "joint_index": j, "every_nth_step": 4, "control_hz": 20,
                     "series": {lab: [round(float(v), 4) for v in load(d, seed).reshape(-1, 29)[::4, j]]
                                for lab, d in {"reference": a.ref, "fp16": a.fp16, **arms}.items()}}
out["per_graph_gates"] = {lab: json.load(open(p)) for lab, p in (x.split("=", 1) for x in a.gate)}
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps({"closed_loop_success": out["closed_loop_success"], "call0_action_error_vs_fp16": c0}, indent=1))
print("wrote", a.out)
