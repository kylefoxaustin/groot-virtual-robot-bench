#!/usr/bin/env python3
"""Score a gate run (drone/drone_fulltrt.py gate) with the published rule, and compare it with our boards.

  python3 drone/score_gate.py GATE_OUT_DIR

The gate is in ACTION space: the setpoints the drone would actually fly (first 4 of 16, vx / vz / yaw_rate), in units of
the training set's action standard deviation (sigma), against the RTX 5090's eager bf16 reference on the same 256 frames.
  G1  pooled RMS deviation <= 0.07 sigma          G2  per-frame RMS, 95th percentile <= 0.20 sigma
  G3  a zero-image null run deviates >= 10x G1     G4  a planted fault (each frame scored with another frame's
                                                       features, same instruction) must FAIL G1
PASS = G1 and G2 and G3, and G4 fails as planted. Needs python3 + numpy."""
import gzip, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import drone_metrics as M

if len(sys.argv) != 2: sys.exit(__doc__)
RAW = os.path.join(HERE, "..", "results", "drone", "raw", "gate")
J = lambda p: json.load(gzip.open(p, "rt"))
ref = J(os.path.join(RAW, "reference_actions.json.gz")); A_B, A_32 = np.array(ref["bf16_5090_eager"]), np.array(ref["fp32_5090_eager"])
ysd = np.array(ref["ysd_gated_dims"])
g = J(os.path.join(sys.argv[1], "gate_actions.json.gz")); A = np.array(g["real"]); ni = np.array(g["null_frames"])
per, g1 = M.dev_sigma(A, A_B, ysd); _, vs32 = M.dev_sigma(A, A_32, ysd)
_, gn = M.dev_sigma(np.array(g["null_zero_image"]), A_B[ni], ysd); _, g4 = M.dev_sigma(np.array(g["planted_swap"]), A_B, ysd)
p95 = float(np.percentile(per, 95))
ok = g1 <= M.GATE_SIGMA and p95 <= M.GATE_P95 and gn >= 10 * g1 and g4 > M.GATE_SIGMA
print(f"G1 deviation vs RTX 5090 bf16 : {g1:.4f} sigma  (gate <= 0.07)   [vs fp32: {vs32:.4f}]")
print(f"G2 per-frame p95              : {p95:.4f} sigma  (gate <= 0.20)")
print(f"G3 zero-image null            : {gn:.4f} sigma  (must be >= 10 x G1 = {10 * g1:.3f})")
print(f"G4 planted frame swap         : {g4:.4f} sigma  (must FAIL, i.e. > 0.07)")
print(f"pad-id swap max |diff| {g['padswap_max_abs']}, rerun max |diff| {g['rerun_max_abs']} (both 0.0 on our boards)")
print("VERDICT:", "PASS" if ok else "FAIL")
print("\nDistance to our boards' gate runs (same metric, your setpoints vs theirs):")
for b in ("rtx5090", "thor", "orin"):
    o = J(os.path.join(RAW, f"{b}_gate_actions.json.gz"))
    print(f"  vs our {b:8s}: {M.dev_sigma(A, np.array(o['real']), ysd)[1]:.4f} sigma   (ours vs the reference: {M.dev_sigma(np.array(o['real']), A_B, ysd)[1]:.4f})")
print("Our three boards differ from each other by 0.011-0.019 sigma; bit-identical engines give 0.0000.")
