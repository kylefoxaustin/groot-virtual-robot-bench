#!/usr/bin/env python3
"""(host) Check a compiled stand-in's outputs on the NPU against the fp32 references (onnxruntime CPU) before timing it.
Inputs: the --keep dir of `npu_worker.py check` (outputs.npy, repeat0.npy) and that worker's JSON.

  python3 drone/iq9/contention/score_check.py KEEP_DIR WORKER_JSON seg512|seg1024|radar [--standins DIR]

seg512 / seg1024 : per tile argmax agreement over the 21 classes + rel_mae; PASS iff every tile has agreement >= 0.95 and
                   rel_mae <= 0.10 (references: check_ref_fp32.npz / seg1024/check1024_ref_fp32.npz of the CANONICAL dilated
                   model, so a dilation-free build is checked against the original)
radar            : per cloud rel_mae + input-centred rel_mae (minus the mean of the 8 reference clouds); PASS iff <= 0.005 and
                   <= 0.05
Planted fault: each output scored against the NEXT item's reference must FAIL the same gate (else the gate is inert).
Repeat: input 0 run twice must be bit-identical. Needs python3 + numpy."""
import argparse, json, os, sys
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("keep"); ap.add_argument("worker_json"); ap.add_argument("kind", choices=["seg512", "seg1024", "radar"])
ap.add_argument("--standins", default=None); a = ap.parse_args()
A = a.standins or os.environ.get("DRONE_STANDINS") or sys.exit("set --standins or source drone/env.sh")
w = json.load(open(a.worker_json)); out = np.load(f"{a.keep}/outputs.npy"); rep = np.load(f"{a.keep}/repeat0.npy"); n = len(out)
rel = lambda o, r: float(np.abs(o - r).mean() / np.abs(r).mean())
res = {"kind": a.kind, "n": int(n), "repeat_bit_identical": bool(np.array_equal(out[0], rep)), "orch_rc": w.get("orch_rc")}
if a.kind.startswith("seg"):
    R = 512 if a.kind == "seg512" else 1024
    refp = f"{A}/check_ref_fp32.npz" if R == 512 else f"{A}/seg1024/check1024_ref_fp32.npz"
    ref = np.load(refp)["seg_out"].reshape(n, 21, R, R); o = out.reshape(n, 21, R, R)
    ag = [float((o[i].argmax(0) == ref[i].argmax(0)).mean()) for i in range(n)]; rm = [rel(o[i], ref[i]) for i in range(n)]
    fa = [float((o[i].argmax(0) == ref[(i + 1) % n].argmax(0)).mean()) for i in range(n)]; fr = [rel(o[i], ref[(i + 1) % n]) for i in range(n)]
    ok = lambda x, r: x >= 0.95 and r <= 0.10
    res.update({"argmax_agreement": [round(x, 5) for x in ag], "rel_mae": [round(x, 5) for x in rm],
                "verdict": "PASS" if all(ok(x, r) for x, r in zip(ag, rm)) else "FAIL",
                "planted_fault_next_tile": {"argmax_agreement": [round(x, 4) for x in fa], "rel_mae": [round(x, 4) for x in fr]},
                "planted_fault_verdict": "fails as planted" if not any(ok(x, r) for x, r in zip(fa, fr)) else "PASSED (the gate is inert!)",
                "argmax_classes_board": int(len(np.unique(o.argmax(1))))})
else:
    ref = np.load(f"{A}/check_ref_fp32.npz")["radar_out"].reshape(n, -1); o = out.reshape(n, -1); mu = ref.mean(0)
    rm = [rel(o[i], ref[i]) for i in range(n)]; rc = [rel(o[i] - mu, ref[i] - mu) for i in range(n)]
    fr = [rel(o[i], ref[(i + 1) % n]) for i in range(n)]; fc = [rel(o[i] - mu, ref[(i + 1) % n] - mu) for i in range(n)]
    ok = lambda r, c: r <= 0.005 and c <= 0.05
    res.update({"rel_mae": [round(x, 6) for x in rm], "rel_mae_centered": [round(x, 5) for x in rc],
                "verdict": "PASS" if all(ok(x, y) for x, y in zip(rm, rc)) else "FAIL",
                "planted_fault_next_cloud": {"rel_mae": [round(x, 4) for x in fr], "rel_mae_centered": [round(x, 4) for x in fc]},
                "planted_fault_verdict": "fails as planted" if not any(ok(x, y) for x, y in zip(fr, fc)) else "PASSED (the gate is inert!)"})
res["per_call_wall_ms"] = w.get("wall_ms"); res["per_call_exec_us"] = w.get("exec_us")
print(json.dumps(res, indent=1))
