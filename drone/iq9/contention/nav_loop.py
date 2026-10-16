#!/usr/bin/env python3
"""(board) The drone nav policy back to back on one NSP: the full-cost decision path of policy/drone_bframe_board.py
(BoardPolicy.decide, unchanged: GR00T's transforms on the A78 -> raw write -> BFRAME on --device -> numpy head), cycling the 256
gate frames of drone/assets/gate_set.npz, until --stop FILE appears or --duration s pass.
Writes the files of the GPU loop (drone/drone_fulltrt.py loop), so drone/contention/score_run.py and bin_session.py bin and
gate it the same way: <out>/ready (time.time() when warm; shared clock with the NSP workers), <out>/loop.npz (ts = decision
end, tstart, j, actions [n, 16, 4], plus per-decision stage timers stage_ms / stage_names) and <out>/loop.json.
usage: nav_loop.py --out DIR [--duration S] [--stop FILE] [--device 0] [--arm fix] [--warmup 3] [--blas-threads 6]
"""
import os, sys, time, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "policy"))
import drone_bframe_board as D   # sets the BLAS thread env before numpy loads
import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", required=True); ap.add_argument("--duration", type=float, default=3600.0, help="upper bound, s")
ap.add_argument("--stop", default=None, help="end after the decision in flight when this file exists")
ap.add_argument("--device", type=int, default=0); ap.add_argument("--arm", default="fix", choices=["fix", "aaf0"])
ap.add_argument("--warmup", type=int, default=3); ap.add_argument("--blas-threads", type=int, default=6)
ap.add_argument("--ready", default=None, help="default <out>/ready")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); ready = a.ready or f"{a.out}/ready"
z = np.load(f"{D.ASSETS}/gate_set.npz")
IMGS, INS, STS = z["images"], [str(x) for x in z["instructions"]], z["states"]
pol = D.BoardPolicy(a.arm, a.out, device=a.device, blas_threads=a.blas_threads, shm_tag="drone_nav")
STG = ["transform_ms", "rawwrite_ms", "npu_policy_ms", "head_ms", "srv_wall_ms", "srv_vit_ms", "srv_mlp1_ms", "srv_gluevlm_ms",
       "srv_llm_ms", "srv_vlln_ms", "srv_pool_ms"]
ts, tst, js, acts, stg = [], [], [], [], []
stop = lambda: a.stop is not None and os.path.exists(a.stop)
try:
    for k in range(a.warmup): pol.decide(IMGS[k % len(IMGS)], INS[k % len(IMGS)], STS[k % len(IMGS)])
    T0 = time.time(); open(ready, "w").write(str(T0))
    k = 0
    while time.time() - T0 < a.duration and not stop():
        j = k % len(IMGS); s0 = time.time()
        act, f, nv, ms = pol.decide(IMGS[j], INS[j], STS[j]); te = time.time()
        tst.append(s0); ts.append(te); js.append(j); acts.append(act); stg.append([ms.get(s, -1.0) for s in STG]); k += 1
finally:
    rc = pol.close()
np.savez(f"{a.out}/loop.npz", ts=np.array(ts), tstart=np.array(tst), j=np.array(js), actions=np.array(acts),
         stage_ms=np.array(stg), stage_names=np.array(STG))
json.dump({"tag": os.path.basename(a.out.rstrip("/")), "board": "IQ-9075", "gpu": f"IQ-9075 NSP device {a.device}", "device": a.device, "arm": a.arm,
           "duration_s": ts[-1] - T0 if ts else 0.0, "n": len(ts), "orch_rc": rc, "loaded": pol.R.loaded,
           "torch_threads": pol.torch_threads, "blas_threads": a.blas_threads}, open(f"{a.out}/loop.json", "w"), indent=1)
print("NAV_DONE n", len(ts), "rc", rc, flush=True)
