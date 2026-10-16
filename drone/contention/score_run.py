#!/usr/bin/env python3
"""Score a contention run (drone/contention/run_contention.sh) exactly as the published runs were scored, and optionally
print one of ours next to it.

  python3 drone/contention/score_run.py RUN_DIR [--compare e1_scan_rtx5090|e1_scan_thor|...]

Writes RUN_DIR/raw.json.gz in the published raw format (results/drone/raw/contention/), then bins it with
drone_metrics.bin_contention: nav rate alone, inside each co-tenant window (the worker's own [start, end]) and after it,
and the action-space gate on every window's decisions. Published run names: ls results/drone/raw/contention/
Needs python3 + numpy."""
import glob, gzip, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import drone_metrics as M

args = sys.argv[1:]; cmp = None
if "--compare" in args: i = args.index("--compare"); cmp = args[i + 1]; args = args[:i] + args[i + 2:]
if len(args) != 1: sys.exit(__doc__)
D = args[0]
RAW = os.path.join(HERE, "..", "..", "results", "drone", "raw")
J = lambda p: json.load(gzip.open(p, "rt"))
ref = J(os.path.join(RAW, "gate", "reference_actions.json.gz")); A_B = np.array(ref["bf16_5090_eager"]); ysd = np.array(ref["ysd_gated_dims"])
swap = M.swap_index(ref["instructions"])

# ---- convert the run directory to the published raw format ----
L = np.load(f"{D}/loop.npz"); ts, js, acts = L["ts"], L["j"], L["actions"]
ready = float(open(f"{D}/ready").read())
per, nonid = {}, 0
for j in np.unique(js):
    m = np.where(js == j)[0]; a0 = acts[m[0]]
    nonid += sum(1 for i in m if not np.array_equal(acts[i], a0))
    per[str(int(j))] = np.round(M.exec_setpoints(a0[None]), 6)[0].tolist()
orch = [l.split() for l in open(f"{D}/orch.log")]
mode = next(l[1] for l in orch if l and l[0] == "mode")
workers = {}
for f in sorted(glob.glob(f"{D}/*.json")):
    bn = os.path.basename(f)[:-5]
    if not bn.startswith(("vis_", "radar", "bg_")): continue
    v = json.load(open(f))
    w = {k: v[k] for k in ("spec", "n", "wall_s", "per_inf_ms_mean", "per_inf_ms_p95", "per_inf_ms_max", "achieved_hz", "duty_pct") if k in v}
    w["start_rel_s"] = round(v["start"] - ready, 6); w["end_rel_s"] = round(v["end"] - ready, 6)
    workers[bn] = w
lj = json.load(open(f"{D}/loop.json"))
run = {"board": lj.get("gpu"), "mode": mode, "run": os.path.basename(D.rstrip("/")),
       "loop_meta": {k: lj.get(k) for k in ("trt", "torch", "gpu", "torch_threads", "duration_s", "clocks_post")},
       "decision_end_rel_s": [round(float(x - ready), 6) for x in ts], "j": js.tolist(), "exec_by_frame": per,
       "spawns_rel_s": [round(float(l[3]) - ready, 6) for l in orch if len(l) >= 4 and l[2] == "spawn"],
       "bg_spawn_rel_s": [round(float(l[2]) - ready, 6) for l in orch if len(l) >= 3 and l[0] == "bg" and l[1] == "spawn"],
       "workers": workers}
with gzip.open(f"{D}/raw.json.gz", "wt") as f: json.dump(run, f)


def show(name, res):
    print(f"\n{name}: {res['n_decisions']} decisions, gate on all {res['all_decisions_gate_sigma']} sigma "
          f"({'PASS' if res['all_decisions_gate_sigma'] <= M.GATE_SIGMA else 'FAIL'}), planted fault {res['planted_fault_sigma']} "
          f"({'fails as it must' if res['planted_fault_fails'] else 'DID NOT FAIL: the gate is not working'})")
    print(f"  {'window kind':14s} {'windows':>7} {'decisions':>9} {'time s':>8} {'nav Hz':>8} {'% solo':>7} {'per-window Hz':>16} {'gate sigma':>10}")
    for k, p in res["pooled"].items():
        v = f" ({p['void_windows']} VOID: < 5 decisions)" if p["void_windows"] else ""
        print(f"  {k:14s} {p['windows']:7d} {p['n']:9d} {p['time_s']:8.3f} {p['rate_hz']:8.2f} {p['pct_of_solo']:7.1f} "
              f"{p['per_window_rate_min']:7.2f}-{p['per_window_rate_max']:<8.2f} {p['gate_sigma']!s:>10}{v}")


res = M.bin_contention(run, A_B, ysd, swap)
show(f"your run {run['run']} ({mode}, {run['loop_meta']['gpu']})", res)
if nonid: print(f"  note: {nonid} decisions differ from the first decision on the same frame (ours: 0 in every run)")
if cmp:
    ours = J(os.path.join(RAW, "contention", cmp + ".json.gz"))
    show(f"ours: {cmp} ({ours['mode']}, {ours['loop_meta']['gpu']}, torch threads {ours['loop_meta']['torch_threads']})",
         M.bin_contention(ours, A_B, ysd, swap))
