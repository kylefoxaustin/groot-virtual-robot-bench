#!/usr/bin/env python3
"""(host) Bin an IQ-9075 contention session (run_session.sh) over windows you name, and gate every window's decisions.
For sessions whose worker names match the GPU runs (mode scan / all), drone/contention/score_run.py scores them directly.

Method (as drone_metrics.bin_contention): a nav decision counts in a window if its END time falls inside it; rate = count /
window length. Solo = [ready + 1 s, first worker spawn]. A co-tenant window is the intersection of the named workers' OWN
[start, end] (after their context load and warm-up), never the spawn/exit bracket. A window with fewer than 5 decisions is
VOID. Each window is gated in action space against the RTX 5090 bf16 reference on the same gate frames (drone/score_gate.py
units, pass <= 0.07 sigma), and a planted fault (each decision scored against another frame with the same instruction) must
FAIL. Also reports the nav's mean stage times inside each window and the workers' own records.

  python3 drone/iq9/contention/bin_session.py RUN_DIR WINDOWS.json
WINDOWS.json = [{"label": "scan", "workers": ["vis_burstA"]},                       (intersection of their [start, end])
                {"label": "after", "after": "vis_burstA", "offset_s": 1, "len_s": 20}]   (from a worker's end)
Needs python3 + numpy."""
import gzip, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); DRONE = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, DRONE)
import drone_metrics as M

if len(sys.argv) != 3: sys.exit(__doc__)
D, WS = sys.argv[1], json.load(open(sys.argv[2]))
ref = json.load(gzip.open(os.path.join(DRONE, "..", "results", "drone", "raw", "gate", "reference_actions.json.gz"), "rt"))
A_B, ysd = np.array(ref["bf16_5090_eager"]), np.array(ref["ysd_gated_dims"]); swap = M.swap_index(ref["instructions"])
L = np.load(f"{D}/loop.npz"); ts, js = L["ts"], L["j"]; EX = M.exec_setpoints(L["actions"])
SU, SN = L["stage_ms"], [str(x) for x in L["stage_names"]]
ready = float(open(f"{D}/ready").read())
orch = [l.split() for l in open(f"{D}/orch.log")]
spawns = sorted([float(l[3]) for l in orch if len(l) >= 4 and l[0] == "arm" and l[2] == "spawn"] +
                [float(l[2]) for l in orch if len(l) >= 3 and l[0] == "bg" and l[1] == "spawn"])
W = {f[:-5]: json.load(open(f"{D}/{f}")) for f in os.listdir(D) if f.endswith(".json") and f != "loop.json"}
W = {k: v for k, v in W.items() if "start" in v or "mode" in v}
void = []


def win(a, b, label):
    m = (ts >= a) & (ts < b); n = int(m.sum())
    g = M.dev_sigma(EX[m], A_B[js[m]], ysd)[1] if n else None
    gf = M.dev_sigma(EX[m], A_B[swap[js[m]]], ysd)[1] if n else None
    if n < 5: void.append(f"{label}: {n} < 5 decisions")
    if len(ts) and b > ts[-1]: void.append(f"{label}: window ends after the nav loop's last decision")
    r = {"label": label, "t0_rel_s": round(a - ready, 3), "t1_rel_s": round(b - ready, 3), "len_s": round(b - a, 3), "n": n,
         "rate_per_s": round(n / (b - a), 3) if b > a else None,
         "gate_sigma": round(g, 5) if g is not None else None, "gate_pass": bool(g is not None and g <= M.GATE_SIGMA),
         "planted_fault_sigma": round(gf, 4) if gf is not None else None, "planted_fault_fails": bool(gf is not None and gf > M.GATE_SIGMA),
         "nav_stage_ms_mean": {k: round(float(np.mean(SU[m][:, i])), 3) for i, k in enumerate(SN)} if n else {}}
    if n < 5: r["status"] = "VOID"
    return r


out = {"run": os.path.basename(D.rstrip("/")), "n_decisions": int(len(ts)), "loop_meta": json.load(open(f"{D}/loop.json")),
       "governor_log": [" ".join(l) for l in orch if l and l[0] in ("gov", "restored")], "windows": []}
out["windows"].append(win(ready + 1.0, spawns[0] if spawns else ts[-1], "solo"))
for w in WS:
    if "workers" in w:
        a = max(W[k]["start"] for k in w["workers"]); b = min(W[k]["end"] for k in w["workers"])
    else:
        a = W[w["after"]]["end"] + w.get("offset_s", 1.0); b = min(a + w.get("len_s", 20.0), ts[-1])
    r = win(a, b, w["label"])
    r["workers"] = {k: {kk: W[k].get(kk) for kk in ("bin", "cmd", "spec", "sleep_us", "graph", "device", "n", "wall_s", "per_inf_ms_mean",
                                                   "per_inf_ms_p95", "per_inf_ms_max", "exec_ms_mean", "achieved_hz", "duty_pct",
                                                   "gap_ms", "stage_us_mean", "cpu_busy_pct", "orch_rc")} for k in w.get("workers", [])}
    out["windows"].append(r)
solo = out["windows"][0]["rate_per_s"]
for r in out["windows"]:
    if solo and r["rate_per_s"] is not None: r["pct_of_solo"] = round(100 * r["rate_per_s"] / solo, 1)
scored = [r for r in out["windows"] if r.get("status") != "VOID"]
out.update({"void": void or None, "gate_all_pass": all(r["gate_pass"] for r in scored),
            "planted_fault_fails_everywhere": all(r["planted_fault_fails"] for r in scored),
            "worker_error_lines": [" ".join(l) for l in orch if any("graphExecute" in x or "ERR" in x for x in l)]})
json.dump(out, open(f"{D}/bin_session.json", "w"), indent=1)
print(f"{out['run']}: {out['n_decisions']} nav decisions")
print(f"  {'window':18s} {'len s':>8} {'n':>5} {'per s':>8} {'% solo':>7} {'gate sigma':>11} {'fault':>8}")
for r in out["windows"]:
    print(f"  {r['label']:18s} {r['len_s']:8.2f} {r['n']:5d} {r['rate_per_s']!s:>8} {r.get('pct_of_solo', '')!s:>7} "
          f"{r['gate_sigma']!s:>11} {('fails' if r['planted_fault_fails'] else 'PASSED'):>8} {r.get('status', '')}")
print("gate on every scored window:", "PASS" if out["gate_all_pass"] else "FAIL",
      "| planted fault fails everywhere:", out["planted_fault_fails_everywhere"], "| void:", out["void"])
print(f"full record: {D}/bin_session.json")
