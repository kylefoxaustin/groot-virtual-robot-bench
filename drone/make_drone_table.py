#!/usr/bin/env python3
"""Rebuild every published drone number (RTX 5090, Jetson AGX Thor, Jetson AGX Orin) from the raw data in results/drone/raw/.

  fullcost/<board>_r<n>.json          per-frame full-cost timing (raw 224 px frame + prompt + state -> 16 setpoints)
  fullcost/rtx5090_pinned_r<n>*.json  the same on the RTX 5090 with the process pinned to P-cores (+ its actions)
  loops/rtx5090_{pinned,unpinned}_75s.json.gz   75 s back-to-back decision loops (decision end times, actions per frame)
  gate/<board>_gate_actions.json.gz   executed setpoints on the 256 gate frames: real, zero-image null, planted frame swap
  gate/reference_actions.json.gz      the RTX 5090 eager bf16 (reference) and fp32 (control) setpoints, dataset action std
  gate/<board>_gate_pass_timings.json per-frame synced engine timings from the gate pass (for the host + glue share)
  closed_loop/<arm>.jsonl             one row per episode (seeds 7000-7039): reached, decisions, final distance
  contention/<run>.json.gz            nav decision end times + frame per decision, actions per frame, co-tenant windows

Everything is computed here (medians, rates, gate deviations, reach counts, exact tests, window binning). One exception:
the engines-alone times (each engine timed alone, 200 iterations) are copied from the gate-pass summary in gate/.   usage: python3 drone/make_drone_table.py [OUT_JSON]   (needs python3 + numpy)"""
import glob, gzip, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import drone_metrics as M

ROOT = os.path.abspath(os.path.join(HERE, ".."))
RAW = os.path.join(ROOT, "results", "drone", "raw")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "drone", "drone_table_2026-10-15.json")
BOARDS = ["rtx5090", "thor", "orin"]


def J(p):
    p = os.path.join(RAW, p)
    return json.load(gzip.open(p, "rt") if p.endswith(".gz") else open(p))


def r(x, n=3): return round(float(x), n)


REF = J("gate/reference_actions.json.gz")
A_B, A_32 = np.array(REF["bf16_5090_eager"]), np.array(REF["fp32_5090_eager"])
YSD = np.array(REF["ysd_gated_dims"]); SWAP = M.swap_index(REF["instructions"])
T = {"_about": "Rebuilt by drone/make_drone_table.py from results/drone/raw/. Tags: MEASURED = we ran it; DERIVED = computed "
               "from measurements here. Rates in Hz are decisions per second (1000 / median ms)."}

# ---------------- full cost (MEASURED) ----------------
def fullcost(prefix):
    runs = [J(f"fullcost/{prefix}_r{k}.json") for k in (1, 2, 3)]
    med = [float(np.median([x["total_ms"] for x in R["rows"]])) for R in runs]
    st = {s: r(np.median([np.median([x[s] for x in R["rows"]]) for R in runs])) for s in ("transform_ms", "gpu_policy_ms", "head_ms", "total_ms")}
    mm = float(np.median(med))
    return {"tag": "MEASURED", "runs": 3, "frames_per_run": runs[0]["frames"], "warmup_per_run": runs[0]["warmup"],
            "offsets": [R["offset"] for R in runs], "run_medians_ms": [r(x) for x in med], "median_of_run_medians_ms": r(mm),
            "hz": r(1e3 / mm, 2), "run_spread_pct": r(100 * (max(med) - min(med)) / mm, 2),
            "within_run_std_ms": [r(np.std([x["total_ms"] for x in R["rows"]])) for R in runs],
            "p95_ms_per_run": [r(np.percentile([x["total_ms"] for x in R["rows"]], 95)) for R in runs],
            "stage_ms_median_of_run_medians": st, "software": runs[0]["software"],
            "clocks_nvidia_smi_pre_post": [R["clocks_nvidia_smi_pre_post"] for R in runs]}, runs


FC = {}
for b in BOARDS:
    FC[b], _ = fullcost(b)
    gp = J(f"gate/{b}_gate_pass_timings.json"); rows = gp["rows_us"][2:]
    eng = {k: float(np.median([x[k] for x in rows]) / 1e3) for k in ("vit", "llm", "vlln")}
    es = sum(round(v, 3) for v in eng.values())
    host = FC[b]["median_of_run_medians_ms"] - es
    FC[b]["engines_in_pipeline_ms_gate_pass"] = {**{k: r(v) for k, v in eng.items()}, "sum": r(es)}
    FC[b]["host_and_glue_DERIVED"] = {"ms": r(host), "share_pct": r(100 * host / FC[b]["median_of_run_medians_ms"], 2),
                                      "basis": "full-cost median minus the gate pass's synced ViT + LLM + vlln engine medians (two passes, so DERIVED)"}
    FC[b]["standalone_engine_ms"] = gp["standalone_engine_ms"]
pin, pruns = fullcost("rtx5090_pinned")
pin["config"] = pruns[0]["config"]
pin["gate_sigma_per_run"] = []
for k in (1, 2, 3):
    a = J(f"fullcost/rtx5090_pinned_r{k}_actions.json.gz"); jj = np.array(a["j"])
    pin["gate_sigma_per_run"].append(r(M.dev_sigma(np.array(a["exec"]), A_B[jj], YSD)[1], 5))
FC["rtx5090_pinned"] = pin
for k in ("pinned_75s", "unpinned_75s"):
    L = J(f"loops/rtx5090_{k}.json.gz"); ts = np.array(L["decision_end_rel_s"]); jj = np.array(L["j"])
    acts = np.array([L["exec_by_frame"][str(j)] for j in jj]); iv = np.diff(ts) * 1e3
    bins = [int(((ts >= a) & (ts < a + 10)).sum()) / 10 for a in range(0, int(ts[-1]) - 9, 10)]
    FC[f"rtx5090_{k}"] = {"tag": "MEASURED", "config": L["config"], "torch_threads": L["torch_threads"], "n": int(len(ts)),
                          "rate_hz": r(len(ts) / (ts[-1] - ts[0]), 2),
                          "interval_ms_p10_p50_p90_p99": [r(x, 2) for x in np.percentile(iv, [10, 50, 90, 99])],
                          "fraction_intervals_below_7p8ms": r(np.mean(iv < 7.8)),
                          "per_10s_bin_hz": [r(x, 1) for x in bins], "per_10s_bin_spread_pct": r(100 * (max(bins) - min(bins)) / np.median(bins), 2),
                          "gate_sigma": r(M.dev_sigma(acts, A_B[jj], YSD)[1], 5)}
T["full_cost"] = FC

# ---------------- output gate (MEASURED; deviations DERIVED from the measured actions) ----------------
G = {"tag": "MEASURED", "rule": {"G1": "pooled RMS deviation vs the RTX 5090 eager bf16 reference <= 0.07 sigma",
                                  "G2": "per-frame RMS p95 <= 0.20 sigma", "G3": "zero-image null >= 10x the real deviation",
                                  "G4": "planted frame swap must FAIL G1", "pass": "G1 and G2 and G3, and G4 fails as planted"},
     "reference_bf16_vs_fp32_sigma": r(M.dev_sigma(A_B, A_32, YSD)[1], 4), "boards": {}}
AB = {}
for b in BOARDS:
    g = J(f"gate/{b}_gate_actions.json.gz"); A = np.array(g["real"]); AB[b] = A; ni = np.array(g["null_frames"])
    per, g1 = M.dev_sigma(A, A_B, YSD); _, vs32 = M.dev_sigma(A, A_32, YSD)
    _, gnull = M.dev_sigma(np.array(g["null_zero_image"]), A_B[ni], YSD); _, g4 = M.dev_sigma(np.array(g["planted_swap"]), A_B, YSD)
    p95 = float(np.percentile(per, 95)); d = (A - A_B) / YSD
    G["boards"][b] = {"G1_sigma": r(g1, 4), "G2_p95_sigma": r(p95, 4), "G3_null_sigma": r(gnull, 4), "G3_ratio": r(gnull / g1, 1),
                      "G4_planted_swap_sigma": r(g4, 4), "verdict": "PASS" if (g1 <= M.GATE_SIGMA and p95 <= M.GATE_P95 and gnull >= 10 * g1 and g4 > M.GATE_SIGMA) else "FAIL",
                      "vs_fp32_sigma": r(vs32, 4), "per_dim_sigma": {k: r(np.sqrt((d[..., i] ** 2).mean()), 4) for i, k in enumerate(("vx", "vz", "yaw_rate"))},
                      "per_frame_sigma_median": r(np.median(per), 4), "per_frame_sigma_max": r(per.max(), 4),
                      "padswap_max_abs": g["padswap_max_abs"], "rerun_max_abs": g["rerun_max_abs"]}
G["cross_board_sigma"] = {f"{a}_vs_{b}": r(M.dev_sigma(AB[a], AB[b], YSD)[1], 4) for i, a in enumerate(BOARDS) for b in BOARDS[i + 1:]}
T["output_gate"] = G

# ---------------- closed loop (reach MEASURED; agreement and tests DERIVED) ----------------
def arm(name):
    rows = [json.loads(l) for l in open(os.path.join(RAW, "closed_loop", name + ".jsonl"))]
    assert [x["e"] for x in rows] == list(range(40)), name
    return rows


CL = {"tag": "MEASURED (reach); DERIVED (same outcome, McNemar, Fisher)", "arms": {}, "pairs": {}}
ARMS = {p[:-6]: arm(p[:-6]) for p in sorted(os.listdir(os.path.join(RAW, "closed_loop")))}
for k, rows in ARMS.items():
    rr = [x for x in rows if x["reached"]]
    a = {"reached": len(rr), "n": 40, "failed_episodes": [x["e"] for x in rows if not x["reached"]],
         "failed_final_dist_m": [x["final_dist_m"] for x in rows if not x["reached"]],
         "median_decisions_reached": float(np.median([x["n_decisions"] for x in rr])) if rr else None}
    if "board_wall_ms_median" in rows[0]:
        a["board_wall_ms_median_of_episode_medians"] = r(np.median([x["board_wall_ms_median"] for x in rows]), 2)
        dh = {}
        for x in rows:
            for kk, v in x["d_hist"].items(): dh[kk] = dh.get(kk, 0) + v
        a["d_ticks_histogram"] = dh
    if "applied_obs_age_ticks_mean" in rows[0]:
        a["hover_ticks_values"] = sorted({x["hover_ticks"] for x in rows})
    CL["arms"][k] = a
OUTC = {k: {x["e"]: x["reached"] for x in v} for k, v in ARMS.items()}


def pair(x, y):
    b, c, p = M.mcnemar(OUTC[x], OUTC[y]); rx, ry = CL["arms"][x]["reached"], CL["arms"][y]["reached"]
    return {"a": x, "b": y, "reach_a": rx, "reach_b": ry, "same_outcome": 40 - b - c, "a_only": b, "b_only": c,
            "mcnemar_exact_p": r(p, 4), "fisher_totals_p": r(M.fisher(rx, 40, ry, 40), 4),
            "discordant_episodes": [e for e in range(40) if OUTC[x][e] != OUTC[y][e]]}


P5 = "rtx5090_"
PAIRS = [(P5 + "trt_np_K4_llvmpipe", P5 + "eager_np_K4_llvmpipe"), (P5 + "trt_np_K15_llvmpipe", P5 + "eager_np_K15_llvmpipe"),
         (P5 + "trt_np_K4_llvmpipe", P5 + "trt_np_K15_llvmpipe"), (P5 + "eager_np_K4_llvmpipe", P5 + "eager_np_K15_llvmpipe"),
         (P5 + "trt_np_K1_llvmpipe", P5 + "trt_np_K15_llvmpipe"),
         (P5 + "eager_torch_K4_nvidia", P5 + "rerun1_eager_torch_K4_nvidia"), (P5 + "eager_torch_K4_nvidia", P5 + "rerun2_eager_torch_K4_nvidia"),
         ("thor_K4_lockstep", P5 + "trt_np_K4_llvmpipe"), ("orin_K4_lockstep", P5 + "trt_np_K4_llvmpipe"),
         ("thor_K4_lockstep", P5 + "eager_np_K4_llvmpipe"), ("orin_K4_lockstep", P5 + "eager_np_K4_llvmpipe"),
         ("thor_naive_backtoback", P5 + "delay_trt_K2_d2_naive"), ("orin_naive_backtoback", P5 + "delay_trt_K3_d3_naive"),
         ("thor_naive_backtoback", P5 + "delay_trt_K2_d0_naive"), ("orin_naive_backtoback", P5 + "delay_trt_K3_d0_naive"),
         ("thor_K4_lockstep", "orin_K4_lockstep"), ("thor_naive_backtoback", "orin_naive_backtoback")]
for x, y in PAIRS:
    CL["pairs"][f"{x}__vs__{y}"] = pair(x, y)
CL["k_sweep_rtx5090_trt"] = {int(k.split("_K")[1].split("_")[0]): v["reached"] for k, v in CL["arms"].items()
                             if k.startswith(P5 + "trt_np_K") and k.endswith("_llvmpipe")}
CL["k_sweep_rtx5090_trt"] = dict(sorted(CL["k_sweep_rtx5090_trt"].items()))
# delay sweep (features from the RTX 5090; the delay d is injected): each delayed arm vs its d = 0 control at the same K
F2 = []
for k in sorted(CL["arms"]):
    if not k.startswith(P5 + "delay_trt_K") or "_d0_" in k: continue
    K, d, mode = k[len(P5 + "delay_trt_K"):].split("_")[0], k.split("_d")[-1].split("_")[0], k.rsplit("_", 1)[1]
    ctl = f"{P5}delay_trt_K{K}_d0_naive"
    pp = pair(k, ctl)
    F2.append({"K": int(K), "d_ticks": int(d), "d_ms": int(d) * 20, "mode": mode, "reached": CL["arms"][k]["reached"],
               "control_d0_reached": CL["arms"][ctl]["reached"], "mcnemar_vs_control_p": pp["mcnemar_exact_p"],
               "same_outcome_vs_control": pp["same_outcome"]})
CL["delay_sweep_rtx5090"] = {"tag": "MEASURED (reach); DERIVED (McNemar vs the d = 0 arm at the same K)",
                             "rows": sorted(F2, key=lambda x: (x["K"] != x["d_ticks"], -x["K"], x["d_ticks"], x["mode"])),
                             "min_mcnemar_p": min(x["mcnemar_vs_control_p"] for x in F2),
                             "any_delayed_arm_worse_at_p_lt_0p05": any(x["mcnemar_vs_control_p"] < 0.05 and x["reached"] < x["control_d0_reached"] for x in F2)}
T["closed_loop"] = CL

# ---------------- contention (MEASURED rates; % and gate DERIVED) ----------------
C = {"tag": "MEASURED (rates); DERIVED (% of solo, gate deviations)", "runs": {}}
for p in sorted(glob.glob(os.path.join(RAW, "contention", "*.json.gz"))):
    run = J(os.path.relpath(p, RAW)); res = M.bin_contention(run, A_B, YSD, SWAP)
    w = run["workers"]
    res.update({"board": run["board"], "mode": run["mode"], "torch_threads": run["loop_meta"]["torch_threads"]})
    if "vis_alone" in w: res["scan_alone_per_tile_ms"] = w["vis_alone"]["per_inf_ms_mean"]
    sb = [x for x in res["windows"] if x["kind"] in ("scan_burst", "inside_scan")]
    if sb:
        res["scan_burst_window_s"] = [x["len_s"] for x in sb]
        res["scan_per_tile_ms_under_nav"] = [x["worker_per_inf_ms_mean"] for x in sb]
    for k in ("bg_radar", "bg_plan"):
        if k in w: res[k] = {kk: w[k][kk] for kk in ("spec", "n", "per_inf_ms_mean", "per_inf_ms_p95", "per_inf_ms_max", "achieved_hz")}
    if run["mode"] in ("radar", "radarflat"):
        po = res["pooled"]; rec = po["recovery"]["rate_hz"]
        for k in ("radar20", "radarflat4s"):
            if k in po: po[k]["pct_of_recovery"] = round(100 * po[k]["rate_hz"] / rec, 1)
        rw = [x for x in res["windows"] if x["kind"] in ("radar20", "radarflat4s")]
        res["radar_fuse_ms_under_nav"] = [x["worker_per_inf_ms_mean"] for x in rw]
        if "radar_alone" in w: res["radar_alone_ms"] = w["radar_alone"]["per_inf_ms_mean"]
    C["runs"][os.path.basename(p)[:-8]] = res
T["contention"] = C

# ---------------- the headline (DERIVED from the sections above) ----------------
cr = C["runs"]
T["headline"] = {b: {"full_cost_ms": FC[b]["median_of_run_medians_ms"], "full_cost_hz": FC[b]["hz"], "gate": G["boards"][b]["verdict"],
                     "gate_sigma": G["boards"][b]["G1_sigma"],
                     "reach_lockstep_K4": (CL["arms"][f"{b}_K4_lockstep"] if b != "rtx5090" else CL["arms"][P5 + "trt_np_K4_llvmpipe"])["reached"],
                     "reach_backtoback": (CL["arms"][f"{b}_naive_backtoback"] if b != "rtx5090" else CL["arms"][P5 + "delay_trt_K1_d1_naive"])["reached"],
                     "scan_hz": cr[f"e1_scan_{b}"]["pooled"]["scan_burst"]["rate_hz"],
                     "scan_solo_hz": cr[f"e1_scan_{b}"]["pooled"]["solo"]["rate_hz"],
                     "scan_pct_of_solo": cr[f"e1_scan_{b}"]["pooled"]["scan_burst"]["pct_of_solo"],
                     "all_three_inside_scan_hz": cr[f"e3_all_{b}"]["pooled"]["inside_scan"]["rate_hz"]} for b in BOARDS}
T["headline"]["rtx5090"]["full_cost_hz_pinned"] = FC["rtx5090_pinned"]["hz"]
T["headline"]["rtx5090"]["scan_hz_rep2"] = cr["e1_scan_rtx5090_rep2"]["pooled"]["scan_burst"]["rate_hz"]
T["headline"]["rtx5090"]["scan_hz_pinned"] = cr["e1_scan_rtx5090_pinned"]["pooled"]["scan_burst"]["rate_hz"]

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(T, open(OUT, "w"), indent=1)
print("wrote", OUT)
