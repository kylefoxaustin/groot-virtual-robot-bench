#!/usr/bin/env python3
"""Rebuild the GEAR-SONIC same-span table from the per-run raw files in results/sonic/raw/<board>/.

Inputs per run (written by the harness, made run-relative before publishing):
  <run>__ticks.csv.gz    one row per 50 Hz control tick: t_rel_ms (tick start, ms after control start), step_us, enc_us, dec_us, planner_active
  <run>__planner.csv.gz  one row per planner call: t_rel_ms, total_ms (UpdatePlanning/Initialize wall), model_ms (TensorRT inference), kind
  <run>__run.json        band-off / stop times, the simulator's fall-resets (times), fall flags inside the rule window, tracking per phase

Every timing column is computed here from the tick log; nothing is copied from a summary:
  step            = obs_start -> after CurrentFrameAdvancement (the whole 50 Hz control step)
  compute miss    = step > 20 ms
  late start      = tick start-to-start gap > 21 ms
  slot miss       = tick start-to-start gap > 25 ms
  fall (rule)     = pelvis z < 0.30 m OR tilt > 60 deg OR a sim fall-reset, inside [band_off + 1 s, stop)
  band-drop fall  = a sim fall-reset within 3 s after the band-off command (a reset later than +1.0 s is also in the rule window);
                    the 1.5 s window used in the original records is kept as band_drop_fall_le1p5s (same counts on this data)
  tracking ok     = per-phase mean joint error within +/-30% of the RTX 5090 range, displacement +/-15%, turn yaw +/-15 deg
usage: python3 sonic/make_sonic_table.py [OUT_JSON]   (default: results/sonic/sonic_same_span_table_2026-10-14.json)
Needs python3 + numpy."""
import csv, glob, gzip, json, math, os, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RAW = os.path.join(ROOT, "results", "sonic", "raw")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "sonic", "sonic_same_span_table_2026-10-14.json")
BOARDS = ["rtx5090", "thor", "orin"]
ARMS = ["unloaded", "cpu_saturated", "gpu_cotenant", "diag_cpu_saturated_no_fifo"]


def col(path):
    with gzip.open(path, "rt") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]} if rows else {}


def st(x):
    x = np.asarray(x, float)
    return dict(n=int(x.size), p50=float(np.percentile(x, 50)), p99=float(np.percentile(x, 99)), max=float(x.max()), mean=float(x.mean()))


def run_metrics(board, name):
    base = os.path.join(RAW, board, name)
    tk, pl, rj = col(base + "__ticks.csv.gz"), col(base + "__planner.csv.gz"), json.load(open(base + "__run.json"))
    gap = np.diff(tk["t_rel_ms"]); step = tk["step_us"] / 1e3
    bo, end = rj["band_off_rel_s"], rj["stop_rel_s"]; resets = rj["sim"]["resets_rel_s"]
    in_window = [t for t in resets if bo + 1.0 < t < end]
    ff = rj["fall_flags_in_rule_window"]
    return dict(arm=rj["arm"], ticks=int(step.size), step_ms=st(step), enc_us=st(tk["enc_us"]), dec_us=st(tk["dec_us"]),
                compute_misses_step_gt20ms=int((step > 20).sum()), late_starts_gap_gt21ms=int((gap > 21).sum()),
                slot_misses_gap_gt25ms=int((gap > 25).sum()), max_start_gap_ms=float(gap.max()),
                planner_calls=int(pl["model_ms"].size) if pl else 0, planner_model_ms=st(pl["model_ms"]) if pl else None,
                planner_total_ms=st(pl["total_ms"]) if pl else None,
                fall_rule_window=bool(ff["pelvis_z_lt_0p30"] or ff["tilt_gt60"] or len(in_window) > 0),
                band_drop_fall=any(0 <= t - bo <= 3.0 for t in resets), band_drop_fall_le1p5s=any(0 <= t - bo <= 1.5 for t in resets),
                band_drop_reset_after_band_off_s=[round(t - bo, 2) for t in resets if 0 <= t - bo <= 3.0],
                sim_resets=len(resets), sim_overrun_steps=rj["sim"]["overrun_steps"], _tracking=rj["tracking"])


def fisher_two_sided(a, b, c, d):
    """2x2 [[a, b], [c, d]] two-sided Fisher exact p (sum of tables no more likely than the observed one)."""
    n1, n2, k = a + b, c + d, a + c
    pr = lambda x: math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n1 + n2, k)
    p0 = pr(a)
    return sum(pr(x) for x in range(max(0, k - n2), min(k, n1) + 1) if pr(x) <= p0 * (1 + 1e-9))


runs = {b: {} for b in BOARDS}
for b in BOARDS:
    for p in sorted(glob.glob(os.path.join(RAW, b, "*__run.json"))):
        n = os.path.basename(p)[:-len("__run.json")]
        runs[b][n] = run_metrics(b, n)

# tracking tolerance, defined on the RTX 5090 unloaded runs
ref = [r["_tracking"] for r in runs["rtx5090"].values() if r["arm"] == "unloaded"]
rng = lambda v: (min(v), max(v))
ref_ph = {p: rng([t["phase_joint_err_mean_rad"][p] for t in ref]) for p in ref[0]["phase_joint_err_mean_rad"]}
DISP = ["walk", "turn_walk", "run", "slow_walk"]
ref_disp = {k: rng([t["planar_displacement_m"][k] for t in ref]) for k in DISP}
ref_yaw = rng([t["turn_yaw_deg"] for t in ref])
for b in BOARDS:
    for n, r in runs[b].items():
        t = r.pop("_tracking")
        ok = [ref_ph[p][0] * 0.7 <= t["phase_joint_err_mean_rad"][p] <= ref_ph[p][1] * 1.3 for p in ref_ph]
        ok += [ref_disp[k][0] * 0.85 <= t["planar_displacement_m"][k] <= ref_disp[k][1] * 1.15 for k in DISP]
        ok += [ref_yaw[0] - 15 <= t["turn_yaw_deg"] <= ref_yaw[1] + 15]
        r["tracking_ok"] = "reference" if b == "rtx5090" else bool(all(ok))

LABEL = {"rtx5090": "RTX 5090", "thor": "Jetson AGX Thor", "orin": "Jetson AGX Orin 64 GB"}
r3 = lambda x: x   # full precision; round only for display (avoids double rounding)


def pool(b, names):
    R = [runs[b][n] for n in names]
    tr = [r["tracking_ok"] for r in R]
    return dict(board=LABEL[b], runs=len(R), run_names=names, ticks=sum(r["ticks"] for r in R),
                step_p50_ms_range=[r3(min(r["step_ms"]["p50"] for r in R)), r3(max(r["step_ms"]["p50"] for r in R))],
                step_p99_ms_range=[r3(min(r["step_ms"]["p99"] for r in R)), r3(max(r["step_ms"]["p99"] for r in R))],
                step_max_ms=r3(max(r["step_ms"]["max"] for r in R)),
                compute_misses_step_gt20ms=sum(r["compute_misses_step_gt20ms"] for r in R),
                late_starts_gap_gt21ms=sum(r["late_starts_gap_gt21ms"] for r in R),
                slot_misses_gap_gt25ms=sum(r["slot_misses_gap_gt25ms"] for r in R),
                max_start_gap_ms=r3(max(r["max_start_gap_ms"] for r in R)),
                falls_rule_window=sum(r["fall_rule_window"] for r in R), band_drop_falls=sum(r["band_drop_fall"] for r in R),
                band_drop_falls_le1p5s=sum(r["band_drop_fall_le1p5s"] for r in R),
                band_drop_fall_runs=[n for n, r in zip(names, R) if r["band_drop_fall"]],
                planner_model_ms_p50_range=[r3(min(r["planner_model_ms"]["p50"] for r in R)), r3(max(r["planner_model_ms"]["p50"] for r in R))],
                sim_overrun_steps_per_run=[r["sim_overrun_steps"] for r in R],
                tracking_ok="reference" if b == "rtx5090" else f"{sum(t is True for t in tr)}/{len(tr)} runs")


order = lambda names: sorted(names, key=lambda n: (n.rsplit("_r", 1)[0], int(n.rsplit("_r", 1)[1]) if n.rsplit("_r", 1)[-1].isdigit() else 0))
groups = {}
for b in BOARDS:
    for arm in ARMS:
        names = order([n for n, r in runs[b].items() if r["arm"] == arm])
        if names:
            groups[f"{b}/{arm}"] = pool(b, names)
for b in ("thor", "orin"):
    groups[f"{b}/unloaded_preregistered_r1-r3"] = pool(b, ["cl_r1", "cl_r2", "cl_r3"])

fisher = {}
g5 = groups["rtx5090/unloaded"]
for b in ("thor", "orin"):
    g = groups[f"{b}/unloaded"]
    fisher[b] = dict(board_band_drop=f"{g['band_drop_falls']}/{g['runs']}", rtx5090_band_drop=f"{g5['band_drop_falls']}/{g5['runs']}",
                     p_two_sided=round(fisher_two_sided(g["band_drop_falls"], g["runs"] - g["band_drop_falls"],
                                                        g5["band_drop_falls"], g5["runs"] - g5["band_drop_falls"]), 3))

out = dict(
    _schema="groot-virtual-robot-bench/sonic-same-span/1", tag="MEASURED",
    tag_note="every count and percentile is computed by sonic/make_sonic_table.py from the per-tick logs in results/sonic/raw/; "
             "falls from each run's simulator record (resets and fall flags); the Fisher p-values are DERIVED",
    generated_by="sonic/make_sonic_table.py",
    same_span="same upstream program (gear_sonic_deploy g1_deploy_onnx_ref @ b042411), same measurement probe, same step span, same "
              "fixed keyboard script and the same scoring code on every machine; only the computer (and its TensorRT build) differs",
    rules=dict(step="obs_start -> after CurrentFrameAdvancement (whole 50 Hz control step)", compute_miss="step > 20 ms",
               late_start="tick start-to-start gap > 21 ms", slot_miss="tick start-to-start gap > 25 ms",
               fall_rule_window="pelvis z < 0.30 m OR tilt > 60 deg OR a sim fall-reset, inside [band_off + 1 s, stop)",
               band_drop_fall="a sim fall-reset within 3 s after the band-off command (a reset later than +1.0 s is counted in both columns)",
               tracking="per-phase mean joint error within +/-30% of the RTX 5090 v1_r1-r3 range, displacement +/-15%, turn yaw +/-15 deg"),
    tracking_reference_rtx5090=dict(phase_joint_err_mean_range_rad=ref_ph, displacement_range_m=ref_disp, turn_yaw_range_deg=ref_yaw),
    groups=groups,
    band_drop_vs_rtx5090_fisher_DERIVED=fisher,
    runs=runs)
json.dump(out, open(OUT, "w"), indent=1, sort_keys=False)
print("wrote", OUT)
