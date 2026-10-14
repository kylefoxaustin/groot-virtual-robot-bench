#!/usr/bin/env python3
"""Consistency check: every published record must be reproducible from the raw data shipped in results/.
  * speed:    speed/make_e2e_record.py on results/speed/raw/res_<board>_{A,B}.json   == the committed e2e records' numbers
              speed/make_sweep_record.py on results/speed/raw/denoise_sweep_{A,B}.json == the committed sweep/stage records
  * accuracy: accuracy/finalize.py on results/accuracy/runs/*                        == the committed finalize_*.json numbers
              accuracy/make_summary.py on the finalize_*.json files                   == results/accuracy/closed_loop_summary_*.json
  * profiling: profiling/ddr_bandwidth/derive_gbps.py                               == results/profiling/groot_ddr_bandwidth_* GB/s
  * sonic:    sonic/make_sonic_table.py on results/sonic/raw/<board>/*              == results/sonic/sonic_same_span_table_2026-10-14.json
              (and the per-board records' deadline / band-drop numbers must agree with that table)
  * docs:     tools/make_results_md.py output                                       == the committed RESULTS.md + README headline and SONIC table
Records marked "published record" in docs/BENCHMARKS.md have no raw data in this repo to rebuild from, so they are not checked.
Everything is regenerated into a temporary directory: the check never modifies your checkout.
Needs only python3 + numpy. Exit code 1 on any mismatch.   usage: python3 tools/check_results.py"""
import json, os, shutil, subprocess, sys, tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RS, RA = os.path.join(ROOT, "results", "speed"), os.path.join(ROOT, "results", "accuracy")
PY = sys.executable
bad = []


def run(args, env=None):
    r = subprocess.run([PY] + args, capture_output=True, text=True, env=dict(os.environ, **(env or {})))
    if r.returncode:
        bad.append(f"{args[0]} failed: {r.stderr[-400:]}")
    return r


def same(a, b, label, keys):
    for k in keys:
        if json.dumps(a.get(k), sort_keys=True) != json.dumps(b.get(k), sort_keys=True):
            bad.append(f"{label}: '{k}' differs from a fresh regeneration")


with tempfile.TemporaryDirectory() as td:
    for b in ("5090", "thor", "orin"):
        pub = json.load(open(f"{RS}/groot_{b}_fulltrt_e2e_2026-10-12.json"))
        out = f"{td}/{b}.json"
        run([f"{ROOT}/speed/make_e2e_record.py", pub["host"], f"{RS}/raw/res_{b}_A.json", f"{RS}/raw/res_{b}_B.json", out, pub["clocks"]])
        if os.path.exists(out):
            same(pub, json.load(open(out)), f"e2e {b}", ["results", "gate", "gate_build_B", "build_variance"])
    run([f"{ROOT}/speed/make_sweep_record.py", f"{RS}/raw/denoise_sweep_A.json", f"{RS}/raw/denoise_sweep_B.json",
         f"{td}/d.json", f"{td}/s.json"])
    if os.path.exists(f"{td}/d.json"):
        same(json.load(open(f"{RS}/n15_5090_denoise_fulltrt_2026-10-12.json")), json.load(open(f"{td}/d.json")), "denoise sweep",
             ["latency_ms", "deviation_vs_ref_pct", "latency_decomposition", "output_gate", "build_variance", "speedup_vs_same_session_eager"])
        same(json.load(open(f"{RS}/n15_5090_stagetime_fulltrt_2026-10-12.json")), json.load(open(f"{td}/s.json")), "stage split",
             ["stages_ms", "per_component_vs_steps_fit", "fixed_vs_scaling"])
    runs = f"{RA}/runs"
    jobs = [("orin_fp16_vs_5090", "Jetson AGX Orin", "fp16", "orin_fp16", "rtx5090_bf16_reference", "thor_bf16", "RTX 5090 (bf16)"),
            ("thor_fp8_vs_5090", "Jetson AGX Thor", "fp8", "thor_fp8", "rtx5090_bf16_reference", "orin_fp16", "RTX 5090 (bf16)"),
            ("thor_bf16_vs_5090", "Jetson AGX Thor", "bf16", "thor_bf16", "rtx5090_bf16_reference", "orin_fp16", "RTX 5090 (bf16)"),
            ("thor_fp8_vs_thor_bf16", "Jetson AGX Thor", "fp8", "thor_fp8", "thor_bf16", "thor_bf16", "Jetson AGX Thor (bf16)"),
            ("thor_fp16_vs_5090", "Jetson AGX Thor", "fp16", "thor_fp16", "rtx5090_bf16_reference", "orin_fp16", "RTX 5090 (bf16)"),
            ("thor_fp16_vs_thor_bf16_rerun", "Jetson AGX Thor", "fp16", "thor_fp16", "thor_bf16_rerun_2026-10-12", "orin_fp16",
             "Jetson AGX Thor (bf16, 2026-10-12 rerun)")]
    for name, board, prec, d, ref, scene, refname in jobs:
        args = [f"{ROOT}/accuracy/finalize.py", "--board_name", board, "--precision", prec, "--ref_name", refname,
                "--dir_board", f"{runs}/{d}", "--dir_5090", f"{runs}/{ref}", "--board_calllog", f"{runs}/{d}/server_calllog.json",
                "--scene_ref", f"{runs}/{scene}", "--out", f"{td}/{name}.json"]
        if os.path.exists(f"{runs}/{d}/fp8_proof.json"):
            args += ["--fp8_proof", f"{runs}/{d}/fp8_proof.json"]
        run(args)
        if os.path.exists(f"{td}/{name}.json"):
            same(json.load(open(f"{RA}/finalize_{name}.json")), json.load(open(f"{td}/{name}.json")), f"finalize {name}",
                 ["success_rate", "PARITY_headline", "fidelity_arm_joints_board_vs_5090", "scene_bit_match", "in_loop_hz", "verdict"])
    # ---- DDR GB/s derivation
    run([f"{ROOT}/profiling/ddr_bandwidth/derive_gbps.py", f"{ROOT}/results/profiling/groot_ddr_bandwidth_2026-09-24.json"])
    # ---- closed-loop summary from the finalize records
    sd = f"{td}/acc"; os.makedirs(f"{sd}/runs/thor_fp8")
    for f in os.listdir(RA):
        if f.startswith("finalize_"):
            shutil.copy(f"{RA}/{f}", sd)
    shutil.copy(f"{runs}/thor_fp8/fp8_proof.json", f"{sd}/runs/thor_fp8/")
    run([f"{ROOT}/accuracy/make_summary.py", sd])
    for f in os.listdir(RA):
        if f.startswith("closed_loop_summary_") and os.path.exists(f"{sd}/{f}"):
            if json.load(open(f"{RA}/{f}")) != json.load(open(f"{sd}/{f}")):
                bad.append(f"{f} differs from a fresh accuracy/make_summary.py run")
    # ---- GEAR-SONIC same-span table from the per-tick logs
    RSN = os.path.join(ROOT, "results", "sonic"); stab = "sonic_same_span_table_2026-10-14.json"
    run([f"{ROOT}/sonic/make_sonic_table.py", f"{td}/{stab}"])
    if os.path.exists(f"{td}/{stab}"):
        pub, new = json.load(open(f"{RSN}/{stab}")), json.load(open(f"{td}/{stab}"))
        for k in sorted(set(pub) | set(new)):
            if json.dumps(pub.get(k), sort_keys=True) != json.dumps(new.get(k), sort_keys=True):
                bad.append(f"sonic table: '{k}' differs from a fresh sonic/make_sonic_table.py run")
        # cross-check: numbers the per-board records state must match the rebuilt table
        g = new["groups"]
        r5 = json.load(open(f"{RSN}/sonic_rtx5090_2026-10-13.json"))["results"]["deadline"]
        if (r5["compute_misses_gt20ms"], r5["slot_misses_gt25ms"], r5["worst_step_ms"], r5["worst_period_ms"]) != (
                g["rtx5090/unloaded"]["compute_misses_step_gt20ms"], g["rtx5090/unloaded"]["slot_misses_gap_gt25ms"],
                round(g["rtx5090/unloaded"]["step_max_ms"], 3), round(g["rtx5090/unloaded"]["max_start_gap_ms"], 3)):
            bad.append("sonic: RTX 5090 record deadline block disagrees with the rebuilt table")
        for b_ in ("thor", "orin"):
            rec = json.load(open(f"{RSN}/sonic_{b_}_2026-10-14.json"))
            n_bd = sum(1 for v in rec["A3_closed_loop"]["band_drop_resets_rel_s"].values() if any(0 <= x <= 3.0 for x in v["after_band_off_s"]))
            if n_bd != g[f"{b_}/unloaded"]["band_drop_falls"]:
                bad.append(f"sonic: {b_} record band-drop resets disagree with the rebuilt table")
    # ---- RESULTS.md + README headline, generated into the temp dir and compared
    run([f"{ROOT}/tools/make_results_md.py", td])
    for f, what in (("RESULTS.md", "RESULTS.md"), ("README.md", "README.md headline + SONIC tables")):
        if os.path.exists(f"{td}/{f}") and open(f"{td}/{f}").read() != open(f"{ROOT}/{f}").read():
            bad.append(f"{what} is stale -- run python3 tools/make_results_md.py and commit the result")

print("\n".join(bad) if bad else "OK: every record regenerates from the shipped raw data; RESULTS.md is current")
sys.exit(1 if bad else 0)
