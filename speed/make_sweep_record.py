"""Fold the two engine-build runs of speed/denoise_sweep_fulltrt.py into the two published records
(results/speed/n15_5090_denoise_fulltrt_*.json and n15_5090_stagetime_fulltrt_*.json), field-compatible with the
eager sweep records (latency_ms / deviation_vs_ref_pct / latency_decomposition; stages_ms["4"].{transform,backbone,head}).
usage: python3 speed/make_sweep_record.py RUN_A RUN_B OUT_DENOISE OUT_STAGETIME [EAGER_SWEEP_JSON]   (needs only numpy)
EAGER_SWEEP_JSON defaults to results/speed/n15_5090_denoise_eager_2026-09-23.json (the published eager sweep).
env: CLOCKS_NOTE   free-text GPU clock state during the run, e.g. 'locked SM 2700 / MEM 14001' or 'boost, unlocked'
                   (written to clocks_pinned)
     TENANCY_NOTE  free-text note on other GPU processes / host load during the run (written to tenancy.note)"""
import json, sys, statistics as S
import numpy as np
ra, rb, out_d, out_s = sys.argv[1:5]
R = {"A": json.load(open(ra)), "B": json.load(open(rb))}
import os
V2 = json.load(open(sys.argv[5] if len(sys.argv) > 5 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "speed", "n15_5090_denoise_eager_2026-09-23.json")))
STEPS = R["A"]["steps"]; ks = [str(n) for n in STEPS]
def fit(xs, ys):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float); b, a = np.polyfit(xs, ys, 1)
    return float(a), float(b), float(1 - ((ys - (a + b * xs)) ** 2).sum() / ((ys - ys.mean()) ** 2).sum())
r3 = lambda x: round(float(x), 3)

lat = {}
for k in ks:
    a, b = R["A"]["latency_full"][k], R["B"]["latency_full"][k]
    lat[k] = {"median_ms": r3((a["median_ms"] + b["median_ms"]) / 2), "per_build_median_ms": [a["median_ms"], b["median_ms"]],
              "per_build_pass_medians_ms": [R["A"]["latency_full_pass_medians"][k], R["B"]["latency_full_pass_medians"][k]],
              "p10_ms": r3(min(a["p10_ms"], b["p10_ms"])), "p90_ms": r3(max(a["p90_ms"], b["p90_ms"])), "min_ms": r3(min(a["min_ms"], b["min_ms"])),
              "reps": a["reps"] + b["reps"], "estimator": "mean over 2 engine builds of (median of 3 pass medians, 40 reps each)"}
ia, sa, r2 = fit(STEPS, [lat[k]["median_ms"] for k in ks])
per_build_fit = {B: R[B]["fit_full"] for B in R}

dev = {}
for k in ks:
    a, b = R["A"]["deviation_full"][k], R["B"]["deviation_full"][k]
    dev[k] = {"mean_pct": round((a["mean_pct"] + b["mean_pct"]) / 2, 4), "std_pct": round((a["std_pct"] + b["std_pct"]) / 2, 4),
              "min_pct": min(a["min_pct"], b["min_pct"]), "max_pct": max(a["max_pct"], b["max_pct"]), "n_samples": a["n_samples"],
              "per_build_mean_pct": [a["mean_pct"], b["mean_pct"]]}
C = float(np.mean([dev[k]["mean_pct"] * n for k, n in zip(ks, STEPS)]))
eager_same = R["A"]["deviation_eager_same_session"]
v2_repro = {k: {"eager_sweep_mean_pct": V2["deviation_vs_ref_pct"][k]["mean_pct"], "same_session_eager_mean_pct": eager_same[k]["mean_pct"]} for k in ks}
eag_ctrl = {k: {"median_ms_per_build": [R["A"]["latency_eager_ctrl"][k]["median_ms"], R["B"]["latency_eager_ctrl"][k]["median_ms"]],
                "eager_sweep_median_ms": V2["latency_ms"][k]["median_ms"]} for k in R["A"]["latency_eager_ctrl"]}
ec = [(int(k), S.mean(v["median_ms_per_build"])) for k, v in eag_ctrl.items()]
ea, es, er2 = fit([x for x, _ in ec], [y for _, y in ec])

denoise = {
 "tag": "MEASURED",
 "study": "GR00T N1.5 denoise-steps sweep on NVIDIA's FULL-PIPELINE TensorRT path, replicating the eager sweep protocol",
 "model": "nvidia/GR00T-N1.5-3B", "device": R["A"].get("gpu", "NVIDIA RTX 5090"),
 "path": "NVIDIA Isaac-GR00T-n15 deployment_scripts/trt_model_forward: eagle_tensorrt_forward + action_head_tensorrt_forward — ViT, LLM, "
         "vlln_vl_self_attention, state_encoder, action_encoder, DiT, action_decoder all fp16 TensorRT; pixel-shuffle/mlp1/embedding merge, "
         "transforms and the Euler loop in torch (as shipped: per-engine stream sync, no CUDA graphs)",
 "trt": R["A"]["trt"], "torch": R["A"]["torch"],
 "engines": "two independent fp16 strongly-typed builds (engA, engB) from the fp16 ONNX (speed/export_onnx.sh), fixed batch-1 shapes, LLM/DiT/vlln seq 296 "
            "(= the PickNPlace prompt; all five sweep episodes share the same instruction, so the fixed shape is valid — runtime shapes asserted by TRT)",
 "clocks_under_load_sm_mem": {B: R[B]["clocks_during_latency"] for B in R},
 "clocks_pinned": os.environ.get("CLOCKS_NOTE", "unknown -- set CLOCKS_NOTE (e.g. 'locked SM 2700 / MEM 14001' or 'boost, unlocked')") +
                  "; mitigation: 3 passes per step count (ascending / descending / ascending), median of pass medians, same-session eager control.",
 "data_config": "fourier_gr1_arms_only", "embodiment": "gr1", "dof": 26, "action_horizon": 16,
 "config_match_vs_eager_sweep": "IDENTICAL: same checkpoint, data config, embodiment, 26-DOF, 16-step chunk, episodes, seeds, 64-step reference, metric. "
                       "Only the compute path changed (eager bf16 autocast -> full TensorRT fp16).",
 "input_provenance": "recorded (simulated) GR1 observations from NVIDIA demo_data/robot_sim.PickNPlace, not synthetic data: 5 episodes [0, 20, 40, 60, 80] x 3 seeds [0, 1, 2] (same as the eager sweep); latency on ds[0]",
 "noise_matching": "the eager torch.randn draw for each seed is captured (fp32, episode-independent — verified) and fed to NVIDIA's path via "
                   "action_head.init_actions (cast to fp16), for every step count and the 64-step reference: full-TRT N-step vs full-TRT 64-step "
                   "differ only by solver truncation, and full-TRT vs eager share the identical noise draw",
 "scope": "policy.get_action() full e2e (transforms + ViT + LLM + action head; the video frames are already decoded by the dataset loader), full TensorRT, warm",
 "ref_steps": 64, "reference_note": "64 steps = near-converged (NOT exact), computed on the SAME full-TensorRT path",
 "latency_ms": lat, "deviation_vs_ref_pct": dev,
 "latency_decomposition": {"backbone_intercept_ms": r3(ia), "per_denoise_step_ms": r3(sa), "r2": round(r2, 5), "per_build_fit": per_build_fit,
     "note": "full TensorRT; per-step = action_encoder + DiT + action_decoder engines + Euler-loop torch glue; intercept = denoise-independent "
             "transforms + ViT + LLM + vlln/state engines + glue (stage record has the measured split)"},
 "deviation_1overN_fit": {"model": "dev_pct ~= C/steps", "C": round(C, 3), "eager_sweep_C": V2["deviation_1overN_fit"]["C"],
     "note": "dev ~ C/N (Euler truncation); compare C with the eager sweep's C"},
 "eager_same_session_control": {
     "latency": eag_ctrl, "fit_1_4_16": {"intercept_ms": r3(ea), "per_step_ms": r3(es), "r2": round(er2, 5)},
     "deviation_reproduces_eager_sweep": v2_repro,
     "note": "Eager latency drifts between sessions (clocks, driver); the eager DEVIATION is deterministic and should reproduce the "
             "published eager sweep to ~0.01 pp. Full-TRT vs eager speedup is quoted against THIS same-session eager."},
 "speedup_vs_same_session_eager": {k: r3(S.mean(eag_ctrl[k]["median_ms_per_build"]) / lat[k]["median_ms"]) for k in eag_ctrl},
 "output_gate": {B: R[B]["gate"] for B in R},
 "output_gate_note": "full TensorRT vs eager, identical noise. Expect rel_mae ~0.3-1% on keys with real motion; an idle key (|ref| mean "
                     "~0.01, e.g. left_hand in this episode) shows a large rel_mae from its tiny denominator -- judge it by abs_mae, which should "
                     "match the other keys' fp16 floor. The eager-vs-eager same-seed control must be exactly 0.0.",
 "build_variance": {"per_step_ms": [per_build_fit["A"]["per_step_ms"], per_build_fit["B"]["per_step_ms"]],
                    "intercept_ms": [per_build_fit["A"]["intercept_ms"], per_build_fit["B"]["intercept_ms"]],
                    "max_step_count_spread_pct": r3(max(abs(a - b) / ((a + b) / 2) * 100 for a, b in (lat[k]["per_build_median_ms"] for k in ks)))},
 "torch_version": R["A"]["torch"], "utc": R["B"]["utc"],
 "tenancy": {"note": os.environ.get("TENANCY_NOTE", "not recorded -- set TENANCY_NOTE (other GPU processes during the run)")},
 "noise_floor": {"reps": "40 per pass x 3 passes x 2 builds per step count", "estimator": "median of 3 pass medians, mean over builds", "warmup": 10,
     "deviation_replication": "5 episodes x 3 seeds = 15 samples per step (mean±std); compare per_build_mean_pct across builds"},
 "metric_span": "Latency: full policy.get_action() wall on NVIDIA's full-TensorRT N1.5 path. Deviation: trajectory rel-MAE of the 26-DOF x 16-step action "
                "vs a 64-step near-converged reference on the same path, matched noise, 5 episodes x 3 seeds. Compiled-path counterpart of "
                "the eager sweep (speed/denoise_sweep_eager.py).",
 "raw": [ra, rb],
 "harness": "speed/denoise_sweep_fulltrt.py + speed/make_sweep_record.py",
}
json.dump(denoise, open(out_d, "w"), indent=1)

# ---------------- stage record ----------------
def comp(st):
    e = st["engines_ms"]
    return {"transform": st["transform_ms"], "backbone": st["backbone_ms"], "head": st["head_ms"], "other": st["outside_model_ms"],
            "vit": e["vit_fp16"], "llm": e["llm_fp16"], "backbone_torch_glue": st["backbone_torch_glue_ms"],
            "vlln_vl_self_attention": e["vlln_vl_self_attention"], "state_encoder": e["state_encoder"],
            "action_encoder_total": e["action_encoder"], "dit_total": e["DiT_fp16"], "action_decoder_total": e["action_decoder"],
            "head_torch_glue": st["head_torch_glue_ms"], "engines_sum": st["engines_sum_ms"], "host_and_glue": st["host_and_glue_ms"],
            "e2e_instrumented_median": st["e2e_instrumented_median_ms"]}
stg = {}
for n in ("4", "16"):
    ca, cb = comp(R["A"]["stages_full"][n]), comp(R["B"]["stages_full"][n])
    stg[n] = {k: r3((ca[k] + cb[k]) / 2) for k in ca}
def compfit(key):
    out = {}
    for B in R:
        ys = [comp(R[B]["stages_full"][k])[key] for k in ks]; a, b, r = fit(STEPS, ys); out[B] = (a, b, r)
    return {"intercept_ms": r3(S.mean(v[0] for v in out.values())), "per_step_ms": round(S.mean(v[1] for v in out.values()), 4),
            "r2_per_build": [round(v[2], 3) for v in out.values()]}
fits = {k: compfit(k) for k in ["transform", "backbone", "head", "other", "vit", "llm", "backbone_torch_glue", "vlln_vl_self_attention",
                                "state_encoder", "action_encoder_total", "dit_total", "action_decoder_total", "head_torch_glue",
                                "engines_sum", "host_and_glue", "e2e_instrumented_median"]}
stage = {
 "tag": "MEASURED", "study": "GR00T N1.5 stage timing on NVIDIA's FULL-PIPELINE TensorRT path — compiled counterpart of speed/stage_time_eager.py",
 "device": R["A"].get("gpu", "NVIDIA RTX 5090"), "trt": R["A"]["trt"],
 "stages_ms": stg,
 "stages_note": "transform = policy.apply_transforms (on already-decoded frames); backbone = eagle_tensorrt_forward (ViT + LLM engines + pixel-shuffle/mlp1/"
                "embedding-merge torch glue); head = action_head_tensorrt_forward (vlln + state engines, then per step: action_encoder + DiT + "
                "action_decoder engines + torch glue); other = everything else in get_action (obs copy, prepare_input/collate, unapply/denormalize). "
                "*_total = summed over the denoise steps. Mean of two engine builds.",
 "per_component_vs_steps_fit": fits,
 "action_head_per_step_ms": fits["head"]["per_step_ms"], "action_head_fixed_ms": fits["head"]["intercept_ms"],
 "fixed_vs_scaling": {
     "fixed_ms_at_any_step_count": r3(fits["transform"]["intercept_ms"] + fits["backbone"]["intercept_ms"] + fits["head"]["intercept_ms"] + fits["other"]["intercept_ms"]),
     "per_step_ms": round(fits["transform"]["per_step_ms"] + fits["backbone"]["per_step_ms"] + fits["head"]["per_step_ms"] + fits["other"]["per_step_ms"], 4),
     "fixed": {k: fits[k]["intercept_ms"] for k in ["transform", "vit", "llm", "backbone_torch_glue", "vlln_vl_self_attention", "state_encoder", "other"]},
     "scales_per_step": {k: fits[k]["per_step_ms"] for k in ["dit_total", "action_encoder_total", "action_decoder_total", "head_torch_glue"]},
     "host_and_glue": {"intercept_ms": fits["host_and_glue"]["intercept_ms"], "per_step_ms": fits["host_and_glue"]["per_step_ms"],
                       "note": "host + glue = e2e - engine time; on this path it grows with steps (Euler loop + 3 engine handoffs per step)"}},
 "instrumentation_overhead": {"e2e_at_4_instrumented_ms": stg["4"]["e2e_instrumented_median"], "e2e_at_4_uninstrumented_ms": lat["4"]["median_ms"],
     "note": "per-stage cuda-synced timers add ~0.3-0.5 ms; the stage split is read from the instrumented pass, the latency law from the uninstrumented one"},
 "cross_validation_result": "Stage fit (instrumented): fixed %.2f ms + %.3f ms/step vs the uninstrumented sweep law %.2f + %.3f ms/step "
                            "(the instrumented pass is expected to sit slightly higher: per-stage syncs)." % (
                                fits["e2e_instrumented_median"]["intercept_ms"], fits["e2e_instrumented_median"]["per_step_ms"], ia, sa),
 "utc": R["B"]["utc"],
 "tenancy": denoise["tenancy"],
 "noise_floor": {"reps": 25, "warmup": 5, "estimator": "mean of 25 synced reps per grid point per build (median e2e also kept); two builds averaged"},
 "metric_span": "Per-stage cuda-synced wall time within policy.get_action() on the full-TensorRT path, every grid step count; each engine's "
                "in-pipeline time summed per inference.",
 "input_provenance": "a recorded (simulated) GR1 observation, not synthetic data: NVIDIA demo_data/robot_sim.PickNPlace ds[0] (same step as the sweep latency and as the eager stage timing).",
}
json.dump(stage, open(out_s, "w"), indent=1)
print(json.dumps({"law": (r3(ia), r3(sa), round(r2, 5)), "C": round(C, 3), "stage4": stg["4"], "fixed": stage["fixed_vs_scaling"]}, indent=1))
