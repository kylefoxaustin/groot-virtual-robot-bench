"""Fold the A/B result JSONs (two independent engine builds) into a fleet-bench-result/1 record.
usage: python3 speed/make_e2e_record.py BOARD RES_A RES_B OUT_JSON CLOCKS_NOTE   (needs only the standard library)"""
import json, sys, statistics as S
board, ra, rb, out, clocks = sys.argv[1:6]
R = [json.load(open(ra)), json.load(open(rb))]
def blk(r, m): return [S.median(x) for x in r["blocks_ms"][m]]
def allms(m): return [v for r in R for v in blk(r, m)]
SUM3 = lambda a: a["vit_fp16"] + a["llm_fp16"] + 4 * a["DiT_fp16"]
SUM7 = lambda a: SUM3(a) + a["vlln_vl_self_attention"] + a["state_encoder"] + 4 * (a["action_encoder"] + a["action_decoder"])
full = allms("full"); hyb = allms("hybrid"); eag = allms("eager")
def med(x): return round(S.median(x), 2)
eng_in = {m: [sum(d.values()) for r in R for d in r["engine_ms_per_call_in_pipeline"][m]] for m in ("hybrid", "full")}
s3 = [SUM3(r["standalone_engine_ms"]) for r in R]; s7 = [SUM7(r["standalone_engine_ms"]) for r in R]
f, h = S.median(full), S.median(hyb)
rec = {
 "_schema": "fleet-bench-result/1",
 "title": f"GR00T N1.5 MEASURED end-to-end on {board}: NVIDIA's FULL-PIPELINE TensorRT path vs our hybrid, same session",
 "date": __import__("time").strftime("%Y-%m-%d"), "provenance": "MEASURED",
 "why": "How much of the engine-only rate survives the whole policy call, on the hybrid path (3 big engines in TRT, rest eager) vs NVIDIA's own N1.5 full-pipeline path (all 7 modules in TRT)?",
 "method": ("One process, one loaded policy, alternating blocks eager/hybrid/full/hybrid/full/eager (20 timed after 5 warmup each) so session drift cancels. "
            "HYBRID = ViT/LLM/DiT TRT engines patched at the module seams . FULL = NVIDIA deployment_scripts/trt_model_forward "
            "eagle_tensorrt_forward + action_head_tensorrt_forward (ViT, LLM, vlln+vl_self_attention, state_encoder, action_encoder, DiT, action_decoder all TRT; "
            "pixel-shuffle/mlp1/embedding merge + Euler loop in torch) — module deletion skipped so the A/B can alternate. Both modes share the SAME ViT/LLM/DiT engine "
            "objects. fp16 strongly-typed engines, fixed batch-1 shapes (LLM seq 296), built from the fp16 ONNX (speed/export_onnx.sh; md5-identical across boards). "
            "Two INDEPENDENT engine builds (A, B), each with its own A/B run. Noise: torch.manual_seed(1234) per call; FULL gets the identical eager noise draw as init_actions (bf16->fp16 exact)."),
 "host": board, "trt": R[0]["trt"], "torch": R[0]["torch"], "gpu": R[0]["gpu"], "clocks": clocks,
 "input_provenance": R[0]["input"] + " — a recorded (simulated) GR1 PickNPlace observation (camera image + proprio state + instruction), not synthetic data.",
 "results": {
  "full_trt_e2e_ms": {"median": med(full), "min": round(min(min(x) for r in R for x in r["blocks_ms"]["full"]), 2),
                      "max": round(max(max(x) for r in R for x in r["blocks_ms"]["full"]), 2), "block_medians": [round(x, 2) for x in full], "n_per_block": 20},
  "full_trt_hz": round(1000 / f, 1),
  "hybrid_e2e_ms_same_session": {"median": med(hyb), "block_medians": [round(x, 2) for x in hyb]},
  "hybrid_hz_same_session": round(1000 / h, 1),
  "eager_bf16_e2e_ms_same_session": {"median": med(eag), "block_medians": [round(x, 2) for x in eag]},
  "full_vs_hybrid_speedup": round(h / f, 3),
  "engines_in_pipeline_ms_per_inference": {m: {"median_sum": med(v), "detail_build_A_first_block": R[0]["engine_ms_per_call_in_pipeline"][m][0]} for m, v in eng_in.items()},
  "host_and_glue_ms_MEASURED": {"full": round(f - S.median(eng_in["full"]), 2), "hybrid": round(h - S.median(eng_in["hybrid"]), 2),
      "note": "e2e minus synced in-pipeline engine time, same runs. FULL host+glue = pixel shuffle, mlp1, embedding merge, transforms, Euler loop, 7-engine handoffs + per-engine stream syncs (NVIDIA's Engine.forward syncs after every call)."},
  "engine_sum_standalone_ms_DERIVED": {"vit_llm_4xdit": round(S.median(s3), 2), "all_7_modules": round(S.median(s7), 2),
      "per_engine_standalone_ms_build_A": {k: round(v, 3) for k, v in R[0]["standalone_engine_ms"].items()},
      "note": "each engine timed alone (CUDA events, 200 iters after 50 warm, same session); sum DERIVED. vit_llm_4xdit is the engine-sum basis of rate_lost_vs_engine_sum_pct."},
  "rate_lost_vs_engine_sum_pct": {"full": round(100 * (1 - S.median(s3) / f), 1), "hybrid": round(100 * (1 - S.median(s3) / h), 1),
      "basis": "1 - e2e_Hz / (1000 / standalone ViT+LLM+4xDiT ms), same session"},
 },
 "gate": {k: R[0]["gate"][k] for k in R[0]["gate"]},
 "gate_build_B": {k: R[1]["gate"][k] for k in R[1]["gate"]},
 "gate_basis": "rel_mae per action key vs eager bf16, same seed; gate: rel_mae < 0.04 on keys with meaningful magnitude. action.left_hand is idle in this episode (|ref| mean ~0.008) -> rel_mae is a DENOMINATOR ARTIFACT; its abs_mae equals the other keys' fp16 floor (small denominator: compare abs_mae).",
 "build_variance": {"builds": 2, "full_e2e_median_per_build": [round(S.median(blk(r, "full")), 2) for r in R],
                    "hybrid_e2e_median_per_build": [round(S.median(blk(r, "hybrid")), 2) for r in R]},
 "scope": "Policy call only (policy.get_action on a recorded observation); excludes camera capture and robot I/O. NVIDIA's N1.5 full path as shipped (per-engine stream sync, torch Euler loop, no CUDA graph); a CUDA-graphed/fused export could be faster still — not measured.",
}
json.dump(rec, open(out, "w"), indent=1, ensure_ascii=False); print(json.dumps(rec["results"], indent=1, ensure_ascii=False)[:2500])
