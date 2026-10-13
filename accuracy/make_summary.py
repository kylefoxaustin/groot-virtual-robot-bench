#!/usr/bin/env python3
"""Rebuild results/accuracy/closed_loop_summary_2026-09-18.json from the finalize_*.json outputs
(which accuracy/finalize.py computes from the raw runs in results/accuracy/runs/).
usage: make_summary.py [RESULTS_ACCURACY_DIR]"""
import json, os, sys
R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "accuracy")
F = {k: json.load(open(os.path.join(R, f"finalize_{k}.json")))
     for k in ["orin_fp16_vs_5090", "thor_fp8_vs_5090", "thor_bf16_vs_5090", "thor_fp8_vs_thor_bf16"]}
ref = F["orin_fp16_vs_5090"]["success_rate"]["ref_5090"]


def row(k):
    x = F[k]; s = x["success_rate"]["board"]; p = x["PARITY_headline"]
    fd = x["fidelity_arm_joints_board_vs_5090"]["arm_rel_mae_distribution"]
    return {"success": f'{s["k"]}/{s["n"]}', "rate": s["rate"], "wilson95": s["wilson95"], "success_seeds": s["success_seeds"],
            "agreement_with_reference": f'{p["n_agree"]}/{p["n_agree"] + p["n_disagree"]}', "flip_seeds": [f["seed"] for f in p["flips"]],
            "arm_joint_rel_mae_vs_reference": {"median": fd["median"], "max": fd["max"]},
            "scene_hash_match": f'{x["scene_bit_match"]["n_match"]}/{x["scene_bit_match"]["n_seeds"]}',
            "board_compute_ms_median": x["in_loop_hz"]["board_compute_ms_median"],
            "in_loop_ms_median_incl_zmq": x["in_loop_hz"]["in_loop_net_ms_median"],
            "source": f"results/accuracy/finalize_{k}.json"}


iso = row("thor_fp8_vs_thor_bf16")
summ = {
    "tag": "MEASURED", "date": "2026-09-18",
    "task": "RoboCasa GR1 tabletop PnPMilkToMicrowaveClose (Fourier GR1, arms+waist+hands, 29-DoF action), 20 seeds (0-19) x 720 sim "
            "steps, 20 Hz control, 16-step action chunks, 4 denoise steps",
    "policy": "youliangtan/gr00t-n1.5-robocasa-tabletop-posttrain @ b46fb3c (GR00T N1.5 post-trained on the RoboCasa tabletop tasks); "
              "PyTorch (not TensorRT) on every backend",
    "pairing": "one diffusion-noise bank (seed 5000, 20x50x1x16x32 fp32) injected bit-identically into every backend; scenes deep-reseeded "
               "per seed; per-(episode, call) noise pairing",
    "reference": {"board": "RTX 5090", "precision": "bf16", "success": f'{ref["k"]}/{ref["n"]}', "rate": ref["rate"], "wilson95": ref["wilson95"],
                  "success_seeds": ref["success_seeds"], "run": "results/accuracy/runs/rtx5090_bf16_reference"},
    "boards": {"Jetson AGX Orin fp16": row("orin_fp16_vs_5090"), "Jetson AGX Thor fp8": row("thor_fp8_vs_5090"),
               "Jetson AGX Thor bf16 (control)": row("thor_bf16_vs_5090")},
    "fp8_isolation_same_board": {
        "comparison": "Thor fp8 vs Thor bf16", "agreement": iso["agreement_with_reference"], "flip_seeds": iso["flip_seeds"],
        "arm_joint_rel_mae_median": iso["arm_joint_rel_mae_vs_reference"]["median"],
        "reading": "Thor fp8's flips vs the 5090 decompose into one hardware/stack flip (seed 7, which also flips under Thor bf16) and "
                   "two fp8-precision flips (seeds 8 and 10).",
        "source": "results/accuracy/finalize_thor_fp8_vs_thor_bf16.json"},
    "fp8_engagement_proof": json.load(open(os.path.join(R, "runs", "thor_fp8", "fp8_proof.json"))),
    "fp8_engagement_accounting": "664,200 e4m3 GEMMs / 900 policy calls = 738 per call = 165 (SigLIP) + 84 (Qwen LLM, select_layer 12) + "
                                 "1 (mlp1) + 24 (VL self-attn) + 4 x 116 (DiT per Euler step). Attention, norms, patch-embed and the "
                                 "per-embodiment state/action encoder+decoder stay bf16.",
    "latency_note": "closed-loop latencies are PyTorch eager policy calls (not TensorRT): board compute, and in-loop incl. host<->board ZMQ. "
                    "On Thor, fp8 is slower than bf16 at batch 1 (dynamic-quantisation overhead): in-loop incl. ZMQ ~141 vs ~76 ms; board compute in the table. Use results/speed for deployable latency.",
    "caveats": [
        "The 5090 reference is bf16, not fp32 ground truth.",
        "n = 20 seeds, one noise draw per seed: Wilson 95% CIs are about +/-20 points and overlap for every backend (all ~40-50%). "
        "The data rule out a collapse, not a 10-20 point degradation; the fp8 > bf16 rate ordering is noise.",
        "The checkpoint authors report 0.60 for this task (SOURCED: Isaac-GR00T examples/RoboCasa/README.md at the pinned commit); our 45% (Wilson CI 26-66%) is consistent with it.",
        "Seed 14's initial observation is nondeterministic on the host simulator (3 distinct call-0 hashes over 7 host sweeps; not the "
        "seeded layout RNG). It is a win on every backend and in no flip; excluding it leaves the conclusions unchanged "
        "(Orin 8/19 = 5090 8/19; Thor fp8 9/19 vs 8/19).",
        "The 5090 reference run predates call-0 hashing, so its scene identity is inferred (host probe + call-0 action agreement <=0.013 "
        "vs ~0.30 for a different scene). The 2026-09-18 board runs hash-match each other on 19/20 seeds (all but 14); the 2026-10-12 Thor bf16 re-run also differs at seed 5, which is not a flip seed.",
        "Thor fp8 determinism was not tested (no repeat run); Orin fp16 is bit-exact across two processes.",
        "Thor fp8's flips are heterogeneous: seed 8 is a true boundary case; seed 10 is a large-divergence success (per-call rel_mae "
        "peaks at 1.70) that won the seed the 5090 lost."],
    "reproduce": "accuracy/run_closed_loop.py per backend, then accuracy/finalize.py and accuracy/make_summary.py (docs/ACCURACY.md)"}
out = os.path.join(R, "closed_loop_summary_2026-09-18.json")
json.dump(summ, open(out, "w"), indent=1)
print("wrote", out)
