#!/usr/bin/env python3
"""M3 finalize: the bit-matched paired-rollout attribution.

Arms (each a pair of rollout dirs; per-episode seed pairing, per-call index pairing,
min n_calls; rel_mae = mean|a-b| / (mean|b|+eps)):
  1 determinism  : R1(5090,noiseA) vs R2(5090,noiseA) -> must be ~0 (gates everything)
  2 noise_envelope: a=R3(5090,noiseB) b=R1(5090,noiseA)
  3 fp16_effect   : a=R4(iq9,noiseA)  b=R1(5090,noiseA)   [SAME denominator as arm2]
  4 inloop_null   : a=R5(iq9,noiseA,zeroed) b=R4(iq9,noiseA) -> must be LARGE

Decisive ratio = fp16_effect / noise_envelope, per-call and overall, whole-vector
and per joint-group (arms [0:14] / hands [14:26] / waist [26:29]).
"""
import argparse, json, os
import numpy as np

GROUPS = {"arms": (0, 14), "hands": (14, 26), "waist": (26, 29)}


def rel_mae(a, b, lo=None, hi=None):
    a = a.astype(np.float64); b = b.astype(np.float64)
    if lo is not None:
        a = a[..., lo:hi]; b = b[..., lo:hi]
    return float(np.mean(np.abs(a - b)) / (np.mean(np.abs(b)) + 1e-12))


def load_ep(d, ep, seed):
    return np.load(os.path.join(d, f"episode_{ep:02d}_seed{seed}_actions.npy"))


def seeds_of(d):
    return json.load(open(os.path.join(d, "rollout_summary.json")))["seeds"]


def per_call_arm(dir_a, dir_b, group=None):
    """Return list-of-episodes; each episode a dict with per_call rel_mae + cumulative."""
    lo, hi = (GROUPS[group] if group else (None, None))
    seeds = seeds_of(dir_b)
    eps = []
    for i, seed in enumerate(seeds):
        a = load_ep(dir_a, i, seed); b = load_ep(dir_b, i, seed)
        n = min(a.shape[0], b.shape[0])
        a, b = a[:n], b[:n]
        pc = [rel_mae(a[k], b[k], lo, hi) for k in range(n)]
        cum = [rel_mae(a[:k + 1], b[:k + 1], lo, hi) for k in range(n)]
        eps.append({"episode": i, "seed": seed, "n_calls": int(n),
                    "per_call_rel_mae": [round(v, 5) for v in pc],
                    "cumulative_rel_mae": [round(v, 5) for v in cum],
                    "rel_mae_call0": round(pc[0], 5),
                    "rel_mae_mean": round(float(np.mean(pc)), 5),
                    "rel_mae_last": round(pc[-1], 5),
                    "cumulative_last": round(cum[-1], 5)})
    return eps


def overall(eps):
    allv = [v for e in eps for v in e["per_call_rel_mae"]]
    return {"median": round(float(np.median(allv)), 5),
            "mean": round(float(np.mean(allv)), 5),
            "call0_mean": round(float(np.mean([e["rel_mae_call0"] for e in eps])), 5),
            "last_mean": round(float(np.mean([e["rel_mae_last"] for e in eps])), 5)}


def paired_call_series(eps_a, eps_b):
    """Per-call ratio series (arm3/arm2), matched by episode+call over common length."""
    ratios = []
    per_ep = []
    for ea, eb in zip(eps_a, eps_b):
        n = min(len(ea["per_call_rel_mae"]), len(eb["per_call_rel_mae"]))
        r = [ea["per_call_rel_mae"][k] / (eb["per_call_rel_mae"][k] + 1e-12) for k in range(n)]
        ratios.extend(r)
        per_ep.append({"seed": ea["seed"], "ratio_mean": round(float(np.mean(r)), 3),
                       "ratio_median": round(float(np.median(r)), 3)})
    return {"per_call_ratio_median": round(float(np.median(ratios)), 3),
            "per_call_ratio_mean": round(float(np.mean(ratios)), 3),
            "per_episode": per_ep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir holding r1..r5 rollout subdirs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    R = lambda n: os.path.join(args.root, n)
    R1, R2, R3 = R("r1_5090_noiseA"), R("r2_5090_noiseA_2"), R("r3_5090_noiseB")
    R4, R5 = R("r4_iq9_noiseA"), R("r5_iq9_noiseA_zero")

    def succ(d):
        s = json.load(open(os.path.join(d, "rollout_summary.json")))
        return {"success_rate": s["success_rate"],
                "success_seeds": [e["seed"] for e in s["episodes"] if e["success"]]}
    outcomes = {"R1_5090_noiseA": succ(R1), "R2_5090_noiseA_rerun": succ(R2),
                "R3_5090_noiseB": succ(R3), "R4_iq9_noiseA": succ(R4),
                "R5_iq9_noiseA_zeroimg": succ(R5)}
    ref_seeds = set(outcomes["R1_5090_noiseA"]["success_seeds"])
    outcome_cmp = {k: {"same_success_seeds_as_R1": set(v["success_seeds"]) == ref_seeds,
                       "seeds_differing_from_R1": sorted(ref_seeds ^ set(v["success_seeds"]))}
                   for k, v in outcomes.items() if k != "R1_5090_noiseA"}

    # arm 1 determinism
    a1 = per_call_arm(R1, R2)
    a1o = overall(a1)
    # arm 2 noise envelope (b = R1)
    a2 = per_call_arm(R3, R1)
    a2o = overall(a2)
    # arm 3 fp16 effect (b = R1, same denom as arm2)
    a3 = per_call_arm(R4, R1)
    a3o = overall(a3)
    # arm 4 in-loop null (b = R4)
    a4 = per_call_arm(R5, R4)
    a4o = overall(a4)

    # decisive ratio (whole vector)
    ratio_overall = round(a3o["median"] / (a2o["median"] + 1e-12), 3)
    ratio_series = paired_call_series(a3, a2)

    # joint-group breakdown for arm2 and arm3 + per-group ratios
    groups = {}
    for g in GROUPS:
        g2 = overall(per_call_arm(R3, R1, group=g))
        g3 = overall(per_call_arm(R4, R1, group=g))
        groups[g] = {"noise_envelope_median": g2["median"], "fp16_effect_median": g3["median"],
                     "ratio_fp16_over_noise": round(g3["median"] / (g2["median"] + 1e-12), 3)}

    det = a1o["median"]
    det_ok = det < 0.01
    arms_ratio = groups["arms"]["ratio_fp16_over_noise"]
    # verdict on the task-relevant ARM joints (whole-vector can be dominated by the hand joints)
    if not det_ok:
        verdict = ("DETERMINISM CONTROL FAILED (median rel_mae %.4f !~ 0) -- harness/env "
                   "not deterministic; downstream arms NOT trustworthy." % det)
    elif arms_ratio > 1.3:
        verdict = ("iq9 closed-loop divergence IS partly attributable to Hexagon fp16: on the "
                   "task-relevant ARM joints the fp16 effect is %.2fx the policy's own "
                   "diffusion-noise envelope (>1 with separation)." % arms_ratio)
    elif arms_ratio < 0.77:
        verdict = ("iq9 fp16 is BELOW the noise envelope on the ARM joints (ratio %.2f<1): "
                   "driving on the iq9 perturbs the trajectory LESS than re-seeding the "
                   "policy's own diffusion noise does." % arms_ratio)
    else:
        verdict = ("iq9 closed-loop divergence is WITHIN the policy's own diffusion-noise "
                   "envelope on the ARM joints (ratio %.2f ~ 1): the iq9 drives the task as "
                   "well as re-drawing the diffusion noise does; fp16 is NOT the driver." % arms_ratio)

    out = {
        "tag": "MEASURED",
        "title": "bit-matched paired closed-loop rollout: is the iq9 GR00T "
                 "trajectory divergence attributable to Hexagon v73 fp16, or within the "
                 "policy's own diffusion-noise envelope?",
        "env_name": json.load(open(os.path.join(R1, "rollout_summary.json")))["env_name"],
        "checkpoint": "GR00T-N1.5-3B base (zero-shot)",
        "design": "4 arms from injected FIXED noise banks A/B, denoising_steps=4 (Euler dt=0.25) "
                  "on BOTH backends, prefix-stripped 296-token instruction common-mode, "
                  "per-(episode,call) bit-matched pairing.",
        "arms": {
            "1_determinism_control": {"pair": "5090 noiseA vs 5090 noiseA (rerun)",
                                      "overall": a1o, "per_episode": a1, "pass_~0": det_ok},
            "2_noise_envelope": {"pair": "5090 noiseB vs 5090 noiseA", "overall": a2o,
                                 "per_episode": a2},
            "3_fp16_effect": {"pair": "iq9 noiseA vs 5090 noiseA", "overall": a3o,
                              "per_episode": a3},
            "4_inloop_obs_null": {"pair": "iq9 noiseA zeroed-image vs iq9 noiseA normal",
                                  "overall": a4o, "per_episode": a4},
        },
        "decisive_ratio_fp16_over_noise": {
            "whole_vector_overall_median": ratio_overall,
            "whole_vector_per_call": ratio_series,
            "note": "whole-vector rel_mae can be dominated by the hand joints (small "
                    "denominator); the ARM-joint ratio is the task-relevant number.",
        },
        "joint_group_breakdown": groups,
        "determinism_control_median": round(det, 6),
        "inloop_null_median": a4o["median"],
        "task_success_outcomes": outcomes,
        "outcome_comparison": outcome_cmp,
        "methodology_note":
            "RoboCasa reset(seed) alone may not reproduce the scene: robosuite's scene-generation "
            "RNG (layout/style/textures/object placement) is seeded at construction and is not "
            "reseeded by gym reset(seed). The rollout clients therefore reseed every wrapper "
            "layer's rng=default_rng(seed) + np.random.seed(seed) before reset (deep reseed); "
            "probe_env_det.py / probe_fix_det.py check this on your install. Arm 1 "
            "(determinism control) must be ~0 for the other arms to be meaningful.",
        "verdict": verdict,
        "committed": False,
    }
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps({
        "determinism_control_median (arm1, must ~0)": round(det, 6),
        "noise_envelope_median (arm2)": a2o["median"],
        "fp16_effect_median (arm3)": a3o["median"],
        "inloop_null_median (arm4, must be large)": a4o["median"],
        "ratio_whole_vector": ratio_overall,
        "ratio_per_call_median": ratio_series["per_call_ratio_median"],
        "joint_groups": groups,
        "verdict": verdict}, indent=1))
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
