#!/usr/bin/env python3
"""M5 finalize: 20-seed FINE-TUNED GR00T success-rate + parity between the
RTX-5090 (bf16) reference and the iq9 Hexagon-v73 (fp16) board, on ONE fresh
natural diffusion-noise draw per seed injected bit-identically into both backends.

Metrics:
  * success RATE + Wilson 95% CI for each backend
  * PARITY: per-seed agreement (both win / both lose); disagreement count + seeds
  * fidelity: arm-joint (dims 0:14) rel_mae iq9-vs-5090, per-seed + pooled distribution
  * determinism spot-check: bit-exact max|diff| between two 5090 runs on the spot seeds
  * in-loop Hz from the iq9 server call log
"""
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_W = _gvrb_os.path.expanduser(_gvrb_os.environ.get("GVRB_WORK", "~/gvrb"))
_B = _gvrb_os.path.expanduser(_gvrb_os.environ.get("IQ9_WORK", "/opt/gvrb"))
import argparse, json, math, os
import numpy as np

GROUPS = {"arms": (0, 14), "hands": (14, 26), "waist": (26, 29)}


def rel_mae(a, b, lo=None, hi=None):
    a = a.astype(np.float64); b = b.astype(np.float64)
    if lo is not None:
        a = a[..., lo:hi]; b = b[..., lo:hi]
    return float(np.mean(np.abs(a - b)) / (np.mean(np.abs(b)) + 1e-12))


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(p, 4), round(max(0.0, center - half), 4), round(min(1.0, center + half), 4))


def load_summary(d):
    return json.load(open(os.path.join(d, "rollout_summary.json")))


def succ_map(summ):
    """seed -> bool success."""
    return {e["seed"]: bool(e["success"]) for e in summ["episodes"]}


def load_ep(d, ep, seed):
    return np.load(os.path.join(d, f"episode_{ep:02d}_seed{seed}_actions.npy"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir_5090", required=True)
    ap.add_argument("--dir_iq9", required=True)
    ap.add_argument("--dir_det_a", required=True, help="5090 run #1 (spot seeds)")
    ap.add_argument("--dir_det_b", required=True, help="5090 run #2 (spot seeds, rerun)")
    ap.add_argument("--iq9_calllog", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    s5 = load_summary(args.dir_5090)
    si = load_summary(args.dir_iq9)
    m5 = succ_map(s5)
    mi = succ_map(si)
    # common seeds actually completed by BOTH backends
    seeds = sorted(set(m5) & set(mi))
    N = len(seeds)

    # ---- success rates + Wilson CI ----
    k5 = sum(1 for s in seeds if m5[s])
    ki = sum(1 for s in seeds if mi[s])
    r5 = wilson(k5, N)
    ri = wilson(ki, N)
    win5 = sorted([s for s in seeds if m5[s]])
    wini = sorted([s for s in seeds if mi[s]])

    # ---- parity ----
    agree = [s for s in seeds if m5[s] == mi[s]]
    disagree = [s for s in seeds if m5[s] != mi[s]]
    disagree_detail = [{"seed": s, "5090": m5[s], "iq9": mi[s]} for s in disagree]

    # ---- fidelity: arm-joint rel_mae iq9 vs 5090, per seed + pooled ----
    per_seed_arm, per_seed_all = {}, {}
    pooled_a_arm, pooled_b_arm, pooled_a_all, pooled_b_all = [], [], [], []
    ep_index_5 = {e["seed"]: e["episode"] for e in s5["episodes"]}
    ep_index_i = {e["seed"]: e["episode"] for e in si["episodes"]}
    for s in seeds:
        a = load_ep(args.dir_iq9, ep_index_i[s], s)      # iq9
        b = load_ep(args.dir_5090, ep_index_5[s], s)     # 5090 ref (denominator)
        n = min(a.shape[0], b.shape[0])
        a, b = a[:n], b[:n]
        per_seed_arm[str(s)] = round(rel_mae(a, b, 0, 14), 5)
        per_seed_all[str(s)] = round(rel_mae(a, b), 5)
        pooled_a_arm.append(a[..., 0:14].reshape(-1)); pooled_b_arm.append(b[..., 0:14].reshape(-1))
        pooled_a_all.append(a.reshape(-1)); pooled_b_all.append(b.reshape(-1))
    pa_arm = np.concatenate(pooled_a_arm); pb_arm = np.concatenate(pooled_b_arm)
    pa_all = np.concatenate(pooled_a_all); pb_all = np.concatenate(pooled_b_all)
    pooled_arm = rel_mae(pa_arm, pb_arm)
    pooled_all = rel_mae(pa_all, pb_all)
    arm_vals = list(per_seed_arm.values())
    both_win = [s for s in seeds if m5[s] and mi[s]]
    top3 = sorted(seeds, key=lambda s: -per_seed_arm[str(s)])[:3]
    def cls(s):
        if s in disagree: return "FLIP"
        if s in both_win: return "both_win"
        return "both_lose"
    agree_vals = [per_seed_arm[str(s)] for s in seeds if s not in disagree]
    flip_vals = {str(s): per_seed_arm[str(s)] for s in disagree}
    fidelity = {
        "arm_joints_rel_mae_pooled": round(pooled_arm, 5),
        "all_joints_rel_mae_pooled": round(pooled_all, 5),
        "per_seed_arm_joint_rel_mae": per_seed_arm,
        "per_seed_all_joint_rel_mae": per_seed_all,
        "arm_rel_mae_distribution": {
            "min": round(float(np.min(arm_vals)), 5),
            "median": round(float(np.median(arm_vals)), 5),
            "mean": round(float(np.mean(arm_vals)), 5),
            "max": round(float(np.max(arm_vals)), 5),
        },
        "outlier_characterization": {
            "top3_seeds_by_arm_rel_mae": [{"seed": s, "arm_rel_mae": per_seed_arm[str(s)],
                                           "class": cls(s)} for s in top3],
            "flip_seed_arm_rel_mae": flip_vals,
            "agree_only_median": round(float(np.median(agree_vals)), 5),
            "agree_only_mean": round(float(np.mean(agree_vals)), 5),
            "note": "The median is the typical seed; a mean/pooled value inflated by late-episode path divergence on seeds whose outcome did not change is not evidence of a flip."},
        "noise_envelope_note": "one noise draw per seed (no per-seed second draw), so the run-to-run "
                               "noise envelope must come from a separate two-draw run (closed_loop_m4).",
    }

    # ---- determinism spot-check ----
    da = load_summary(args.dir_det_a)
    db = load_summary(args.dir_det_b)
    det_seeds = sorted(set(succ_map(da)) & set(succ_map(db)))
    epa = {e["seed"]: e["episode"] for e in da["episodes"]}
    epb = {e["seed"]: e["episode"] for e in db["episodes"]}
    max_abs = 0.0
    for s in det_seeds:
        a = load_ep(args.dir_det_a, epa[s], s)
        b = load_ep(args.dir_det_b, epb[s], s)
        n = min(a.shape[0], b.shape[0])
        max_abs = max(max_abs, float(np.max(np.abs(a[:n].astype(np.float64) - b[:n].astype(np.float64)))))
    det_ok = max_abs == 0.0

    # ---- in-loop Hz (iq9) ----
    cl = json.load(open(args.iq9_calllog))
    board_wall = np.array([c["board_wall_ms"] for c in cl], dtype=float)
    board_gframe = np.array([c["board_gframe_ms"] for c in cl], dtype=float)
    server_ms = np.array([c["server_ms"] for c in cl], dtype=float)
    hz = {
        "n_calls": len(cl),
        "board_wall_ms_median": round(float(np.median(board_wall)), 1),
        "board_gframe_ms_median": round(float(np.median(board_gframe)), 1),
        "server_ms_median": round(float(np.median(server_ms)), 1),
        "in_loop_policy_call_hz": round(1000.0 / float(np.median(board_wall)), 3),
        "board_gframe_floor_hz": round(1000.0 / float(np.median(board_gframe)), 3),
    }

    # ---- verdict ----
    if not det_ok:
        verdict = ("DETERMINISM SPOT-CHECK FAILED (max|diff|=%.3g != 0) -- harness NOT "
                   "deterministic; parity comparison VOID." % max_abs)
    elif len(disagree) == 0:
        verdict = ("YES -- across N=%d seeds the iq9 (fp16) and 5090 (bf16) fine-tuned GR00T "
                   "agree on EVERY episode's outcome (0 flips). iq9 rate %d/%d=%.0f%%, "
                   "5090 rate %d/%d=%.0f%%. The iq9 tracks the 5090 across the full sweep."
                   % (N, ki, N, 100 * ki / N, k5, N, 100 * k5 / N))
    else:
        verdict = ("iq9 tracks the 5090 on %d/%d seeds; %d DISAGREEMENT(S) at seeds %s "
                   "(fp16 flipped the outcome). iq9 rate %d/%d=%.0f%%, 5090 rate %d/%d=%.0f%%. "
                   "Report the flip(s) plainly."
                   % (len(agree), N, len(disagree), disagree,
                      ki, N, 100 * ki / N, k5, N, 100 * k5 / N))

    out = {
        "experiment": "iq9 Hexagon v73 fp16 vs RTX-5090 bf16: FINE-TUNED GR00T closed-loop "
                    "success-RATE + per-seed PARITY over a 20-seed sweep (one fresh natural "
                    "diffusion-noise draw per seed, injected bit-identically into both backends)",
        "tag": "MEASURED",
        "N_seeds_completed": N,
        "seeds": seeds,
        "checkpoint": {
            "fine_tuned_5090": _gvrb_os.environ.get("GROOT_FT_MODEL", _W+"/models/gr00t-n1.5-robocasa-tabletop-posttrain"),
            "fine_tuned_iq9_bins": "$IQ9_HOST:$IQ9_WORK/optionB/{ctx_ft,glue_ft,io_ft} (built by closed_loop_m4/build)",
            "env": s5["env_name"],
            "denoise": 4, "dt": 0.25, "published_ceiling": "~60% (NVIDIA)"},
        "noise": {"bank": _gvrb_os.environ.get("GVRB_NOISE_BANK", _W+"/noise_bank.npy"), "shape": list(np.load(
            _gvrb_os.environ.get("GVRB_NOISE_BANK", _W+"/noise_bank.npy")).shape),
            "seed": 5000, "draw": "fresh natural N(0,1), one per (episode,call); distinct from "
                                   "the closed_loop_m3 noiseA/noiseB banks; identical fp32 values fed to both backends"},
        "success_rate": {
            "iq9_FT": {"k": ki, "n": N, "rate": ri[0], "wilson95": [ri[1], ri[2]],
                       "success_seeds": wini},
            "ref_5090_FT": {"k": k5, "n": N, "rate": r5[0], "wilson95": [r5[1], r5[2]],
                            "success_seeds": win5},
            "published_anchor": "~60% NVIDIA"},
        "PARITY_headline": {
            "n_agree": len(agree), "n_disagree": len(disagree),
            "disagreement_seeds": disagree, "disagreement_detail": disagree_detail,
            "agreement_rate": round(len(agree) / N, 4) if N else 0.0,
            "note": "a disagreement = fp16 flipped an outcome vs bf16 on bit-identical scene+noise"},
        "fidelity_arm_joints_iq9_vs_5090": fidelity,
        "determinism_spotcheck": {
            "seeds": det_seeds, "max_abs_diff": max_abs, "bit_exact_0.0": det_ok,
            "note": "two independent 5090-FT runs on the spot seeds; must be 0.0 or comparison void"},
        "in_loop_hz_iq9": hz,
        "caveats": [
            "5090 reference is bf16, NOT fp32 -- part of any small residual is the 5090's own bf16.",
            "ONE natural diffusion-noise draw per seed (not many draws per scene): the rate is over "
            "scene+noise samples, not a per-scene expectation.",
            "n=%d episodes; Wilson CI reflects the modest n." % N,
            "iq9 runs the same fine-tuned checkpoint as the 5090 reference."],
        "verdict": verdict,
        "committed": False,
    }
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps({
        "N": N,
        "iq9_rate": f"{ki}/{N}={ri[0]}  CI{ri[1:]}",
        "5090_rate": f"{k5}/{N}={r5[0]}  CI{r5[1:]}",
        "iq9_win_seeds": wini, "5090_win_seeds": win5,
        "n_disagree": len(disagree), "disagree_seeds": disagree,
        "arm_fid_pooled": round(pooled_arm, 5),
        "arm_fid_dist": fidelity["arm_rel_mae_distribution"],
        "determinism_max_abs": max_abs, "det_ok": det_ok,
        "in_loop_hz": hz["in_loop_policy_call_hz"],
        "verdict": verdict}, indent=1))
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
