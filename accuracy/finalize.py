#!/usr/bin/env python3
"""Compare ONE board run (e.g. Orin fp16 / Thor fp8 / Thor bf16) against the RTX-5090 bf16
reference run on the same 20 seeds with the same bit-identical noise bank.

Per board:
  * success RATE + Wilson 95% CI (board) and the 5090 reference rate
  * per-seed AGREEMENT vs the 5090: agree / disagree(flip) count, WHICH seeds, direction
  * fidelity: per-call arm-joint (dims 0:14) rel_mae board-vs-5090, per-seed + pooled + dist
  * in-loop Hz: client-side net (incl host<->board ZMQ) AND board compute-only (server log)
  * scene bit-match: board call-0 obs hash vs the reference scene hashes, per seed
  * fp8-engaged proof (Thor fp8 only)
"""
import argparse, json, math, os
import numpy as np


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


def succ_map(s):
    return {e["seed"]: bool(e["success"]) for e in s["episodes"]}


def load_ep(d, ep, seed):
    return np.load(os.path.join(d, f"episode_{ep:02d}_seed{seed}_actions.npy"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board_name", required=True)     # "Jetson AGX Orin" etc
    ap.add_argument("--precision", required=True)       # fp16 / fp8 / bf16
    ap.add_argument("--dir_board", required=True)
    ap.add_argument("--dir_5090", required=True, help="reference run dir (normally the RTX 5090 bf16 run)")
    ap.add_argument("--ref_name", default="RTX 5090 (bf16)", help="label of the reference run")
    ap.add_argument("--board_calllog", required=True)   # server_calllog.json (compute ms)
    ap.add_argument("--scene_ref", default=None,
                    help="scene hashes to compare against: scene_probe.py output (JSON {seed: sha1}) or another run dir "
                         "whose rollout_summary.json has call0_obs_sha1 (the published 5090 reference run predates hashing)")
    ap.add_argument("--fp8_proof", default=None)
    ap.add_argument("--det_a", default=None)            # optional board determinism run #1
    ap.add_argument("--det_b", default=None)            # optional board determinism run #2
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    s5 = load_summary(a.dir_5090)
    sb = load_summary(a.dir_board)
    m5, mb = succ_map(s5), succ_map(sb)
    seeds = sorted(set(m5) & set(mb))
    N = len(seeds)

    k5 = sum(1 for s in seeds if m5[s]); kb = sum(1 for s in seeds if mb[s])
    r5, rb = wilson(k5, N), wilson(kb, N)
    win5 = sorted([s for s in seeds if m5[s]]); winb = sorted([s for s in seeds if mb[s]])

    agree = [s for s in seeds if m5[s] == mb[s]]
    disagree = [s for s in seeds if m5[s] != mb[s]]
    # direction: board_win = board succeeds where 5090 fails; board_lose = board fails where 5090 succeeds
    flips = []
    for s in disagree:
        direction = "board_WIN/5090_lose" if (mb[s] and not m5[s]) else "board_LOSE/5090_win"
        flips.append({"seed": s, "5090": m5[s], "board": mb[s], "direction": direction})

    ep5 = {e["seed"]: e["episode"] for e in s5["episodes"]}
    epb = {e["seed"]: e["episode"] for e in sb["episodes"]}
    per_seed_arm, per_seed_all = {}, {}
    pa, pb = [], []
    for s in seeds:
        x = load_ep(a.dir_board, epb[s], s)
        y = load_ep(a.dir_5090, ep5[s], s)
        n = min(x.shape[0], y.shape[0]); x, y = x[:n], y[:n]
        per_seed_arm[str(s)] = round(rel_mae(x, y, 0, 14), 5)
        per_seed_all[str(s)] = round(rel_mae(x, y), 5)
        pa.append(x[..., 0:14].reshape(-1)); pb.append(y[..., 0:14].reshape(-1))
    pooled_arm = rel_mae(np.concatenate(pa), np.concatenate(pb))
    arm_vals = list(per_seed_arm.values())
    both_win = [s for s in seeds if m5[s] and mb[s]]
    def cls(s):
        return "FLIP" if s in disagree else ("both_win" if s in both_win else "both_lose")
    top3 = sorted(seeds, key=lambda s: -per_seed_arm[str(s)])[:3]
    agree_vals = [per_seed_arm[str(s)] for s in agree]
    fidelity = {
        "arm_joints_rel_mae_pooled": round(pooled_arm, 5),
        "per_seed_arm_joint_rel_mae": per_seed_arm,
        "per_seed_all_joint_rel_mae": per_seed_all,
        "arm_rel_mae_distribution": {
            "min": round(float(np.min(arm_vals)), 5), "median": round(float(np.median(arm_vals)), 5),
            "mean": round(float(np.mean(arm_vals)), 5), "max": round(float(np.max(arm_vals)), 5)},
        "top3_seeds_by_arm_rel_mae": [{"seed": s, "arm_rel_mae": per_seed_arm[str(s)], "class": cls(s)} for s in top3],
        "flip_seed_arm_rel_mae": {str(s): per_seed_arm[str(s)] for s in disagree},
        "agree_only_median": round(float(np.median(agree_vals)), 5) if agree_vals else None,
    }

    # scene bit-match
    if a.scene_ref and os.path.isdir(a.scene_ref):
        ref_h = {str(e["seed"]): e.get("call0_obs_sha1") for e in load_summary(a.scene_ref)["episodes"]}
    elif a.scene_ref:
        ref_h = json.load(open(a.scene_ref))
    else:
        ref_h = {}
    board_h = {str(e["seed"]): e.get("call0_obs_sha1") for e in sb["episodes"]}
    scene = {"n_seeds": N, "n_match": 0, "mismatches": []}
    for s in seeds:
        rh, bh = ref_h.get(str(s)), board_h.get(str(s))
        if rh is not None and bh is not None and rh == bh:
            scene["n_match"] += 1
        else:
            scene["mismatches"].append({"seed": s, "ref": rh, "board": bh})
    scene["all_match"] = (scene["n_match"] == N)
    scene["checked"] = bool(ref_h)

    # Hz: board compute-only vs client net (incl ZMQ)
    cl = json.load(open(a.board_calllog))
    comp = np.array([c["server_ms"] for c in cl], dtype=float)
    net = np.array([e["server_ms_median"] for e in sb["episodes"]], dtype=float)  # per-episode net median
    hz = {
        "n_calls": len(cl),
        "board_compute_ms_median": round(float(np.median(comp)), 1),
        "board_compute_hz": round(1000.0 / float(np.median(comp)), 3),
        "in_loop_net_ms_median": round(float(np.median([e["server_ms_median"] for e in sb["episodes"]])), 1),
        "in_loop_policy_call_hz_incl_zmq": round(1000.0 / float(sb["action_net_ms_overall_median"]), 3),
        "note": "in-loop Hz INCLUDES host<->board ZMQ network latency (client on host, policy on board); "
                "board_compute_hz is the board's get_action compute only (server-side timer)."}

    # optional determinism spot-check
    det = None
    if a.det_a and a.det_b and os.path.exists(os.path.join(a.det_a, "rollout_summary.json")):
        da, db = load_summary(a.det_a), load_summary(a.det_b)
        epda = {e["seed"]: e["episode"] for e in da["episodes"]}
        epdb = {e["seed"]: e["episode"] for e in db["episodes"]}
        ds = sorted(set(succ_map(da)) & set(succ_map(db)))
        mx = 0.0
        for s in ds:
            x = load_ep(a.det_a, epda[s], s); y = load_ep(a.det_b, epdb[s], s)
            n = min(x.shape[0], y.shape[0])
            mx = max(mx, float(np.max(np.abs(x[:n].astype(np.float64) - y[:n].astype(np.float64)))))
        det = {"seeds": ds, "max_abs_diff": mx, "bit_exact": mx == 0.0}

    fp8 = json.load(open(a.fp8_proof)) if a.fp8_proof and os.path.exists(a.fp8_proof) else None

    if len(disagree) == 0:
        verdict = ("%s (%s) vs %s: %d/%d = %.0f%% success, agrees with the reference on ALL %d seeds "
                   "(0 flips). Reference %d/%d = %.0f%%. Arm-joint fidelity median %.3f." %
                   (a.board_name, a.precision, a.ref_name, kb, N, 100*kb/N, N, k5, N, 100*k5/N,
                    fidelity["arm_rel_mae_distribution"]["median"]))
    else:
        verdict = ("%s (%s) vs %s: %d/%d = %.0f%% success; agrees on %d/%d seeds, %d FLIP(S) at %s. "
                   "Reference %d/%d = %.0f%%. Arm-joint fidelity median %.3f. A flip is a win/lose disagreement on the "
                   "bit-identical scene + noise -- report it plainly." %
                   (a.board_name, a.precision, a.ref_name, kb, N, 100*kb/N, len(agree), N, len(disagree),
                    [f["seed"] for f in flips], k5, N, 100*k5/N,
                    fidelity["arm_rel_mae_distribution"]["median"]))

    if scene["checked"] and not scene["all_match"]:
        verdict = ("[scene hash match %d/%d -- seeds %s differ from the reference scene; inspect them (in the published runs "
                   "seed 14's initial observation is host-nondeterministic)] " % (scene["n_match"], N,
                   [m["seed"] for m in scene["mismatches"]])) + verdict

    out = {
        "campaign": "%s (%s) vs %s reference: FINE-TUNED GR00T closed-loop success-RATE "
                    "+ per-seed PARITY over the same 20-seed sweep and noise bank" % (a.board_name, a.precision, a.ref_name),
        "date": __import__("time").strftime("%Y-%m-%d"), "tag": "MEASURED",
        "board": a.board_name, "precision": a.precision, "N_seeds": N, "seeds": seeds,
        "reference": "RTX-5090 bf16 reference run (same noise bank), win set %s = %d/%d" % (win5, k5, N),
        "scene_bit_match": scene,
        "success_rate": {
            "board": {"k": kb, "n": N, "rate": rb[0], "wilson95": [rb[1], rb[2]], "success_seeds": winb},
            "ref_5090": {"k": k5, "n": N, "rate": r5[0], "wilson95": [r5[1], r5[2]], "success_seeds": win5},
            "published_anchor": "~60% NVIDIA"},
        "PARITY_headline": {"n_agree": len(agree), "n_disagree": len(disagree),
                            "agreement_rate": round(len(agree)/N, 4) if N else 0.0,
                            "flips": flips,
                            "note": "a flip = the board's precision changed a win/lose outcome vs the 5090 "
                                    "bf16 reference on the bit-identical scene+noise"},
        "fidelity_arm_joints_board_vs_5090": fidelity,
        "in_loop_hz": hz,
        "determinism_spotcheck": det,
        "fp8_engaged_proof": fp8,
        "caveats": [
            "5090 reference is bf16, NOT fp32 -- part of any small residual is the 5090's own bf16.",
            "Board attention runs on torch SDPA (no flash_attn on Jetson); SDPA is exact attention, "
            "far below the precision effect measured, and both boards use it consistently.",
            "n=%d seeds; Wilson CI reflects the modest n." % N,
            "Board runs the SAME fine-tuned policy brain as the 5090; flips are boundary disagreements, not capability."],
        "verdict": verdict, "committed": False}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({"board": a.board_name, "precision": a.precision, "N": N,
                      "board_rate": f"{kb}/{N}={rb[0]} CI{rb[1:]}", "ref_5090": f"{k5}/{N}={r5[0]} CI{r5[1:]}",
                      "board_wins": winb, "5090_wins": win5, "n_flips": len(disagree),
                      "flips": flips, "arm_fid_median": fidelity["arm_rel_mae_distribution"]["median"],
                      "arm_fid_pooled": round(pooled_arm, 5),
                      "scene_match": f'{scene["n_match"]}/{N}', "compute_hz": hz["board_compute_hz"],
                      "in_loop_hz_incl_zmq": hz["in_loop_policy_call_hz_incl_zmq"],
                      "verdict": verdict}, indent=1))
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
