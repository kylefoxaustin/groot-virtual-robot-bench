#!/usr/bin/env python3
"""Score IQ-9075 closed-loop runs against a reference run, from the raw traces (numpy only, no board needed).

Each run dir is what the closed-loop client writes: rollout_summary.json + episode_XX_seedS_actions.npy.
Per run: success k/n + Wilson 95% CI, same-outcome-as-reference agreement + flipped seeds, and arm-joint (dims 0:14)
rel_mae vs the reference per seed, pooled and as a distribution (same definitions as
iq9/work/closed_loop_m5/finalize_m5.py and closed_loop_m7/finalize_m7.py).

usage: python iq9/accuracy/score_iq9_runs.py --ref RUNS/rtx5090_bf16 \\
           --run fp16=RUNS/iq9_fp16 --run int8_head=RUNS/iq9_int8 --run int16_head=RUNS/iq9_int16 [--out scores.json]"""
import argparse, json, math, os
import numpy as np


def rel_mae(a, b, lo=None, hi=None):
    if lo is not None: a, b = a[..., lo:hi], b[..., lo:hi]
    return float(np.mean(np.abs(a - b)) / (np.mean(np.abs(b)) + 1e-12))


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def summ(d): return json.load(open(os.path.join(d, "rollout_summary.json")))
def succ(s): return {e["seed"]: bool(e["success"]) for e in s["episodes"]}
def epi(s): return {e["seed"]: e["episode"] for e in s["episodes"]}
def load(d, ep, seed): return np.load(os.path.join(d, f"episode_{ep:02d}_seed{seed}_actions.npy")).astype(np.float64)


def score(ref, run):
    rs, s = summ(ref), summ(run); rsucc, rep, sc, ep = succ(rs), epi(rs), succ(s), epi(s)
    seeds = sorted(set(sc) & set(rsucc)); k = sum(sc[x] for x in seeds); flips = [x for x in seeds if sc[x] != rsucc[x]]
    per, pa, pb = {}, [], []
    for x in seeds:
        a, b = load(run, ep[x], x), load(ref, rep[x], x); n = min(len(a), len(b)); a, b = a[:n], b[:n]
        per[str(x)] = round(rel_mae(a, b, 0, 14), 5); pa.append(a[..., 0:14].reshape(-1)); pb.append(b[..., 0:14].reshape(-1))
    v = list(per.values())
    return {"success": f"{k}/{len(seeds)}", "wilson95": wilson(k, len(seeds)), "success_seeds": [x for x in seeds if sc[x]],
            "reference_success": f"{sum(rsucc[x] for x in seeds)}/{len(seeds)}",
            "agreement_with_reference": f"{len(seeds) - len(flips)}/{len(seeds)}", "flip_seeds": flips,
            "arm_rel_mae_vs_reference": {"pooled": round(rel_mae(np.concatenate(pa), np.concatenate(pb)), 5),
                                         "median": round(float(np.median(v)), 5), "mean": round(float(np.mean(v)), 5),
                                         "max": round(float(np.max(v)), 5)}, "per_seed_arm_rel_mae": per}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ref", required=True)
    ap.add_argument("--run", action="append", required=True, help="label=dir (repeatable)"); ap.add_argument("--out")
    a = ap.parse_args()
    out = {"tag": "MEASURED (scored from raw closed-loop traces)", "reference": a.ref, "runs": {}}
    for r in a.run:
        label, d = r.split("=", 1); out["runs"][label] = v = score(a.ref, d)
        print(f"{label:24s} success {v['success']:6s} (ref {v['reference_success']})  agree {v['agreement_with_reference']:6s} "
              f"flips {v['flip_seeds']}  arm rel_mae median {v['arm_rel_mae_vs_reference']['median']}")
    if a.out: json.dump(out, open(a.out, "w"), indent=1); print("wrote", a.out)
