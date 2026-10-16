#!/usr/bin/env python3
"""Compare your closed-loop run with one of ours, episode by episode.

  python3 drone/compare_closed_loop.py YOUR_episodes.jsonl results/drone/raw/closed_loop/<arm>.jsonl

Prints, for every episode you ran: reached / decisions / final distance, yours and ours, and whether they match exactly
(same outcome, same number of decisions, same final distance to 0.01 m). An exact match means your run reproduced ours bit
for bit (same renders, same policy outputs). Same outcome with different decision counts means the trajectory differed
but the result did not; that is expected across GPUs, TensorRT versions or renderers. At 40 episodes, only a large change in
reach is detectable (exact McNemar test on the episodes both runs have). Needs python3 + numpy."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import drone_metrics as M

if len(sys.argv) != 3: sys.exit(__doc__)
load = lambda p: {r["e"]: r for r in map(json.loads, open(p))}
mine, ours = load(sys.argv[1]), load(sys.argv[2])
eps = sorted(set(mine) & set(ours))
if not eps: sys.exit("no common episodes")
exact = same = 0
print(f"{'ep':>3} {'colour':7} {'yours':>22} {'ours':>22}  match")
for e in eps:
    a, b = mine[e], ours[e]
    fa, fb = round(a["final_dist_m"], 2), round(b["final_dist_m"], 2)
    s = a["reached"] == b["reached"]; x = s and a["n_decisions"] == b["n_decisions"] and fa == fb
    same += s; exact += x
    f = lambda r, fd: f"{'R' if r['reached'] else 'F'} {r['n_decisions']:4d} dec {fd:6.2f} m"
    print(f"{e:3d} {a['color']:7} {f(a, fa):>22} {f(b, fb):>22}  {'exact' if x else ('same outcome' if s else 'DIFFERENT')}")
b_, c_, p = M.mcnemar({e: mine[e]["reached"] for e in eps}, {e: ours[e]["reached"] for e in eps})
print(f"\nepisodes {len(eps)}: reached yours {sum(mine[e]['reached'] for e in eps)}, ours {sum(ours[e]['reached'] for e in eps)}; "
      f"same outcome {same}/{len(eps)}, exact match {exact}/{len(eps)}; McNemar exact p = {p:.3g} ({b_} yours only, {c_} ours only)")
