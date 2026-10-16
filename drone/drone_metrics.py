"""Shared numpy-only metrics for the drone benchmark: the action-space output gate, exact McNemar / Fisher tests, and the
contention window binning. Used by make_drone_table.py (rebuild of the published numbers) and by the run scripts
(scoring your own runs the same way). Needs only numpy."""
import math
from decimal import Decimal, ROUND_HALF_UP
import numpy as np


def rhu(x, n=2):
    """round half up (22.25 -> 22.3 at n = 1), on the shortest decimal repr of the float; Python's round() would give 22.2"""
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-n), rounding=ROUND_HALF_UP))

GATE_SIGMA = 0.07        # G1: pooled RMS deviation vs the RTX 5090 bf16 reference, in dataset-action standard deviations
GATE_P95 = 0.20          # G2: per-frame RMS, 95th percentile
EXEC_STEPS = 4           # the gate scores the first 4 of the 16 setpoints (the ones a K = 4 loop executes) ...
DIMS = [0, 2, 3]         # ... in vx, vz and yaw_rate (vy has zero variance in the training data; the head emits ~0)


def exec_setpoints(actions):
    """[N, 16, 4] action chunks -> [N, 4, 3] executed setpoints in the gated dims."""
    return np.asarray(actions, np.float64)[:, :EXEC_STEPS][:, :, DIMS]


def dev_sigma(a_exec, ref_exec, ysd3):
    """Per-frame RMS deviation and the pooled RMS deviation of executed setpoints, in units of the dataset action std.
    a_exec, ref_exec: [N, 4, 3]; ysd3: the action std of [vx, vz, yaw_rate]."""
    d = (np.asarray(a_exec, np.float64) - np.asarray(ref_exec, np.float64)) / np.asarray(ysd3, np.float64)
    return np.sqrt((d ** 2).mean(axis=(1, 2))), float(np.sqrt((d ** 2).mean()))


def swap_index(instructions):
    """Planted-fault index: for frame i, the next frame at least 7 later (wrapping) with the same instruction."""
    n = len(instructions)
    return np.array([next(j for j in list(range(i + 7, n)) + list(range(0, i)) if instructions[j] == instructions[i] and j != i)
                     for i in range(n)])


def mcnemar(x, y):
    """Exact two-sided McNemar on paired per-episode outcomes. x, y: {episode: bool reached}. Returns (b, c, p)."""
    b = sum(1 for e in x if x[e] and not y[e]); c = sum(1 for e in x if not x[e] and y[e]); n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n)
    return b, c, p


def fisher(a, n1, b, n2):
    """Two-sided Fisher exact test of a/n1 vs b/n2 (sum of tables no more likely than the observed one)."""
    K = a + b; N = n1 + n2
    pr = lambda x: math.comb(n1, x) * math.comb(n2, K - x) / math.comb(N, K)
    po = pr(a); lo, hi = max(0, K - n2), min(K, n1)
    return min(1.0, sum(pr(x) for x in range(lo, hi + 1) if pr(x) <= po * (1 + 1e-9)))


def bin_contention(run, ref_exec, ysd3, swap):
    """Bin one contention run (the published raw format: decision end times relative to the nav loop's ready stamp,
    frame index per decision, executed setpoints per frame, co-tenant workers with their own [start, end]).

    Method: a nav decision counts in a window if its END time falls inside it; rate = count / window length.
    Solo = [ready + 1 s, first co-tenant spawn]. A co-tenant window is the worker's OWN interval (after its engine load
    and warm-up), never the spawn/exit bracket. Recovery (or 'outside the scan') = from each co-tenant end + 1 s to the
    next spawn (or + 5 s for the last), kept if longer than 0.5 s. A window with fewer than 5 decisions is VOID.
    Every window's decisions are gated against the RTX 5090 bf16 reference (dev_sigma <= 0.07)."""
    ts = np.asarray(run["decision_end_rel_s"]); js = np.asarray(run["j"])
    acts = np.array([run["exec_by_frame"][str(j)] for j in js])
    spawns = sorted(run["spawns_rel_s"]); first = min(spawns + run["bg_spawn_rel_s"])
    W, void = [], []

    def win(a, b, kind):
        m = (ts >= a) & (ts < b); n = int(m.sum())
        g = dev_sigma(acts[m], ref_exec[js[m]], ysd3)[1] if n else None
        if n < 5: void.append(f"{kind} window [{a:.2f}, {b:.2f}] has {n} < 5 decisions")
        if b > ts[-1]: void.append(f"{kind} window ends after the nav loop's last decision")
        w = {"kind": kind, "t0_rel_s": rhu(a, 4), "t1_rel_s": rhu(b, 4), "len_s": rhu(b - a, 4), "n": n,
             "rate_hz": rhu(n / (b - a), 2), "gate_sigma": (rhu(g, 5) if g is not None else None),
             "gate_pass": bool(g is not None and g <= GATE_SIGMA), "_m": m}
        if n < 5: w["status"] = "VOID"
        return w

    def workers(prefix, kind):
        out = []
        for name in sorted((k for k in run["workers"] if k.startswith(prefix)), key=lambda k: int("".join(c for c in k if c.isdigit()) or 0)):
            v = run["workers"][name]; w = win(v["start_rel_s"], v["end_rel_s"], kind)
            w.update({"worker_n": v["n"], "worker_per_inf_ms_mean": v["per_inf_ms_mean"]})
            for k in ("per_inf_ms_p95", "per_inf_ms_max", "achieved_hz"):
                if k in v: w["worker_" + k] = v[k]
            out.append(w)
        return out

    def gaps(cot, kind, upper):
        out = []
        for w in cot:
            e = w["t1_rel_s"]; nxt = [s for s in spawns if s > e]; hi = nxt[0] if nxt else min(e + 5.0, upper)
            if hi - (e + 1.0) > 0.5: out.append(win(e + 1.0, hi, kind))
        return out

    W.append(win(1.0, first, "solo"))
    mode = run["mode"]
    if mode == "scan":
        B = workers("vis_burst", "scan_burst"); W += B + gaps(B, "recovery", ts[-1])
    elif mode == "radar":
        R20 = workers("radar20_win", "radar20"); RF = workers("radarflat_win", "radarflat400")
        W += R20 + gaps(R20, "recovery", ts[-1]) + RF + gaps(RF, "recovery", ts[-1])
    elif mode == "radarflat":
        RF = workers("radarflat_win", "radarflat4s"); W += RF + gaps(RF, "recovery", ts[-1])
    elif mode == "all":
        bgr, bgp = run["workers"]["bg_radar"], run["workers"]["bg_plan"]
        bg0, bg1 = max(bgr["start_rel_s"], bgp["start_rel_s"]), min(bgr["end_rel_s"], bgp["end_rel_s"])
        B = workers("vis_burst", "inside_scan")
        W += [win(bg0 + 0.5, spawns[0], "outside_scan")] + B + gaps(B, "outside_scan", bg1)
    else:
        raise ValueError(mode)
    kinds = list(dict.fromkeys(w["kind"] for w in W)); pooled = {}
    for k in kinds:
        ws = [w for w in W if w["kind"] == k]; m = np.zeros(len(ts), bool)
        for w in ws: m |= w["_m"]
        n = sum(w["n"] for w in ws); t = sum(w["len_s"] for w in ws)
        pooled[k] = {"windows": len(ws), "n": n, "time_s": rhu(t, 4), "rate_hz": rhu(n / t, 2),
                     "per_window_rate_min": min(w["rate_hz"] for w in ws), "per_window_rate_max": max(w["rate_hz"] for w in ws),
                     "gate_sigma": rhu(dev_sigma(acts[m], ref_exec[js[m]], ysd3)[1], 5) if m.any() else None,
                     "void_windows": sum(1 for w in ws if w.get("status") == "VOID")}
    solo = pooled["solo"]["rate_hz"]
    for k in pooled: pooled[k]["pct_of_solo"] = rhu(100 * pooled[k]["rate_hz"] / solo, 1)
    for w in W: w.pop("_m")
    g_all = dev_sigma(acts, ref_exec[js], ysd3)[1]; g_fault = dev_sigma(acts, ref_exec[swap[js]], ysd3)[1]
    return {"windows": W, "pooled": pooled, "n_decisions": int(len(ts)), "all_decisions_gate_sigma": rhu(g_all, 5),
            "every_scored_window_passes": all(w["gate_pass"] for w in W if w.get("status") != "VOID"),
            "planted_fault_sigma": rhu(g_fault, 4), "planted_fault_fails": bool(g_fault > GATE_SIGMA), "void": void or None}
