"""The closed-loop drone episode, with an optional observation-to-actuation delay ("moving while deciding").

Per control tick of DT = 20 ms (sim/drone_sim.step_kin):
  * decision n renders the observation at tick t_n;
  * the chunk it produces lands at tick t_n + d_n; until then the drone keeps applying the previous chunk;
  * before the first chunk lands the drone hovers: a = [0, 0, 0, 0] (body-frame velocities and yaw rate; the sim is
    kinematic, so zero holds position and heading exactly);
  * arrival mode: naive starts the landed chunk at index 0, aligned at index d_n; any index past 15 (the last of the
    16-step chunk) holds setpoint 15 -- in both modes, also when the previous chunk runs out before the next lands;
  * next decision at t_n + max(K_n, d_n): K setpoints per decision, or the accelerator free, whichever is later
    (K = d is back-to-back: the accelerator never idles);
  * reach (xy < 0.8 m) is checked at every decision tick and at the budget end; budget 1280 ticks (25.6 s of sim time).
With d = 0 this is the lockstep loop: the simulator waits for every decision, and K setpoints are flown per decision.
The simulator never runs during a decision (it is in the same loop); the delay is applied in sim time.

Logged per decision: obs tick, scheduled arrival tick, landed tick, start index, the chunk; per tick: applied setpoint
(float64), its decision id (-1 = hover), its chunk index, held flag.
"""
import numpy as np

DT_TICK_MS = 20.0
HORIZON = 16


def run_episode(dc, sim, e, policy, K, d, mode, budget=320 * 4):
    """policy(img, instr, st) -> (chunk [16,4] float64, extra dict). K, d: int, or callables K(d_n), d(extra)."""
    assert mode in ("naive", "aligned")
    colors = list(dc.TARGETS); color = colors[e % 4]; tgt = np.array(dc.TARGETS[color], float)
    rng = np.random.default_rng(7000 + e)
    pos = np.array([rng.uniform(-4, 4), rng.uniform(-4, 4), rng.uniform(1.5, 3.0)]); yaw = rng.uniform(-np.pi, np.pi)
    instr = f"fly to the {color} pillar"
    steps = 0; traj = [[*pos, yaw]]
    chunks, states, decs, extras = [], [], [], []
    app, app_dec, app_idx, app_held = [], [], [], []
    cur = None; pending = None; next_dec = 0
    reached = False; reached_any = False; first_any_step = None
    def land():
        nonlocal pending, cur
        if pending is not None and pending[2] == steps:
            n, ch, arr, i0 = pending; pending = None
            cur = {"n": n, "ch": ch, "landed": steps, "i0": i0}; decs[n]["landed_tick"] = steps
    while steps < budget:
        land()   # a chunk landing on a decision tick (K = d) swaps in before the next decision is issued
        if steps == next_dec:
            sim.set(pos, yaw); img = sim.img()
            st = np.array([*pos, yaw], np.float32)
            ch, extra = policy(img, instr, st)
            dn = int(d(extra)) if callable(d) else int(d)
            Kn = int(K(dn)) if callable(K) else int(K)
            assert dn >= 0 and Kn >= 1
            assert pending is None, "a chunk is still in flight at a decision tick (period < d)"
            n = len(decs); i0 = 0 if mode == "naive" else dn
            decs.append({"n": n, "obs_tick": steps, "arrival_tick": steps + dn, "landed_tick": None, "d": dn, "K": Kn,
                         "start_index": i0, "ticks_applied": 0, "held_ticks": 0})
            chunks.append(np.asarray(ch, np.float64)); states.append(st); extras.append(extra)
            pending = (n, chunks[-1], steps + dn, i0)
            next_dec = steps + max(Kn, dn)
        land()   # d = 0: the chunk just decided lands on its own obs tick
        if cur is None:
            a = np.zeros(4); di, ix, held = -1, -1, False
        else:
            j = cur["i0"] + (steps - cur["landed"]); held = j > HORIZON - 1; ix = min(j, HORIZON - 1)
            a = cur["ch"][ix]; di = cur["n"]
            decs[di]["ticks_applied"] += 1; decs[di]["held_ticks"] += int(held)
        pos, yaw = dc.step_kin(pos, yaw, a); steps += 1; traj.append([*pos, yaw])
        app.append(np.asarray(a, np.float64)); app_dec.append(di); app_idx.append(ix); app_held.append(held)
        if not reached_any and np.linalg.norm((tgt - pos)[:2]) < 0.8: reached_any = True; first_any_step = steps
        if steps == next_dec or steps == budget:
            if np.linalg.norm((tgt - pos)[:2]) < 0.8: reached = True; break
    app_dec = np.array(app_dec); app_idx = np.array(app_idx); app_held = np.array(app_held)
    obs_tick = np.array([dd["obs_tick"] for dd in decs])
    age = np.where(app_dec >= 0, np.arange(len(app_dec)) - obs_tick[np.maximum(app_dec, 0)], -1)
    row = {"e": e, "seed": 7000 + e, "color": color, "K": K if not callable(K) else "per-call", "d": d if not callable(d) else "per-call",
           "mode": mode, "reached": reached, "reached_any_step": reached_any, "first_any_step": first_any_step,
           "n_decisions": len(decs), "n_landed": int(sum(dd["landed_tick"] is not None for dd in decs)),
           "sim_steps": steps, "sim_time_s": steps * dc.DT, "final_dist_m": float(np.linalg.norm((tgt - pos)[:2])),
           "hover_ticks": int((app_dec < 0).sum()), "held_ticks": int(app_held.sum()),
           "chunks_exhausted_before_next_landed": int(sum(dd["held_ticks"] > 0 for dd in decs)),
           "applied_obs_age_ticks_mean": float(age[age >= 0].mean()) if (age >= 0).any() else None,
           "applied_obs_age_ticks_min": int(age[age >= 0].min()) if (age >= 0).any() else None,
           "applied_obs_age_ticks_max": int(age[age >= 0].max()) if (age >= 0).any() else None,
           "d_values": sorted({dd["d"] for dd in decs})}
    arrays = {"traj": np.array(traj), "acts": np.array(chunks, np.float32), "acts64": np.array(chunks), "states": np.array(states),
              "applied": np.array(app), "applied_dec": app_dec, "applied_idx": app_idx, "applied_held": app_held,
              "dec_obs_tick": obs_tick, "dec_arrival_tick": np.array([dd["arrival_tick"] for dd in decs]),
              "dec_landed_tick": np.array([-1 if dd["landed_tick"] is None else dd["landed_tick"] for dd in decs]),
              "dec_d": np.array([dd["d"] for dd in decs]), "dec_K": np.array([dd["K"] for dd in decs]),
              "dec_start_index": np.array([dd["start_index"] for dd in decs])}
    return row, arrays, decs, extras


def check_arrivals(arrays, d_expected, mode):
    """Planted-check target: every landed chunk landed at obs_tick + d_expected (d_expected = int or per-decision array),
    and its first applied tick used index 0 (naive) or d (aligned, capped at 15). Returns (ok, n_checked, n_bad, first_bad)."""
    obs = arrays["dec_obs_tick"]; land = arrays["dec_landed_tick"]
    dexp = np.broadcast_to(np.asarray(d_expected), obs.shape)
    m = land >= 0
    bad = np.where(m & (land != obs + dexp))[0].tolist()
    ad, ai = arrays["applied_dec"], arrays["applied_idx"]
    for n in np.where(m)[0]:
        t = land[n]
        want_ix = 0 if mode == "naive" else min(int(dexp[n]), HORIZON - 1)
        if ad[t] != n or ai[t] != want_ix:
            bad.append(int(n))
    bad = sorted(set(int(b) for b in bad))
    return len(bad) == 0, int(m.sum()), len(bad), (bad[0] if bad else None)
