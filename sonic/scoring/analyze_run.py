"""Analyse one SONIC sim2sim run dir -> summary json (latency, deadline misses, planner, behaviour)."""
import json, sys, os, numpy as np
d = sys.argv[1]
TICK = np.dtype([('tick','<u8'),('t','<i8'),('enc','<i8'),('dec','<i8'),('step','<i8'),('mode','<i4'),('frame','<i4'),
                 ('planner','<i4'),('play','<i4'),('ref_q','<f4',29),('meas_q','<f4',29),('ref_root','<f4',3),('pad','<f4')])
meta = dict(l.split() for l in open(f'{d}/probe_meta.txt'))
assert int(meta['sizeof_tick']) == TICK.itemsize, (meta, TICK.itemsize)
tk = np.fromfile(f'{d}/ticks.bin', dtype=TICK)
PL = np.dtype([('t','<i8'),('total','<i8'),('model','<i8'),('kind','<i4'),('pad','<i4')])
pl = np.fromfile(f'{d}/planner.bin', dtype=PL)
ev = json.load(open(f'{d}/run_events.json'))
sim = np.load(f'{d}/sim_state.npz')
acts = [e for e in ev['events'] if e['kind'] == 'action']
t0 = acts[0]['t_mono_ns']           # ']' control start
def T(s): return t0 + int(s * 1e9)
def st(x):
    x = np.asarray(x, float)
    return dict(n=int(x.size), p50=float(np.percentile(x,50)), p90=float(np.percentile(x,90)), p99=float(np.percentile(x,99)),
                max=float(x.max()), mean=float(x.mean()), min=float(x.min()))
us = lambda a: a / 1e3
out = dict(run=os.path.basename(d), ticks=int(tk.size), duration_s=float((tk['t'][-1]-tk['t'][0])/1e9))
out['enc_us'] = st(us(tk['enc'])); out['dec_us'] = st(us(tk['dec'])); out['step_us'] = st(us(tk['step']))
out['enc_plus_dec_us'] = st(us(tk['enc'] + tk['dec']))
per = np.diff(tk['t']) / 1e6
out['period_ms'] = st(per)
out['deadline_miss_compute_gt20ms'] = int((tk['step'] > 20e6).sum())
out['deadline_miss_period_gt25ms'] = int((per > 25).sum())
out['period_gt21ms'] = int((per > 21).sum())
# warm-up exclusion view: first 50 ticks (engine/CUDA-graph first calls)
out['step_us_excl_first50'] = st(us(tk['step'][50:]))
if pl.size:
    upd = pl[pl['kind'] == 1]
    out['planner_calls'] = int(pl.size); out['planner_update_calls'] = int(upd.size)
    out['planner_total_ms'] = st(pl['total'] / 1e6); out['planner_model_ms'] = st(pl['model'] / 1e6)
    pa = tk['planner'] == 1
    act_s = pa.sum() * 0.02
    out['planner_active_s'] = float(act_s)
    out['planner_call_rate_hz_while_active'] = float(upd.size / act_s) if act_s else None
# --- behaviour ---
band_off = [e for e in acts if e['act'] == 'sim:band_off'][0]['t_mono_ns']
ts = sim['t_mono_ns']; qp = sim['qpos']; qv = sim['qvel6']
after = ts > band_off + int(1.0e9)  # 1 s for the drop to the ground
w, x, y, z = qp[:,3], qp[:,4], qp[:,5], qp[:,6]
upz = 1 - 2*(x*x + y*y)              # world-z component of body z-axis
tilt = np.degrees(np.arccos(np.clip(upz, -1, 1)))
end = [e for e in acts if e['act'] == 'o'][0]['t_mono_ns']
win = after & (ts < end)
fall_z = bool((qp[win,2] < 0.30).any()); fall_tilt = bool((tilt[win] > 60).any())
resets = int(sim['resets'][-1])
out['behaviour'] = dict(fall=bool(fall_z or fall_tilt or resets > 0), fall_z_lt_0p30=fall_z, fall_tilt_gt60=fall_tilt, sim_resets=resets,
    pelvis_z_min=float(qp[win,2].min()), pelvis_z_p50=float(np.median(qp[win,2])), tilt_max_deg=float(tilt[win].max()),
    sim_overrun_steps=int(sim['overruns'][-1]), sim_rtf=float((sim['sim_time'][-1]-sim['sim_time'][0])/((ts[-1]-ts[0])/1e9)))
# phases (seconds after ']'); steady windows skip transients
PH = {'stand_ref_frame0': (4.0, 7.0), 'planner_idle': (8.0, 10.0), 'walk': (12.0, 18.0), 'turn_walk': (20.5, 25.0),
      'run': (27.0, 31.0), 'idle_after_run': (33.0, 36.0), 'slow_walk': (38.0, 41.0), 'ref_clip_macarena': (49.5, 76.5)}
if ev['script'] != 'v1': PH = {'stand_ref_frame0': (4.0, 7.5)}
ph = {}
for name, (a, b) in PH.items():
    m = (tk['t'] >= T(a)) & (tk['t'] < T(b))
    sm = (ts >= T(a)) & (ts < T(b))
    if m.sum() == 0: continue
    err = np.abs(tk['ref_q'][m] - tk['meas_q'][m])
    sp = np.hypot(qv[sm,0], qv[sm,1])
    # reference root speed from consecutive ticks whose frame advanced by exactly 1 (same motion, no replan jump)
    rr = tk['ref_root'][m]; fr = tk['frame'][m]
    ok = (np.diff(fr) == 1)
    dr = np.hypot(np.diff(rr[:,0]), np.diff(rr[:,1]))[ok]
    dr = dr[dr < 0.2]
    ref_speed = float(dr.mean() / 0.02) if dr.size > 10 else None
    ph[name] = dict(ticks=int(m.sum()), joint_err_mean_rad=float(err.mean()), joint_err_p95_rad=float(np.percentile(err,95)),
                    joint_err_max_rad=float(err.max()), achieved_speed_mps=float(sp.mean()), ref_root_speed_mps=ref_speed,
                    pelvis_z_mean=float(qp[sm,2].mean()), planner_active_frac=float(tk['planner'][m].mean()), enc_mode=int(np.bincount(tk['mode'][m]+3).argmax()-3))
out['phases'] = ph
json.dump(out, open(f'{d}/summary.json', 'w'), indent=1)
print(json.dumps({k: out[k] for k in ['ticks','enc_us','dec_us','step_us','deadline_miss_compute_gt20ms','deadline_miss_period_gt25ms','behaviour']}, indent=0)[:3000])
