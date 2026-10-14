"""Run analyze_run.py on one run dir, then add: (a) sim fall-resets counted INSIDE the control window only (the sim keeps
running after 'o' while control is stopped, so post-stop collapses are not falls under control), (b) heading (yaw)
change through the turn, (c) planar displacement per phase. Rewrites <run_dir>/summary.json.
usage: python3 sonic/scoring/analyze_cl.py <run_dir>"""
import json, sys, os, subprocess, numpy as np
d = sys.argv[1]
subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analyze_run.py'), d], check=True, capture_output=True)
out = json.load(open(f'{d}/summary.json'))
ev = json.load(open(f'{d}/run_events.json')); acts = [e for e in ev['events'] if e['kind'] == 'action']
sim = np.load(f'{d}/sim_state.npz'); ts = sim['t_mono_ns']; qp = sim['qpos']
t0 = acts[0]['t_mono_ns']; T = lambda s: t0 + int(s * 1e9)
band_off = [e for e in acts if e['act'] == 'sim:band_off'][0]['t_mono_ns']; end = [e for e in acts if e['act'] == 'o'][0]['t_mono_ns']
win = (ts > band_off + int(1e9)) & (ts < end)
r = sim['resets']; rin = int(r[win][-1] - r[ts <= band_off + int(1e9)][-1]) if win.any() else None
b = out['behaviour']; b['sim_resets_total_incl_after_stop'] = b['sim_resets']; b['sim_resets_in_control_window'] = rin
b['fall'] = bool(b['fall_z_lt_0p30'] or b['fall_tilt_gt60'] or (rin or 0) > 0)
b['fall_rule'] = 'pelvis z < 0.30 m OR tilt > 60 deg OR a sim fall-reset, inside [band_off+1 s, o)'
cw = (ts >= t0) & (ts < end)
b['sim_rtf_control_window'] = float((sim['sim_time'][cw][-1] - sim['sim_time'][cw][0]) / ((ts[cw][-1] - ts[cw][0]) / 1e9))
w, x, y, z = qp[:, 3], qp[:, 4], qp[:, 5], qp[:, 6]
yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
def at(s): return int(np.argmin(np.abs(ts - T(s))))
out['heading'] = {'yaw_change_turn_deg_17to25s': float(np.degrees(yaw[at(25.0)] - yaw[at(17.5)])),
                  'yaw_change_walk_10to18s_deg': float(np.degrees(yaw[at(18.0)] - yaw[at(10.0)])),
                  'commanded_turn_deg': 90.0}
disp = {}
for n, (a_, b_) in {'walk': (10.0, 18.0), 'turn_walk': (18.0, 25.0), 'run': (25.0, 31.0), 'slow_walk': (36.0, 41.0), 'macarena': (49.0, 77.0)}.items():
    i, j = at(a_), at(b_); disp[n] = float(np.hypot(qp[j, 0] - qp[i, 0], qp[j, 1] - qp[i, 1]))
out['planar_displacement_m'] = disp
json.dump(out, open(f'{d}/summary.json', 'w'), indent=1)
print(json.dumps({k: out[k] for k in ['ticks', 'step_us', 'deadline_miss_compute_gt20ms', 'deadline_miss_period_gt25ms', 'behaviour', 'heading', 'planar_displacement_m']}, indent=0))
