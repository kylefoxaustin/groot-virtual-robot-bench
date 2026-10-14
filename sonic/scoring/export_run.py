"""Turn one analysed run dir (probe output + run_events.json + sim_state.npz + summary.json from analyze_cl.py) into the
published raw format that sonic/make_sonic_table.py reads: <name>__ticks.csv.gz, <name>__planner.csv.gz, <name>__run.json.
Every time becomes relative to the control-start action (']'); no absolute clock, pid, path or command line is kept.
usage: python3 sonic/scoring/export_run.py <run_dir> <out_dir> <board_label> [<name>]"""
import gzip, json, os, sys
import numpy as np
d, od, board = sys.argv[1], sys.argv[2], sys.argv[3]
n = sys.argv[4] if len(sys.argv) > 4 else os.path.basename(d.rstrip('/'))
TICK = np.dtype([('tick','<u8'),('t','<i8'),('enc','<i8'),('dec','<i8'),('step','<i8'),('mode','<i4'),('frame','<i4'),
                 ('planner','<i4'),('play','<i4'),('ref_q','<f4',29),('meas_q','<f4',29),('ref_root','<f4',3),('pad','<f4')])
PL = np.dtype([('t','<i8'),('total','<i8'),('model','<i8'),('kind','<i4'),('pad','<i4')])
ARM = lambda n: 'unloaded' if n.startswith(('cl_', 'v1_')) else 'cpu_saturated' if n.startswith('a4cpu') else 'gpu_cotenant' if n.startswith('a4gpu') else 'diag_cpu_saturated_no_fifo'
def gz(path, text):
    with open(path, 'wb') as f:
        with gzip.GzipFile(filename='', mode='wb', fileobj=f, mtime=0) as g: g.write(text.encode())
os.makedirs(od, exist_ok=True)
tk = np.fromfile(f'{d}/ticks.bin', dtype=TICK); pl = np.fromfile(f'{d}/planner.bin', dtype=PL)
ev = json.load(open(f'{d}/run_events.json')); acts = [e for e in ev['events'] if e['kind'] == 'action']; t0 = acts[0]['t_mono_ns']
bo = [e for e in acts if e['act'] == 'sim:band_off'][0]['t_mono_ns']; end = [e for e in acts if e['act'] == 'o'][0]['t_mono_ns']
s = json.load(open(f'{d}/summary.json')); sim = np.load(f'{d}/sim_state.npz')
ts, rs = sim['t_mono_ns'], sim['resets']; idx = [i for i in range(1, len(rs)) if rs[i] > rs[i - 1]]
rows = ['t_rel_ms,step_us,enc_us,dec_us,planner_active']
rows += [f"{(tk['t'][i]-t0)/1e6:.6f},{tk['step'][i]/1e3:.3f},{tk['enc'][i]/1e3:.3f},{tk['dec'][i]/1e3:.3f},{tk['planner'][i]}" for i in range(tk.size)]
gz(f'{od}/{n}__ticks.csv.gz', '\n'.join(rows) + '\n')
rows = ['t_rel_ms,total_ms,model_ms,kind'] + [f"{(r['t']-t0)/1e6:.6f},{r['total']/1e6:.6f},{r['model']/1e6:.6f},{r['kind']}" for r in pl]
gz(f'{od}/{n}__planner.csv.gz', '\n'.join(rows) + '\n')
b = s['behaviour']
run = dict(board=board, run=n, arm=ARM(n), tag='MEASURED', script=ev['script'], schedule=ev['schedule'],
           band_off_rel_s=round((bo - t0) / 1e9, 6), stop_rel_s=round((end - t0) / 1e9, 6),
           sim=dict(resets_rel_s=[round(float((ts[i] - t0) / 1e9), 4) for i in idx], resets_total=int(rs[-1]),
                    overrun_steps=int(b['sim_overrun_steps']), rtf_control_window=b['sim_rtf_control_window'],
                    rtf_note='a sim fall-reset rewinds sim time, so the real-time factor is not meaningful in a run with a reset'),
           fall_flags_in_rule_window=dict(pelvis_z_lt_0p30=b['fall_z_lt_0p30'], tilt_gt60=b['fall_tilt_gt60'],
                                          pelvis_z_min_m=b['pelvis_z_min'], tilt_max_deg=b['tilt_max_deg']),
           tracking=dict(phase_joint_err_mean_rad={p: v['joint_err_mean_rad'] for p, v in s['phases'].items()},
                         phases=s['phases'], planar_displacement_m=s['planar_displacement_m'], turn_yaw_deg=s['heading']['yaw_change_turn_deg_17to25s']),
           load_order=ev['load_order'])
ref = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'sonic', 'raw', 'rtx5090', 'v1_r1__run.json')
run['load_order_matches_rtx5090'] = ev['load_order'] == json.load(open(ref))['load_order']
json.dump(run, open(f'{od}/{n}__run.json', 'w'), indent=1)
print('wrote', od, n, 'ticks', tk.size, 'resets', len(idx))
