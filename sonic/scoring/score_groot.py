"""GPU co-tenant rate: GR00T alone (the >= 60 s before the first SONIC run's control start, after .ready) vs during each
SONIC closed-loop control window [']', 'o'] (CLOCK_MONOTONIC, shared by host and container).
usage: score_groot.py <groot_cotenant OUT_JSON> <out.json> <run_dir> [<run_dir> ...]"""
import json, sys, numpy as np
g = json.load(open(sys.argv[1])); out = sys.argv[2]; runs = sys.argv[3:]
t = g['t_mono_ns_first'] + np.array(g['t_rel_s_series']) * 1e9; ms = np.array(g['call_ms_series']); tend = t + ms * 1e6
def win(a, b):
    m = (t >= a) & (tend <= b); n = int(m.sum())
    return dict(calls=n, rate_hz=float(n / ((b - a) / 1e9)), call_ms_p50=float(np.median(ms[m])) if n else None, call_ms_p99=float(np.percentile(ms[m], 99)) if n else None, call_ms_max=float(ms[m].max()) if n else None)
res = {'groot_meta': {k: g[k] for k in ('trt', 'torch', 'gpu', 'input', 'calls', 'duration_s', 'rate_hz', 'call_ms', 'gate_full_vs_eager_rel_mae', 'gate_every25_max_rel_mae_vs_first_full')}, 'windows': {}}
starts = []
for d in runs:
    ev = json.load(open(f'{d}/run_events.json')); acts = [e for e in ev['events'] if e['kind'] == 'action']
    a = acts[0]['t_mono_ns']; b = [e for e in acts if e['act'] == 'o'][0]['t_mono_ns']; starts.append(min(e['t_mono_ns'] for e in ev['events']))
    res['windows'][d.rstrip('/').split('/')[-1] + '_control_window'] = win(a, b)
res['windows']['alone_before_first_run'] = win(t[0], min(starts))
res['windows']['alone_before_first_run']['span_s'] = float((min(starts) - t[0]) / 1e9)
json.dump(res, open(out, 'w'), indent=1); print(json.dumps(res['windows'], indent=0))
