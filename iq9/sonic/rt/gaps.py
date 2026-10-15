"""Score sonic_rt --mode loaded runs from their per-tick log (lat.csv), never from a self-reported counter.
  compute miss = step_us > 20000 (the 50 Hz tick); late start = start-to-start gap > 21000 us; slot miss = gap > 25000 us.
usage: python3 gaps.py <arm dir> [<arm dir> ...]   (each with lat.csv: rep,i,t_start_us,enc_us,dec_us,glue_us,step_us,late_us)"""
import sys, json, numpy as np
for d in sys.argv[1:]:
    a = np.genfromtxt(f'{d}/lat.csv', delimiter=',', names=True)
    g = np.diff(a['t_start_us']); st = a['step_us']
    pc = lambda v, q: float(np.percentile(v, q))
    out = dict(dir=d, ticks=int(len(a)), step_us=dict(p50=pc(st, 50), p99=pc(st, 99), max=float(st.max())),
               gap_us=dict(p50=pc(g, 50), p99=pc(g, 99), max=float(g.max())),
               compute_misses=int((st > 20000).sum()), late_starts_gt21ms=int((g > 21000).sum()), slot_misses_gt25ms=int((g > 25000).sum()))
    json.dump(out, open(f'{d}/gaps.json', 'w'), indent=1); print(json.dumps(out))
