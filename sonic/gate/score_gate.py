"""Score the engine-correctness gate (A2): board engine outputs (engine_gate.py) vs the RTX 5090 reference sets
(make_refsets.py), against the gate written before the Jetson runs. Planted faults must fail it.
usage: python3 score_gate.py <engine_gate out.npz> <out.json> <encdec refset.npz> <planner refset.npz>"""
import json, sys, numpy as np
REF, PREF = sys.argv[3], sys.argv[4]
r = np.load(REF); p = np.load(PREF); b = np.load(sys.argv[1])
G = dict(enc_token_exact_frac_min=0.999, enc_max_flip_levels=1, dec_rel_mae_max=1e-3, dec_max_abs_rad_max=0.02, dec_cos_min=0.99999,
         planner_npf_agree_min=84, planner_p90_max=0.05)
def dec_stats(y, ref):
    d = np.abs(y - ref); return dict(rel_mae=float(d.mean() / np.abs(ref).mean()), max_abs_rad=float(d.max()), p99_abs_rad=float(np.percentile(d, 99)),
                                     cos=float((y * ref).sum() / np.linalg.norm(y) / np.linalg.norm(ref)))
def dec_pass(s): return s['rel_mae'] <= G['dec_rel_mae_max'] and s['max_abs_rad'] <= G['dec_max_abs_rad_max'] and s['cos'] >= G['dec_cos_min']
e = b['enc_out']; er = r['enc_out']; lv = np.abs(e - er) * 16
enc = dict(token_exact_frac=float((e == er).mean()), n_token_flips=int((e != er).sum()), tick_anyflip_rate=float((e != er).any(1).mean()),
           max_flip_levels=float(lv.max()), max_abs_nonflip=float(np.abs(e - er)[lv < 0.5].max()), on_grid=bool(np.allclose(e * 16, np.round(e * 16), atol=1e-5)))
enc['pass'] = enc['token_exact_frac'] >= G['enc_token_exact_frac_min'] and enc['max_flip_levels'] <= G['enc_max_flip_levels'] + 1e-6
dec = dec_stats(b['dec_out'], r['dec_out']); dec['pass'] = dec_pass(dec)
planted = {'dec_vs_ref_shifted_1tick': dec_stats(b['dec_out'][1:], r['dec_out'][:-1]), 'dec_wrong_observation_input': dec_stats(b['dec_out_planted_wrongobs'], r['dec_out'])}
for k in planted: planted[k]['gate_fails_as_required'] = not dec_pass(planted[k])
def pl(ref_q, ref_n):
    q = b['planner_qpos']; n = b['planner_npf']; agree = int((n == ref_n).sum())
    mx = np.array([np.abs(q[i, :n[i]] - ref_q[i, :n[i]]).max() for i in range(len(n))])
    return dict(npf_agree=agree, of=len(n), per_call_max_abs_dqpos_p50=float(np.percentile(mx, 50)), p90=float(np.percentile(mx, 90)), max=float(mx.max()),
                calls_lt_1e3=int((mx < 1e-3).sum()))
plan = {'vs_ort_fp32': pl(p['ort_qpos'], p['ort_npf']), 'vs_5090_trt': pl(p['trt_qpos'], p['trt_npf'])}
plan['pass'] = plan['vs_ort_fp32']['npf_agree'] >= G['planner_npf_agree_min'] and plan['vs_ort_fp32']['p90'] <= G['planner_p90_max']
# planted planner fault: reference shifted by one call
q = b['planner_qpos']; n = b['planner_npf']; sh = np.array([np.abs(q[i + 1, :n[i + 1]] - p['ort_qpos'][i, :n[i + 1]]).max() for i in range(len(n) - 1)])
plan['planted_shifted_1call'] = dict(p90=float(np.percentile(sh, 90)), gate_fails_as_required=bool(np.percentile(sh, 90) > G['planner_p90_max']))
out = dict(gate=G, enc=enc, dec=dec, planted_decoder=planted, planner=plan, verdict='PASS' if (enc['pass'] and dec['pass'] and plan['pass']) else 'FAIL',
           planted_all_fail=all(v['gate_fails_as_required'] for v in planted.values()) and plan['planted_shifted_1call']['gate_fails_as_required'])
json.dump(out, open(sys.argv[2], 'w'), indent=1); print(json.dumps(out, indent=1))
