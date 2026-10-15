"""Put IQ-9075 board outputs into the npz that sonic/gate/score_gate.py scores, so the IQ-9075 is held to EXACTLY the gate the
Jetsons' upstream-built TensorRT engines were held to (same reference sets, same thresholds, same planted faults).
  encoder: sonic_bench --mode correct --enc_kind core output (enc_raw = pre-Round values) -> host Round/16 + mode select
  decoder: dec_raw of the same run; the planted 'wrong observation' decoder run (encdec/run_encdec.sh: npu_wrongobs)
  planner: sonic_rt --mode planner_correct output (qpos.f32, npf.i32)
usage: python3 to_gate_npz.py <encdec refset.npz> <npu_aaf0 dir> <npu_wrongobs dir> <planner correct dir> <out.npz>
then:  python3 sonic/gate/score_gate.py <out.npz> <out.json> <encdec refset.npz> <planner refset.npz>"""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'encdec')); from sonic_host import fsq_select
ref, dn, dw, pdir, out = sys.argv[1:6]
r = np.load(ref); enc_in = r['enc_in'].astype(np.float32); N = len(enc_in)
pre = np.fromfile(f'{dn}/enc_raw.f32', np.float32).reshape(-1, 3, 2, 32); assert len(pre) == N, (len(pre), N)
res = dict(enc_out=fsq_select(pre, enc_in),
           dec_out=np.fromfile(f'{dn}/dec_raw.f32', np.float32).reshape(N, 29),
           dec_out_planted_wrongobs=np.fromfile(f'{dw}/dec_raw.f32', np.float32).reshape(N, 29))
npf = np.fromfile(f'{pdir}/npf.i32', np.int32)
res['planner_qpos'] = np.fromfile(f'{pdir}/qpos.f32', np.float32).reshape(len(npf), 64, 36); res['planner_npf'] = npf
np.savez(out, **res); print('wrote', out, {k: v.shape for k, v in res.items()})
