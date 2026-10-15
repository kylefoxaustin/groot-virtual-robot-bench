# Gate a board planner run (rt/sonic_rt --mode planner_correct) against thresholds written before the run.
# usage: python3 gate.py <board run dir> <planner refset.npz> <pred.json with a 'thresholds' block>
# pred.json comes from predict.py; add a "thresholds" object to it BEFORE the board run (add_thresholds.py writes it; keys: npf_agree_min, pose_code_agree_min,
# root_pos_err_m_max, joint_err_rad_p99_max, fp16_signature_joint_p99_min), derived from the predicted f16store/f16acc bracket.
import numpy as np, json, sys, os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); from pmetrics import metrics
ref=np.load(sys.argv[2]); P=json.load(open(sys.argv[3]))
d=sys.argv[1]; N=len(ref['mode'])
q=np.fromfile(f'{d}/qpos.f32',np.float32).reshape(N,64,36); n=np.fromfile(f'{d}/npf.i32',np.int32)
codes=np.fromfile(f'{d}/plogits.f32',np.float32).reshape(N,16,8,10).argmax(-1)
refc=np.load(sys.argv[3]+'.fp32.npz')['codes']
m=metrics(ref['ort_qpos'],ref['ort_npf'],q,n,refc,codes); th=P['thresholds']
checks={'npf_agree':m['npf_agree']>=th['npf_agree_min'],'pose_code_agree':m['pose_code_agree']>=th['pose_code_agree_min'],
        'root_pos_err_m_max':m['root_pos_err_m']['max']<=th['root_pos_err_m_max'],'joint_err_rad_p99':m['joint_err_rad']['p99']<=th['joint_err_rad_p99_max'],
        'fp16_signature':m['joint_err_rad']['p99']>=th['fp16_signature_joint_p99_min']}
m['checks']=checks; m['verdict']='PASS' if all(checks.values()) else 'FAIL'
m['predicted_bracket']={k:(P['f16store'].get(k),P['f16acc'].get(k)) for k in ('npf_agree','pose_code_agree','root_pos_err_m','joint_err_rad','calls_any_code_flip')}
json.dump(m,open(f'{d}/gate.json','w'),indent=1)
print(json.dumps({k:v for k,v in m.items() if k not in ('predicted_bracket','npf_mismatch_calls')}))
