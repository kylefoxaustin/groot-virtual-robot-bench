# Derive the planner gate thresholds from the fp16 prediction and store them in pred.json (run BEFORE the board run).
# The formulas are the rule; the numbers come from YOUR prediction on YOUR reference calls:
#   npf_agree_min                = max(0.80, min(f16store, f16acc npf agreement) - 0.05)
#   pose_code_agree_min          = f16acc pose-code agreement - 0.02
#   root_pos_err_m_max           = min(2 x f16acc max root-position error, 0.10 m)
#   joint_err_rad_p99_max        = 2 x f16acc p99 joint error
#   fp16_signature_joint_p99_min = 0.1 x f16store p99 joint error (the board error must be at least this: fp16 really ran)
# A planted fault (rt/sonic_rt --plant P) must fail at least one check.     usage: python3 add_thresholds.py pred.json
import json, sys
P = json.load(open(sys.argv[1])); s, a = P['f16store'], P['f16acc']
P['thresholds'] = dict(npf_agree_min=max(0.80, min(s['npf_agree'], a['npf_agree']) - 0.05),
                       pose_code_agree_min=a['pose_code_agree'] - 0.02,
                       root_pos_err_m_max=min(2 * a['root_pos_err_m']['max'], 0.10),
                       joint_err_rad_p99_max=2 * a['joint_err_rad']['p99'],
                       fp16_signature_joint_p99_min=0.1 * s['joint_err_rad']['p99'])
json.dump(P, open(sys.argv[1], 'w'), indent=1); print(json.dumps(P['thresholds'], indent=1))
