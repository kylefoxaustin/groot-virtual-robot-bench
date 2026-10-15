"""Verify-the-output for the LATENCY loop: actions produced inside the timed loop (NPU tokens -> host
Round/select -> NPU decoder) vs host fp32 reference ref_act[i] (decoder on logged dec_in).
Compare to the predictor's dec_coupled arms. Usage: loop_check.py <pred.json> <lat_dir>"""
import sys, json, numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0]); from fp16_predict import dec_metrics
P = json.load(open(sys.argv[1])); R = np.load(sys.argv[1].replace(".json", "_ref.npz"))
d = sys.argv[2]; lat = np.genfromtxt(f"{d}/lat.csv", delimiter=",", names=True)
acts = np.fromfile(f"{d}/lat_actions.f32", np.float32).reshape(-1, 29)
idx = lat["i"].astype(int); m = idx < P["N"]
out = {"loop_vs_fp32": dec_metrics(acts[m], R["ref_act"][idx[m]]), "n": int(m.sum()),
       "pred_dec_coupled_f16store": P["arms"]["f16store"].get("dec_coupled"),
       "pred_dec_coupled_f16acc": P["arms"]["f16acc"].get("dec_coupled")}
json.dump(out, open(f"{d}/loop_check.json", "w"), indent=1); print(json.dumps(out, indent=1))
