"""CPU check that the split is exact: sonic_enc_core + host Round/16 + mode select == the full upstream encoder (onnxruntime fp32,
random inputs). usage: python3 check_core_equiv.py <dir containing onnx/ (make_graphs.py output)>"""
import sys, os, numpy as np, onnxruntime as ort
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from sonic_host import fsq_select
D = sys.argv[1] + "/onnx"
full = ort.InferenceSession(f"{D}/sonic_enc_full.onnx", providers=["CPUExecutionProvider"])
core = ort.InferenceSession(f"{D}/sonic_enc_core.onnx", providers=["CPUExecutionProvider"])
rng = np.random.default_rng(0)
N = 300; obs = rng.standard_normal((N, 1762)).astype(np.float32)
obs[:, 0] = np.arange(N) % 3
f = np.concatenate([full.run(None, {"obs_dict": obs[i:i+1]})[0] for i in range(N)])
p = np.stack([core.run(None, {"obs_dict": obs[i:i+1]})[0] for i in range(N)])
c = fsq_select(p, obs)
print("SYNTHETIC equivalence core+host vs full ORT: max|diff|", np.abs(f - c).max(), "exact", np.array_equal(f, c))
print("token levels sample", np.unique(f)[:10], "n unique", len(np.unique(f)))
