"""usage: python3 npz_to_f32.py <refset.npz> <out dir>.  npz -> contiguous float32 .f32 files for the board (enc_in.f32 [N,1762], dec_in.f32 [N,994])."""
import sys, numpy as np, hashlib
z = np.load(sys.argv[1]); out = sys.argv[2]
for k in ["enc_in", "dec_in"]:
    a = np.ascontiguousarray(z[k].astype(np.float32)); a.tofile(f"{out}/{k}.f32")
    print(k, a.shape, hashlib.sha256(a.tobytes()).hexdigest()[:16])
