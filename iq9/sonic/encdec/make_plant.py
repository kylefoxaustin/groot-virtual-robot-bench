"""usage: python3 make_plant.py <refset.npz> <out dir>
Planted-fault inputs: the board is fed WRONG inputs; the gate (vs the correct-input reference) MUST FAIL.
 plant A: dec_in[:, 64:] reversed column order (wrong observation layout)
 plant B: enc_in[:, 1:] reversed column order (mode column 0 kept; in the core path mode-select is host-side so a mode-column plant would be a no-op on the NPU)"""
import sys, numpy as np
z = np.load(sys.argv[1]); out = sys.argv[2]
e = z["enc_in"].astype(np.float32).copy(); d = z["dec_in"].astype(np.float32).copy()
d[:, 64:] = d[:, 64:][:, ::-1]
e[:, 1:] = e[:, 1:][:, ::-1]
np.ascontiguousarray(e).tofile(f"{out}/enc_in.f32"); np.ascontiguousarray(d).tofile(f"{out}/dec_in.f32")
print("planted inputs written", e.shape, d.shape)
