#!/usr/bin/env python3
"""Route-planning stand-in for the IQ-9075: the SAME seeded seg model at 1024 (seg_1024.onnx from make_standins.py). Writes
into $DRONE_STANDINS/seg1024/:
  - seg_1024_dilfree.onnx via make_dilfree.py --res 1024 (exact dilated-conv rewrite, as at 512)
  - 3 seeded 1024 check tiles seg1024_tile{0,1,2}.raw (ImageNet-normalised uniform pixels, rng seed 20261015 + 1024; the same
    recipe as the 512 tiles)
  - fp32 references from the CANONICAL seg_1024.onnx (onnxruntime CPU), check1024_ref_fp32.npz, and the dilation-free vs
    canonical fp32 equivalence (asserted, max |diff| < 1e-3), with a sha256 manifest.
usage: python3 drone/iq9/contention/make_seg1024_check.py [--standins DIR]   (default $DRONE_STANDINS)"""
import os, sys, json, hashlib, subprocess, time, argparse
import numpy as np, onnxruntime as ort
ap = argparse.ArgumentParser(); ap.add_argument("--standins", default=None); a = ap.parse_args()
A = a.standins or os.environ.get("DRONE_STANDINS") or sys.exit("set --standins or source drone/env.sh")
O = f"{A}/seg1024"; os.makedirs(O, exist_ok=True)
H = os.path.dirname(os.path.abspath(__file__))
subprocess.run([sys.executable, f"{H}/make_dilfree.py", O, "--res", "1024", "--standins", A], check=True,
               stdout=open(f"{O}/make_dilfree_1024.log", "w"))
SEED = 20261015; rng = np.random.default_rng(SEED + 1024)
mean = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]; std = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
tiles = np.stack([((rng.random((3, 1024, 1024), dtype=np.float32) - mean) / std) for _ in range(3)])[:, None]
so = ort.InferenceSession(f"{A}/seg_1024.onnx", providers=["CPUExecutionProvider"])
sd = ort.InferenceSession(f"{O}/seg_1024_dilfree.onnx", providers=["CPUExecutionProvider"])
ref = np.stack([so.run(None, {"img": t})[0] for t in tiles]); dfo = np.stack([sd.run(None, {"img": t})[0] for t in tiles])
eq = {"max_abs_dilfree_vs_canonical_fp32": float(np.abs(dfo - ref).max()), "ref_absmax": float(np.abs(ref).max()),
      "argmax_agreement_dilfree_vs_canonical": float((dfo.argmax(2) == ref.argmax(2)).mean()),
      "ref_argmax_classes_used": int(len(np.unique(ref.argmax(2)))),
      "ref_top_class_share": float(np.bincount(ref.argmax(2).ravel()).max() / ref.argmax(2).size),
      "input_dependence_max_abs_tile0_vs_tile1": float(np.abs(ref[0] - ref[1]).max())}
assert eq["max_abs_dilfree_vs_canonical_fp32"] < 1e-3, eq
np.savez(f"{O}/check1024_ref_fp32.npz", seg_out=ref)
for i, t in enumerate(tiles): t.astype(np.float32).tofile(f"{O}/seg1024_tile{i}.raw")
man = {"made_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seed_inputs": SEED + 1024,
       "model": "seg_1024.onnx (make_standins.py, seed 20261015)", "equivalence": eq,
       "files": {f: hashlib.sha256(open(f"{O}/{f}", "rb").read()).hexdigest() for f in sorted(os.listdir(O))
                 if f.endswith((".onnx", ".raw", ".npz", ".json")) and f != "MANIFEST.json"}}
json.dump(man, open(f"{O}/MANIFEST.json", "w"), indent=1); print(json.dumps(man, indent=1))
