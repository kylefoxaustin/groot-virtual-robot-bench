#!/usr/bin/env python3
"""Is your renderer bit-identical to ours? Renders the 40 evaluation episodes' start views plus the 256 gate-frame views
and prints one sha256 over all of them. Ours (x86, Mesa llvmpipe 'LLVM 15.0.7, 256 bits', MuJoCo 3.8.1):
    EXPECTED below. Same hash => your closed-loop episodes can match ours bit for bit (given the same policy outputs).
A different hash does not make your runs wrong; it means trajectories can drift apart by tiny amounts, so compare
reach (and same-outcome counts), not exact decision counts.
  MUJOCO_GL=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json python3 drone/sim/render_check.py
Needs numpy, mujoco, PyOpenGL (no GPU)."""
import hashlib, os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drone_sim as dc

EXPECTED = "319e12aa7723cc6fffbc272801f76f62ddfeef25234e5c64eec58c253018fea7"
sim = dc.Sim(); rend = dc.renderer_name()
z = np.load(os.path.join(os.environ.get("DRONE_ASSETS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")), "gate_set.npz"))
h = hashlib.sha256(); diffs = []
for e in range(40):
    _, _, pos, yaw, _ = dc.episode_start(e); sim.set(pos, yaw); h.update(sim.img().tobytes())
for s, im in zip(z["states"], z["images"]):
    sim.set(s[:3].astype(np.float64), float(s[3])); r = sim.img(); h.update(r.tobytes())
    diffs.append(np.abs(r.astype(int) - im.astype(int)).mean())
d = h.hexdigest()
print(f"renderer: {rend}\nmujoco {dc.mujoco.__version__}\nrender sha256: {d}\n"
      f"{'IDENTICAL to ours' if d == EXPECTED else 'differs from ours (' + EXPECTED[:16] + '...)'}\n"
      f"gate frames re-rendered vs the shipped NVIDIA-EGL renders: mean |diff| {np.mean(diffs):.4f} LSB (ours with llvmpipe: about 0.15)")
sim.close()
