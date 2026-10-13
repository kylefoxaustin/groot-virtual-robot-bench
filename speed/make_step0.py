#!/usr/bin/env python3
"""Write the speed benchmarks' single input step: NVIDIA demo_data/robot_sim.PickNPlace ds[0]
(a recorded, simulated GR1 observation -- camera frame + proprio state + instruction -- not synthetic data;
data config fourier_gr1_arms_only).
Run once on the x86 host (needs decord for the video), then copy the .npz to the Jetsons and
point GROOT_STEP_NPZ at it, so every board times the IDENTICAL input. The printed content hash
(sha256 over sorted keys + raw array bytes) of the published input is
  3fa53621464390cd327ef6aaecb1fa2c7d9b2b62bd01e5d0d7db13c882a71eb5
usage: $GROOT_PY speed/make_step0.py OUT_NPZ   (env: GROOT_REPO, GROOT_SHIM optional)"""
import os, sys, hashlib
import numpy as np
out = sys.argv[1] if len(sys.argv) > 1 else sys.exit(__doc__)
repo = os.environ.get("GROOT_REPO") or sys.exit("GROOT_REPO is not set -- source setup/env.sh")
sys.path.insert(0, repo)
if os.environ.get("GROOT_SHIM"): sys.path.insert(0, os.environ["GROOT_SHIM"])
from gr00t.experiment.data_config import load_data_config
from gr00t.data.dataset import LeRobotSingleDataset
dc = load_data_config("fourier_gr1_arms_only")
ds = LeRobotSingleDataset(dataset_path=os.path.join(repo, "demo_data/robot_sim.PickNPlace"),
                          modality_configs=dc.modality_config(), video_backend="decord",
                          video_backend_kwargs=None, transforms=None, embodiment_tag="gr1")
step = ds[0]
np.savez(out, **{k: np.asarray(v) for k, v in step.items()})
h = hashlib.sha256()
for k in sorted(step):
    a = np.asarray(step[k]); h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(np.ascontiguousarray(a).tobytes())
print("wrote", out, "keys:", sorted(step), "content sha256:", h.hexdigest())
