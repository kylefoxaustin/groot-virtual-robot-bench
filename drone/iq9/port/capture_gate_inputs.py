#!/usr/bin/env python3
"""(GPU host) Write the drone gate's 256 frames as IQ-9075 board inputs: the exact eagle tensors the NVIDIA path feeds its
backbone, so the board gate isolates the NPU (GR00T's transforms are not part of the gate; they are in the full-cost run).

For each frame of drone/assets/gate_set.npz (224 px image + instruction) this calls drone/policy.py unchanged:
  DronePolicy.transform(img, instr)   GR00T's own input transforms (resize to 256, Eagle processor, tokenizer)
  DronePolicy.vl_inputs(nrm)          to_dev (floats -> the action head's dtype, bf16, as GR00TN15.prepare_input does)
                                      + right-pad to 296 tokens with pad id 151643, mask 0
and writes <out>/NNNN/{input_ids.raw (int64 x 296), attention_mask.raw (float32 x 296, 1 = valid, valid tokens first),
pixel_values.raw (float32 1x3x224x224, the bf16 values upcast exactly)} plus <out>/manifest.json.
The orchestrator's BFRAME reads these three files; the GPU path consumes the same tensors (feature_from_vl).

  python3 drone/iq9/port/capture_gate_inputs.py [--out $DRONE_IQ9_WORK/gate_frames]
env: setup/env.sh + drone/env.sh (GROOT_REPO, GROOT_MODEL, GROOT_STEP_NPZ, DRONE_ASSETS). Loads GR00T eagerly on the GPU
(backend "eager": no TensorRT engines needed); nothing is timed.
"""
import os, sys, json, argparse, hashlib
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); DRONE = os.path.abspath(os.path.join(HERE, "..", ".."))
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", default=None, help="default $DRONE_IQ9_WORK/gate_frames")
ap.add_argument("--assets", default=None, help="dir with gate_set.npz (default $DRONE_ASSETS)")
a = ap.parse_args(); sys.argv = [sys.argv[0]]
out = a.out or os.path.join(os.environ.get("DRONE_IQ9_WORK") or sys.exit("set --out or source drone/iq9/env.sh"), "gate_frames")
assets = a.assets or os.environ.get("DRONE_ASSETS", os.path.join(DRONE, "assets"))
sys.path.insert(0, DRONE)
import numpy as np
import torch
import policy as P

gs = os.path.join(assets, "gate_set.npz")
z = np.load(gs)
imgs, ins = z["images"], [str(x) for x in z["instructions"]]
pol = P.DronePolicy("eager", head=os.path.join(assets, "drone_head.npz"))
IMAGE_TOKEN = int(pol.m.backbone.eagle_model.image_token_index)
os.makedirs(out, exist_ok=True)
nvalid = []
with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):   # as DronePolicy.decide
    for j in range(len(imgs)):
        nrm = pol.transform(imgs[j], ins[j])
        am0 = np.asarray(nrm["eagle_attention_mask"]).reshape(-1)
        assert am0.all(), f"frame {j}: unexpected padding in the transformed input"
        vl, n = pol.vl_inputs(nrm)
        ids = vl["eagle_input_ids"].cpu().numpy().reshape(-1).astype(np.int64)
        am = vl["eagle_attention_mask"].cpu().numpy().reshape(-1).astype(np.float32)
        pv = vl["eagle_pixel_values"].float().cpu().numpy()
        assert ids.shape == (P.NPAD,) and am.shape == (P.NPAD,) and pv.shape == (1, 3, 224, 224), (ids.shape, am.shape, pv.shape)
        assert am[:n].all() and not am[n:].any() and (ids[n:] == P.PAD_ID).all()
        assert int((ids == IMAGE_TOKEN).sum()) == 256, "expected 256 image tokens"
        fd = os.path.join(out, f"{j:04d}"); os.makedirs(fd, exist_ok=True)
        ids.tofile(f"{fd}/input_ids.raw"); am.tofile(f"{fd}/attention_mask.raw"); pv.astype(np.float32).tofile(f"{fd}/pixel_values.raw")
        nvalid.append(int(n))
        if j % 64 == 0: print(f"{j}/{len(imgs)} n_valid={n}", flush=True)
man = {"n": len(imgs), "npad": P.NPAD, "pad_id": P.PAD_ID, "image_token": IMAGE_TOKEN, "nvalid_unique": sorted(set(nvalid)),
       "gate_set_sha256": hashlib.sha256(open(gs, "rb").read()).hexdigest(), "torch": torch.__version__,
       "gpu": torch.cuda.get_device_name(0), "pixel_dtype_before_upcast": str(pol.m.action_head.dtype)}
json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
print(json.dumps(man, indent=1))
if man["nvalid_unique"] != [P.NV]:
    print(f"WARNING: valid lengths {man['nvalid_unique']} != [{P.NV}]: the 289-token vlln build will reject other lengths")
