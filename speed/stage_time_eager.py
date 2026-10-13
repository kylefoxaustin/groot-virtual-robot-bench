#!/usr/bin/env python3
"""Cross-validate the denoise-sweep decomposition by DIRECTLY timing GR00T N1.5 stages:
transforms (on already-decoded frames) | backbone(ViT+LLM) | action_head(DiT, per denoise-step).
If the sweep intercept ~= transforms+backbone, and the sweep slope ~= action_head/steps, the
linear model is confirmed by an independent method (not a fit artifact).
usage: $GROOT_PY speed/stage_time_eager.py OUT_JSON   (env: GROOT_MODEL; Isaac-GR00T on PYTHONPATH)"""
import os, sys, time, json, statistics, numpy as np, torch, warnings
OUT = sys.argv[1] if len(sys.argv) > 1 else sys.exit(__doc__)
warnings.filterwarnings("ignore")
import gr00t
from gr00t.experiment.data_config import load_data_config
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.model.policy import Gr00tPolicy

REPO = os.path.dirname(os.path.dirname(gr00t.__file__))
dc = load_data_config("fourier_gr1_arms_only")
policy = Gr00tPolicy(model_path=os.environ.get("GROOT_MODEL", "nvidia/GR00T-N1.5-3B"), embodiment_tag="gr1",
                     modality_config=dc.modality_config(), modality_transform=dc.transform(),
                     denoising_steps=4, device="cuda")
ds = LeRobotSingleDataset(dataset_path=os.path.join(REPO,"demo_data/robot_sim.PickNPlace"),
                          modality_configs=policy.modality_config, video_backend="decord",
                          video_backend_kwargs=None, transforms=None, embodiment_tag="gr1")
step = ds[0]
m = policy.model
T = {"transform": [], "backbone": [], "head": []}

orig_bb = m.backbone.forward
orig_head = m.action_head.get_action
def timed_bb(*a, **k):
    torch.cuda.synchronize(); t=time.perf_counter(); o=orig_bb(*a,**k); torch.cuda.synchronize()
    T["backbone"].append(time.perf_counter()-t); return o
def timed_head(*a, **k):
    torch.cuda.synchronize(); t=time.perf_counter(); o=orig_head(*a,**k); torch.cuda.synchronize()
    T["head"].append(time.perf_counter()-t); return o
m.backbone.forward = timed_bb
m.action_head.get_action = timed_head
orig_tf = policy.apply_transforms
def timed_tf(x):
    torch.cuda.synchronize(); t=time.perf_counter(); o=orig_tf(x); torch.cuda.synchronize()
    T["transform"].append(time.perf_counter()-t); return o
policy.apply_transforms = timed_tf

def run_at(nstep, reps=25, warm=6):
    m.action_head.num_inference_timesteps = nstep
    for k in T: T[k].clear()
    for _ in range(warm): policy.get_action(step)
    for k in T: T[k].clear()
    for _ in range(reps): policy.get_action(step)
    return {k: round(statistics.median(v)*1000,3) for k,v in T.items()}

out = {}
for n in [4, 16]:
    out[n] = run_at(n)
    print("steps=%d  transform=%.2f  backbone=%.2f  head=%.2f  (sum=%.2f)ms"
          % (n, out[n]["transform"], out[n]["backbone"], out[n]["head"],
             out[n]["transform"]+out[n]["backbone"]+out[n]["head"]), flush=True)
# per-step from the two points
head4, head16 = out[4]["head"], out[16]["head"]
per_step = (head16-head4)/(16-4)
head_intercept = head4 - 4*per_step
fixed = out[4]["transform"] + out[4]["backbone"]
print(json.dumps({
    "direct_stage_timing_ms": out,
    "action_head_per_denoise_step_ms": round(per_step,3),
    "action_head_fixed_ms": round(head_intercept,3),
    "denoise_independent_fixed_ms (transform+backbone+head_fixed)": round(fixed+head_intercept,3),
    "cross_check": "compare to the eager sweep latency_decomposition (intercept, per_denoise_step_ms)"
}, indent=1), flush=True)
json.dump({"tag":"MEASURED","study":"GR00T N1.5 direct stage timing (eager) cross-validation",
           "device":torch.cuda.get_device_name(0),"stages_ms":out,
           "action_head_per_step_ms":round(per_step,3),"action_head_fixed_ms":round(head_intercept,3),
           "utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())},
          open(OUT,"w"),indent=1)
