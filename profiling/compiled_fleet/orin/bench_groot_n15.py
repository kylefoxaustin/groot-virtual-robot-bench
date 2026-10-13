#!/usr/bin/env python3
"""Time GR00T N1.5 exactly as NVIDIA's deployment_scripts/gr00t_inference.py loads it.

Their script prints result SHAPES, not timings — the published Orin chart (220 ms
PyTorch / 93 ms TensorRT = 4.5 / 10.75 Hz) came from separate profiling. This replicates
their load path verbatim (same Gr00tPolicy, same data config, same LeRobotSingleDataset,
same demo_data/robot_sim.PickNPlace episode) and times policy.get_action().

Input: a recorded (simulated) GR1 PickNPlace episode from NVIDIA's own demo dataset
(demo_data/robot_sim.PickNPlace, the one their deployment guide uses) -- recorded data, not synthetic.

Both the replan rate and the action rate are reported (action_horizon 16, 4 denoise steps), so the
two are never confused.

--power: optional Jetson rail sampling through an external power-sampler module (not included in
this repo): set POWER_SAMPLER_DIR to the directory holding a module (POWER_SAMPLER_MODULE, default
"rail_sampler") that exposes RailSampler(paths) with .start(), .stop_flag, .join(), .window(t0, t1).
"""
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_W = _gvrb_os.path.expanduser(_gvrb_os.environ.get("GVRB_WORK", "~/gvrb"))
import argparse, json, os, statistics, sys, time
import numpy as np, torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=_W+"/GR00T-N1.5-3B")
    ap.add_argument("--dataset-path", default=None)
    ap.add_argument("--data-config", default="fourier_gr1_arms_only")
    ap.add_argument("--embodiment-tag", default="gr1")
    ap.add_argument("--denoising-steps", type=int, default=4)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None)
    ap.add_argument("--power", default="")
    a = ap.parse_args()

    import gr00t
    from gr00t.experiment.data_config import load_data_config
    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.model.policy import Gr00tPolicy

    REPO = os.path.dirname(os.path.dirname(gr00t.__file__))
    ds_path = a.dataset_path or os.path.join(REPO, "demo_data/robot_sim.PickNPlace")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dc = load_data_config(a.data_config)
    policy = Gr00tPolicy(model_path=a.model_path, embodiment_tag=a.embodiment_tag,
                         modality_config=dc.modality_config(),
                         modality_transform=dc.transform(),
                         denoising_steps=a.denoising_steps, device=device)
    ds = LeRobotSingleDataset(dataset_path=ds_path, modality_configs=policy.modality_config,
                              video_backend="decord", video_backend_kwargs=None,
                              transforms=None, embodiment_tag=a.embodiment_tag)
    step = ds[0]

    sampler = None
    if a.power:
        sys.path.insert(0, os.environ.get("POWER_SAMPLER_DIR", "."))
        RailSampler = __import__(os.environ.get("POWER_SAMPLER_MODULE", "rail_sampler")).RailSampler
        sampler = RailSampler([x.strip() for x in a.power.split(",") if x.strip()])
        if sampler.paths: sampler.start()
        else: sampler = None

    def one():
        if device == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        act = policy.get_action(step)
        if device == "cuda": torch.cuda.synchronize()
        return time.perf_counter() - t0, act

    for _ in range(a.warmup): one()
    ts, act = [], None
    for _ in range(a.reps):
        d, act = one(); ts.append(d)
    if sampler:
        sampler.stop_flag = True; sampler.join(timeout=2)

    med = statistics.median(ts) * 1000.0
    shapes = {k: list(np.asarray(v).shape) for k, v in act.items()}
    horizon = max((s[0] for s in shapes.values() if s), default=None)
    out = {"label": a.label, "model": a.model_path, "device": device,
           "scope": "TRUE END-TO-END via policy.get_action() — NVIDIA's own load path, "
                    "their demo_data/robot_sim.PickNPlace recorded episode. Includes transforms, "
                    "VLM backbone, and the flow-matching action head; the video frames are decoded "
                    "once by the dataset loader, outside the timed call.",
           "input_provenance": "a recorded (simulated) GR1 PickNPlace observation from NVIDIA's "
                               "demo_data/robot_sim.PickNPlace (the dataset their own deployment "
                               "guide uses), not synthetic data.",
           "denoising_steps": a.denoising_steps,
           "e2e_ms_median": round(med, 3), "e2e_ms_min": round(min(ts)*1000, 3),
           "e2e_ms_max": round(max(ts)*1000, 3),
           "replan_hz": round(1000.0/med, 2),
           "action_horizon_from_output": horizon,
           "action_hz": round(1000.0/med*horizon, 1) if horizon else None,
           "output_shapes": shapes, "reps": a.reps, "warmup": a.warmup,
           "torch_version": torch.__version__,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if sampler: out["power_mW"] = sampler.window(0, 1e18)
    js = json.dumps(out, indent=1); print(js)
    if a.out: open(a.out, "w").write(js)

if __name__ == "__main__":
    main()
