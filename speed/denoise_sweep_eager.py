#!/usr/bin/env python3
"""GR00T N1.5 denoise-steps sweep, PyTorch eager (published run: RTX 5090, clocks locked SM 2700 / MEM 14001).

Protocol:
 - reference at REF_STEPS (default 64), a NEAR-converged endpoint, not 16. Deviation is
   then honest solver-truncation error vs near-converged, and never 0 at 16.
 - deviation measured over MULTIPLE episodes x MULTIPLE seeds -> mean +/- std error band
   (the old n=1 single-draw could not support a 'knee'). Same seed for N-step and the
   reference so the noise draw is matched (pure truncation, not noise variance).
 - latency timed with pinned clocks (set externally), median of many reps.
Reports the 1/N Euler-error decay, not a 'saturation'.
usage: $GROOT_PY speed/denoise_sweep_eager.py --out OUT_JSON   (env: GROOT_MODEL, CLOCKS_NOTE; Isaac-GR00T on PYTHONPATH)
"""
import argparse, json, os, statistics, time
import numpy as np, torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=os.environ.get("GROOT_MODEL", "nvidia/GR00T-N1.5-3B"))
    ap.add_argument("--data-config", default="fourier_gr1_arms_only")
    ap.add_argument("--embodiment-tag", default="gr1")
    ap.add_argument("--steps", default="1,2,3,4,6,8,12,16,24,32")
    ap.add_argument("--ref-steps", type=int, default=64)
    ap.add_argument("--reps", type=int, default=40)      # timing reps (on episode 0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--episodes", default="0,20,40,60,80")  # ds indices for deviation
    ap.add_argument("--seeds", default="0,1,2")               # noise seeds for deviation
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    steps = [int(x) for x in a.steps.split(",")]
    eps = [int(x) for x in a.episodes.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]

    import gr00t
    from gr00t.experiment.data_config import load_data_config
    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.model.policy import Gr00tPolicy

    REPO = os.path.dirname(os.path.dirname(gr00t.__file__))
    ds_path = os.path.join(REPO, "demo_data/robot_sim.PickNPlace")
    dc = load_data_config(a.data_config)
    policy = Gr00tPolicy(model_path=a.model_path, embodiment_tag=a.embodiment_tag,
                         modality_config=dc.modality_config(), modality_transform=dc.transform(),
                         denoising_steps=a.ref_steps, device="cuda")
    ds = LeRobotSingleDataset(dataset_path=ds_path, modality_configs=policy.modality_config,
                              video_backend="decord", video_backend_kwargs=None,
                              transforms=None, embodiment_tag=a.embodiment_tag)
    nds = len(ds)
    eps = [e for e in eps if e < nds] or [0]
    steplist = ds[0]
    def set_steps(n): policy.model.action_head.num_inference_timesteps = n
    def flat(act): return np.concatenate([np.asarray(act[k]).reshape(-1) for k in sorted(act.keys())])
    def action_at(sample, n, seed):
        set_steps(n); torch.manual_seed(seed); np.random.seed(seed)
        return flat(policy.get_action(sample))

    # clocks under load (spot-check)
    try:
        import subprocess
        clk = subprocess.check_output(["nvidia-smi","--query-gpu=clocks.sm,clocks.mem",
                                       "--format=csv,noheader"]).decode().strip()
    except Exception: clk="?"

    # ---- LATENCY (episode 0, pinned clocks, median of many reps) ----
    lat = {}
    for _ in range(a.warmup): policy.get_action(steplist)
    for n in steps:
        set_steps(n)
        for _ in range(a.warmup): policy.get_action(steplist)
        ts = []
        for _ in range(a.reps):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            policy.get_action(steplist)
            torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
        lat[n] = {"median_ms": round(statistics.median(ts)*1000,3),
                  "p10_ms": round(np.percentile(ts,10)*1000,3),
                  "p90_ms": round(np.percentile(ts,90)*1000,3),
                  "min_ms": round(min(ts)*1000,3), "reps": a.reps}
        print("LAT steps=%d median=%.2f (p10 %.2f p90 %.2f)"%(n,lat[n]["median_ms"],lat[n]["p10_ms"],lat[n]["p90_ms"]),flush=True)

    # ---- DEVIATION vs REF_STEPS, over episodes x seeds ----
    samples = {e: ds[e] for e in eps}
    refs = {}   # (e,seed) -> ref vector at ref_steps
    for e in eps:
        for s in seeds:
            refs[(e,s)] = action_at(samples[e], a.ref_steps, s)
    dev = {}
    for n in steps:
        vals = []
        for e in eps:
            for s in seeds:
                vec = action_at(samples[e], n, s)
                ref = refs[(e,s)]
                vals.append(float(np.mean(np.abs(vec-ref))/(np.mean(np.abs(ref))+1e-8)))
        dev[n] = {"mean_pct": round(100*float(np.mean(vals)),4),
                  "std_pct": round(100*float(np.std(vals)),4),
                  "min_pct": round(100*float(np.min(vals)),4),
                  "max_pct": round(100*float(np.max(vals)),4),
                  "n_samples": len(vals)}
        print("DEV steps=%d mean=%.3f%% +/- %.3f (n=%d)"%(n,dev[n]["mean_pct"],dev[n]["std_pct"],dev[n]["n_samples"]),flush=True)

    # linear fit of median latency vs steps
    xs=np.array(steps,float); ys=np.array([lat[n]["median_ms"] for n in steps])
    A=np.vstack([xs,np.ones_like(xs)]).T
    slope,intercept=np.linalg.lstsq(A,ys,rcond=None)[0]
    r2=1-np.sum((ys-(slope*xs+intercept))**2)/np.sum((ys-ys.mean())**2)
    # 1/N model for deviation: dev ~ C/N
    dv=np.array([dev[n]["mean_pct"] for n in steps]); C=float(np.mean(dv*xs))

    out={"tag":"MEASURED","study":"GR00T N1.5 denoise-steps sweep (eager e2e)",
         "model":a.model_path,"device":torch.cuda.get_device_name(0),
         "clocks_under_load_sm_mem":clk,"clocks_pinned":os.environ.get("CLOCKS_NOTE","unknown -- set CLOCKS_NOTE"),
         "data_config":a.data_config,"embodiment":a.embodiment_tag,"dof":26,"action_horizon":16,
         "input_provenance":"recorded (simulated) GR1 observations from NVIDIA demo_data/robot_sim.PickNPlace, not synthetic data: %d episodes %s x %d seeds %s"%(len(eps),eps,len(seeds),seeds),
         "scope":"policy.get_action() full e2e (transforms+ViT+LLM+DiT; the video frames are already decoded by the dataset loader), PyTorch eager, warm",
         "ref_steps":a.ref_steps,"reference_note":"64 steps = near-converged (NOT exact); Euler error ~C/N so 16 steps is itself ~1%% from the infinite-step limit",
         "latency_ms":lat,"deviation_vs_ref_pct":dev,
         "latency_decomposition":{"backbone_intercept_ms":round(float(intercept),3),
             "per_denoise_step_ms":round(float(slope),3),"r2":round(float(r2),5),
             "note":"eager PyTorch; per-step is the DiT flow-matching pass; intercept is denoise-independent SEE+READ+THINK+prep"},
         "deviation_1overN_fit":{"model":"dev_pct ~= C/steps","C":round(C,3),
             "note":"Euler truncation decays ~1/N; there is no saturation knee in the metric — it is a cost/benefit trade vs the linear latency"},
         "torch_version":torch.__version__,"utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
    json.dump(out,open(a.out,"w"),indent=1)
    print("SAVED",a.out,flush=True)
    print("DECOMP intercept=%.2f slope=%.3f r2=%.4f | dev C=%.2f | clocks=%s"%(intercept,slope,r2,C,clk),flush=True)

if __name__=="__main__":
    main()
