"""GR00T N1.5 denoise-steps sweep (run on an RTX 5090) — NVIDIA's FULL-PIPELINE TensorRT path (all 7 engines,
deployment_scripts/trt_model_forward), replicating the eager sweep (speed/denoise_sweep_eager.py) protocol:
  latency  : policy.get_action() on PickNPlace ds[0], per step count 10 warmup + 40 timed, synced, median;
             TWO passes over the grid (ascending, then descending) so drift cancels; same-session EAGER control
             at 1/4/16 steps before and after.
  deviation: rel-MAE of the 26-DOF x 16-step action vs a 64-step reference ON THE SAME PATH, matched noise
             (the eager torch.randn draw for each seed, captured, fed to NVIDIA's path via ah.init_actions),
             5 episodes [0,20,40,60,80] x 3 seeds [0,1,2]. Same-session eager deviation computed too (cross-check of the eager sweep).
  gate     : full-TRT vs eager actions per key at 4 and 64 steps, same noise; eager-vs-eager control.
  stages   : instrumented pass (cuda-synced timers around apply_transforms / backbone / action head / each engine)
             at every grid point -> host+glue vs steps regression.
usage: $GROOT_PY speed/denoise_sweep_fulltrt.py ENGINE_DIR OUT_JSON   (env: GROOT_REPO, GROOT_MODEL, GROOT_SHIM)"""
import os, sys, time, json, statistics, threading, copy
def _env(k):
    v = os.environ.get(k)
    if not v: sys.exit(f"{k} is not set -- run: source setup/env.sh (see docs/SETUP.md)")
    return v
from functools import partial
REPO = _env("GROOT_REPO")
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "deployment_scripts"))
sys.path.insert(0, _env("GROOT_SHIM"))
ENG, OUT = sys.argv[1], sys.argv[2]; sys.argv = [sys.argv[0]]
import numpy as np, torch, tensorrt as trt, pynvml
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy
from gr00t.data.dataset import LeRobotSingleDataset
import trt_torch
from trt_model_forward import eagle_tensorrt_forward, action_head_tensorrt_forward

STEPS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]; REF = 64; REPS, WARM = 40, 10
EPS = [0, 20, 40, 60, 80]; SEEDS = [0, 1, 2]; EAGER_CTRL = [1, 4, 16]

# ---------------- clock / utilisation sampler (no sudo: clocks are NOT pinned; we record what they were) -------------
pynvml.nvmlInit(); H = pynvml.nvmlDeviceGetHandleByIndex(0)
SAMP = {"on": False, "sm": [], "mem": [], "util": []}
def _sampler():
    while True:
        if SAMP["on"]:
            SAMP["sm"].append(pynvml.nvmlDeviceGetClockInfo(H, pynvml.NVML_CLOCK_SM))
            SAMP["mem"].append(pynvml.nvmlDeviceGetClockInfo(H, pynvml.NVML_CLOCK_MEM))
            SAMP["util"].append(pynvml.nvmlDeviceGetUtilizationRates(H).gpu)
        time.sleep(0.25)
threading.Thread(target=_sampler, daemon=True).start()

print("loading policy...", flush=True)
dc = load_data_config("fourier_gr1_arms_only")
policy = Gr00tPolicy(model_path=_env("GROOT_MODEL"), embodiment_tag="gr1", modality_config=dc.modality_config(),
                     modality_transform=dc.transform(), denoising_steps=REF, device="cuda")
ds = LeRobotSingleDataset(dataset_path=os.path.join(REPO, "demo_data/robot_sim.PickNPlace"), modality_configs=policy.modality_config,
                          video_backend="decord", video_backend_kwargs=None, transforms=None, embodiment_tag="gr1")
samples = {e: ds[e] for e in EPS}; step0 = samples[0]
m = policy.model; bb = m.backbone; ah = m.action_head
E = {t: trt_torch.Engine(f"{ENG}/{t}.engine") for t in
     ["vit_fp16", "llm_fp16", "DiT_fp16", "vlln_vl_self_attention", "state_encoder", "action_encoder", "action_decoder"]}
vm = bb.eagle_model.vision_model
bb.num_patches = vm.vision_model.embeddings.num_patches
bb.embedding_layer = bb.eagle_model.language_model.get_input_embeddings()
bb.image_token_index = bb.eagle_model.image_token_index
bb.vit_engine, bb.llm_engine = E["vit_fp16"], E["llm_fp16"]
ah.vlln_vl_self_attention_engine = E["vlln_vl_self_attention"]; ah.state_encoder_engine = E["state_encoder"]
ah.action_encoder_engine = E["action_encoder"]; ah.DiT_engine = E["DiT_fp16"]; ah.action_decoder_engine = E["action_decoder"]
ORIG = dict(bb=bb.forward, ah=ah.get_action)

def mode(name, noise=None):
    bb.forward, ah.get_action = ORIG["bb"], ORIG["ah"]
    if hasattr(ah, "init_actions"): del ah.init_actions
    if name == "full":
        bb.forward = partial(eagle_tensorrt_forward, bb); ah.get_action = partial(action_head_tensorrt_forward, ah)
        if noise is not None: ah.init_actions = noise.to(torch.float16)
def set_steps(n): ah.num_inference_timesteps = n
def flat(a): return np.concatenate([np.asarray(a[k]).reshape(-1) for k in sorted(a.keys())])

# ---------------- capture the eager noise draw per seed (it does not depend on the episode: probe-verified) ---------
_cap = {}; _orandn = torch.randn
def _randn(*a, **k):
    r = _orandn(*a, **k)
    if r.dim() == 3 and tuple(r.shape[1:]) == (ah.config.action_horizon, ah.config.action_dim): _cap["n"] = r.detach().clone()
    return r
mode("eager"); set_steps(1); NOISE = {}
torch.randn = _randn
for s in SEEDS:
    _cap.clear(); torch.manual_seed(s); np.random.seed(s); policy.get_action(step0); NOISE[s] = _cap["n"]
torch.randn = _orandn
print("noise captured per seed", {s: (tuple(v.shape), str(v.dtype)) for s, v in NOISE.items()}, flush=True)

BLOCKLOG = []
def _others():
    try:
        me = os.getpid(); o = []
        for p in pynvml.nvmlDeviceGetProcessUtilization(H, int((time.time() - 2) * 1e6)):
            if p.pid != me and (p.smUtil or 0) > 0: o.append((p.pid, p.smUtil))
        return o
    except Exception as ex: return str(ex)[:60]
def timed_block(name, n):
    la0 = os.getloadavg()[0]
    mode(name); set_steps(n)
    for _ in range(WARM): policy.get_action(step0)
    ts = []
    for _ in range(REPS):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        policy.get_action(step0)
        torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1000)
    BLOCKLOG.append({"mode": name, "steps": n, "median_ms": round(statistics.median(ts), 3), "loadavg1_start": la0,
                     "loadavg1_end": os.getloadavg()[0], "other_gpu_procs_sm_util": _others(), "t": round(time.time(), 1)})
    return ts

# ---------------- LATENCY ----------------
mode("full"); set_steps(4)
for _ in range(WARM): policy.get_action(step0)          # global warmup (as in the eager sweep)
lat = {"full": {n: [] for n in STEPS}, "eager": {n: [] for n in EAGER_CTRL}}
blocks = []
SAMP["on"] = True
for n in EAGER_CTRL: lat["eager"][n] += timed_block("eager", n); blocks.append(("eager", n))
for order in (STEPS, list(reversed(STEPS)), STEPS):
    for n in order:
        ts = timed_block("full", n); lat["full"][n] += ts; blocks.append(("full", n, round(statistics.median(ts), 3)))
        print(f"LAT full steps={n:2d} median {statistics.median(ts):7.3f} ms", flush=True)
for n in EAGER_CTRL: lat["eager"][n] += timed_block("eager", n); blocks.append(("eager", n))
SAMP["on"] = False
clk = {"sm_mhz_median": statistics.median(SAMP["sm"]), "sm_mhz_min": min(SAMP["sm"]), "sm_mhz_max": max(SAMP["sm"]),
       "mem_mhz_median": statistics.median(SAMP["mem"]), "util_pct_median": statistics.median(SAMP["util"]), "n_samples": len(SAMP["sm"])}
print("clocks during latency blocks", clk, flush=True)

def summ(ts):
    return {"median_ms": round(statistics.median(ts), 3), "p10_ms": round(float(np.percentile(ts, 10)), 3),
            "p90_ms": round(float(np.percentile(ts, 90)), 3), "min_ms": round(min(ts), 3), "reps": len(ts)}
pass_medians = {n: [round(statistics.median(lat["full"][n][i * REPS:(i + 1) * REPS]), 3) for i in range(3)] for n in STEPS}
# pre-registered estimator (2026-10-12, after a reproducible ~25% host transient hit one 1-step block in both builds):
# per step count, the MEDIAN OF THE THREE PASS MEDIANS (robust to one contaminated pass); raw + every pass kept.
lat_full = {n: summ(v) for n, v in lat["full"].items()}
for n in STEPS:
    lat_full[n]["all_reps_median_ms"] = lat_full[n]["median_ms"]; lat_full[n]["median_ms"] = round(statistics.median(pass_medians[n]), 3)
    lat_full[n]["estimator"] = "median of 3 pass medians"
lat_eager = {n: summ(v) for n, v in lat["eager"].items()}
def fit(xs, ys):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float); A = np.vstack([xs, np.ones_like(xs)]).T
    b, a = np.linalg.lstsq(A, ys, rcond=None)[0]; r2 = 1 - np.sum((ys - (b * xs + a)) ** 2) / np.sum((ys - ys.mean()) ** 2)
    return float(a), float(b), float(r2)
a_f, b_f, r2_f = fit(STEPS, [lat_full[n]["median_ms"] for n in STEPS])
a_e, b_e, r2_e = fit(EAGER_CTRL, [lat_eager[n]["median_ms"] for n in EAGER_CTRL])
print(f"FIT full: {a_f:.3f} + {b_f:.4f} x steps (r2 {r2_f:.5f}) | eager ctrl: {a_e:.3f} + {b_e:.4f} x steps (r2 {r2_e:.5f})", flush=True)

# ---------------- DEVIATION (full TRT, and same-session eager) ----------------
def action(path, e, n, s):
    mode(path, NOISE[s] if path == "full" else None); set_steps(n)
    torch.manual_seed(s); np.random.seed(s)
    return policy.get_action(samples[e])
ACT = {}
for path in ("full", "eager"):
    for n in STEPS + [REF]:
        for e in EPS:
            for s in SEEDS:
                ACT[(path, n, e, s)] = action(path, e, n, s)
    print(f"actions done: {path}", flush=True)
ctrl = {(e, s): action("eager", e, 4, s) for e in EPS for s in SEEDS}   # eager-vs-eager same-seed control
def devtab(path):
    out = {}
    for n in STEPS:
        vals = []
        for e in EPS:
            for s in SEEDS:
                v = flat(ACT[(path, n, e, s)]); r = flat(ACT[(path, REF, e, s)])
                vals.append(float(np.mean(np.abs(v - r)) / (np.mean(np.abs(r)) + 1e-8)))
        out[n] = {"mean_pct": round(100 * float(np.mean(vals)), 4), "std_pct": round(100 * float(np.std(vals)), 4),
                  "min_pct": round(100 * float(np.min(vals)), 4), "max_pct": round(100 * float(np.max(vals)), 4), "n_samples": len(vals)}
    return out
dev_full, dev_eager = devtab("full"), devtab("eager")
C_full = float(np.mean([dev_full[n]["mean_pct"] * n for n in STEPS])); C_eager = float(np.mean([dev_eager[n]["mean_pct"] * n for n in STEPS]))
for n in STEPS: print(f"DEV steps={n:2d} full {dev_full[n]['mean_pct']:7.3f}±{dev_full[n]['std_pct']:.3f}%  eager {dev_eager[n]['mean_pct']:7.3f}±{dev_eager[n]['std_pct']:.3f}%", flush=True)

def gate_keys(pairs):
    keys = sorted(next(iter(pairs))[0].keys()); o = {}
    for k in keys:
        ra, aa, ref = [], [], []
        for x, y in pairs:
            a = np.asarray(x[k], np.float64); b = np.asarray(y[k], np.float64)
            aa.append(np.abs(b - a).mean()); ref.append(np.abs(a).mean()); ra.append(np.abs(b - a).mean() / max(np.abs(a).mean(), 1e-9))
        o[k] = {"rel_mae": round(float(np.mean(ra)), 5), "abs_mae": round(float(np.mean(aa)), 5), "ref_mean_abs": round(float(np.mean(ref)), 5)}
    return o
gate = {"full_vs_eager_4step": gate_keys([(ACT[("eager", 4, e, s)], ACT[("full", 4, e, s)]) for e in EPS for s in SEEDS]),
        "full_vs_eager_64step_ref": gate_keys([(ACT[("eager", REF, e, s)], ACT[("full", REF, e, s)]) for e in EPS for s in SEEDS]),
        "control_eager_vs_eager_4step": gate_keys([(ACT[("eager", 4, e, s)], ctrl[(e, s)]) for e in EPS for s in SEEDS])}
for g, v in gate.items(): print("GATE", g, json.dumps({k: (x["rel_mae"], x["abs_mae"]) for k, x in v.items()}), flush=True)

# ---------------- STAGES (instrumented pass, full TRT, every grid point) ----------------
T = {}
def wrap(obj, attr, key):
    f = getattr(obj, attr)
    def w(*a, **k):
        torch.cuda.synchronize(); t0 = time.perf_counter(); r = f(*a, **k); torch.cuda.synchronize()
        T[key] = T.get(key, 0.0) + time.perf_counter() - t0; return r
    setattr(obj, attr, w); return f
_ofwd = trt_torch.Engine.forward
def _tfwd(self, *a, **k):
    torch.cuda.synchronize(); t0 = time.perf_counter(); r = _ofwd(self, *a, **k); torch.cuda.synchronize()
    key = "eng:" + os.path.basename(self.file).replace(".engine", ""); T[key] = T.get(key, 0.0) + time.perf_counter() - t0; return r
stages = {}
for n in STEPS:
    mode("full"); set_steps(n)
    o_tr = wrap(policy, "apply_transforms", "transform"); o_bb = wrap(bb, "forward", "backbone"); o_ah = wrap(ah, "get_action", "head")
    trt_torch.Engine.forward = _tfwd
    for _ in range(5): policy.get_action(step0)
    T.clear(); e2e = []
    for _ in range(25):
        torch.cuda.synchronize(); t0 = time.perf_counter(); policy.get_action(step0); torch.cuda.synchronize(); e2e.append((time.perf_counter() - t0) * 1000)
    trt_torch.Engine.forward = _ofwd; policy.apply_transforms = o_tr; bb.forward = o_bb; ah.get_action = o_ah
    per = {k: v * 1000 / 25 for k, v in T.items()}
    eng = {k[4:]: round(v, 4) for k, v in per.items() if k.startswith("eng:")}
    eng_sum = sum(eng.values()); e2e_mean = float(np.mean(e2e))
    stages[n] = {"e2e_instrumented_mean_ms": round(e2e_mean, 3), "e2e_instrumented_median_ms": round(statistics.median(e2e), 3),
                 "transform_ms": round(per.get("transform", 0), 3), "backbone_ms": round(per.get("backbone", 0), 3), "head_ms": round(per.get("head", 0), 3),
                 "engines_ms": eng, "engines_sum_ms": round(eng_sum, 3),
                 "host_and_glue_ms": round(e2e_mean - eng_sum, 3),
                 "backbone_torch_glue_ms": round(per.get("backbone", 0) - eng.get("vit_fp16", 0) - eng.get("llm_fp16", 0), 3),
                 "head_torch_glue_ms": round(per.get("head", 0) - sum(v for k, v in eng.items() if k not in ("vit_fp16", "llm_fp16")), 3),
                 "outside_model_ms": round(e2e_mean - per.get("transform", 0) - per.get("backbone", 0) - per.get("head", 0), 3)}
    print(f"STAGE steps={n:2d}", json.dumps(stages[n]), flush=True)
ag, bg, r2g = fit(STEPS, [stages[n]["host_and_glue_ms"] for n in STEPS])
ae, be, r2e = fit(STEPS, [stages[n]["engines_sum_ms"] for n in STEPS])

def _ser(d): return {str(k): v for k, v in d.items()}
json.dump({"trt": trt.__version__, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "engine_dir": ENG,
           "steps": STEPS, "ref_steps": REF, "episodes": EPS, "seeds": SEEDS, "reps": REPS, "warmup": WARM,
           "clocks_during_latency": clk, "latency_full": _ser(lat_full), "latency_full_pass_medians": _ser(pass_medians),
           "latency_eager_ctrl": _ser(lat_eager), "block_log": BLOCKLOG, "latency_full_raw": _ser(lat["full"]), "latency_eager_raw": _ser(lat["eager"]),
           "fit_full": {"intercept_ms": a_f, "per_step_ms": b_f, "r2": r2_f}, "fit_eager_ctrl": {"intercept_ms": a_e, "per_step_ms": b_e, "r2": r2_e},
           "deviation_full": _ser(dev_full), "deviation_eager_same_session": _ser(dev_eager), "C_full": C_full, "C_eager": C_eager,
           "gate": gate, "stages_full": _ser(stages),
           "host_glue_vs_steps_fit": {"intercept_ms": ag, "per_step_ms": bg, "r2": r2g},
           "engines_vs_steps_fit": {"intercept_ms": ae, "per_step_ms": be, "r2": r2e},
           "noise": {str(s): {"dtype": str(v.dtype), "sum": float(v.float().sum())} for s, v in NOISE.items()},
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, open(OUT, "w"), indent=1)
print("DONE", OUT, flush=True)
