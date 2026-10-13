"""GR00T N1.5 MEASURED end-to-end: HYBRID (ViT/LLM/DiT in TRT, rest eager) vs NVIDIA's FULL-PIPELINE TRT path
(deployment_scripts/trt_model_forward: all 7 modules in TRT) — same process, same engines for the 3 shared modules,
same recorded (simulated) PickNPlace step, alternating A/B blocks so session drift cancels.
Protocol mirrors groot_*_hybrid_e2e_2026-09-06: policy.get_action(), 4 denoise steps, torch.manual_seed(1234) per call,
median of 20 after 5 warmup per block, synced. Gate: rel_mae per action key vs eager bf16, same seed.
usage: $GROOT_PY speed/e2e_eager_hybrid_fulltrt.py ENGINE_DIR OUT_JSON   (env: GROOT_REPO, GROOT_MODEL, GROOT_SHIM, GROOT_STEP_NPZ, JETSON=1)"""
import os, sys, time, json, statistics
def _env(k):
    v = os.environ.get(k)
    if not v: sys.exit(f"{k} is not set -- run: source setup/env.sh (see docs/SETUP.md)")
    return v
from functools import partial
REPO = _env("GROOT_REPO")
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "deployment_scripts"))
sys.path.insert(0, _env("GROOT_SHIM"))
ENG, OUT = sys.argv[1], sys.argv[2]; sys.argv = [sys.argv[0]]
import numpy as np, torch, tensorrt as trt
if os.environ.get("JETSON"):   # flash-attn has no aarch64 wheel: force sdpa (implementation swap only; on Jetson)
    import transformers as _tf
    _oc = _tf.AutoConfig.from_pretrained
    def _pc(*a, **k):
        c = _oc(*a, **k)
        for o in [c] + [v for v in vars(c).values() if hasattr(v, "_attn_implementation")]:
            try: o._attn_implementation = "sdpa"
            except Exception: pass
        return c
    _tf.AutoConfig.from_pretrained = _pc
    from transformers.models.siglip.modeling_siglip import SiglipVisionModel as _SVM
    _os = _SVM.__init__
    def _si(self, cfg, *a, **k):
        cfg._attn_implementation = "sdpa"; cfg._attn_implementation_autoset = True
        return _os(self, cfg, *a, **k)
    _SVM.__init__ = _si
import gr00t
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy
import trt_torch
from trt_model_forward import eagle_tensorrt_forward, action_head_tensorrt_forward
from transformers.modeling_outputs import BaseModelOutputWithPooling, BaseModelOutputWithPast

# ---- per-engine in-pipeline timer around NVIDIA's Engine.forward (it already syncs its stream at the end) ----
ENGT = {}
_ofwd = trt_torch.Engine.forward
def _tfwd(self, *a, **k):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    r = _ofwd(self, *a, **k)
    torch.cuda.synchronize(); ENGT[self.file] = ENGT.get(self.file, 0.0) + time.perf_counter() - t0
    return r
trt_torch.Engine.forward = _tfwd

print("loading policy...", flush=True)
dc = load_data_config("fourier_gr1_arms_only")
policy = Gr00tPolicy(model_path=_env("GROOT_MODEL"), embodiment_tag="gr1",
                     modality_config=dc.modality_config(), modality_transform=dc.transform(), denoising_steps=4, device="cuda")
if os.environ.get("GROOT_STEP_NPZ"):
    z = np.load(os.environ["GROOT_STEP_NPZ"], allow_pickle=True); step = {k: z[k] for k in z.files}; INPUT = "groot_step0.npz (PickNPlace ds[0], written by speed/make_step0.py; its content hash must match across boards)"
else:
    from gr00t.data.dataset import LeRobotSingleDataset
    ds = LeRobotSingleDataset(dataset_path=os.path.join(REPO, "demo_data/robot_sim.PickNPlace"), modality_configs=policy.modality_config,
                              video_backend="decord", video_backend_kwargs=None, transforms=None, embodiment_tag="gr1")
    step = ds[0]; INPUT = "demo_data/robot_sim.PickNPlace ds[0] via LeRobotSingleDataset"
m = policy.model; bb = m.backbone; ah = m.action_head
E = {t: trt_torch.Engine(f"{ENG}/{t}.engine") for t in
     ["vit_fp16", "llm_fp16", "DiT_fp16", "vlln_vl_self_attention", "state_encoder", "action_encoder", "action_decoder"]}

# ---- HYBRID patches (TRT engines patched at the ViT / LLM / DiT module seams), using the SAME engine objects ----
vm = bb.eagle_model.vision_model; lm = bb.eagle_model.language_model.model; dit = ah.model
import inspect
_dit_names = list(inspect.signature(dit.forward).parameters)
POS = torch.arange(256, dtype=torch.int64).reshape(1, 256).cuda()
def _run(e, **feeds):
    for n, x in feeds.items(): e.set_runtime_tensor_shape(n, x.shape)
    return e(**feeds)
def vit_fwd(pixel_values=None, *a, **kw):
    o = _run(E["vit_fp16"], pixel_values=pixel_values.to(torch.float16).contiguous(), position_ids=POS)
    return BaseModelOutputWithPooling(last_hidden_state=o["vit_embeds"].to(torch.bfloat16))
def llm_fwd(*a, inputs_embeds=None, attention_mask=None, **kw):
    am = attention_mask if attention_mask is not None else torch.ones(1, inputs_embeds.shape[1], dtype=torch.int64, device="cuda")
    o = _run(E["llm_fp16"], inputs_embeds=inputs_embeds.to(torch.float16).contiguous(), attention_mask=am.to(torch.int64).contiguous())
    h = o["embeddings"].to(torch.bfloat16); sl = getattr(bb, "select_layer", -1)
    return BaseModelOutputWithPast(last_hidden_state=h, hidden_states=tuple(h for _ in range(sl + 1)) if sl >= 0 else (h,))
def dit_fwd(*a, **kw):
    f = dict(zip(_dit_names, a)); f.update(kw)
    sa = f.get("hidden_states", f.get("sa_embs")); vl = f.get("encoder_hidden_states", f.get("vl_embs"))
    ts = f.get("timestep", f.get("timesteps", f.get("timesteps_tensor")))
    if ts.dim() == 0: ts = ts.reshape(1)
    o = _run(E["DiT_fp16"], sa_embs=sa.to(torch.float16).contiguous(), vl_embs=vl.to(torch.float16).contiguous(), timesteps_tensor=ts.to(torch.int64).contiguous())
    return o["output"].to(torch.bfloat16)
ORIG = dict(vm=vm.forward, lm=lm.forward, dit=dit.forward, bb=bb.forward, ah=ah.get_action)

# ---- FULL-PIPELINE: NVIDIA's own forwards (setup_tensorrt_engines minus the module deletion, so we can A/B) ----
bb.num_patches = vm.vision_model.embeddings.num_patches
bb.embedding_layer = bb.eagle_model.language_model.get_input_embeddings()
bb.image_token_index = bb.eagle_model.image_token_index
bb.vit_engine, bb.llm_engine = E["vit_fp16"], E["llm_fp16"]
ah.vlln_vl_self_attention_engine = E["vlln_vl_self_attention"]; ah.state_encoder_engine = E["state_encoder"]
ah.action_encoder_engine = E["action_encoder"]; ah.DiT_engine = E["DiT_fp16"]; ah.action_decoder_engine = E["action_decoder"]

def mode(name):
    vm.forward, lm.forward, dit.forward, bb.forward, ah.get_action = ORIG["vm"], ORIG["lm"], ORIG["dit"], ORIG["bb"], ORIG["ah"]
    if hasattr(ah, "init_actions"): del ah.init_actions
    if name == "hybrid":
        vm.forward, lm.forward, dit.forward = vit_fwd, llm_fwd, dit_fwd
    elif name == "full":
        bb.forward = partial(eagle_tensorrt_forward, bb); ah.get_action = partial(action_head_tensorrt_forward, ah)
        ah.init_actions = NOISE   # same noise draw eager uses under seed 1234 (bf16 -> fp16 is exact)

def run(reps, warm):
    ts, acts = [], []
    for i in range(warm + reps):
        torch.manual_seed(1234); torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad(): a = policy.get_action(step)
        torch.cuda.synchronize(); dt = (time.perf_counter() - t0) * 1000
        if i >= warm: ts.append(dt); acts.append(a)
    return ts, acts

# the eager noise draw: capture the first torch.randn of the action shape inside eager get_action
_cap = {}
_orandn = torch.randn
def _randn(*a, **k):
    r = _orandn(*a, **k)
    if "noise" not in _cap and r.dim() == 3 and r.shape[1:] == (ah.config.action_horizon, ah.config.action_dim): _cap["noise"] = r.detach().clone()
    return r
torch.randn = _randn
mode("eager"); te0, ae0 = run(3, 1)
torch.randn = _orandn
NOISE = _cap["noise"].to(torch.float16)
print("noise captured", tuple(NOISE.shape), _cap["noise"].dtype, flush=True)

res = {"eager": [], "hybrid": [], "full": []}; acts = {}; engms = {"hybrid": [], "full": []}
order = ["eager", "hybrid", "full", "hybrid", "full", "eager"]
for blk in order:
    mode(blk); ENGT.clear()
    ts, a = run(20, 5)
    n_calls = 25
    if blk != "eager":
        engms[blk].append({os.path.basename(k).replace(".engine", ""): v * 1000 / n_calls for k, v in ENGT.items()})
    res[blk].append(ts); acts.setdefault(blk, a)
    print(f"{blk:6s} median {statistics.median(ts):7.2f} ms  min {min(ts):.2f} max {max(ts):.2f}", flush=True)

def rel(x, y):
    o = {}
    for k in x:
        a = np.asarray(x[k], np.float64); b = np.asarray(y[k], np.float64)
        o[k] = {"rel_mae": float(np.abs(b - a).mean() / max(np.abs(a).mean(), 1e-9)), "abs_mae": float(np.abs(b - a).mean()),
                "ref_mean_abs": float(np.abs(a).mean())}
    return o
ref = acts["eager"][0]
gate = {"control_eager_vs_eager": rel(ref, acts["eager"][-1]), "hybrid_vs_eager": rel(ref, acts["hybrid"][0]),
        "full_vs_eager": rel(ref, acts["full"][0]), "full_vs_hybrid": rel(acts["hybrid"][0], acts["full"][0])}
for g, v in gate.items(): print("GATE", g, json.dumps({k: (round(x["rel_mae"], 5), round(x["abs_mae"], 5)) for k, x in v.items()}), flush=True)

# ---- standalone per-engine timing (isolated, trtexec-like): CUDA events, 200 iters after 50 warm ----
def standalone(e):
    ctx = e.execution_context; bufs = []
    for name, shape, dt in e.in_meta + e.out_meta:
        shp = tuple(ctx.get_tensor_shape(name)) if (name, shape, dt) in e.out_meta else tuple(max(1, s) for s in ctx.get_tensor_shape(name))
        x = (torch.zeros(shp, dtype=dt, device="cuda") if dt in (torch.int64, torch.int32) else torch.randn(shp, device="cuda").to(dt))
        ctx.set_tensor_address(name, x.data_ptr()); bufs.append(x)
    s = torch.cuda.current_stream()
    for _ in range(50): ctx.execute_async_v3(s.cuda_stream)
    st, en = torch.cuda.Event(True), torch.cuda.Event(True); st.record()
    for _ in range(200): ctx.execute_async_v3(s.cuda_stream)
    en.record(); torch.cuda.synchronize(); return st.elapsed_time(en) / 200
alone = {t: standalone(e) for t, e in E.items()}
print("standalone ms:", json.dumps({k: round(v, 3) for k, v in alone.items()}), flush=True)

json.dump({"trt": trt.__version__, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "engine_dir": ENG,
           "input": INPUT, "order": order, "blocks_ms": res, "engine_ms_per_call_in_pipeline": engms,
           "standalone_engine_ms": alone, "gate": gate}, open(OUT, "w"), indent=1)
print("DONE", flush=True)
