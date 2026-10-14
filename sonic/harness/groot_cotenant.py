"""GPU co-tenant for the SONIC loaded arm: GR00T N1.5 on NVIDIA's full-pipeline TensorRT path (deployment_scripts/
trt_model_forward, all 7 engines, the same setup as speed/e2e_eager_hybrid_fulltrt.py) run CONTINUOUSLY back-to-back for
DURATION_S, logging every call's wall (CLOCK_MONOTONIC) and gating the output every 25th call vs the first full-TensorRT
call and once vs eager bf16 (same seed, same noise draw). Writes OUT_JSON.ready when the loop starts; stops early when
OUT_JSON.stop appears (graceful; never SIGKILL a process holding a CUDA context).
usage: $GROOT_PY sonic/harness/groot_cotenant.py ENGINE_DIR OUT_JSON DURATION_S
env (setup/env.sh): GROOT_REPO, GROOT_MODEL, GROOT_SHIM, GROOT_STEP_NPZ, JETSON=1 on Jetsons.  ENGINE_DIR = speed/build_engines.py output."""
import os, sys, time, json
from functools import partial
REPO = os.environ["GROOT_REPO"]
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "deployment_scripts"))
sys.path.insert(0, os.environ["GROOT_SHIM"])
ENG, OUT, DUR = sys.argv[1], sys.argv[2], float(sys.argv[3]); sys.argv = [sys.argv[0]]
import numpy as np, torch, tensorrt as trt
if os.environ.get("JETSON"):   # flash-attn has no aarch64 wheel: force sdpa (implementation swap only, as in the speed harness)
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
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy
import trt_torch
from trt_model_forward import eagle_tensorrt_forward, action_head_tensorrt_forward

print("loading policy...", flush=True)
dc = load_data_config("fourier_gr1_arms_only")
policy = Gr00tPolicy(model_path=os.environ["GROOT_MODEL"], embodiment_tag="gr1",
                     modality_config=dc.modality_config(), modality_transform=dc.transform(), denoising_steps=4, device="cuda")
z = np.load(os.environ["GROOT_STEP_NPZ"], allow_pickle=True); step = {k: z[k] for k in z.files}
INPUT = "GROOT_STEP_NPZ (speed/make_step0.py: PickNPlace ep0 step0, the speed benchmarks' input)"
m = policy.model; bb = m.backbone; ah = m.action_head
E = {t: trt_torch.Engine(f"{ENG}/{t}.engine") for t in
     ["vit_fp16", "llm_fp16", "DiT_fp16", "vlln_vl_self_attention", "state_encoder", "action_encoder", "action_decoder"]}
bb.num_patches = bb.eagle_model.vision_model.vision_model.embeddings.num_patches
bb.embedding_layer = bb.eagle_model.language_model.get_input_embeddings()
bb.image_token_index = bb.eagle_model.image_token_index
bb.vit_engine, bb.llm_engine = E["vit_fp16"], E["llm_fp16"]
ah.vlln_vl_self_attention_engine = E["vlln_vl_self_attention"]; ah.state_encoder_engine = E["state_encoder"]
ah.action_encoder_engine = E["action_encoder"]; ah.DiT_engine = E["DiT_fp16"]; ah.action_decoder_engine = E["action_decoder"]
# eager reference (bf16) + its noise draw
_cap = {}
_orandn = torch.randn
def _randn(*a, **k):
    r = _orandn(*a, **k)
    if "noise" not in _cap and r.dim() == 3 and r.shape[1:] == (ah.config.action_horizon, ah.config.action_dim): _cap["noise"] = r.detach().clone()
    return r
torch.randn = _randn
torch.manual_seed(1234)
with torch.no_grad(): ref_eager = policy.get_action(step)
torch.randn = _orandn
NOISE = _cap["noise"].to(torch.float16)
bb.forward = partial(eagle_tensorrt_forward, bb); ah.get_action = partial(action_head_tensorrt_forward, ah); ah.init_actions = NOISE
def rel(x, y):
    return {k: float(np.abs(np.asarray(y[k], np.float64) - np.asarray(x[k], np.float64)).mean() / max(np.abs(np.asarray(x[k], np.float64)).mean(), 1e-9)) for k in x}
for _ in range(5):
    torch.manual_seed(1234)
    with torch.no_grad(): ref_full = policy.get_action(step)
gate0 = rel(ref_eager, ref_full); print("GATE full_vs_eager", json.dumps({k: round(v, 5) for k, v in gate0.items()}), flush=True)
open(OUT + ".ready", "w").write(str(time.monotonic_ns()))
T, MS, G = [], [], []
t_end = time.monotonic() + DUR; i = 0
while time.monotonic() < t_end and not os.path.exists(OUT + ".stop"):
    torch.manual_seed(1234); torch.cuda.synchronize(); t0 = time.perf_counter(); tm = time.monotonic_ns()
    with torch.no_grad(): a = policy.get_action(step)
    torch.cuda.synchronize(); T.append(tm); MS.append((time.perf_counter() - t0) * 1000)
    if i % 25 == 0: G.append(max(rel(ref_full, a).values()))
    i += 1
ms = np.array(MS)
json.dump({"trt": trt.__version__, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "input": INPUT,
           "calls": int(ms.size), "duration_s": float((T[-1] - T[0]) / 1e9) if T else 0, "rate_hz": float(ms.size / ((T[-1] - T[0]) / 1e9 + ms[-1] / 1e3)) if T else 0,
           "call_ms": {"p50": float(np.median(ms)), "p90": float(np.percentile(ms, 90)), "p99": float(np.percentile(ms, 99)), "max": float(ms.max()), "min": float(ms.min()), "mean": float(ms.mean())},
           "gate_full_vs_eager_rel_mae": gate0, "gate_every25_max_rel_mae_vs_first_full": {"n": len(G), "max": float(max(G)) if G else None},
           "t_mono_ns_first": int(T[0]) if T else None, "t_mono_ns_last": int(T[-1]) if T else None,
           "call_ms_series": [round(x, 3) for x in MS], "t_rel_s_series": [round((t - T[0]) / 1e9, 4) for t in T]}, open(OUT, "w"))
print("DONE", ms.size, "calls", flush=True)
