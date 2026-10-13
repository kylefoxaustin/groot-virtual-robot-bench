#!/usr/bin/env python3
"""Write the per-stage input tensors the compiled_fleet scripts feed to the ViT / LLM / DiT engines
(trtexec --loadInputs, trt_groot.py, mkref.py), captured from one eager policy.get_action() on the
speed benchmarks' input step ($GROOT_STEP_NPZ, written by speed/make_step0.py).

Files written to OUTDIR (default $GVRB_WORK/trtwork/io), raw little-endian, in the engines' binding dtypes:
  vit_pixel_values.raw      float16  (1, 3, 224, 224)   input of the SigLIP vision tower
  vit_position_ids.raw      int64    (1, 256)           arange(256), as the hybrid / full-TensorRT paths feed it
  llm_inputs_embeds.raw     float16  (1, 296, 2048)     input embeddings of the Qwen3 LLM (image tokens merged)
  llm_attention_mask.raw    int64    (1, 296)
  dit_sa_embs.raw           float16  (1, 49, 1536)      DiT inputs at the FIRST denoise step
  dit_vl_embs.raw           float16  (1, 296, 2048)
  dit_timesteps_tensor.raw  int64    (1,)
The tensors are captured at the same module seams speed/e2e_eager_hybrid_fulltrt.py patches for its hybrid mode
(vision_model.forward, language_model.model.forward, action_head.model.forward), cast exactly as that script casts
them. The published runs used input files of this provenance (the same PickNPlace step) but their md5s are not
shipped, so files from this script are equivalent, not provably byte-identical. Timing of these static-shape
engines does not depend on tensor content; the output gates compare against a reference that mkref.py computes
from the SAME files, so a run is self-consistent.

usage: $GROOT_PY profiling/compiled_fleet/dump_stage_inputs.py [OUTDIR]
       (env: GROOT_REPO, GROOT_MODEL, GROOT_SHIM, GROOT_STEP_NPZ, JETSON=1 on a Jetson -- source setup/env.sh)"""
import inspect, os, sys


def _env(k):
    v = os.environ.get(k)
    if not v: sys.exit(f"{k} is not set -- run: source setup/env.sh (see docs/SETUP.md)")
    return v


OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.expanduser(os.environ.get("GVRB_WORK", "~/gvrb")), "trtwork", "io")
NPZ = _env("GROOT_STEP_NPZ")
if not os.path.exists(NPZ):
    sys.exit(f"missing {NPZ} -- run `$GROOT_PY speed/make_step0.py $GROOT_STEP_NPZ` first (docs/SPEED.md)")
REPO = _env("GROOT_REPO")
sys.path.insert(0, REPO); sys.path.insert(0, _env("GROOT_SHIM"))
import numpy as np, torch
if os.environ.get("JETSON"):   # flash-attn has no aarch64 wheel: force sdpa (same swap as the speed harnesses)
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

dc = load_data_config("fourier_gr1_arms_only")
policy = Gr00tPolicy(model_path=_env("GROOT_MODEL"), embodiment_tag="gr1", modality_config=dc.modality_config(),
                     modality_transform=dc.transform(), denoising_steps=4, device="cuda")
z = np.load(NPZ, allow_pickle=True); step = {k: z[k] for k in z.files}
m = policy.model; bb = m.backbone; ah = m.action_head
vm = bb.eagle_model.vision_model; lm = bb.eagle_model.language_model.model; dit = ah.model
CAP = {}
o_vm, o_lm, o_dit = vm.forward, lm.forward, dit.forward
_dit_names = list(inspect.signature(dit.forward).parameters)


def cap_vm(*a, **kw):
    if "vit_pixel_values" not in CAP:
        pv = kw.get("pixel_values", a[0] if a else None)
        CAP["vit_pixel_values"] = pv.to(torch.float16)
        CAP["vit_position_ids"] = torch.arange(256, dtype=torch.int64).reshape(1, 256)
    return o_vm(*a, **kw)


def cap_lm(*a, **kw):
    if "llm_inputs_embeds" not in CAP:
        ie = kw["inputs_embeds"]; am = kw.get("attention_mask")
        if am is None: am = torch.ones(1, ie.shape[1], dtype=torch.int64)
        CAP["llm_inputs_embeds"] = ie.to(torch.float16); CAP["llm_attention_mask"] = am.to(torch.int64)
    return o_lm(*a, **kw)


def cap_dit(*a, **kw):
    if "dit_sa_embs" not in CAP:   # first denoise step
        f = dict(zip(_dit_names, a)); f.update(kw)
        sa = f.get("hidden_states", f.get("sa_embs")); vl = f.get("encoder_hidden_states", f.get("vl_embs"))
        ts = f.get("timestep", f.get("timesteps", f.get("timesteps_tensor")))
        if ts.dim() == 0: ts = ts.reshape(1)
        CAP["dit_sa_embs"] = sa.to(torch.float16); CAP["dit_vl_embs"] = vl.to(torch.float16)
        CAP["dit_timesteps_tensor"] = ts.to(torch.int64)
    return o_dit(*a, **kw)


vm.forward, lm.forward, dit.forward = cap_vm, cap_lm, cap_dit
torch.manual_seed(1234)
with torch.no_grad():
    policy.get_action(step)

WANT = {"vit_pixel_values": ((1, 3, 224, 224), np.float16), "vit_position_ids": ((1, 256), np.int64),
        "llm_inputs_embeds": ((1, 296, 2048), np.float16), "llm_attention_mask": ((1, 296), np.int64),
        "dit_sa_embs": ((1, 49, 1536), np.float16), "dit_vl_embs": ((1, 296, 2048), np.float16),
        "dit_timesteps_tensor": ((1,), np.int64)}
os.makedirs(OUT, exist_ok=True); bad = []
for k, (shape, dt) in WANT.items():
    if k not in CAP: bad.append(f"{k}: not captured (module seam not called?)"); continue
    a = CAP[k].detach().cpu().numpy().astype(dt)
    if a.shape != shape: bad.append(f"{k}: captured shape {a.shape}, engines expect {shape}"); continue
    np.ascontiguousarray(a).tofile(os.path.join(OUT, k + ".raw"))
    print(f"wrote {OUT}/{k}.raw  {a.dtype} {a.shape}  absmean={float(np.abs(a.astype(np.float64)).mean()):.6g}")
if bad:
    sys.exit("ERROR: " + "; ".join(bad))
