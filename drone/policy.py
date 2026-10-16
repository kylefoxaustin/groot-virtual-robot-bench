"""The GR00T drone policy: GR00T N1.5's vision-language backbone (frozen, unchanged) + a small trained action head.

One decision = raw 224 x 224 FPV frame + "fly to the <colour> pillar" + drone state [x, y, z, yaw]
  -> GR00T's own input transforms (host CPU; the frame is resized to 256 x 256 first, as in training)
  -> SigLIP ViT (TensorRT fp16) -> pixel shuffle + mlp1 + embedding + image-token scatter (torch; NVIDIA's
     deployment_scripts/trt_model_forward.eagle_tensorrt_forward, unmodified)
  -> Qwen LLM (TensorRT fp16, the 296-token engine; the 289-token drone prompt is right-padded with pad id 151643, mask 0)
  -> rows [:289] -> vlln + vl_self_attention (TensorRT fp16, a 289-token engine) -> mean over the 289 tokens (GPU, fp32)
  -> action head (numpy float64 on the host CPU, single-threaded BLAS; drone/assets/drone_head.npz)
  -> 16 x [vx, vy, vz, yaw_rate] body-frame velocity setpoints, 20 ms apart.
No diffusion transformer (DiT): the drone head replaces GR00T's action head. The pooled feature includes GR00T's
vlln + 4-layer vl_self_attention, because that is what the head was trained on.

backend "trt"   = the path above (the one every published rate, gate and closed-loop run of the NVIDIA boards used).
backend "eager" = Gr00tPolicy.get_action in bf16 with a hook on the backbone (the path the training features came from).

Environment (setup/env.sh sets all of them): GROOT_REPO, GROOT_MODEL (base N1.5), GROOT_SHIM, GROOT_STEP_NPZ, JETSON=1 on Jetson.
Set OPENBLAS_NUM_THREADS=1 BEFORE importing numpy (the entry scripts do): spinning BLAS workers steal host cores."""
import os, sys, time
import numpy as np

NPAD, NV, PAD_ID = 296, 289, 151643
HERE = os.path.dirname(os.path.abspath(__file__))


def _need(var):
    v = os.environ.get(var)
    if not v: sys.exit(f"{var} is not set -- run: source setup/env.sh && source drone/env.sh (see drone/README.md)")
    return v


REPO = _need("GROOT_REPO")
for p in (os.path.join(REPO, "deployment_scripts"), REPO, os.environ.get("GROOT_SHIM", "")):
    if p and p not in sys.path: sys.path.insert(0, p)
import torch, cv2  # noqa: E402

if os.environ.get("JETSON"):   # no flash-attn wheel on aarch64: ask transformers for SDPA (implementation swap only)
    import transformers as _tf
    _oc = _tf.AutoConfig.from_pretrained

    def _pc(*aa, **k):
        c = _oc(*aa, **k)
        for o in [c] + [v for v in vars(c).values() if hasattr(v, "_attn_implementation")]:
            try: o._attn_implementation = "sdpa"
            except Exception: pass
        return c
    _tf.AutoConfig.from_pretrained = _pc
    from transformers.models.siglip.modeling_siglip import SiglipVisionModel as _SVM
    _os = _SVM.__init__

    def _si(self, cfg, *aa, **k):
        cfg._attn_implementation = "sdpa"; cfg._attn_implementation_autoset = True
        return _os(self, cfg, *aa, **k)
    _SVM.__init__ = _si

from gr00t.experiment.data_config import load_data_config  # noqa: E402
from gr00t.model.policy import Gr00tPolicy, unsqueeze_dict_values  # noqa: E402
sys.path.insert(0, HERE)
import drone_head_np  # noqa: E402


class DronePolicy:
    def __init__(self, backend="trt", eng=None, vlln=None, head=None):
        self.backend = backend
        dc = load_data_config("fourier_gr1_arms_only")
        self.policy = Gr00tPolicy(model_path=_need("GROOT_MODEL"), embodiment_tag="gr1", modality_config=dc.modality_config(),
                                  modality_transform=dc.transform(), denoising_steps=4, device="cuda")
        self.m = self.policy.model; self.bb = self.m.backbone; self.dev = torch.device("cuda")
        self.head = drone_head_np.Head(head or os.path.join(os.environ.get("DRONE_ASSETS", os.path.join(HERE, "assets")), "drone_head.npz"))
        self.base = {k: v for k, v in np.load(_need("GROOT_STEP_NPZ"), allow_pickle=True).items() if not k.startswith("action.")}
        if os.environ.get("DRONE_TORCH_THREADS"):   # host co-tenant control only; default = torch's own default, as published
            torch.set_num_threads(int(os.environ["DRONE_TORCH_THREADS"]))
        if backend == "trt":
            import tensorrt as trt, trt_torch
            from trt_model_forward import eagle_tensorrt_forward
            self._etf = eagle_tensorrt_forward
            eng = eng or _need("DRONE_ENG")
            vlln = vlln or os.path.join(eng, "vlln289", "vlln_vl_self_attention.engine")
            for f in (f"{eng}/vit_fp16.engine", f"{eng}/llm_fp16.engine", vlln):
                if not os.path.exists(f): sys.exit(f"missing {f} -- build the engines first: bash drone/setup.sh (drone/README.md)")
            self.E = {"vit_fp16": trt_torch.Engine(f"{eng}/vit_fp16.engine"), "llm_fp16": trt_torch.Engine(f"{eng}/llm_fp16.engine"),
                      "vlln289": trt_torch.Engine(vlln)}
            # NVIDIA setup_tensorrt_engines attributes (module deletion skipped; unused eager modules stay resident)
            vm = self.bb.eagle_model.vision_model
            self.bb.num_patches = vm.vision_model.embeddings.num_patches
            self.bb.embedding_layer = self.bb.eagle_model.language_model.get_input_embeddings()
            self.bb.image_token_index = self.bb.eagle_model.image_token_index
            self.bb.vit_engine, self.bb.llm_engine = self.E["vit_fp16"], self.E["llm_fp16"]
            self.desc = f"TensorRT {trt.__version__}: vit_fp16 + llm_fp16 (296, right-padded) + vlln289 fp16"
        else:
            self._feat = {}; orig = self.bb.forward

            def cap(*a, **k):
                o = orig(*a, **k); self._feat["out"] = o; return o
            self.bb.forward = cap
            self.desc = "eager Gr00tPolicy.get_action (bf16, 4 denoise steps), backbone output pooled after vlln + vl_self_attention"
        self.meta = {"backend": backend, "desc": self.desc, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                     "torch_threads": torch.get_num_threads(), "openblas_threads_env": os.environ.get("OPENBLAS_NUM_THREADS")}
        if backend == "trt": self.meta["trt"] = trt.__version__; self.meta["engines"] = {k: e.file for k, e in self.E.items()}

    # ---- the decision path, split so the timed stages match the published records ----
    def obs(self, img224, instr):
        o = dict(self.base)
        o["video.ego_view"] = cv2.resize(img224, (256, 256)).reshape(1, 256, 256, 3).astype(np.uint8)
        o["annotation.human.action.task_description"] = np.array([instr])
        return o

    def transform(self, img224, instr):
        """exactly Gr00tPolicy.get_action's observation path"""
        c = unsqueeze_dict_values(self.obs(img224, instr))
        for k, v in c.items():
            if not isinstance(v, np.ndarray): c[k] = np.array(v)
        return self.policy.apply_transforms(c)

    def to_dev(self, x):   # GR00TN15.prepare_input: floats -> action-head dtype (bf16)
        return x.to(self.dev, dtype=self.m.action_head.dtype) if torch.is_floating_point(x) else x.to(self.dev)

    def pad_vl(self, vl):
        ids, am = vl["eagle_input_ids"], vl["eagle_attention_mask"]
        n = ids.shape[1]; assert n <= NPAD, n
        if n < NPAD:
            vl["eagle_input_ids"] = torch.cat([ids, torch.full((1, NPAD - n), PAD_ID, dtype=ids.dtype, device=self.dev)], 1)
            vl["eagle_attention_mask"] = torch.cat([am, torch.zeros((1, NPAD - n), dtype=am.dtype, device=self.dev)], 1)
        return n

    def feature_from_vl(self, vl, n):
        """backbone (eagle_tensorrt_forward) -> rows[:n] -> vlln289 -> mean. Pooled fp32 GPU tensor [1, 2048]."""
        bo = self._etf(self.bb, vl)
        llm = bo["backbone_features"][:, :n].contiguous()
        if llm.dtype != torch.float16: llm = llm.to(torch.float16)
        self.E["vlln289"].set_runtime_tensor_shape("backbone_features", llm.shape)
        return self.E["vlln289"](llm)["output"].float().mean(1)

    def vl_inputs(self, nrm, zero_pixels=False, padswap=False):
        vl = {k: self.to_dev(torch.as_tensor(v)) for k, v in nrm.items() if k.startswith("eagle_")}
        if zero_pixels: vl["eagle_pixel_values"] = torch.zeros_like(vl["eagle_pixel_values"])
        n = self.pad_vl(vl)
        if padswap: vl["eagle_input_ids"][:, n:] = 0
        return vl, n

    def feature(self, img224, instr, nrm=None, **kw):
        """pooled 2048-d feature (numpy fp32) of one frame + prompt"""
        if self.backend == "eager":
            with torch.inference_mode(): self.policy.get_action(self.obs(img224, instr))
            x = self._feat["out"]["backbone_features"]
            return x.float().mean(1).squeeze(0).cpu().numpy()
        nrm = self.transform(img224, instr) if nrm is None else nrm
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            vl, n = self.vl_inputs(nrm, **kw)
            return self.feature_from_vl(vl, n).squeeze(0).cpu().numpy()   # .cpu() syncs

    def decide(self, img224, instr, state):
        """one full-cost decision. Returns (actions [16, 4] float64, feature [2048] fp32, n_valid, stage ms)."""
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if self.backend == "eager":
            f = self.feature(img224, instr); n = NV; t1 = t0
        else:
            nrm = self.transform(img224, instr)
            t1 = time.perf_counter()
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                vl, n = self.vl_inputs(nrm)
                f = self.feature_from_vl(vl, n).squeeze(0).cpu().numpy()
        t2 = time.perf_counter()
        act = self.head(f, state)[0]
        t3 = time.perf_counter()
        return act, f, n, {"transform_ms": (t1 - t0) * 1e3, "gpu_policy_ms": (t2 - t1) * 1e3, "head_ms": (t3 - t2) * 1e3,
                           "total_ms": (t3 - t0) * 1e3}


def clocks():
    try:
        import subprocess
        return subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:
        return f"n/a ({e})"
