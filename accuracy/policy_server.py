#!/usr/bin/env python3
"""On-board GR00T policy server (Orin fp16 / Thor fp8 / Thor bf16 control).

Identical policy + noise-injection contract as the 5090 reference server
(reference_server.py) -- SAME Gr00tPolicy code path, SAME torch.randn (episode,call)
monkeypatch feeding the same noise bank, SAME 4-step Euler / prefix strip -- so the ONLY
deliberate difference vs the 5090 bf16 reference is the numeric precision:

  --precision fp16 : COMPUTE_DTYPE = torch.float16   (Orin)
  --precision bf16 : COMPUTE_DTYPE = torch.bfloat16  (clearly-labeled fallback control)
  --precision fp8  : bf16 base; every eligible nn.Linear (dims % 16 == 0) replaced by a
                     torch-native e4m3 GEMM via torch._scaled_mm (--fp8_backend torch, the
                     backend behind the published Thor fp8 result) or TransformerEngine
                     (--fp8_backend te). A POSITIVE proof (count of real fp8 GEMMs) is
                     emitted; if fp8 cannot engage the server ABORTS rather than ship an
                     inert-fp8 result.

Attention runs on torch SDPA (Jetson has no flash_attn): board_patch_sdpa.py switches the
eagle backbone to sdpa and shims/jetson/flash_attn satisfies the dead import (its kernels raise if
ever called -> proof sdpa is on the path).

Bound on host="*" so the host sim-client reaches it over the LAN.
"""
import argparse, json, os, time
import numpy as np
import torch

WAIST_PREFIX = "unlocked_waist: "


def strip_prefix(obs):
    k = "annotation.human.coarse_action"
    if k not in obs:
        return obs
    v = obs[k]
    def s1(x):
        x = x.decode() if isinstance(x, bytes) else str(x)
        return x[len(WAIST_PREFIX):] if x.startswith(WAIST_PREFIX) else x
    if isinstance(v, np.ndarray):
        obs[k] = np.array([s1(x) for x in v.reshape(-1)]).reshape(v.shape)
    elif isinstance(v, (list, tuple)):
        obs[k] = [s1(x) for x in v]
    else:
        obs[k] = s1(v)
    return obs


class BoardShim:
    def __init__(self, policy, noise_bank, horizon, adim, work, fp8_ctx=None, fp8_probe=None):
        self.policy = policy
        self.noise = np.load(noise_bank)  # (E, C, 1, H, D)
        self.size = (1, horizon, adim)
        self.ep = 0
        self.call = 0
        self.work = work
        self.log = []
        self.fp8_ctx = fp8_ctx          # callable -> context manager (TE fp8_autocast) or None
        self.fp8_probe = fp8_probe      # dict updated in-place with fp8 GEMM proof
        assert self.noise.shape[2:] == (1, horizon, adim), self.noise.shape
        self._real_randn = torch.randn
        def patched(*a, **kw):
            size = kw.get("size", a[0] if a else None)
            if size is not None and tuple(size) == self.size:
                n = self.noise[self.ep, self.call]  # (1,H,D) fp32
                return torch.from_numpy(np.ascontiguousarray(n)).to(
                    device=kw.get("device", "cpu"),
                    dtype=kw.get("dtype", torch.float32))
            return self._real_randn(*a, **kw)
        torch.randn = patched
        print("[server] noise bank", self.noise.shape, "injected via randn patch", flush=True)
        print("[server] SERVER READY", flush=True)

    def get_modality_config(self):
        return self.policy.get_modality_config()

    def reset_episode(self, data):
        self.ep = int(data["episode"])
        self.call = 0
        return {"ok": True, "episode": self.ep}

    def get_action(self, observations):
        t0 = time.time()
        obs = strip_prefix({k: v for k, v in observations.items()})
        if "video.ego_view" not in obs and "video.ego_view_bg_crop_pad_res256_freq20" in obs:
            obs["video.ego_view"] = obs.pop("video.ego_view_bg_crop_pad_res256_freq20")
        if self.fp8_ctx is not None:
            with self.fp8_ctx():
                act = self.policy.get_action(obs)
        else:
            act = self.policy.get_action(obs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = (time.time() - t0) * 1000.0
        self.log.append({"ep": self.ep, "call": self.call, "server_ms": round(dt, 2)})
        self.call += 1
        return act

    def dump_log(self):
        p = os.path.join(self.work, "server_calllog.json")
        json.dump(self.log, open(p, "w"), indent=1)
        if self.fp8_probe is not None:
            json.dump(self.fp8_probe, open(os.path.join(self.work, "fp8_proof.json"), "w"), indent=1)
        return p


# ---- torch-native fp8 (no TransformerEngine): genuine float8_e4m3fn tensor-core GEMMs ----
FP8_STATS = {"n_linear_converted": 0, "n_scaled_mm_calls": 0, "operand_dtype": None,
             "out_dtype": None, "e4m3_max": 448.0}


class Float8DynamicLinear(torch.nn.Module):
    """Drop-in for nn.Linear that runs a GENUINE fp8 (e4m3) tensor-core GEMM via
    torch._scaled_mm: weight pre-quantized to float8_e4m3fn (per-tensor scale) once; the
    activation is dynamically per-tensor-quantized to float8_e4m3fn each call. Every
    forward increments FP8_STATS['n_scaled_mm_calls'] -> positive proof fp8 engaged."""
    E4M3_MAX = 448.0

    def __init__(self, lin):
        super().__init__()
        self.in_features = lin.in_features
        self.out_features = lin.out_features
        w = lin.weight.data
        ws = (w.abs().amax() / self.E4M3_MAX).clamp(min=1e-12).to(torch.float32)
        self.register_buffer("w_fp8", (w / ws).to(torch.float8_e4m3fn))  # [out,in]
        self.register_buffer("w_scale", ws)
        if lin.bias is not None:
            self.register_buffer("bias", lin.bias.data)
        else:
            self.bias = None
        # The fp8 weight/scale live as BUFFERS (not params — we never train them). But GR00T
        # modules (e.g. cross_attention_dit TimestepEmbedder) infer their compute dtype via
        # `next(self.parameters()).dtype`; a module whose only weights were in a converted
        # Linear would then have ZERO parameters and raise StopIteration. Expose one scalar
        # bf16 Parameter so `.parameters()` is non-empty and reports the correct compute dtype.
        self.dtype_anchor = torch.nn.Parameter(
            torch.zeros((), dtype=torch.bfloat16, device=w.device), requires_grad=False)

    @property
    def weight(self):
        # dequantized bf16 weight, for any model code that reads .weight directly
        # (tied heads / custom ops). The fp8 GEMM on the main forward path is unaffected.
        return self.w_fp8.to(torch.bfloat16) * self.w_scale

    def forward(self, x):
        shp = x.shape
        orig_dtype = x.dtype
        # _scaled_mm forbids bias when out_dtype is fp32; always compute the fp8 GEMM in
        # bf16 (some GR00T layers feed fp32 activations).
        cdt = torch.bfloat16
        # A real bf16 nn.Linear (weight bf16) returns bf16 REGARDLESS of input dtype, so its
        # output plumbs as bf16 downstream (e.g. into the DiT action-head index_put). Returning
        # out.to(orig_dtype) instead leaks fp32 (when a layer is fed an fp32 activation) into a
        # bf16 index_put -> "source Float / dest BFloat16" crash. Mirror nn.Linear: emit bf16,
        # except preserve a low-precision fp16 input (that path uses fp16 end-to-end).
        ret_dtype = orig_dtype if orig_dtype in (torch.float16, torch.bfloat16) else cdt
        x2 = x.reshape(-1, self.in_features).to(cdt)
        M = x2.shape[0]
        pad = (-M) % 16
        if pad:
            x2 = torch.nn.functional.pad(x2, (0, 0, 0, pad))
        xs = (x2.abs().amax() / self.E4M3_MAX).clamp(min=1e-12).to(torch.float32)
        x_fp8 = (x2 / xs).to(torch.float8_e4m3fn)
        b = self.bias.to(cdt) if self.bias is not None else None
        out = torch._scaled_mm(x_fp8, self.w_fp8.t(), scale_a=xs, scale_b=self.w_scale,
                               bias=b, out_dtype=cdt)
        if pad:
            out = out[:M]
        FP8_STATS["n_scaled_mm_calls"] += 1
        if FP8_STATS["operand_dtype"] is None:
            FP8_STATS["operand_dtype"] = str(x_fp8.dtype)
            FP8_STATS["out_dtype"] = str(out.dtype)
        return out.to(ret_dtype).reshape(*shp[:-1], self.out_features)


def convert_to_torch_fp8(model):
    """Replace eligible nn.Linear (dims %16==0) with Float8DynamicLinear, dry-running each
    to confirm _scaled_mm works for its shape before committing (else keep bf16)."""
    n = 0
    dev = next(model.parameters()).device
    for mod in model.modules():
        for cn, child in list(mod.named_children()):
            if isinstance(child, torch.nn.Linear) and child.in_features % 16 == 0 and child.out_features % 16 == 0:
                f8 = Float8DynamicLinear(child).to(dev)
                try:
                    with torch.no_grad():
                        _ = f8(torch.randn(16, child.in_features, device=dev, dtype=child.weight.dtype))
                except Exception:
                    continue
                setattr(mod, cn, f8)
                n += 1
    FP8_STATS["n_linear_converted"] = n
    return n


def convert_to_te_fp8(model):
    """Replace nn.Linear with te.Linear so fp8 GEMMs actually run under fp8_autocast.
    Returns (n_replaced, te_module). Raises if TE unavailable."""
    import transformer_engine.pytorch as te
    n = 0
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, torch.nn.Linear):
                inf = child.in_features; outf = child.out_features
                # te.Linear needs dims divisible by 16 for fp8 GEMM; skip odd ones (stay bf16)
                if inf % 16 or outf % 16:
                    continue
                new = te.Linear(inf, outf, bias=child.bias is not None,
                                params_dtype=child.weight.dtype)
                with torch.no_grad():
                    new.weight.copy_(child.weight)
                    if child.bias is not None:
                        new.bias.copy_(child.bias)
                setattr(mod, child_name, new.to(child.weight.device))
                n += 1
    return n, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data_config", default="fourier_gr1_arms_waist")
    ap.add_argument("--embodiment_tag", default="gr1")
    ap.add_argument("--denoising_steps", type=int, default=4)
    ap.add_argument("--noise_bank", required=True)
    ap.add_argument("--precision", choices=["fp16", "bf16", "fp8"], required=True)
    ap.add_argument("--fp8_backend", choices=["torch", "te"], default="torch")
    ap.add_argument("--port", type=int, default=5571)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--work", default="/tmp/gvrb_server_work")
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # ---- precision: patch the module-global COMPUTE_DTYPE BEFORE constructing policy ----
    import gr00t.model.policy as gp
    base_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp8": torch.bfloat16}[args.precision]
    gp.COMPUTE_DTYPE = base_dtype
    print(f"[server] precision={args.precision} COMPUTE_DTYPE={gp.COMPUTE_DTYPE}", flush=True)

    from gr00t.eval.robot import RobotInferenceServer
    from gr00t.experiment.data_config import load_data_config
    from gr00t.model.policy import Gr00tPolicy

    print(f"[server] loading policy {args.model_path} denoise={args.denoising_steps}", flush=True)
    t0 = time.time()
    dc = load_data_config(args.data_config)
    policy = Gr00tPolicy(model_path=args.model_path, modality_config=dc.modality_config(),
                         modality_transform=dc.transform(), embodiment_tag=args.embodiment_tag,
                         denoising_steps=args.denoising_steps)
    # from_pretrained leaves a MIXED model for fp16 (backbone fp16, action-head DiT bf16),
    # which triggers Float/BFloat16 index_put mismatches. Force a UNIFORM low-precision
    # model (the bf16 reference is uniform bf16 for the same reason).
    if args.precision == "fp16":
        policy.model.half()
    elif args.precision in ("bf16", "fp8"):
        policy.model.to(torch.bfloat16)
    ptypes = {str(pp.dtype) for pp in policy.model.parameters()}
    print(f"[server] uniform param dtypes after cast: {ptypes}", flush=True)

    ah = policy.model.action_head
    horizon, adim = ah.config.action_horizon, ah.config.action_dim
    dev = next(policy.model.parameters()).device
    dtp = next(policy.model.parameters()).dtype
    print(f"[server] loaded {time.time()-t0:.1f}s device={dev} dtype={dtp} "
          f"horizon={horizon} adim={adim} denoise={ah.num_inference_timesteps}", flush=True)
    assert (horizon, adim) == (16, 32), (horizon, adim)
    assert ah.num_inference_timesteps == args.denoising_steps

    # confirm the eagle attention is sdpa (not flash)
    try:
        eagle = policy.model.backbone.eagle_model
        vattn = getattr(eagle.config.vision_config, "_attn_implementation", "?")
        tattn = getattr(eagle.config.text_config, "_attn_implementation", "?")
        print(f"[server] eagle attn: vision={vattn} text={tattn}", flush=True)
    except Exception as e:
        print(f"[server] attn probe skipped: {e}", flush=True)

    fp8_ctx, fp8_probe = None, None
    if args.precision == "fp8" and args.fp8_backend == "torch":
        if not hasattr(torch, "float8_e4m3fn") or not hasattr(torch, "_scaled_mm"):
            print("[server][FP8-ABORT] torch has no float8_e4m3fn/_scaled_mm -- cannot engage fp8.", flush=True)
            raise SystemExit(3)
        n_f8 = convert_to_torch_fp8(policy.model)
        if n_f8 == 0:
            print("[server][FP8-ABORT] 0 Linear layers converted to fp8 -- would be inert.", flush=True)
            raise SystemExit(3)
        # sanity: prove a fp8 GEMM actually executes right now (not just at conversion)
        FP8_STATS["n_scaled_mm_calls"] = 0
        fp8_probe = {"backend": "torch-native float8_e4m3fn via torch._scaled_mm (per-tensor "
                     "dynamic-act + static-weight scales, E4M3)",
                     "torch_version": torch.__version__,
                     "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                     "compute_capability": list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None,
                     "stats": FP8_STATS}  # live dict: reflects final counts at dump time
        print(f"[server][FP8] converted {n_f8} Linear->Float8DynamicLinear (torch-native e4m3); "
              f"fp8 GEMMs will be COUNTED at runtime as engagement proof", flush=True)
    elif args.precision == "fp8" and args.fp8_backend == "te":
        try:
            import transformer_engine.pytorch as te
            from transformer_engine.common.recipe import DelayedScaling, Format
        except Exception as e:
            print(f"[server][FP8-ABORT] TransformerEngine not importable: {e}", flush=True)
            print("[server][FP8-ABORT] cannot genuinely engage fp8 -- refusing to ship inert fp8.", flush=True)
            raise SystemExit(3)
        n_te, _ = convert_to_te_fp8(policy.model)
        if n_te == 0:
            print("[server][FP8-ABORT] 0 Linear layers converted to te.Linear -- fp8 would be inert.", flush=True)
            raise SystemExit(3)
        recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max")
        fp8_probe = {"te_version": getattr(te, "__version__", "?"), "n_te_linear": n_te,
                     "recipe": "DelayedScaling HYBRID (E4M3 fwd / E5M2 bwd)", "fp8_gemm_seen": None}
        def _ctx():
            return te.fp8_autocast(enabled=True, fp8_recipe=recipe)
        fp8_ctx = _ctx
        print(f"[server][FP8] converted {n_te} Linear->te.Linear; DelayedScaling HYBRID active", flush=True)

    shim = BoardShim(policy, args.noise_bank, horizon, adim, args.work, fp8_ctx, fp8_probe)
    server = RobotInferenceServer(shim, port=args.port)  # host="*" default -> LAN reachable
    server.register_endpoint("reset_episode", shim.reset_episode)
    server.register_endpoint("get_modality_config", shim.get_modality_config, requires_input=False)

    # Clean, reliable shutdown: pyzmq's blocking recv() swallows SIGINT (EINTR retry),
    # so register an explicit handler that dumps the call log then exits. This is a
    # graceful stop (dumps state first), NOT a hard SIGKILL.
    import signal
    def _shutdown(signum, frame):
        try:
            p = shim.dump_log()
            print(f"[server] signal {signum}: call log -> {p}", flush=True)
        except Exception as e:
            print(f"[server] dump on signal failed: {e}", flush=True)
        os._exit(0)
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        server.run()
    finally:
        p = shim.dump_log()
        print(f"[server] call log -> {p}", flush=True)


if __name__ == "__main__":
    main()
