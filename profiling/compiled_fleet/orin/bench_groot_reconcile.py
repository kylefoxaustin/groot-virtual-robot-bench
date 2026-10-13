#!/usr/bin/env python3
"""GR00T N1.5 latency RECONCILIATION — every part measured in ONE pass, summing to the rate.

Every part of policy.get_action() timed separately, in the same pass:
    transforms  policy.apply_transforms(obs)          -- normalisation, image prep
    backbone    model.backbone(backbone_inputs)       -- Eagle VLM (SigLIP2-400M + Qwen3-1.7B)
    action_head model.action_head.get_action(...)     -- flow-matching DiT, 4 denoise steps
    post        _get_unnormalized_action(...)         -- unnormalise + device->host
    residual    e2e - sum(parts), MEASURED not inferred

Input: a recorded (simulated) GR1 PickNPlace observation from NVIDIA's demo_data/robot_sim.PickNPlace
-- recorded data, not synthetic.

16-action horizon, 4 denoise steps. Both rates are emitted so replan and action Hz are never confused.

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
    ap.add_argument("--dump-action", default=None,
                    help="save the seeded action_pred of rep 0 as .npy for the cross-board output gate")
    ap.add_argument("--power", default="")
    ap.add_argument("--attn-impl", default="",
                    help="force the Eagle backbone's attention implementation (sdpa|eager). "
                         "The checkpoint asks for flash_attention_2, which has no aarch64 wheels, "
                         "so the Jetsons need this. It is an implementation swap, and it is RECORDED "
                         "in the output rather than assumed away.")
    ap.add_argument("--step-npz", default=None,
                    help="pre-extracted recorded observation (npz, speed/make_step0.py). Lets a board run without a video "
                         "decoder installed — decode happens outside the timed region anyway, so "
                         "this removes a dependency without changing what is measured.")
    a = ap.parse_args()

    # NOTE: GR00T CANNOT BE IMPORTED AT ALL ON A CPU-ONLY TORCH WITHOUT THIS.
    # NVIDIA's Eagle remote code runs replace_vit_attn_with_flash_attn() at MODULE SCOPE
    # (radio_model.py:181), and that function's first line is
    #     cuda_major, cuda_minor = torch.cuda.get_device_capability()
    # which raises "Torch not compiled with CUDA enabled" before any of our code runs.
    # It patches the RADIO vision Attention class — which this checkpoint never uses; it is
    # a siglip_vision_model. So the whole call is dead code for us, executed at import.
    # Returning (0,0) takes the documented "capability < 8" branch: it warns and proceeds.
    if not torch.cuda.is_available():
        torch.cuda.get_device_capability = lambda *a_, **k_: (0, 0)

    if a.attn_impl:
        # Force the attention implementation at CONFIG-LOAD time. EagleBackbone builds its
        # model from AutoConfig.from_pretrained(...trust_remote_code=True), and the remote
        # config requests flash_attention_2. flash-attn has no aarch64 wheel, so on Orin and
        # Thor the choice is this override or nothing. sdpa is mathematically the same
        # attention; it is an IMPLEMENTATION swap, not a model change — but it is recorded
        # in the output so no reader has to infer it.
        import transformers
        _orig_cfg = transformers.AutoConfig.from_pretrained
        def _patched(*args, **kw):
            c = _orig_cfg(*args, **kw)
            for obj in [c] + [v for v in vars(c).values() if hasattr(v, "_attn_implementation")]:
                try: obj._attn_implementation = a.attn_impl
                except Exception: pass
            return c
        transformers.AutoConfig.from_pretrained = _patched

        # NOTE: THE VISION TOWER HARDCODES FLASH ATTENTION AND MUST BE INTERCEPTED SEPARATELY.
        # NVIDIA's Eagle remote code (modeling_eagle2_5_vl.py:111) does
        #     config.vision_config._attn_implementation = "flash_attention_2"
        # unconditionally for a siglip_vision_model, overriding whatever the config said.
        # flash-attn has no aarch64 wheel, so on Jetson that line alone blocks the model.
        # Note the LLM is NOT affected: the hard assert at :128 applies only to
        # Qwen2ForCausalLM, and this checkpoint is Qwen3 (:131), which carries no such
        # requirement. So this patch touches ONE line's effect, on ONE submodule.
        # sdpa is the same attention by a different kernel — an implementation swap.
        from transformers.models.siglip.modeling_siglip import SiglipVisionModel as _SVM
        _orig_svm_init = _SVM.__init__
        def _svm_init(self, cfg, *args, **kw):
            cfg._attn_implementation = a.attn_impl
            cfg._attn_implementation_autoset = True
            return _orig_svm_init(self, cfg, *args, **kw)
        _SVM.__init__ = _svm_init

    import gr00t
    from gr00t.experiment.data_config import load_data_config
    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.model.policy import Gr00tPolicy, unsqueeze_dict_values, squeeze_dict_values

    REPO = os.path.dirname(os.path.dirname(gr00t.__file__))
    ds_path = a.dataset_path or os.path.join(REPO, "demo_data/robot_sim.PickNPlace")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dc = load_data_config(a.data_config)
    policy = Gr00tPolicy(model_path=a.model_path, embodiment_tag=a.embodiment_tag,
                         modality_config=dc.modality_config(), modality_transform=dc.transform(),
                         denoising_steps=a.denoising_steps, device=dev)
    if a.step_npz:
        z = np.load(a.step_npz, allow_pickle=True)
        step = {k: z[k] for k in z.files}
        vid = [k for k in step if k.startswith("video")]
        if not vid:
            sys.exit("  ERROR: the npz carries no video.* key — refusing to benchmark a VLM with no image")
        print(f"  step from npz: {sorted(step)}; video {vid[0]} {step[vid[0]].shape} "
              f"range=[{step[vid[0]].min()},{step[vid[0]].max()}]", file=sys.stderr)
    else:
        ds = LeRobotSingleDataset(dataset_path=ds_path, modality_configs=policy.modality_config,
                                  video_backend="decord", video_backend_kwargs=None,
                                  transforms=None, embodiment_tag=a.embodiment_tag)
        step = ds[0]
    model = policy.model
    COMPUTE_DTYPE = torch.bfloat16
    cuda = dev == "cuda"
    def sync():
        if cuda: torch.cuda.synchronize()

    # SPLIT THE EAGLE BACKBONE into vision and LLM. In PyTorch the backbone runs as one
    # call, so the reconciliation reports a single "backbone_vlm" number — but the summary
    # table wants ViT and LLM separately, and on the Hexagon NPU they ARE separate graphs.
    # Timed with forward hooks on the two submodules so the split is measured on the same
    # basis as the fused figure, inside the same pass. Their sum will be slightly LESS than
    # backbone_vlm: the connector (mlp1) and the embedding merge live between them and
    # belong to neither — that difference shows up as backbone_other.
    _vt = {"vit": 0.0, "llm": 0.0}
    _mark = {}
    def _pre(tag):
        def f(m, i):
            if cuda: torch.cuda.synchronize()
            _mark[tag] = time.perf_counter()
        return f
    def _post(tag):
        def f(m, i, o):
            if cuda: torch.cuda.synchronize()
            _vt[tag] += time.perf_counter() - _mark[tag]
        return f
    try:
        _em = model.backbone.eagle_model
        _em.vision_model.register_forward_pre_hook(_pre("vit"))
        _em.vision_model.register_forward_hook(_post("vit"))
        _em.language_model.register_forward_pre_hook(_pre("llm"))
        _em.language_model.register_forward_hook(_post("llm"))
        _split_ok = True
    except Exception as _e:
        print(f"  WARNING: backbone split unavailable: {type(_e).__name__} {_e}", file=sys.stderr)
        _split_ok = False

    sampler = None
    if a.power:
        sys.path.insert(0, os.environ.get("POWER_SAMPLER_DIR", "."))
        RailSampler = __import__(os.environ.get("POWER_SAMPLER_MODULE", "rail_sampler")).RailSampler
        sampler = RailSampler([x.strip() for x in a.power.split(",") if x.strip()])
        if sampler.paths: sampler.start()
        else: sampler = None

    def one():
        t = {}
        obs = {k: (v if isinstance(v, np.ndarray) else np.array(v)) for k, v in step.items()}
        sync(); t0 = time.perf_counter()
        obs_b = unsqueeze_dict_values(obs)
        norm = policy.apply_transforms(obs_b)
        sync(); t["transforms_ms"] = (time.perf_counter()-t0)*1000

        with torch.inference_mode(), torch.autocast(device_type="cuda" if cuda else "cpu",
                                                    dtype=COMPUTE_DTYPE):
            sync(); s = time.perf_counter()
            bb_in, ah_in = model.prepare_input(norm)
            sync(); t["prepare_input_ms"] = (time.perf_counter()-s)*1000

            _vt["vit"] = 0.0; _vt["llm"] = 0.0
            sync(); s = time.perf_counter()
            bb_out = model.backbone(bb_in)
            sync(); t["backbone_vlm_ms"] = (time.perf_counter()-s)*1000
            t["backbone_vit_ms"] = _vt["vit"]*1000
            t["backbone_llm_ms"] = _vt["llm"]*1000
            t["backbone_other_ms"] = t["backbone_vlm_ms"] - t["backbone_vit_ms"] - t["backbone_llm_ms"]

            # SEEDED for the cross-board OUTPUT GATE. The action head is flow-matching:
            # it starts from random noise, so two boards disagree even in exact arithmetic
            # unless the noise is identical. Seeding immediately before get_action makes
            # the action chunk a deterministic function of (weights, input) and therefore
            # comparable across boards. Seeding does NOT change the work done, so the
            # timing in this same call remains valid.
            torch.manual_seed(1234)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(1234)
            sync(); s = time.perf_counter()
            ah_out = model.action_head.get_action(bb_out, ah_in)
            sync(); t["action_head_dit_ms"] = (time.perf_counter()-s)*1000

        sync(); s = time.perf_counter()
        act = policy._get_unnormalized_action(ah_out["action_pred"].float())
        act = squeeze_dict_values(act)
        if a.dump_action:
            import numpy as _np
            _a = ah_out["action_pred"].float().detach().cpu().numpy()
            _np.save(a.dump_action, _a)
            print(f"[gate] action_pred {_a.shape} -> {a.dump_action}", flush=True)
        sync(); t["post_ms"] = (time.perf_counter()-s)*1000
        sync(); t["e2e_ms"] = (time.perf_counter()-t0)*1000
        parts = sum(t[k] for k in ("transforms_ms","prepare_input_ms","backbone_vlm_ms",
                                   "action_head_dit_ms","post_ms"))
        # backbone_vit/llm/other are a BREAKDOWN of backbone_vlm, not additional parts —
        # excluded from the sum so the residual stays meaningful.
        t["residual_ms"] = t["e2e_ms"] - parts
        return t, act

    for _ in range(a.warmup): one()
    runs, act = [], None
    for _ in range(a.reps):
        r, act = one(); runs.append(r)
    if sampler: sampler.stop_flag = True; sampler.join(timeout=2)

    med = {k: round(statistics.median(r[k] for r in runs), 3) for k in runs[0]}
    arr = np.concatenate([np.asarray(v).ravel() for v in act.values()])
    horizon = max((np.asarray(v).shape[0] for v in act.values()), default=None)
    out = {"label": a.label, "device": dev, "denoising_steps": a.denoising_steps,
           "attn_implementation_forced": a.attn_impl or "(checkpoint default: flash_attention_2)",
           "cpu_import_patch": (None if torch.cuda.is_available() else
              "torch.cuda.get_device_capability stubbed to (0,0). NVIDIA's Eagle code calls it at "
              "MODULE SCOPE via replace_vit_attn_with_flash_attn (radio_model.py:181), so GR00T "
              "cannot be IMPORTED on a CPU-only torch without it. The function patches the RADIO "
              "vision Attention, which this siglip_vision_model checkpoint never uses."),
           "_attn_substitution_note": ("NVIDIA's Eagle code hardcodes flash_attention_2 for the SigLIP "
              "vision tower (modeling_eagle2_5_vl.py:111) regardless of config. flash-attn has no "
              "aarch64 wheel, so on Jetson that line blocks the model outright. Intercepted at "
              "SiglipVisionModel.__init__ and set to the requested implementation. The LLM is "
              "untouched — the hard flash assert at :128 applies only to Qwen2 and this checkpoint is "
              "Qwen3. sdpa is the same attention computed by a different kernel."
              ) if a.attn_impl else None,
           "breakdown_ms_median": med,
           "sum_of_parts_ms": round(sum(med[k] for k in ("transforms_ms","prepare_input_ms",
                                        "backbone_vlm_ms","action_head_dit_ms","post_ms")), 3),
           "residual_ms": med["residual_ms"],
           "residual_pct_of_e2e": round(100.0*med["residual_ms"]/med["e2e_ms"], 2),
           "pct_of_e2e": {k: round(100.0*med[k]/med["e2e_ms"], 2)
                          for k in ("transforms_ms","prepare_input_ms","backbone_vlm_ms",
                                    "action_head_dit_ms","post_ms","residual_ms")},
           "replan_hz": round(1000.0/med["e2e_ms"], 2),
           "action_horizon": horizon,
           "action_hz": round(1000.0/med["e2e_ms"]*horizon, 1) if horizon else None,
           "OUTPUT_COHERENCE": {"finite": bool(np.isfinite(arr).all()),
                                "all_zero": bool(np.all(arr == 0)),
                                "range": [float(arr.min()), float(arr.max())],
                                "mean_abs": float(np.abs(arr).mean())},
           "input_provenance": "a recorded (simulated) GR1 PickNPlace observation from NVIDIA's "
                               "demo_data/robot_sim.PickNPlace (the dataset their own deployment "
                               "guide uses), not synthetic data.",
           "reps": a.reps, "warmup": a.warmup, "torch_version": torch.__version__,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if sampler: out["power_mW"] = sampler.window(0, 1e18)
    js = json.dumps(out, indent=1); print(js)
    if a.out: open(a.out, "w").write(js)

if __name__ == "__main__":
    main()
