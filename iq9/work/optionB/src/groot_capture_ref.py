#!/usr/bin/env python3
"""GR00T N1.5-3B — HOST capture + e2e reference for the iq9 board gate (x86 host, $GROOT_PY).

Resolves host-side blockers:
  G1  capture REAL board inputs (pixel/position_ids/input_ids/attention_mask/state/
      embodiment_id) AND the init-noise the golden uses (torch.randn intercepted).
  G2  build the board FUNCTION on host (fp16 ONNX graphs + numpy glue) and gate it e2e
      against a fresh same-stack golden.
  G3  read image_token_index from the model (not a guess).
  G4  confirm num_video_views / patch count from the real pixel_values.
  G5  DiT precision on host with MATCHED inputs/precision (part of the e2e gate).

Outputs (optionB/io/): pixel_values.raw, position_ids.raw, input_ids.raw,
attention_mask.raw, state.raw, embodiment_id.raw, init_actions.raw, and
groot_fresh_golden.npy (normalized action_pred, 1x16x32) + host_e2e_action.npy.
"""
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_W = _gvrb_os.path.expanduser(_gvrb_os.environ.get("GVRB_WORK", "~/gvrb"))
import os, sys, json, numpy as np, torch

REPO = _gvrb_os.environ.get("GROOT_REPO", _W+"/Isaac-GR00T")
sys.path.insert(0, REPO)
MODEL = _gvrb_os.environ.get("GROOT_MODEL", _W+"/models/GR00T-N1.5-3B")
IO = _W+"/optionB/io"
ONNX = _gvrb_os.environ.get("GROOT_ONNX", _W+"/onnx")
GLUE = _W+"/optionB/glue_constants"
os.makedirs(IO, exist_ok=True)

import gr00t
from gr00t.experiment.data_config import load_data_config
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.model.policy import Gr00tPolicy, COMPUTE_DTYPE, unsqueeze_dict_values

def npsave_raw(name, arr, dt):
    a = np.ascontiguousarray(arr).astype(dt)
    a.tofile(f"{IO}/{name}.raw")
    print(f"  saved {name}.raw shape={arr.shape} dt={dt.__name__} -> {a.nbytes} B")
    return a

def main():
    dc = load_data_config("fourier_gr1_arms_only")
    policy = Gr00tPolicy(model_path=MODEL, embodiment_tag="gr1",
                         modality_config=dc.modality_config(),
                         modality_transform=dc.transform(),
                         denoising_steps=4, device="cuda")
    model = policy.model
    ds_path = os.path.join(os.path.dirname(os.path.dirname(gr00t.__file__)),
                           "demo_data/robot_sim.PickNPlace")
    ds = LeRobotSingleDataset(dataset_path=ds_path, modality_configs=policy.modality_config,
                              video_backend="decord", video_backend_kwargs=None,
                              transforms=None, embodiment_tag="gr1")
    step = ds[0]

    eagle = model.backbone.eagle_model
    cfg = {}
    cfg["image_token_index"] = int(eagle.image_token_index)
    cfg["use_pixel_shuffle"] = bool(getattr(eagle, "use_pixel_shuffle", False))
    cfg["downsample_ratio"] = float(getattr(model.backbone, "downsample_ratio", -1))
    cfg["select_layer"] = int(model.backbone.select_layer)
    cfg["mlp1_type"] = type(eagle.mlp1).__name__
    cfg["mlp1_children"] = [f"{n}:{type(m).__name__}" for n, m in eagle.mlp1.named_children()]
    cfg["num_llm_layers"] = len(eagle.language_model.model.layers)
    cfg["compute_dtype"] = str(COMPUTE_DTYPE)
    print("MODEL CONFIG (G3/G4/pixel_shuffle):", json.dumps(cfg, indent=1))

    # ---- build normalized_input exactly as policy.get_action does ----
    obs = step.copy()
    is_batch = policy._check_state_is_batched(obs)
    if not is_batch:
        obs = unsqueeze_dict_values(obs)
    for k, v in obs.items():
        if not isinstance(v, np.ndarray):
            obs[k] = np.array(v)
    normalized_input = policy.apply_transforms(obs)

    # ---- capture the REAL board front-end inputs (G1) ----
    binp = model.backbone.prepare_input(normalized_input)
    ainp = model.action_head.prepare_input(normalized_input)
    def to_np(x):
        return x.detach().cpu().numpy()
    pixel_values = to_np(binp["eagle_pixel_values"]).astype(np.float32)   # (V,3,224,224)
    input_ids = to_np(binp["eagle_input_ids"]).astype(np.int64)           # (1,N)
    attention_mask = to_np(binp["eagle_attention_mask"]).astype(np.int64) # (1,N)
    state = to_np(ainp["state"]).astype(np.float32)                       # (1,1,64)
    embodiment_id = to_np(ainp["embodiment_id"]).astype(np.int64)         # (1,)
    n_views = pixel_values.shape[0]
    N = input_ids.shape[1]
    n_image_tokens = int((input_ids.reshape(-1) == cfg["image_token_index"]).sum())
    print(f"CAPTURE: views={n_views} pixel={pixel_values.shape} N={N} "
          f"n_image_tokens={n_image_tokens} state={state.shape} embodiment_id={embodiment_id.tolist()}")
    cfg["num_video_views"] = n_views; cfg["N_tokens"] = N; cfg["n_image_tokens"] = n_image_tokens
    assert n_views == 1, f"expected 1 view (gr1 ego_view), got {n_views}"
    assert n_image_tokens == 256, f"expected 256 image tokens, got {n_image_tokens}"

    # position_ids for the ViT patches = arange(256)
    position_ids = np.arange(256, dtype=np.int64).reshape(1, 256)

    # ---- fixed init-noise (seeded), intercepted so the golden is reproducible (G1) ----
    torch.manual_seed(0)
    ah = model.action_head
    B = 1
    init_noise = torch.randn(B, ah.config.action_horizon, ah.config.action_dim,
                             dtype=torch.float32)  # (1,16,32)
    init_noise_np = init_noise.numpy().astype(np.float32)

    real_randn = torch.randn
    def patched_randn(*a, **k):
        size = k.get("size", a[0] if a else None)
        if size is not None and tuple(size) == (B, ah.config.action_horizon, ah.config.action_dim):
            dev = k.get("device", "cpu"); dt = k.get("dtype", torch.float32)
            return init_noise.to(device=dev, dtype=dt)
        return real_randn(*a, **k)
    torch.randn = patched_randn
    try:
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
            model_pred = model.get_action(normalized_input)
        fresh_golden = model_pred["action_pred"].float().cpu().numpy()  # (1,16,32) normalized
    finally:
        torch.randn = real_randn
    print(f"FRESH GOLDEN (normalized action_pred): {fresh_golden.shape} std={fresh_golden.std():.4f}")

    # ---- save board inputs (G1) ----
    npsave_raw("pixel_values", pixel_values, np.float32)
    npsave_raw("position_ids", position_ids, np.int32)      # board baked int; verified via dumpGraphIO
    npsave_raw("input_ids", input_ids, np.int64)            # host-glue gather (int64)
    npsave_raw("attention_mask", attention_mask, np.int32)  # board LLM graph
    npsave_raw("state", state, np.float32)
    npsave_raw("embodiment_id", embodiment_id, np.int32)
    npsave_raw("init_actions", init_noise_np, np.float32)
    np.save(f"{IO}/groot_fresh_golden.npy", fresh_golden)
    np.save(f"{IO}/init_actions.npy", init_noise_np)

    # ==================================================================
    # HOST e2e reference = the board FUNCTION on host (fp16 ONNX + numpy glue) (G2/G5)
    # ==================================================================
    import onnxruntime as ort
    so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    def sess(p): return ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])
    s_vit = sess(f"{ONNX}/eagle2/vit_fp16.onnx")
    s_llm = sess(f"{ONNX}/eagle2/llm_fp16.onnx")
    s_vlln = sess(f"{ONNX}/action_head/vlln_vl_self_attention.onnx")
    s_state = sess(f"{ONNX}/action_head/state_encoder.onnx")
    s_aenc = sess(f"{ONNX}/action_head/action_encoder.onnx")
    s_adec = sess(f"{ONNX}/action_head/action_decoder.onnx")
    s_dit = sess(f"{ONNX}/action_head/DiT_fp16.onnx")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from host_glue_groot import GrootGlue
    g = GrootGlue(GLUE)

    # 1. ViT
    vit = s_vit.run(None, {"pixel_values": pixel_values.astype(np.float16),
                           "position_ids": position_ids.astype(np.int64)})[0].astype(np.float32)
    print("host ViT", vit.shape)
    # 2-3. mlp1 + embed/scatter
    proj = g.mlp1_project(vit.reshape(1, 256, 1152))
    inputs_embeds = g.embed_and_scatter(input_ids, proj, cfg["image_token_index"])
    # 4. LLM
    backbone = s_llm.run(None, {"inputs_embeds": inputs_embeds.astype(np.float16),
                                "attention_mask": attention_mask.astype(np.int64)})[0].astype(np.float32)
    # 5. vlln -> vl_embs
    vl_embs = s_vlln.run(None, {"backbone_features": backbone.astype(np.float16)})[0].astype(np.float32)
    # 6. state
    state_features = s_state.run(None, {"state": state.astype(np.float16),
                                        "embodiment_id": embodiment_id.astype(np.int64)})[0].astype(np.float32)
    # 7. denoise x4
    actions = init_noise_np.copy()
    for stp in range(g.num_steps):
        td = np.array([g.t_discretized(stp)], np.int64)
        af = s_aenc.run(None, {"actions": actions.astype(np.float16), "timesteps_tensor": td,
                               "embodiment_id": embodiment_id.astype(np.int64)})[0].astype(np.float32)
        af = g.add_action_pos_embed(af.reshape(1, 16, 1536))
        sa = g.assemble_sa_embs(state_features.reshape(1, 1, 1536), af)
        mo = s_dit.run(None, {"sa_embs": sa.astype(np.float16), "vl_embs": vl_embs.astype(np.float16),
                              "timesteps_tensor": td})[0].astype(np.float32)
        pred = s_adec.run(None, {"model_output": mo.astype(np.float16),
                                 "embodiment_id": embodiment_id.astype(np.int64)})[0].astype(np.float32)
        actions = g.euler(actions, pred.reshape(1, 49, 32))
    host_e2e = actions
    np.save(f"{IO}/host_e2e_action.npy", host_e2e)

    def gate(a, b):
        a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
        rel = float(np.mean(np.abs(a - b)) / (np.mean(np.abs(b)) + 1e-12))
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        return rel, cos
    rel, cos = gate(host_e2e, fresh_golden)
    a26, b26 = host_e2e[..., :26], fresh_golden[..., :26]
    rel26, cos26 = gate(a26, b26)
    print(f"HOST e2e (ONNX fp16 + numpy glue) vs FRESH bf16 golden: "
          f"rel_mae_32={rel:.5f} cos={cos:.6f} | rel_mae_26real={rel26:.5f} cos26={cos26:.6f}")

    cfg.update({"host_e2e_vs_golden_rel_mae_32": round(rel, 6),
                "host_e2e_vs_golden_cos": round(cos, 6),
                "host_e2e_vs_golden_rel_mae_26": round(rel26, 6),
                "golden_std": round(float(fresh_golden.std()), 5)})
    json.dump(cfg, open(f"{IO}/capture_config.json", "w"), indent=1)
    print("WROTE", f"{IO}/capture_config.json")
    print("DONE_CAPTURE")

if __name__ == "__main__":
    main()
