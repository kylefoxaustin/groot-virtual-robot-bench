#!/usr/bin/env python3
"""What actually scales when a robot grows: arms vs cameras (RTX 5090, fp16 TensorRT, standalone engines).

  part 1  every action-head module timed alone (state encoder, action encoder, action decoder, VL self-attention, DiT)
          -> only the action encoder/decoder depend on the action dimension (= number of arms); each runs once per
             denoise step, so per inference the DoF-dependent cost is 4 x (encoder + decoder)
  part 2  the ViT with 1 / 2 / 4 camera views (batch of 224x224 views, 256 patches each)
  part 3  the LLM with 256 / 512 / 1024 vision tokens in context (1 / 2 / 4 views x 256)

Timing: one strongly-typed fp16 engine per shape, CUDA events around execute_async_v3, 200 iterations after 50 warm-up,
median. Engines are rebuilt here (build-to-build spread is a few %; run twice and compare).

usage: source setup/env.sh && $GROOT_PY speed/multiarm/camera_token_scaling.py OUT.json
NOTE: the 2026-10-10 published record (results/speed/multiarm_scaling_2026-10-10.json) came from an earlier, unpreserved
version of this measurement; this script re-implements its method. Expect agreement within build-to-build variance."""
import json, os, sys, time
import numpy as np, tensorrt as trt, torch

ONNX = os.environ.get("GROOT_ONNX") or sys.exit("GROOT_ONNX is not set -- source setup/env.sh")
OUT = sys.argv[1] if len(sys.argv) > 1 else "camera_token_scaling.json"
L = trt.Logger(trt.Logger.ERROR)


def build(onnx_path, shapes):
    b = trt.Builder(L); net = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    p = trt.OnnxParser(net, L)
    assert p.parse_from_file(onnx_path), [p.get_error(i) for i in range(p.num_errors)]
    cfg = b.create_builder_config(); prof = b.create_optimization_profile()
    for i in range(net.num_inputs):
        n = net.get_input(i).name; prof.set_shape(n, shapes[n], shapes[n], shapes[n])
    cfg.add_optimization_profile(prof)
    blob = b.build_serialized_network(net, cfg); assert blob is not None, onnx_path
    return trt.Runtime(L).deserialize_cuda_engine(blob)


def time_engine(eng, shapes, iters=200, warm=50):
    ctx = eng.create_execution_context(); keep = []
    for i in range(eng.num_io_tensors):
        n = eng.get_tensor_name(i)
        if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT: ctx.set_input_shape(n, shapes[n])
    for i in range(eng.num_io_tensors):
        n = eng.get_tensor_name(i); sh = tuple(ctx.get_tensor_shape(n))
        dt = torch.from_numpy(np.zeros(1, dtype=trt.nptype(eng.get_tensor_dtype(n)))).dtype
        t = (torch.randn(sh, device="cuda").to(dt) if dt.is_floating_point else torch.zeros(sh, dtype=dt, device="cuda"))
        if "position_ids" in n: t = torch.arange(sh[-1], device="cuda").expand(*sh).contiguous().to(dt)
        if "attention_mask" in n: t = torch.ones(sh, dtype=dt, device="cuda")
        keep.append(t); ctx.set_tensor_address(n, t.data_ptr())
    s = torch.cuda.Stream(); a, b = torch.cuda.Event(True), torch.cuda.Event(True); ms = []
    for k in range(warm + iters):
        a.record(s); ctx.execute_async_v3(s.cuda_stream); b.record(s); b.synchronize()
        if k >= warm: ms.append(a.elapsed_time(b))
    return round(float(np.median(ms)), 4)


def one(onnx_rel, shapes):
    return time_engine(build(f"{ONNX}/{onnx_rel}", shapes), shapes)


SEQ = 296
res = {"date": time.strftime("%Y-%m-%d"), "device": torch.cuda.get_device_name(0), "trt": trt.__version__,
       "method": "standalone fp16 strongly-typed engines, CUDA events, median of 200 after 50 warm-up", "tag": "MEASURED"}
head = {
    "state_encoder": ("action_head/state_encoder.onnx", {"state": (1, 1, 64), "embodiment_id": (1,)}),
    "action_encoder": ("action_head/action_encoder.onnx", {"actions": (1, 16, 32), "timesteps_tensor": (1,), "embodiment_id": (1,)}),
    "action_decoder": ("action_head/action_decoder.onnx", {"model_output": (1, 49, 1024), "embodiment_id": (1,)}),
    "vlln_vl_self_attention": ("action_head/vlln_vl_self_attention.onnx", {"backbone_features": (1, SEQ, 2048)}),
    "DiT_step": ("action_head/DiT_fp16.onnx", {"sa_embs": (1, 49, 1536), "vl_embs": (1, SEQ, 2048), "timesteps_tensor": (1,)}),
}
comp = {k: one(*v) for k, v in head.items()}
dof = comp["action_encoder"] + comp["action_decoder"]
res["part1_action_head_ms_per_call"] = comp
res["part1_dof_dependent"] = {"per_call_ms": round(dof, 4), "per_inference_ms_4_steps": round(4 * dof, 4),
                              "share_of_action_head_4_steps_pct": round(100 * 4 * dof / (comp["state_encoder"] + comp["vlln_vl_self_attention"]
                                                                                       + 4 * (dof + comp["DiT_step"])), 2)}
print(json.dumps(res["part1_action_head_ms_per_call"]), flush=True)
res["part2_vit_ms_by_views"] = {str(v): one("eagle2/vit_fp16.onnx", {"pixel_values": (v, 3, 224, 224), "position_ids": (v, 256)})
                                for v in (1, 2, 4)}
print(json.dumps(res["part2_vit_ms_by_views"]), flush=True)
res["part3_llm_ms_by_vision_tokens"] = {str(n): one("eagle2/llm_fp16.onnx", {"inputs_embeds": (1, n, 2048), "attention_mask": (1, n)})
                                        for n in (256, 512, 1024)}
print(json.dumps(res["part3_llm_ms_by_vision_tokens"]), flush=True)
json.dump(res, open(OUT, "w"), indent=1); print("wrote", OUT)
