"""Build the 7 GR00T N1.5 fp16 TensorRT engines NVIDIA's full-pipeline path uses (deployment_scripts/
trt_model_forward.setup_tensorrt_engines names), from the fp16 ONNX written by speed/export_onnx.sh, strongly typed, fixed batch-1 shapes.
usage: $GROOT_PY speed/build_engines.py OUTDIR [LLM_SEQ]
env:   GROOT_ONNX  directory holding eagle2/*.onnx and action_head/*.onnx (setup/env.sh sets it)
       ONLY        optional comma-separated subset of engine names to build, e.g. ONLY=vit_fp16,DiT_fp16
                   (names: vit_fp16, llm_fp16, vlln_vl_self_attention, state_encoder, action_encoder, DiT_fp16, action_decoder)"""
import os, sys, time, json
import tensorrt as trt
OUT = sys.argv[1]; SEQ = int(sys.argv[2]) if len(sys.argv) > 2 else 296
ONNX = os.environ.get("GROOT_ONNX") or sys.exit("GROOT_ONNX is not set -- run: source setup/env.sh (see docs/SETUP.md)")
SPEC = {
 "vit_fp16":   (f"{ONNX}/eagle2/vit_fp16.onnx", {"pixel_values": (1,3,224,224), "position_ids": (1,256)}),
 "llm_fp16":   (f"{ONNX}/eagle2/llm_fp16.onnx", {"inputs_embeds": (1,SEQ,2048), "attention_mask": (1,SEQ)}),
 "vlln_vl_self_attention": (f"{ONNX}/action_head/vlln_vl_self_attention.onnx", {"backbone_features": (1,SEQ,2048)}),
 "state_encoder":  (f"{ONNX}/action_head/state_encoder.onnx", {"state": (1,1,64), "embodiment_id": (1,)}),
 "action_encoder": (f"{ONNX}/action_head/action_encoder.onnx", {"actions": (1,16,32), "timesteps_tensor": (1,), "embodiment_id": (1,)}),
 "DiT_fp16":   (f"{ONNX}/action_head/DiT_fp16.onnx", {"sa_embs": (1,49,1536), "vl_embs": (1,SEQ,2048), "timesteps_tensor": (1,)}),
 "action_decoder": (f"{ONNX}/action_head/action_decoder.onnx", {"model_output": (1,49,1024), "embodiment_id": (1,)}),
}
only = os.environ.get("ONLY")
missing = [p for t, (p, _) in SPEC.items() if (not only or t in only.split(",")) and not os.path.exists(p)]
if missing:
    sys.exit("missing " + ", ".join(missing) + " -- run `bash speed/export_onnx.sh` and "
             "`$GROOT_PY speed/make_step0.py $GROOT_STEP_NPZ` first (docs/SPEED.md)")
os.makedirs(OUT, exist_ok=True)
L = trt.Logger(trt.Logger.ERROR); meta = {"trt": trt.__version__, "seq": SEQ, "engines": {}}
for tag, (onnx_path, shapes) in SPEC.items():
    if only and tag not in only.split(","): continue
    t0 = time.time()
    b = trt.Builder(L); net = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    p = trt.OnnxParser(net, L)
    if not p.parse_from_file(onnx_path):
        for i in range(p.num_errors): print("parse:", p.get_error(i))
        raise SystemExit(f"parse failed {tag}")
    cfg = b.create_builder_config(); prof = b.create_optimization_profile()
    for i in range(net.num_inputs):
        n = net.get_input(i).name; prof.set_shape(n, shapes[n], shapes[n], shapes[n])
    cfg.add_optimization_profile(prof)
    blob = b.build_serialized_network(net, cfg); assert blob is not None, f"build failed {tag}"
    open(f"{OUT}/{tag}.engine", "wb").write(blob)
    meta["engines"][tag] = {"onnx": onnx_path, "build_s": round(time.time() - t0, 1), "bytes": blob.nbytes}
    print(f"built {tag} in {time.time()-t0:.0f}s", flush=True)
json.dump(meta, open(f"{OUT}/build_meta.json", "w"), indent=1)
