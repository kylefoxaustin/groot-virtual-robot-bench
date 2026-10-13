#!/usr/bin/env python3
"""GR00T N1.5-3B stage engines: build + time + GATE, TensorRT Python API (RTX 5090).

Mirrors the Orin trtexec recipe:
  build variants: --stronglyTyped  /  --fp16   (+ builderOptimizationLevel)
  timing metric  : GPU Compute Time  (cudaEvents around enqueue only, NO H2D/D2H)
  iterations=50, warmUp>=500ms, median reported
  output exported in the SAME invocation as the timing and gated vs an
  onnxruntime-CPU fp32 reference on the SAME inputs.
"""
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_W = _gvrb_os.path.expanduser(_gvrb_os.environ.get("GVRB_WORK", "~/gvrb"))
import argparse, hashlib, json, os, time
import numpy as np, tensorrt as trt, torch

IO   = _gvrb_os.environ.get("GVRB_STAGE_IO", _W+"/trtwork/io")   # stage inputs: dump_stage_inputs.py
REF  = _W+"/trtwork"
SHAPES = {   # min=opt=max, identical to the Orin --minShapes/--optShapes/--maxShapes
 "vit": {"pixel_values": (1,3,224,224), "position_ids": (1,256)},
 "llm": {"inputs_embeds": (1,296,2048), "attention_mask": (1,296)},
 "dit": {"sa_embs": (1,49,1536), "vl_embs": (1,296,2048), "timesteps_tensor": (1,)},
}
ONNX = {
 "vit": _W+"/onnx/eagle2/vit_fp16.onnx",
 "llm": _W+"/onnx/eagle2/llm_fp16.onnx",
 "dit": _W+"/onnx/action_head/DiT_fp16.onnx",
}

def build(stage, strongly, oplevel, logger, workspace_gb=16):
    b = trt.Builder(logger)
    flags = (1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)) if strongly else 0
    net = b.create_network(flags)
    p = trt.OnnxParser(net, logger)
    if not p.parse_from_file(ONNX[stage]):          # resolves the .onnx.data sidecar
        for i in range(p.num_errors): print("PARSE:", p.get_error(i))
        raise SystemExit(f"parse failed {stage}")
    cfg = b.create_builder_config()
    prof = b.create_optimization_profile()
    for nm, shp in SHAPES[stage].items():
        prof.set_shape(nm, shp, shp, shp)          # min = opt = max, static
    cfg.add_optimization_profile(prof)
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if not strongly:
        cfg.set_flag(trt.BuilderFlag.FP16)
    if oplevel is not None:
        cfg.builder_optimization_level = oplevel
    t0 = time.time()
    plan = b.build_serialized_network(net, cfg)
    if plan is None: raise SystemExit(f"build failed {stage}")
    blob = bytes(plan)
    return blob, time.time() - t0

def load_inputs(stage, engine, ctx):
    """Feed the SAME .raw stage-input tensors the Orin/Thor runs use (dump_stage_inputs.py), in the BINDING dtype."""
    host = {}
    for i in range(engine.num_io_tensors):
        nm = engine.get_tensor_name(i)
        if engine.get_tensor_mode(nm) != trt.TensorIOMode.INPUT: continue
        dt = trt.nptype(engine.get_tensor_dtype(nm))
        shape = tuple(ctx.get_tensor_shape(nm))
        path = f"{IO}/{stage}_{nm}.raw"
        a = np.fromfile(path, dtype=dt)
        want = int(np.prod(shape))
        assert a.size == want, f"{stage}/{nm}: raw has {a.size} elems of {np.dtype(dt)}, binding wants {want} {shape}"
        host[nm] = a.reshape(shape)
    return host

def run(stage, tag, strongly, oplevel, iters, warmup_ms, logger, outdir):
    blob, build_s = build(stage, strongly, oplevel, logger)
    md5 = hashlib.md5(blob).hexdigest()
    eng = trt.Runtime(logger).deserialize_cuda_engine(blob)
    ctx = eng.create_execution_context()

    # --- layer-precision census: the RMSNorm-demotion tell from the fp16 trap ---
    try:
        insp = eng.create_engine_inspector(); insp.execution_context = ctx
        li = insp.get_engine_information(trt.LayerInformationFormat.JSON)
        n_float = li.count('"Float"')
        n_half  = li.count('"Half"')
    except Exception as e:
        n_float = n_half = -1

    host = load_inputs(stage, eng, ctx)
    dev, outs = {}, {}
    for i in range(eng.num_io_tensors):
        nm = eng.get_tensor_name(i)
        dt = trt.nptype(eng.get_tensor_dtype(nm))
        shape = tuple(ctx.get_tensor_shape(nm))
        t = torch.empty(shape, dtype=getattr(torch, np.dtype(dt).name), device="cuda")
        if eng.get_tensor_mode(nm) == trt.TensorIOMode.INPUT:
            t.copy_(torch.from_numpy(np.ascontiguousarray(host[nm])).cuda())
        else:
            outs[nm] = t
        dev[nm] = t
        ctx.set_tensor_address(nm, int(t.data_ptr()))

    st = torch.cuda.Stream()
    # ---- warmup: at least warmup_ms of real work (trtexec --warmUp semantics) ----
    torch.cuda.synchronize(); t0 = time.perf_counter(); nw = 0
    with torch.cuda.stream(st):
        while (time.perf_counter() - t0) * 1000 < warmup_ms or nw < 10:
            ctx.execute_async_v3(st.cuda_stream); nw += 1
            if nw % 20 == 0: torch.cuda.synchronize()
    torch.cuda.synchronize()

    # ---- timing: cudaEvents bracket the ENQUEUE only -> GPU Compute Time ----
    evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    with torch.cuda.stream(st):
        for a, b_ in evs:
            a.record(st); ctx.execute_async_v3(st.cuda_stream); b_.record(st)
    torch.cuda.synchronize()
    ms = sorted(a.elapsed_time(b_) for a, b_ in evs)

    # ---- gate: export output from the SAME run, compare to fp32 ORT-CPU ref ----
    onm = [n for n in outs][0]
    got = outs[onm].detach().float().cpu().numpy()
    ref = np.load(f"{REF}/ref_{stage}.npy").astype(np.float32).reshape(got.shape)
    a = got.ravel().astype(np.float64); b_ = ref.ravel().astype(np.float64)
    cos = float(a @ b_ / (np.linalg.norm(a) * np.linalg.norm(b_)))
    rel = float(np.abs(a - b_).mean() / np.abs(b_).mean())
    np.save(f"{outdir}/out_{tag}.npy", got)

    r = dict(tag=tag, stage=stage,
             build_flags=("stronglyTyped" if strongly else "fp16") + (f" builderOptimizationLevel={oplevel}" if oplevel is not None else ""),
             engine_md5=md5, engine_bytes=len(blob), build_s=round(build_s, 1),
             median_ms=round(ms[len(ms)//2], 4), min_ms=round(ms[0], 4), max_ms=round(ms[-1], 4),
             spread_pct=round(100*(ms[-1]-ms[0])/ms[len(ms)//2], 2),
             n_float_tensors=n_float, n_half_tensors=n_half,
             out_name=onm, out_shape=list(got.shape),
             all_zero=bool(not np.any(got)), n_nan=int(np.isnan(got).sum()), n_inf=int(np.isinf(got).sum()),
             absmean=float(np.abs(got).mean()),
             cosine=round(cos, 6), rel_mae=round(rel, 6),
             iters=iters, warmup_iters=nw)
    del ctx, eng, dev, outs; torch.cuda.empty_cache()
    return r

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="vit,llm,dit")
    ap.add_argument("--variants", default="st,st5,fp16")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup-ms", type=int, default=500)
    ap.add_argument("--outdir", default=_W+"/trtwork5090")
    ap.add_argument("--out", default=_W+"/trtwork5090/r_5090_groot_trt.json")
    a = ap.parse_args()
    L = trt.Logger(trt.Logger.ERROR)
    VAR = {"st": (True, None), "st5": (True, 5), "fp16": (False, None), "fp165": (False, 5), "st3": (True, 3), "st4": (True, 4)}
    res = []
    for s in a.stages.split(","):
        for v in a.variants.split(","):
            strongly, opl = VAR[v]
            tag = f"{s}_{v}"
            print(f"\n=== {tag} ===", flush=True)
            try:
                r = run(s, tag, strongly, opl, a.iters, a.warmup_ms, L, a.outdir)
            except SystemExit as e:
                r = dict(tag=tag, stage=s, ERROR=str(e)); print("FAIL", e, flush=True)
            except Exception as e:
                r = dict(tag=tag, stage=s, ERROR=f"{type(e).__name__}: {e}"); print("FAIL", r["ERROR"], flush=True)
            print(json.dumps(r, indent=1), flush=True)
            res.append(r)
            json.dump(res, open(a.out, "w"), indent=1)
    print("\nWROTE", a.out)
