"""2 arms (action_dim 32) vs 4 arms (synthetic action_dim 64): time the two DoF-dependent action-head graphs.
Multi-build on purpose: a single TensorRT build is a tactic lottery for these tiny graphs (one engine build can
swing ~2x), so each (stage, dim) gets NB fresh builds, each timed as the CUDA-event mean of 300 executes.
Reports per-build samples, the across-build mean and the build spread; a delta inside the spread is "FREE".
usage: $GROOT_PY speed/arms_2v4/bench_2v4.py BOARD_LABEL [N_BUILDS=5]   (env: GROOT_ONNX; run widen_action_head_onnx.py first)
Needs torch + tensorrt python. On a board without torch (our Thor host), use bench_2v4_trtexec.sh instead."""
import os, sys, json, numpy as np, tensorrt as trt, torch, statistics as st
BOARD=sys.argv[1] if len(sys.argv)>1 else "GPU"; NB=int(sys.argv[2]) if len(sys.argv)>2 else 5
L=trt.Logger(trt.Logger.ERROR)
AH=os.path.join(os.environ.get("GROOT_ONNX") or sys.exit("GROOT_ONNX is not set -- run: source setup/env.sh"), "action_head")
STAGES={"action_encoder":{"2arm":f"{AH}/action_encoder.onnx","4arm":f"{AH}/action_encoder_4arm.onnx"},
        "action_decoder":{"2arm":f"{AH}/action_decoder.onnx","4arm":f"{AH}/action_decoder_4arm.onnx"}}
def build(onnx_path):
    b=trt.Builder(L); net=b.create_network(0); p=trt.OnnxParser(net,L); assert p.parse_from_file(onnx_path)
    cfg=b.create_builder_config(); cfg.set_flag(trt.BuilderFlag.FP16)
    pr=b.create_optimization_profile(); dyn=False
    for i in range(net.num_inputs):
        t=net.get_input(i); sh=list(t.shape)
        if any(d<0 for d in sh):
            dyn=True; fx=tuple(1 if d<0 else d for d in sh); pr.set_shape(t.name,fx,fx,fx)
    if dyn: cfg.add_optimization_profile(pr)
    return b.build_serialized_network(net,cfg)
def timeit(ser):
    eng=trt.Runtime(L).deserialize_cuda_engine(ser); ctx=eng.create_execution_context(); tn={}
    for i in range(eng.num_io_tensors):
        nm=eng.get_tensor_name(i); sh=tuple(ctx.get_tensor_shape(nm)); sh=tuple(1 if d<0 else d for d in sh)
        if any(d<0 for d in eng.get_tensor_shape(nm)): ctx.set_input_shape(nm,sh)
        dt=trt.nptype(eng.get_tensor_dtype(nm)); t=torch.zeros(sh,dtype=getattr(torch,np.dtype(dt).name),device="cuda")
        tn[nm]=t; ctx.set_tensor_address(nm,t.data_ptr())
    s=torch.cuda.Stream()
    for _ in range(30): ctx.execute_async_v3(s.cuda_stream)
    torch.cuda.synchronize(); e0,e1=torch.cuda.Event(True),torch.cuda.Event(True); e0.record(s)
    for _ in range(300): ctx.execute_async_v3(s.cuda_stream)
    e1.record(s); torch.cuda.synchronize(); return e0.elapsed_time(e1)/300
res={}
for stage,d in STAGES.items():
    res[stage]={}
    for dim,onx in d.items():
        ts=[timeit(build(onx)) for _ in range(NB)]
        res[stage][dim]={"builds":ts,"min":min(ts),"mean":st.mean(ts),"spread":max(ts)-min(ts)}
out={"board":BOARD,"method":"multi-build fp16 TRT, 5 fresh builds x 300-rep CUDA-event mean","n_builds":NB,"stages":res}
for stage in STAGES:
    m2=res[stage]["2arm"]["mean"]; m4=res[stage]["4arm"]["mean"]; sp=max(res[stage]["2arm"]["spread"],res[stage]["4arm"]["spread"])
    print(f"  {stage:16s} 2arm {m2:.4f}  4arm {m4:.4f}  Δ {m4-m2:+.4f} ms  (build spread ±{sp:.4f}) -> {'FREE' if abs(m4-m2)<=max(sp,0.002) else 'COST'}")
tot2=sum(res[s]['2arm']['mean'] for s in STAGES); tot4=sum(res[s]['4arm']['mean'] for s in STAGES)
out["action_head_dof_total_2arm"]=tot2; out["action_head_dof_total_4arm"]=tot4; out["delta_total"]=tot4-tot2
print(f"  >>> DoF total: 2arm {tot2:.4f} ms  4arm {tot4:.4f} ms  Δ {tot4-tot2:+.4f} ms")
json.dump(out,open(f"final_2v4_{BOARD}.json","w"),indent=2); print(f"wrote final_2v4_{BOARD}.json")
