#!/usr/bin/env python3
# fp16 engine build pinning named dynamic dims to TARGET values.
# Usage: build_fp16_engine_pin.py <onnx> <out.engine> name=VAL [name=VAL ...]
# Any dynamic dim whose dim_param is not listed is pinned to 1.
import sys, tensorrt as trt
onnx=sys.argv[1]; out=sys.argv[2]
targets={}
for a in sys.argv[3:]:
    k,v=a.split("="); targets[k]=int(v)
L=trt.Logger(trt.Logger.WARNING)
b=trt.Builder(L); net=b.create_network(0); p=trt.OnnxParser(net,L)
if not p.parse_from_file(onnx):
    for i in range(p.num_errors): print(p.get_error(i))
    sys.exit(1)
cfg=b.create_builder_config(); cfg.set_flag(trt.BuilderFlag.FP16)
# read dim_param names straight from onnx to map dynamic positions -> target
import onnx as O
m=O.load(onnx, load_external_data=False)
pm={i.name:[ (d.dim_value if (d.dim_value and not d.dim_param) else d.dim_param) for d in i.type.tensor_type.shape.dim] for i in m.graph.input}
prof=b.create_optimization_profile(); dyn=False
for i in range(net.num_inputs):
    t=net.get_input(i); sh=list(t.shape); params=pm.get(t.name, sh)
    if any(d<0 for d in sh):
        dyn=True; fixed=[]
        for pos,d in enumerate(sh):
            if d>=0: fixed.append(d); continue
            pname=params[pos] if pos<len(params) else None
            fixed.append(targets.get(pname,1))
        fixed=tuple(fixed)
        prof.set_shape(t.name, fixed, fixed, fixed)
        print("pinned", t.name, sh, "params", params, "->", fixed)
if dyn: cfg.add_optimization_profile(prof)
eng=b.build_serialized_network(net,cfg)
assert eng, "build failed"
open(out,"wb").write(eng); print("wrote", out)
