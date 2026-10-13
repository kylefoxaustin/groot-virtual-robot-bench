#!/usr/bin/env python3
# Fresh default-fusion fp16 engine build (matches trtexec fusion) for apples-to-apples roofline.
import sys, tensorrt as trt
onnx=sys.argv[1]; out=sys.argv[2]
L=trt.Logger(trt.Logger.WARNING)
b=trt.Builder(L); net=b.create_network(0); p=trt.OnnxParser(net,L)
if not p.parse_from_file(onnx):
    for i in range(p.num_errors): print(p.get_error(i))
    sys.exit(1)
cfg=b.create_builder_config()
cfg.set_flag(trt.BuilderFlag.FP16)
# pin any dynamic dims to 1 (batch) via optimization profile
prof=b.create_optimization_profile(); dyn=False
for i in range(net.num_inputs):
    t=net.get_input(i); sh=list(t.shape)
    if any(d<0 for d in sh):
        dyn=True; fixed=tuple(1 if d<0 else d for d in sh)
        prof.set_shape(t.name, fixed, fixed, fixed)
        print("pinned", t.name, sh, "->", fixed)
if dyn: cfg.add_optimization_profile(prof)
eng=b.build_serialized_network(net,cfg)
assert eng, "build failed"
open(out,"wb").write(eng); print("wrote", out, len(eng),"bytes")
