#!/usr/bin/env python3
# fp16 TensorRT engine builder for hosts without trtexec (our RTX 5090 stand-in engines were built with it).
# usage: python3 drone/contention/build_trt.py ONNX ENGINE
import sys, tensorrt as trt
onnx, out = sys.argv[1], sys.argv[2]
LOG=trt.Logger(trt.Logger.WARNING)
b=trt.Builder(LOG)
net=b.create_network(0)                      # TRT 10: explicit batch is default
p=trt.OnnxParser(net,LOG)
if not p.parse(open(onnx,"rb").read()):
    for i in range(p.num_errors): print("PARSE ERR:",p.get_error(i))
    sys.exit(1)
cfg=b.create_builder_config()
cfg.set_flag(trt.BuilderFlag.FP16)
cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4<<30)
ser=b.build_serialized_network(net,cfg)
if ser is None: print("BUILD FAILED"); sys.exit(1)
data=bytes(ser)                              # IHostMemory has no len(); materialize to bytes
open(out,"wb").write(data); print("built",out,len(data),"bytes")
