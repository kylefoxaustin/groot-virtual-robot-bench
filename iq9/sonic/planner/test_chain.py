# CPU check that the split is exact: host stages + fp32 blocks == the full planner ONNX (run in the planner work dir).
import sys, os, numpy as np, onnxruntime as ort
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); from inputs import make; from chain import Chain
c=Chain()
so=ort.SessionOptions()
full=ort.InferenceSession('planner_sonic.onnx',so,providers=['CPUExecutionProvider'])
rng=np.random.default_rng(0)
for tag,inp in [('walk',make(2,-1.0)),('idle',make(0,-1.0,(0,0,0))),('run',make(3,-1.0)),('seed7',make(2,-1.0,seed=7)),('slow',make(1,-1.0,(0.7,0.7,0),(0,1,0)))]:
    q,n,t=c(inp); q0,n0=full.run(None,inp)
    print(tag,'npf',n,n0,'maxdiff',np.abs(q-q0).max(),{k:round(v*1e3,1) for k,v in t.items()})
