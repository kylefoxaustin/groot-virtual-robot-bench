#!/usr/bin/env python3
"""Gate the trtexec-exported outputs against the fp32 onnxruntime-CPU reference."""
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_W = _gvrb_os.path.expanduser(_gvrb_os.environ.get("GVRB_WORK", "~/gvrb"))
import json, glob, os, hashlib, re, sys
import numpy as np
OUT=_W+"/trtout"; REF=_W+"/trtwork"
SHAPE={"vit":(1,256,1152),"llm":(1,296,2048),"dit":(1,49,1024)}
res=[]
for f in sorted(glob.glob(f"{OUT}/*.out.json")):
    tag=os.path.basename(f)[:-len(".out.json")]; stage=tag.split("_")[0]
    d=json.load(open(f))
    got=np.array(d[0]["values"],dtype=np.float32).reshape(SHAPE[stage])
    def _m(x,y):
        x=x.ravel().astype(np.float64); y=y.ravel().astype(np.float64)
        return (float(x@y/(np.linalg.norm(x)*np.linalg.norm(y))), float(np.abs(x-y).mean()/np.abs(y).mean()))
    ref=np.load(f"{REF}/ref_{stage}.npy").astype(np.float32).reshape(SHAPE[stage])
    cos,rel=_m(got,ref)
    fp=f"{REF}/freshref_{stage}.npy"
    cos2=rel2=None
    if os.path.exists(fp):
        cos2,rel2=_m(got,np.load(fp).astype(np.float32).reshape(SHAPE[stage]))
    log=open(f"{OUT}/{tag}.log").read()
    m=re.search(r"GPU Compute Time: min = ([\d.]+) ms, max = ([\d.]+) ms, mean = ([\d.]+) ms, median = ([\d.]+) ms",log)
    eng=f"{OUT}/{tag}.engine"
    md5=hashlib.md5(open(eng,'rb').read()).hexdigest() if os.path.exists(eng) else None
    nbytes=os.path.getsize(eng) if os.path.exists(eng) else None
    li=f"{OUT}/{tag}.layers.json"
    nf=nh=None
    if os.path.exists(li):
        t=open(li).read(); nf=t.count('"Float"'); nh=t.count('"Half"')
    r=dict(tag=tag,stage=stage,
        build_flags=("stronglyTyped builderOptimizationLevel=5" if tag.endswith("_st5") else
                     "fp16" if tag.endswith("_fp16") else "stronglyTyped"),
        engine_md5=md5, engine_bytes=nbytes,
        min_ms=float(m.group(1)) if m else None, max_ms=float(m.group(2)) if m else None,
        mean_ms=float(m.group(3)) if m else None, median_ms=float(m.group(4)) if m else None,
        n_float_tensors=nf, n_half_tensors=nh,
        out_shape=list(SHAPE[stage]), all_zero=bool(not np.any(got)),
        n_nan=int(np.isnan(got).sum()), n_inf=int(np.isinf(got).sum()),
        absmean=float(np.abs(got).mean()),
        cosine=round(cos,6), rel_mae=round(rel,6),
        cosine_vs_freshORT=(round(cos2,6) if cos2 is not None else None),
        rel_mae_vs_freshORT=(round(rel2,6) if rel2 is not None else None),
        trtexec_verdict="PASSED" if "&&&& PASSED" in log else "FAILED")
    res.append(r); print(json.dumps(r))
json.dump(res,open(f"{REF}/r_thor_groot_trt.json","w"),indent=1)
print("WROTE",f"{REF}/r_thor_groot_trt.json")
