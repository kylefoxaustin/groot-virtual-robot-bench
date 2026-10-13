import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_W = _gvrb_os.path.expanduser(_gvrb_os.environ.get("GVRB_WORK", "~/gvrb"))
import json,sys,numpy as np,re,os,hashlib
tag=sys.argv[1]
stage=re.match(r"(vit|llm|dit)",tag).group(1)
O=_W+"/trt"; REF=_W+"/trtwork"   # REF: ref_<stage>.npy from mkref.py
d=json.load(open(f"{O}/{tag}.out.json"))
t=d[0]
o=np.array(t["values"],dtype=np.float64)
ref=np.load(f"{REF}/ref_{stage}.npy").astype(np.float64).ravel()
assert o.size==ref.size, (o.size,ref.size)
nan=int(np.isnan(o).sum()); inf=int(np.isinf(o).sum())
allzero = bool(np.all(o==0))
den=np.linalg.norm(o)*np.linalg.norm(ref)
cos=float(o@ref/den) if den>0 else float("nan")
rel=float(np.abs(o-ref).mean()/np.abs(ref).mean())
log=open(f"{O}/{tag}.log").read()
m=re.search(r"GPU Compute Time:.*?median = ([\d.]+) ms",log)
med=float(m.group(1)) if m else float("nan")
eng=f"{O}/{tag}.engine"
md5=hashlib.md5(open(eng,"rb").read()).hexdigest() if os.path.exists(eng) else "NA"
print(json.dumps(dict(tag=tag,stage=stage,out_name=t["name"],dims=t["dimensions"],
  median_gpu_ms=med,engine_md5=md5,engine_bytes=os.path.getsize(eng) if os.path.exists(eng) else 0,
  all_zero=allzero,n_nan=nan,n_inf=inf,cosine=round(cos,6),rel_mae=round(rel,6),
  out_absmean=round(float(np.abs(o).mean()),6),ref_absmean=round(float(np.abs(ref).mean()),6),
  out_min=round(float(o.min()),4),out_max=round(float(o.max()),4)),indent=1))
