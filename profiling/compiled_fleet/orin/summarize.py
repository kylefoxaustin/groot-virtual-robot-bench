import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_W = _gvrb_os.path.expanduser(_gvrb_os.environ.get("GVRB_WORK", "~/gvrb"))
import json,subprocess,sys,os,re,hashlib,numpy as np
O=_W+"/trt"; REF=_W+"/trtwork"   # REF: ref_<stage>.npy from mkref.py
def gate(tag):
    stage=re.match(r"(vit|llm|dit)",tag).group(1)
    d=json.load(open(f"{O}/{tag}.out.json"))[0]
    o=np.array(d["values"],dtype=np.float64)
    ref=np.load(f"{REF}/ref_{stage}.npy").astype(np.float64).ravel()
    log=open(f"{O}/{tag}.log").read()
    med=float(re.search(r"GPU Compute Time:.*?median = ([\d.]+) ms",log).group(1))
    mn=float(re.search(r"GPU Compute Time: min = ([\d.]+) ms",log).group(1))
    mx=float(re.search(r"GPU Compute Time:.*?max = ([\d.]+) ms",log).group(1))
    flags=re.search(r"# /usr/src/tensorrt/bin/trtexec (.*)",log.split("\n")[0]).group(1)
    extra=" ".join([f for f in flags.split() if f.startswith("--stronglyTyped") or f.startswith("--fp16") or f.startswith("--builderOptimizationLevel")])
    eng=f"{O}/{tag}.engine"
    return dict(tag=tag,stage=stage,build_flags=extra or "(none)",
      engine_md5=hashlib.md5(open(eng,"rb").read()).hexdigest(),
      engine_bytes=os.path.getsize(eng),
      median_ms=round(med,4),min_ms=round(mn,4),max_ms=round(mx,4),
      all_zero=bool(np.all(o==0)),n_nan=int(np.isnan(o).sum()),n_inf=int(np.isinf(o).sum()),
      cosine=round(float(o@ref/(np.linalg.norm(o)*np.linalg.norm(ref))),6),
      rel_mae=round(float(np.abs(o-ref).mean()/np.abs(ref).mean()),6))
tags=[t for t in ["vit_st","vit_st5","vit_fp16","llm_st","llm_st5","llm_fp16","dit_st","dit_st5","dit_fp16"] if os.path.exists(f"{O}/{t}.out.json")]
res=[gate(t) for t in tags]
GATE=lambda r: r["cosine"]>=0.99 and not r["all_zero"] and r["n_nan"]==0 and r["n_inf"]==0
best={}
for r in res:
    if not GATE(r): continue
    s=r["stage"]
    if s not in best or r["median_ms"]<best[s]["median_ms"]: best[s]=r
tot=best["vit"]["median_ms"]+best["llm"]["median_ms"]+4*best["dit"]["median_ms"]
eager=dict(vit=42.196,llm=55.857,dit4=125.068,e2e=243.69,transforms=13.358,prepare=2.85,other=1.231,post=1.02)
out=dict(
  board="Jetson AGX Orin 64GB devkit", trt="10.3.0.30", nvpmodel="MAXN(0)", jetson_clocks="NOT applied (no sudo); governor=schedutil",
  method="trtexec --iterations=50 --warmUp=500 --avgRuns=50; metric = median GPU Compute Time; output exported in the SAME run (--exportOutput) and gated vs an onnxruntime-CPU fp32 reference on the SAME inputs",
  engines=res, gate_rule="cosine>=0.99 and not all_zero and no nan/inf",
  best_correct_per_stage={k:dict(tag=v["tag"],flags=v["build_flags"],median_ms=v["median_ms"],cosine=v["cosine"],rel_mae=v["rel_mae"]) for k,v in best.items()},
  groot_compiled_total_ms=round(tot,3),
  formula="ViT + LLM + 4 x DiT_step",
  eager_orin=eager,
  eager_to_compiled={
    "vit":round(eager["vit"]/best["vit"]["median_ms"],3),
    "llm":round(eager["llm"]/best["llm"]["median_ms"],3),
    "dit4":round(eager["dit4"]/(4*best["dit"]["median_ms"]),3),
    "stage_sum":round((eager["vit"]+eager["llm"]+eager["dit4"])/tot,3)},
  )
json.dump(out,open(_W+"/r_orin_groot_trt.json","w"),indent=1)
print(json.dumps({k:out[k] for k in ["best_correct_per_stage","groot_compiled_total_ms","eager_to_compiled"]},indent=1))
