import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_W = _gvrb_os.path.expanduser(_gvrb_os.environ.get("GVRB_WORK", "~/gvrb"))
import json,os,re,hashlib,statistics as S
import numpy as np
OUT=_W+"/trtout"; REF=_W+"/trtwork"
SHAPE={"vit":(1,256,1152),"llm":(1,296,2048),"dit":(1,49,1024)}
def m(x,y):
    x=x.ravel().astype(np.float64); y=y.ravel().astype(np.float64)
    return float(x@y/(np.linalg.norm(x)*np.linalg.norm(y))), float(np.abs(x-y).mean()/np.abs(y).mean())
res={}
for stage in ["vit","llm","dit"]:
    _fp=REF+"/freshref_"+stage+".npy"   # optional second reference; falls back to mkref.py's ref_<stage>.npy
    fresh=np.load(_fp if os.path.exists(_fp) else REF+"/ref_"+stage+".npy").astype(np.float32).reshape(SHAPE[stage])
    fleet=np.load(REF+"/ref_"+stage+".npy").astype(np.float32).reshape(SHAPE[stage])
    builds=[]
    for tag in [stage+"_st5", stage+"_st5b", stage+"_st5c"]:
        oj=OUT+"/"+tag+".out.json"; lg=OUT+"/"+tag+".log"; en=OUT+"/"+tag+".engine"
        if not os.path.exists(oj): continue
        got=np.array(json.load(open(oj))[0]["values"],dtype=np.float32).reshape(SHAPE[stage])
        log=open(lg).read()
        mm=re.search(r"GPU Compute Time: min = ([\d.]+) ms, max = ([\d.]+) ms, mean = ([\d.]+) ms, median = ([\d.]+) ms",log)
        cf,rf=m(got,fleet); co,ro=m(got,fresh)
        li=OUT+"/"+tag+".layers.json"
        nf=open(li).read().count('"Float"') if os.path.exists(li) else None
        builds.append(dict(tag=tag, engine_md5=hashlib.md5(open(en,"rb").read()).hexdigest(),
            engine_bytes=os.path.getsize(en),
            median_ms=float(mm.group(4)), min_ms=float(mm.group(1)), max_ms=float(mm.group(2)),
            cosine_vs_fleet_ref=round(cf,6), rel_mae_vs_fleet_ref=round(rf,6),
            cosine_vs_freshORT=round(co,6), rel_mae_vs_freshORT=round(ro,6),
            all_zero=bool(not np.any(got)), n_nan=int(np.isnan(got).sum()), n_inf=int(np.isinf(got).sum()),
            n_float_tensors=nf, trtexec_verdict="PASSED" if "&&&& PASSED" in log else "FAILED"))
    ms=[b["median_ms"] for b in builds]
    res[stage]=dict(builds=builds, n_builds=len(ms), medians_ms=[round(x,4) for x in ms],
        canonical_median_ms=round(S.median(ms),4),
        build_to_build_spread_pct=round(100*(max(ms)-min(ms))/S.median(ms),2))
v=res["vit"]["canonical_median_ms"]; l=res["llm"]["canonical_median_ms"]; d=res["dit"]["canonical_median_ms"]
ss=v+l+4*d
eager={"vit":10.335,"llm":15.088,"dit4":34.28,"e2e":69.869}; eager["stage_sum"]=round(eager["vit"]+eager["llm"]+eager["dit4"],3)
carry=round(eager["e2e"]-eager["stage_sum"],3)
out=dict(board="NVIDIA Jetson AGX Thor (Blackwell sm_110)", trt="10.16.2",
  method="trtexec --iterations=50 --warmUp=500 --avgRuns=50; median GPU Compute Time; --exportOutput in the SAME invocation as the timing; canonical = MEDIAN of 3 independent --stronglyTyped --builderOptimizationLevel=5 builds",
  per_stage=res, groot_compiled_stage_sum_ms=round(ss,3), formula="ViT + LLM + 4 x DiT_step",
  eager_thor=eager,
  eager_to_compiled_speedup={"vit":round(eager["vit"]/v,3),"llm":round(eager["llm"]/l,3),
     "dit4":round(eager["dit4"]/(4*d),3),"stage_sum":round(eager["stage_sum"]/ss,3)},
  nongraph_carryover_ms=carry, groot_compiled_e2e_ms_DERIVED=round(ss+carry,3),
  e2e_speedup_DERIVED=round(eager["e2e"]/(ss+carry),3))
json.dump(out,open(REF+"/r_thor_groot_FINAL.json","w"),indent=1)
for s in ["vit","llm","dit"]:
    r=res[s]; b=r["builds"][0]
    print("%-4s n=%d %s -> %.4f ms  b2b=%.2f%%  cos_fleet=%.6f cos_fresh=%.6f allzero=%s nFloat=%s" % (
        s, r["n_builds"], r["medians_ms"], r["canonical_median_ms"], r["build_to_build_spread_pct"],
        b["cosine_vs_fleet_ref"], b["cosine_vs_freshORT"], b["all_zero"], b["n_float_tensors"]))
print()
print("STAGE-SUM", out["groot_compiled_stage_sum_ms"], "ms")
print("speedup  ", out["eager_to_compiled_speedup"])
print("e2e DERIV", out["groot_compiled_e2e_ms_DERIVED"], "ms  speedup", out["e2e_speedup_DERIVED"])
