#!/usr/bin/env python3
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_W = _gvrb_os.path.expanduser(_gvrb_os.environ.get("GVRB_WORK", "~/gvrb"))
import json, statistics as S
W=_W+"/trtwork5090"
recs=[]
for f in ["r_vit.json","r_llm.json","r_dit.json","r_rep2.json","r_rep3.json","r_rep4.json","r_rep5.json"]:
    recs+= json.load(open(f"{W}/{f}"))
r2r=json.load(open(f"{W}/r_5090_run2run.json"))
by={}
for r in recs:
    if "ERROR" in r: continue
    by.setdefault((r["stage"],r["tag"].split("_",1)[1]),[]).append(r)

GATE=lambda r: (r["cosine"]>=0.99 and not r["all_zero"] and r["n_nan"]==0 and r["n_inf"]==0)
out={"board":"NVIDIA GeForce RTX 5090","trt":"10.16.1.11 (Python API — NO trtexec on this host)",
     "torch":"2.11.0+cu130",
     "method":("TensorRT Python API: build from ONNX with an explicit min=opt=max optimization profile "
               "identical to the Orin --minShapes/--optShapes/--maxShapes; timing = cudaEvents bracketing "
               "execute_async_v3 ONLY (no H2D/D2H) == trtexec 'GPU Compute Time'; warmUp>=500 ms, "
               "iterations=50, median reported; output read back and gated in the SAME process as the timing "
               "against an onnxruntime-CPU fp32 reference on the SAME .raw inputs the Orin/iq9 runs used."),
     "gate_rule":"cosine>=0.99 and not all_zero and no nan/inf",
     "engines":[r for r in recs if "ERROR" not in r],
     "run_to_run_same_engine": r2r}

# canonical = st5 (the variant the Orin headline used), median across independent BUILDS
canon={}
for st in ["vit","llm","dit"]:
    ms=[r["median_ms"] for r in by[(st,"st5")]] + [r2r[st]["mean"]]
    g=[r for r in by[(st,"st5")]]
    canon[st]=dict(flags="stronglyTyped builderOptimizationLevel=5",
        builds=len(ms), medians_ms=sorted(round(x,4) for x in ms),
        canonical_median_ms=round(S.median(ms),4),
        build_to_build_spread_pct=round(100*(max(ms)-min(ms))/S.median(ms),2),
        run_to_run_spread_pct=r2r[st]["run_to_run_spread_pct"],
        engine_md5s=[r["engine_md5"] for r in g],
        cosine=g[0]["cosine"], rel_mae=g[0]["rel_mae"], all_zero=g[0]["all_zero"],
        n_float_tensors=g[0]["n_float_tensors"], gate_pass=GATE(g[0]))
out["canonical_per_stage"]=canon

v,l,d = canon["vit"]["canonical_median_ms"], canon["llm"]["canonical_median_ms"], canon["dit"]["canonical_median_ms"]
ss = v + l + 4*d
out["groot_compiled_stage_sum_ms"]=round(ss,3)
out["formula"]="ViT + LLM + 4 x DiT_step"

eager={"vit":4.97,"llm":7.0,"dit4":15.32,"e2e":30.65}
eager["stage_sum"]=eager["vit"]+eager["llm"]+eager["dit4"]
out["eager_5090"]=eager
out["eager_to_compiled_speedup"]={
  "vit":round(eager["vit"]/v,3), "llm":round(eager["llm"]/l,3),
  "dit4":round(eager["dit4"]/(4*d),3), "stage_sum":round(eager["stage_sum"]/ss,3)}
carry = round(eager["e2e"]-eager["stage_sum"],3)
out["nongraph_carryover_ms_from_eager"]=carry
out["groot_compiled_e2e_ms_DERIVED"]=round(ss+carry,3)
out["e2e_speedup_DERIVED"]=round(eager["e2e"]/(ss+carry),3)

# the fp16 trap
out["fp16_trap"]={st:{
   "fp16_cosine":by[(st,"fp16")][0]["cosine"], "fp16_rel_mae":by[(st,"fp16")][0]["rel_mae"],
   "fp16_n_float_tensors":by[(st,"fp16")][0]["n_float_tensors"],
   "fp16_median_ms":by[(st,"fp16")][0]["median_ms"],
   "st5_cosine":by[(st,"st5")][0]["cosine"], "st5_n_float_tensors":by[(st,"st5")][0]["n_float_tensors"],
   "st5_median_ms":by[(st,"st5")][0]["median_ms"],
   "fp16_all_zero":by[(st,"fp16")][0]["all_zero"], "fp16_n_nan":by[(st,"fp16")][0]["n_nan"],
   "TRAP_BITES": by[(st,"fp16")][0]["cosine"]<0.99,
   "fp16_speed_advantage_pct": round(100*(by[(st,"st5")][0]["median_ms"]-by[(st,"fp16")][0]["median_ms"])/by[(st,"st5")][0]["median_ms"],2),
  } for st in ["vit","llm","dit"]}
json.dump(out,open(f"{W}/r_5090_groot_trt.json","w"),indent=1)
print(json.dumps({k:out[k] for k in ["canonical_per_stage","groot_compiled_stage_sum_ms",
  "eager_to_compiled_speedup","groot_compiled_e2e_ms_DERIVED","e2e_speedup_DERIVED",
  "fp16_trap","nongraph_carryover_ms_from_eager"]},indent=1))
