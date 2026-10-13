#!/usr/bin/env python3
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_W = _gvrb_os.path.expanduser(_gvrb_os.environ.get("GVRB_WORK", "~/gvrb"))
import json, statistics as S
W=_W+"/trtwork"
recs=json.load(open(f"{W}/r_thor_groot_trt.json"))
by={r["tag"]:r for r in recs}
GATE=lambda r: (r["cosine"]>=0.99 and not r["all_zero"] and r["n_nan"]==0 and r["n_inf"]==0)
out={"board":"NVIDIA Jetson AGX Thor (Blackwell sm_110)","trt":"10.16.2",
 "method":("/usr/src/tensorrt/bin/trtexec --iterations=50 --warmUp=500 --avgRuns=50; metric = median "
           "GPU Compute Time; --exportOutput in the SAME invocation as the timing; gated vs an "
           "onnxruntime-CPU fp32 reference on the SAME .raw inputs used by the Orin and iq9 runs "
           "(refs byte-identical to the iq9-side refs)."),
 "gate_rule":"cosine>=0.99 and not all_zero and no nan/inf","engines":recs}
canon={}
for st in ["vit","llm","dit"]:
    r=by.get(f"{st}_st5")
    if not r: continue
    canon[st]=dict(flags="--stronglyTyped --builderOptimizationLevel=5",
        engine_md5=r["engine_md5"], median_ms=r["median_ms"], min_ms=r["min_ms"], max_ms=r["max_ms"],
        within_run_spread_pct=round(100*(r["max_ms"]-r["min_ms"])/r["median_ms"],2),
        cosine=r["cosine"], rel_mae=r["rel_mae"],
        cosine_vs_freshORT=r.get("cosine_vs_freshORT"), rel_mae_vs_freshORT=r.get("rel_mae_vs_freshORT"),
        all_zero=r["all_zero"],
        n_float_tensors=r["n_float_tensors"], gate_pass=GATE(r))
out["canonical_per_stage"]=canon
if len(canon)==3:
    v,l,d=canon["vit"]["median_ms"],canon["llm"]["median_ms"],canon["dit"]["median_ms"]
    ss=v+l+4*d
    out["groot_compiled_stage_sum_ms"]=round(ss,3); out["formula"]="ViT + LLM + 4 x DiT_step"
    eager={"vit":10.34,"llm":15.09,"dit4":34.28,"e2e":69.87}
    eager["stage_sum"]=eager["vit"]+eager["llm"]+eager["dit4"]
    out["eager_thor"]=eager
    out["eager_to_compiled_speedup"]={"vit":round(eager["vit"]/v,3),"llm":round(eager["llm"]/l,3),
        "dit4":round(eager["dit4"]/(4*d),3),"stage_sum":round(eager["stage_sum"]/ss,3)}
    carry=round(eager["e2e"]-eager["stage_sum"],3)
    out["nongraph_carryover_ms_from_eager"]=carry
    out["groot_compiled_e2e_ms_DERIVED"]=round(ss+carry,3)
    out["e2e_speedup_DERIVED"]=round(eager["e2e"]/(ss+carry),3)
trap={}
for st in ["vit","llm","dit"]:
    a,b=by.get(f"{st}_fp16"),by.get(f"{st}_st5")
    if not(a and b): continue
    trap[st]=dict(fp16_cosine=a["cosine"],fp16_rel_mae=a["rel_mae"],fp16_n_float_tensors=a["n_float_tensors"],
        fp16_median_ms=a["median_ms"],fp16_all_zero=a["all_zero"],fp16_n_nan=a["n_nan"],
        fp16_cosine_vs_freshORT=a.get("cosine_vs_freshORT"),st5_cosine_vs_freshORT=b.get("cosine_vs_freshORT"),
        fp16_trtexec_verdict=a["trtexec_verdict"],
        st5_cosine=b["cosine"],st5_n_float_tensors=b["n_float_tensors"],st5_median_ms=b["median_ms"],
        TRAP_BITES=a["cosine"]<0.99,
        fp16_speed_advantage_pct=round(100*(b["median_ms"]-a["median_ms"])/b["median_ms"],2))
out["fp16_trap"]=trap
json.dump(out,open(f"{W}/r_thor_groot_summary.json","w"),indent=1)
print(json.dumps({k:v for k,v in out.items() if k not in("engines",)},indent=1))
