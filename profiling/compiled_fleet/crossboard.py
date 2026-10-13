#!/usr/bin/env python3
"""Cross-board GR00T N1.5-3B COMPILED row (stage sum = ViT + LLM + 4 x DiT step). Prints TBD for any board not yet measured."""
import os as _gvrb_os  # paths: set by setup/env.sh (or export them yourself)
_W = _gvrb_os.path.expanduser(_gvrb_os.environ.get("GVRB_WORK", "~/gvrb"))
import json, os
W=_W+"/trtwork"
rows=[]
o=json.load(open(f"{W}/r_orin_groot_trt.json"))
b=o["best_correct_per_stage"]
rows.append(dict(board="Jetson AGX Orin 64GB", trt=o["trt"],
  vit=12.1953, llm=15.3326, dit=8.1194,   # the st5 variant the headline used
  eager_ss=o["eager_orin"]["vit"]+o["eager_orin"]["llm"]+o["eager_orin"]["dit4"]))
f=json.load(open(_W+"/trtwork5090/r_5090_groot_trt.json"))
c=f["canonical_per_stage"]
rows.append(dict(board="RTX 5090", trt=f["trt"].split()[0],
  vit=c["vit"]["canonical_median_ms"], llm=c["llm"]["canonical_median_ms"], dit=c["dit"]["canonical_median_ms"],
  eager_ss=f["eager_5090"]["stage_sum"]))
tp=f"{W}/r_thor_groot_FINAL.json"
if os.path.exists(tp):
    t=json.load(open(tp)); tc=t["per_stage"]
    rows.append(dict(board="Jetson AGX Thor", trt=t["trt"],
      vit=tc["vit"]["canonical_median_ms"], llm=tc["llm"]["canonical_median_ms"], dit=tc["dit"]["canonical_median_ms"],
      eager_ss=t["eager_thor"]["stage_sum"]))
else:
    rows.append(dict(board="Jetson AGX Thor", trt="10.16.2", vit=None,llm=None,dit=None,
                     eager_ss=10.34+15.09+34.28))
print(f"{'board':22s} {'TRT':10s} {'ViT':>8s} {'LLM':>8s} {'DiT/st':>8s} {'x4 DiT':>8s} {'STAGE-SUM':>10s} {'eager x':>8s}")
print("-"*100)
for r in rows:
    if r["vit"] is None:
        print(f"{r['board']:22s} {r['trt']:10s} {'TBD':>8s} {'TBD':>8s} {'TBD':>8s} {'TBD':>8s} {'TBD (GPU busy)':>10s}")
        continue
    ss=r["vit"]+r["llm"]+4*r["dit"]
    print(f"{r['board']:22s} {r['trt']:10s} {r['vit']:8.3f} {r['llm']:8.3f} {r['dit']:8.3f} {4*r['dit']:8.3f} {ss:10.3f} {r['eager_ss']/ss:8.2f}")
print("\nAll rows: --stronglyTyped --builderOptimizationLevel=5; median GPU Compute Time; ViT + LLM + 4 x DiT_step;")
print("gated vs the SAME onnxruntime-CPU fp32 reference (byte-identical to the iq9-side reference).")
